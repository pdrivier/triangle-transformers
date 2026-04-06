"""
train.py  –  Multi-GPU DDP training for CANINEPhonemeLM.

Launch (multi-GPU, recommended):
    torchrun --standalone --nproc_per_node=NUM_GPUS train.py [OPTIONS]

Launch (single GPU / CPU):
    python train.py [OPTIONS]

Features
--------
* torch.distributed / DistributedDataParallel (DDP)
* Gradient accumulation  →  large effective batch sizes without extra VRAM
* torch.amp mixed precision  (bf16 on Ampere+, fp16 otherwise)
* Cosine LR schedule with linear warmup
* Periodic checkpointing + resume from checkpoint
* Per-step loss + perplexity logging to stdout (+ optional Weights & Biases)
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from dataset import IGNORE_INDEX, build_dataloader
from model import CANINELMConfig, CANINEPhonemeLM
from vocab import IPAVocab


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CANINE Phoneme LM")

    # ── Data ──────────────────────────────────────────────────────────────
    p.add_argument("--corpus",     default="wikipedia_ipa.jsonl",
                   help="JSONL corpus from wikipedia_streaming.py")
    p.add_argument("--vocab_path", default="phoneme_vocab.json",
                   help="Vocab JSON from wikipedia_streaming.py")

    # ── Model ─────────────────────────────────────────────────────────────
    p.add_argument("--d_model",          type=int,   default=256)
    p.add_argument("--n_local_layers",   type=int,   default=4)
    p.add_argument("--n_global_layers",  type=int,   default=8)
    p.add_argument("--local_n_heads",    type=int,   default=4)
    p.add_argument("--global_n_heads",   type=int,   default=8)
    p.add_argument("--local_window",     type=int,   default=32)
    p.add_argument("--ffn_multiplier",   type=int,   default=4)
    p.add_argument("--dropout",          type=float, default=0.1)
    p.add_argument("--seq_len",          type=int,   default=512)

    # ── Training ──────────────────────────────────────────────────────────
    p.add_argument("--batch_size",       type=int,   default=32,
                   help="Per-GPU batch size")
    p.add_argument("--grad_accum_steps", type=int,   default=4,
                   help="Gradient accumulation steps. "
                        "Effective batch = batch_size × grad_accum × world_size")
    p.add_argument("--max_steps",        type=int,   default=100_000)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--weight_decay",     type=float, default=0.01)
    p.add_argument("--warmup_steps",     type=int,   default=2_000)
    p.add_argument("--grad_clip",        type=float, default=1.0)
    p.add_argument("--num_workers",      type=int,   default=4,
                   help="DataLoader worker processes per GPU")

    # ── Logging / checkpointing ───────────────────────────────────────────
    p.add_argument("--log_every",    type=int, default=50)
    p.add_argument("--save_every",   type=int, default=2_000)
    p.add_argument("--ckpt_dir",     default="checkpoints")
    p.add_argument("--resume",       default=None,
                   help="Path to a checkpoint to resume from")
    p.add_argument("--wandb",        action="store_true",
                   help="Enable Weights & Biases logging")
    p.add_argument("--wandb_project", default="phoneme-lm")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed() -> tuple[int, int, torch.device]:
    """Initialise the process group; return (rank, world_size, device)."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        device     = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        # Single-process fallback
        rank       = 0
        world_size = 1
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, device


def is_main(rank: int) -> bool:
    return rank == 0


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Learning-rate schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, max_steps: int, lr: float) -> float:
    if step < warmup_steps:
        return lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: CANINEPhonemeLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    config: CANINELMConfig,
) -> None:
    # Always save the underlying module, not the DDP wrapper
    raw = model.module if isinstance(model, DDP) else model
    torch.save({
        "step":          step,
        "loss":          loss,
        "model_state":   raw.state_dict(),
        "optim_state":   optimizer.state_dict(),
        "config":        config,
    }, path)


def load_checkpoint(
    path: Path,
    model: CANINEPhonemeLM,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    ckpt  = torch.load(path, map_location=device)
    raw   = model.module if isinstance(model, DDP) else model
    raw.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    print(f"  Resumed from step {ckpt['step']} (loss={ckpt['loss']:.4f})")
    return ckpt["step"]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    rank, world_size, device = setup_distributed()
    main = is_main(rank)

    # ── Vocab ──────────────────────────────────────────────────────────────
    vocab = IPAVocab(args.vocab_path)
    if main:
        print(f"Vocab: {vocab}")

    # ── Model config ───────────────────────────────────────────────────────
    config = CANINELMConfig(
        vocab_size      = vocab.vocab_size,
        d_model         = args.d_model,
        dropout         = args.dropout,
        n_local_layers  = args.n_local_layers,
        local_window    = args.local_window,
        local_n_heads   = args.local_n_heads,
        n_global_layers = args.n_global_layers,
        global_n_heads  = args.global_n_heads,
        ffn_multiplier  = args.ffn_multiplier,
        max_seq_len     = args.seq_len,
        pad_id          = vocab.pad_id,
    )

    model = CANINEPhonemeLM(config).to(device)

    if main:
        n_params = model.num_parameters()
        print(f"Model: {n_params:,} trainable parameters")

    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)

    # ── Optimizer ──────────────────────────────────────────────────────────
    # Separate weight-decayed and non-decayed parameters
    decay_params     = [p for n, p in model.named_parameters()
                        if p.requires_grad and p.dim() >= 2]
    no_decay_params  = [p for n, p in model.named_parameters()
                        if p.requires_grad and p.dim() < 2]
    optimizer = AdamW(
        [
            {"params": decay_params,    "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        fused=torch.cuda.is_available(),  # fused AdamW is faster on CUDA
    )

    # ── Mixed precision ────────────────────────────────────────────────────
    # Use bf16 on Ampere+ (A100, RTX 30xx+) for best stability/speed;
    # fall back to fp16 on older CUDA, or skip on CPU.
    if device.type == "cuda":
        amp_dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
        scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))
    else:
        amp_dtype = torch.float32
        scaler    = torch.cuda.amp.GradScaler(enabled=False)

    autocast = torch.amp.autocast(device_type=device.type, dtype=amp_dtype)

    # ── DataLoader ─────────────────────────────────────────────────────────
    loader = build_dataloader(
        jsonl_path  = args.corpus,
        vocab       = vocab,
        seq_len     = args.seq_len,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        repeat      = True,
        rank        = rank,
        world_size  = world_size,
    )
    data_iter = iter(loader)

    # ── Checkpoint dir ─────────────────────────────────────────────────────
    ckpt_dir = Path(args.ckpt_dir)
    if main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Weights & Biases ───────────────────────────────────────────────────
    if args.wandb and main:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    # ── Resume ─────────────────────────────────────────────────────────────
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(Path(args.resume), model, optimizer, device)

    # ── Training loop ──────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()

    running_loss = 0.0
    t0 = time.time()

    for step in range(start_step, args.max_steps):

        # ── LR schedule ────────────────────────────────────────────────────
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr)
        set_lr(optimizer, lr)

        # ── Gradient accumulation ──────────────────────────────────────────
        # Accumulate over `grad_accum_steps` micro-batches before stepping.
        # DDP: only synchronise gradients on the last accumulation step to
        # avoid all-reduce overhead on every micro-batch.
        accum_loss = 0.0
        for micro in range(args.grad_accum_steps):
            batch = next(data_iter)
            input_ids  = batch["input_ids"].to(device, non_blocking=True)
            target_ids = batch["target_ids"].to(device, non_blocking=True)
            attn_mask  = batch["attn_mask"].to(device, non_blocking=True)

            # Only sync gradients on the final accumulation step
            sync_ctx = (
                model.no_sync()
                if world_size > 1 and micro < args.grad_accum_steps - 1
                else contextlib_nullcontext()
            )

            with sync_ctx:
                with autocast:
                    logits = model(input_ids, attn_mask)        # [B, T, V]
                    B, T, V = logits.shape
                    loss = F.cross_entropy(
                        logits.view(B * T, V),
                        target_ids.view(B * T),
                        ignore_index=IGNORE_INDEX,
                    )
                    # Scale loss for accumulation so gradients are correct
                    loss = loss / args.grad_accum_steps

                scaler.scale(loss).backward()
                accum_loss += loss.item()

        # ── Optimiser step ─────────────────────────────────────────────────
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        running_loss += accum_loss

        # ── Logging ────────────────────────────────────────────────────────
        if main and (step + 1) % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            ppl      = math.exp(min(avg_loss, 20))   # cap to avoid overflow
            elapsed  = time.time() - t0
            tokens_per_sec = (
                args.log_every * args.grad_accum_steps
                * args.batch_size * world_size * args.seq_len
                / elapsed
            )
            print(
                f"step {step+1:>7d} | loss {avg_loss:.4f} | ppl {ppl:.1f} "
                f"| lr {lr:.2e} | {tokens_per_sec/1e3:.1f}k tok/s"
            )
            if args.wandb:
                import wandb
                wandb.log({"loss": avg_loss, "ppl": ppl, "lr": lr,
                           "tok_per_sec": tokens_per_sec}, step=step + 1)
            running_loss = 0.0
            t0 = time.time()

        # ── Checkpointing ──────────────────────────────────────────────────
        if main and (step + 1) % args.save_every == 0:
            raw = model.module if isinstance(model, DDP) else model
            ckpt_path = ckpt_dir / f"step_{step+1:07d}.pt"
            save_checkpoint(ckpt_path, model, optimizer, step + 1, accum_loss, config)
            print(f"  Saved checkpoint → {ckpt_path}")

    if main:
        final_path = ckpt_dir / "final.pt"
        save_checkpoint(final_path, model, optimizer, args.max_steps, accum_loss, config)
        print(f"Training complete. Final checkpoint → {final_path}")

    cleanup()


# ---------------------------------------------------------------------------
# Tiny context-manager shim so we don't import contextlib at the top
# ---------------------------------------------------------------------------

class contextlib_nullcontext:
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train(parse_args())
