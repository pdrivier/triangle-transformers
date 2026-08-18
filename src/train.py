"""
train.py
Training loop for CANINE-C-style phoneme-level language model.

Pipeline: TrainConfig > build_components > train() > {train_step, evaluate, checkpoint}

"""
import os 
import json 
import random
from dataclasses import dataclass, asdict
from typing import Optional, List

import numpy as np 
import torch
import torch.nn.functional as F 
from torch.utils.data import DataLoader, random_split 

from data import PhonemeDataset, make_collate_fn 
from model import PhonemeLM 

# --- config -----------------------------------------------------------------
from dataclasses import dataclass, asdict

@dataclass
class TrainConfig: 
	# Data
	data_path: str = "data/"
	vocab_path: str = "vocab/phoneme_vocab.json"
	corpus_path: str = "raw/june_desktop_latin_filtering_wikipedia_ipa_170000.jsonl"
	val_fraction: float = 0.1

	# --- Model --- (must match the specs of PhonLM architecture)
	# vocab_size and space_id are placeholders, patched after the vocab loads

	vocab_size: int = 0					# will be overwritten anyway after loading vocab
	space_id: int = 0
	d_model: int = 256
	num_heads: int = 8
	ffn_dim: int = 1024
	num_layers: int = 2
	max_seq_len: int = 512
	max_word_len: int = 128				# max number of word-level positions
	boundary_ids: list = None			# phoneme ids that trigger word boundary pooling
	passthrough_ids: list = None		# phoneme ids that bypass pooling (punctuation, etc.)
	dropout: float = 0.1


	# --- Optimization ---
	batch_size: int = 32
	learning_rate: float = 3e-4
	weight_decay: float = 0.01
	max_epochs: int = 10
	warmup_steps: int = 1000
	grad_clip: float = 1.0

	# --- Checkpointing & logging ---
	checkpoint_dir: str = "checkpoints/"
	log_dir: str = "logs/"
	log_every: int = 100
	eval_every: int = 1000
	save_every: int = 5000
	resume_from: Optional[str] = None

	# --- Misc ---
	seed: int = 42

# ====== 2. Setup utilities ========================================================
def set_seed(seed: int = 42):

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

def get_device():
	if torch.cuda.is_available():
		device = torch.device("cuda")
		print(f"Using GPU: {torch.cuda.get_device_name(0)}")
	elif torch.backends.mps.is_available():
		device = torch.device("mps")
		print("Using MPS (Apple Silicon)")
	else: 
		device = torch.device("cpu")
		print("Using CPU")
	return device

class Logger:
	"""Minimal console + jsonl logger. Swap for W&B/TensorBoard later without touching the loop."""
	def __init__(self, log_dir = "logs/"):
		os.makedirs(log_dir, exist_ok=True)
		self.log_path = os.path.join(log_dir, "train_log.jsonl")

	def log(self, step, metrics: dict):
		parts = [f"step {step}"] + [
		f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
		for k, v in metrics.items()
		]
		print(" | ".join(parts))
		with open(self.log_path, "a") as f:
			f.write(json.dumps({"step": step, **metrics}) + "\n")

# ====== 3. Build components ===========================================================

def build_components(cfg: TrainConfig, device):

	# --- Dataset -----------
	dataset = PhonemeDataset(
		data_path=cfg.data_path,
		vocab_path=cfg.vocab_path,
		corpus_path=cfg.corpus_path,
		)

	# patch vocab-dependent config fields now that the vocab is loaded
	cfg.vocab_size = len(dataset.phoneme_to_id)
	cfg.space_id = dataset.phoneme_to_id["<SPACE>"]

	if cfg.boundary_ids is None:
		# default boundary set: SPACE, SOS, EOS all trigger a word split
		boundary_tokens = ["<SPACE>", "<SOS>", "<EOS>"]
		cfg.boundary_ids = [
			dataset.phoneme_to_id[t] for t in boundary_tokens if t in dataset.phoneme_to_id
			]
	if cfg.passthrough_ids is None: 
		cfg.passthrough_ids = []

	# --- Train/val split -------
	n_val = max(1, int(cfg.val_fraction * len(dataset)))
	n_train = len(dataset) - n_val
	train_set, val_set = random_split(
		dataset, [n_train, n_val],
		generator = torch.Generator().manual_seed(cfg.seed),
		)

	collate = make_collate_fn(dataset.pad_id)
	train_loader = DataLoader(
		train_set, batch_size=cfg.batch_size, shuffle=True,
		collate_fn=collate, num_workers=0, pin_memory=(device.type == "cuda"),
		)
	val_loader = DataLoader(
		val_set, batch_size=cfg.batch_size, shuffle=False, 
		collate_fn=collate, num_workers=0, pin_memory=(device.type == "cuda"),
		)

	# --- Model ------------

	model = PhonemeLM(
		vocab_size=cfg.vocab_size,
		d_model=cfg.d_model,
		num_heads=cfg.num_heads,
		ffn_dim=cfg.ffn_dim,
		max_seq_len=cfg.max_seq_len,
		max_word_len=cfg.max_word_len,
		pad_id=dataset.pad_id,
		space_id=cfg.space_id,
		boundary_ids=cfg.boundary_ids,
		passthrough_ids=cfg.passthrough_ids,
		num_layers=cfg.num_layers,
		dropout=cfg.dropout,
		).to(device)

	n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Model parameters: {n_params:,}")

	# ----- Optimizer: no weight decay on biases/norm params ------------
	decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
	nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
	optimizer = torch.optim.AdamW(
		[
			{"params": decay_params, "weight_decay": cfg.weight_decay},
			{"params": nodecay_params, "weight_decay": 0.0}, 
		],
		lr=cfg.learning_rate,
	)

	# ---- LR scheduler: linear warmup -> cosine decay (via OneCycleLR) --------
	total_steps = cfg.max_epochs * len(train_loader)
	scheduler = torch.optim.lr_scheduler.OneCycleLR(
		optimizer,
		max_lr=cfg.learning_rate,
		total_steps=total_steps,
		pct_start=min(cfg.warmup_steps / max(total_steps, 1), 0.3),
		)

	return model, optimizer, scheduler, train_loader, val_loader, dataset

# ==== 4. Train step ========================================================

def train_step(model, batch, optimizer, scheduler, cfg, device): 
	input_ids, target_ids, attention_mask = [t.to(device) for t in batch]

	# NOTE: PhonemeLM.forward currently takes only input_ids (it builds its own
	# causal/boundary masking internally). attention_mask is loaded here and
	# available if you extend forward() to use it for left-padding masking
	logits = model(input_ids)						# (B, T, vocab_size)

	loss = F.cross_entropy(
		logits.view(-1, logits.size(-1)),			# (B*T, vocab_size)
		target_ids.view(-1),						# (B*T)
		ignore_index=-100,							# padding targets are ignored
		)

	optimizer.zero_grad()
	loss.backward()
	torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
	optimizer.step()
	scheduler.step()

	return loss.item()


# ==== 5. Validation loop ========================================================
@torch.no_grad()
def evaluate(model, val_loader, device): 
	model.eval()
	total_loss = 0.0
	total_tokens = 0 

	for input_ids, target_ids, attention_mask in val_loader: 
		input_ids = input_ids.to(device)
		target_ids = target_ids.to(device)

		logits = model(input_ids)
		loss = F.cross_entropy(
			logits.view(-1, logits.size(-1)),
			target_ids.view(-1),
			ignore_index=-100,
			reduction="sum",          # sum so perplexity is computed per-token, not per-batch-mean
			)
		n_tokens = (target_ids != -100).sum().item()
		total_loss += loss.item()
		total_tokens += n_tokens

	model.train()
	avg_loss = total_loss / max(total_tokens, 1)
	perplexity = float(torch.exp(torch.tensor(avg_loss)))
	return {"val_loss": avg_loss, "val_ppl": perplexity}


# ==== 6. Checkpointing ===============================================================
# TODO: want to additionally save the cumulative number of tokens seen by this checkpoint

def save_checkpoint(path, model, optimizer, scheduler, step, cfg): 
	os.makedirs(os.path.dirname(path), exist_ok=True)
	torch.save({
		"step": step,
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"scheduler_state_dict": scheduler.state_dict(),
		"config": asdict(cfg),
		}, path)
	print(f"Checkpoint saved -> {path}")

def load_checkpoint(path, model, optimizer, scheduler, device): 
	ckpt = torch.load(path, map_location=device)
	model.load_state_dict(ckpt["model_state_dict"])
	optimizer.load_state_dict(ckpt["optimizer_state_dict"])
	scheduler.load_state_dict(ckpt["scheduler_state_dict"])
	return ckpt["step"]


# ===== 7. Main training loop ==========================================================
def train(cfg: TrainConfig): 
	set_seed(cfg.seed)
	device = get_device()
	logger = Logger(cfg.log_dir)

	model, optimizer, scheduler, train_loader, val_loader, dataset = build_components(cfg,device)

	global_step = 0
	if cfg.resume_from: 
		global_step = load_checkpoint(cfg.resume_from, model, optimizer, scheduler, device)
		print(f"Resumed from step {global_step}")

	model.train()
	for epoch in range(cfg.max_epochs): 
		print(f"\n{'='*60}\nEpoch {epoch}\n{'='*60}")
		for batch in train_loader:
			loss = train_step(model, batch, optimizer, scheduler, cfg, device)
			global_step += 1 

			if global_step % cfg.log_every == 0: 
				lr = scheduler.get_last_lr()[0]
				logger.log(global_step, {"train_loss": loss, "lr": lr, "epoch": epoch})

			if global_step % cfg.eval_every == 0: 
				val_metrics = evaluate(model, val_loader, device)
				logger.log(global_step, val_metrics)

			if global_step % cfg.save_every == 0: 
				ckpt_path = os.path.join(cfg.checkpoint_dir, f"ckpt_step{global_step}.pt")
				save_checkpoint(ckpt_path, model, optimizer, scheduler, global_step, cfg)

	# final checkpoint + eval at the end of training
	final_metrics = evaluate(model, val_loader, device)
	logger.log(global_step, final_metrics)
	save_checkpoint(os.path.join(cfg.checkpoint_dir, "ckpt_final.pt"),
					model, optimizer, scheduler, global_step, cfg)
	print("Training complete!!!")


if __name__ == "__main__": 
	cfg = TrainConfig()
	train(cfg)





