"""
model.py  –  CANINE-style autoregressive transformer for phoneme-level LM.

Architecture
------------
CANINE processes raw Unicode codepoints with two stages of attention:
a cheap *local* stage (each token sees only a small window) and an
expensive *global* stage (full sequence attention).  We adapt this
decoder-only for next-token prediction over IPA phoneme IDs.

    Embedding (no positional table – we use RoPE instead)
        ↓
    Local blocks   – windowed causal self-attention (O(n · local_window))
        ↓
    Global blocks  – full causal self-attention     (O(n²))
        ↓
    LayerNorm → LM head (linear, weight-tied to embedding)

Design decisions
----------------
* RoPE (Rotary Position Embeddings): better length generalisation than
  learned absolute positions; applied inside every attention layer.
* Pre-LayerNorm blocks: more stable training than post-norm.
* Weight tying between token embedding and LM head: reduces parameters
  and often improves perplexity.
* Causal mask: standard lower-triangular mask; local blocks additionally
  zero out attention beyond `local_window` steps back.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CANINELMConfig:
    # Set at runtime from IPAVocab.vocab_size
    vocab_size: int = 256

    # Core dims
    d_model: int = 256
    dropout: float = 0.1

    # Local (windowed) attention stage
    n_local_layers: int = 4
    local_window:   int = 32      # tokens each position can look back
    local_n_heads:  int = 4

    # Global (full causal) attention stage
    n_global_layers: int = 8
    global_n_heads:  int = 8

    # Feed-forward
    ffn_multiplier: int = 4       # d_ff = d_model * ffn_multiplier

    # Misc
    max_seq_len: int = 512
    pad_id:      int = 0

    def __post_init__(self) -> None:
        assert self.d_model % self.local_n_heads  == 0, "d_model must be divisible by local_n_heads"
        assert self.d_model % self.global_n_heads == 0, "d_model must be divisible by global_n_heads"


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,   # [B, heads, T, head_dim]
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    T        = q.shape[2]
    head_dim = q.shape[3]
    device   = q.device
    dtype    = q.dtype

    half   = head_dim // 2
    theta  = 1.0 / (10_000 ** (torch.arange(half, device=device).float() / half))
    pos    = torch.arange(T, device=device).float()
    freqs  = torch.outer(pos, theta)                  # [T, half]
    freqs  = torch.cat([freqs, freqs], dim=-1)         # [T, head_dim]
    cos    = freqs.cos()[None, None]                   # [1, 1, T, head_dim]
    sin    = freqs.sin()[None, None]

    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q.to(dtype), k.to(dtype)


# ---------------------------------------------------------------------------
# Causal multi-head self-attention (supports optional local window)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Parameters
    ----------
    local_window : int | None
        If set, each position attends only to the last `local_window` tokens
        (windowed causal attention).  None → full causal attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        local_window: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.n_heads     = n_heads
        self.head_dim    = d_model // n_heads
        self.local_window = local_window

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj  = nn.Linear(d_model,     d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                             # [B, T, C]
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, T] True=pad
    ) -> torch.Tensor:
        B, T, C = x.shape
        h, d    = self.n_heads, self.head_dim

        q, k, v = self.qkv_proj(x).split(C, dim=-1)
        # [B, T, C] → [B, h, T, d]
        q = q.view(B, T, h, d).transpose(1, 2)
        k = k.view(B, T, h, d).transpose(1, 2)
        v = v.view(B, T, h, d).transpose(1, 2)

        q, k = apply_rope(q, k)

        # Build additive bias: 0 where allowed, -inf where masked
        # Start with a standard causal (lower-triangular) mask
        allowed = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()

        if self.local_window is not None and self.local_window < T:
            # Also mask positions further back than the window
            idx = torch.arange(T, device=x.device)
            dist = idx.unsqueeze(0) - idx.unsqueeze(1)   # [T, T]  (col - row)
            allowed &= (dist >= -(self.local_window - 1))

        attn_bias = torch.zeros(T, T, device=x.device, dtype=q.dtype)
        attn_bias.masked_fill_(~allowed, float("-inf"))  # [T, T]

        # Key-padding mask broadcastable to [B, h, T, T]
        if key_padding_mask is not None:
            # key_padding_mask: True where *padded* → −inf bias on those keys
            pad_bias = (
                key_padding_mask.float()
                .masked_fill(key_padding_mask, float("-inf"))
                .masked_fill(~key_padding_mask, 0.0)
            )
            attn_bias = attn_bias + pad_bias[:, None, None, :]   # broadcast over B, h, T_q

        scores  = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d) + attn_bias
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        out = torch.matmul(weights, v)                        # [B, h, T, d]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Feed-forward block
# ---------------------------------------------------------------------------

class FFN(nn.Module):
    def __init__(self, d_model: int, multiplier: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        d_ff = d_model * multiplier
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Pre-norm transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
        local_window: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads, dropout, local_window)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = FFN(d_model, ffn_multiplier, dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class CANINEPhonemeLM(nn.Module):
    """
    Decoder-only CANINE-style phoneme language model.

    Input
    -----
    input_ids  : LongTensor  [B, T]
    attn_mask  : BoolTensor  [B, T]  True = real token, False = pad
                 (same convention as dataset.py; inverted internally)

    Output
    ------
    logits : FloatTensor [B, T, vocab_size]
        Pass directly to F.cross_entropy(logits.view(-1, V), targets.view(-1),
                                         ignore_index=IGNORE_INDEX)
    """

    def __init__(self, config: CANINELMConfig) -> None:
        super().__init__()
        self.config = config
        C = config.d_model

        self.embed      = nn.Embedding(config.vocab_size, C, padding_idx=config.pad_id)
        self.embed_drop = nn.Dropout(config.dropout)

        self.local_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=C,
                n_heads=config.local_n_heads,
                ffn_multiplier=config.ffn_multiplier,
                dropout=config.dropout,
                local_window=config.local_window,
            )
            for _ in range(config.n_local_layers)
        ])

        self.global_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=C,
                n_heads=config.global_n_heads,
                ffn_multiplier=config.ffn_multiplier,
                dropout=config.dropout,
                local_window=None,   # full causal attention
            )
            for _ in range(config.n_global_layers)
        ])

        self.norm_out = nn.LayerNorm(C)
        self.lm_head  = nn.Linear(C, config.vocab_size, bias=False)

        # Weight tying: embedding matrix == output projection matrix
        self.lm_head.weight = self.embed.weight

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embed_drop(self.embed(input_ids))   # [B, T, C]

        # dataset.py attn_mask: True = real  →  invert for key_padding_mask
        kpm: Optional[torch.Tensor] = None
        if attn_mask is not None:
            kpm = ~attn_mask   # True = pad = ignore

        for block in self.local_blocks:
            x = block(x, kpm)

        for block in self.global_blocks:
            x = block(x, kpm)

        x = self.norm_out(x)
        return self.lm_head(x)   # [B, T, vocab_size]

    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = (p for p in self.parameters() if (not trainable_only or p.requires_grad))
        return sum(p.numel() for p in params)
