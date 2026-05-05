# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import PhonemeDataset, make_collate_fn
from model import PhonemeLM

# --- config -----------------------------------------------------------------
from dataclasses import dataclass, asdict

@dataclass
class TrainConfig: 
	# Data
	data_path: str = "data/"
	vocab_path: str = "vocab/phoneme_vocab.json"
	corpus_path: str = "raw/wikipedia_ipa_50000.jsonl"

	# Model (must match the specs of PhonLM architecture)
	vocab_size: int = 64				# will be overwritten anyway after loading vocab
	d_model: int = 256
	num_heads: int = 8
	num_layers: int = 2                 # TODO: need to figure out how to set differently per transformer block
	ffn_dim: int = 4 * d_model
	dropout: float = 0.1
	max_seq_len: int = 512

	# Training
	batch_size: int = 32
	learning_rate: float = 3e-4
	weight_decay: float = 0.01
	max_epochs: int = 10
	warmup_steps: int = 1000
	grad_clip: float = 1.0

	# Checkpointing & logging
	checkpoint_dir: str = "checkpoints/"
	log_every: int = 100 			    # n steps between console logs
	eval_every: int = 1000				# n steps between validation runs
	save_every: int = 5000				# n steps between checkpoint saves
	resume_from: str = None				# path to checkpoint to pick up from


# --- utilities -----------------------------------------------------------------
import torch
import random
import numpy as numpy
import os

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
	elif torch.backend.mps.is_available():
		device = torch.device("mps")
		print("Using MPS (Apple silicon)")
	else: 
		device = torch.device("cpu")
		print("Using CPU")

	return device

class Logger:
	def __init__(self, log_dir="logs/"):
		os.makedirs(log_dir, exist_ok=True)
		self.log_path = os.path.join(log_dir, "train_log.jsonl")

	def log(self, step, metrics: dict):
		# Always print to console
		parts = [f"step {step}"] + [f"{k}={v:.4f}" for k, v in metrics.items()]

		print(" | ".join(parts))
		# Also write to disk as jsonl for later plotting
		with open(self.log_path, "a") as f:
			f.write(json.dumps({"step": step, **metrics}) + "\n")


# --- build components -----------------------------------------------------------------
from torch.utils.data import DataLoader, random_split
from data import PhonemeDataset, make_collate_fn
from model import PhonemeLM   # TODO: this is meant to be what model.py exports, figure out if correct?

def build_components(cfg: TrainConfig, device):
	# --- Dataset -----------------------------------------------------------------
	dataset = PhonemeDataset(
		data_path=cfg.data_path,
		vocab_path=cfg.vocab_path,
		corpus_path=cfg.corpus_path,
		)
	cfg.vocab_size = len(dataset.phoneme_to_id) # patch vocab size

	# Train/val split (90/10)
	n_val = max(1, int(0.1 * len(dataset)))
	n_train = len(dataset) - n_val
	train_set, val_set = random_split(datsaet, [n_train, n_val])

	collate = make_collate_fn(dataset.pad_id)
	train_loader = DataLoader(train_set, batch_size=cfg.batch_size,
		shuffle=True, make_collate_fn=collate, num_workers=4, pin_memory=True)
	val_loader = DataLoader(val_set, batch_size=cfg.batch_size,
		shuffle=False, collate_fn=collate, num_workers=4, pin_memory=True)

	# --- Model -----------------------------------------------------------------
	model = PhonemeLM(cfg).to(device)
	n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Model parameters: {n_params:,}")

	# --- Optimizer -------------------------------------------------------------
	# AdamW with weight decay only on non-bias/norm params (standard practice)
	decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
	nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
	optimizer = torch.optim.AdamW([
		{"params": decay_params, "weight_decay": cfg.weight_decay},
		{"params": nodecay_params, "weight_decay": 0.0},
		], lr=cfg.learning_rate)

	# --- LR Scheduler ---------------------------------------------------------
	# Linear warmup then cosine decay 
	total_steps = cfg.max_epochs * len(train_loader)
	scheduler = torch.optim.lr_scheduler.OneCycleLR(
		optimizer,
		max_lr=cfg.learning_rate,
		total_steps=total_steps,
		pct_start=cfg.warmup_steps / total_steps,
		)

	return model, optimizer, scheduler, train_loader, val_loader, dataset

# --- training!! -----------------------------------------------------------------
def train_step(model, batch, optimizer, scheduler, cfg, device):
	input_ids, target_ids, attention_mask = [t.to(device) for t in batch]

	logits = model(input_ids, attention_mask=attention_mask)
	# logits: (B, T, vocab_size)

	# Cross-entropy loss; reshape for pytorch
	# -100 targets (padding) are automatically ignored
	loss = torch.nn.functional.cross_entropy(
		logits.view(-1, logits.size(-1)),	# (B*T, vocab_size)
		target_ids.view(-1),				# (B*T, )
		)

	optimizer.zero_grad()
	loss.backward()
	torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
	optimizer.step()
	scheduler.step()

	return loss.item()

# --- validation!! ----------------------------------------------------------------
@torch.no_grad()
def evaluate(model, val_loader, device):
	model.eval()
	total_loss = 0.0
	total_tokens = 0
	for input_ids, target_ids, attention_mask in val_loader:
		input_ids, target_ids, attention_mask = [t.to(device) for t in (input_ids, target_ids, attention_mask)]
		logits = model(input_ids, attention_mask=attention_mask)
		loss = torch.nn.functional.cross_entropy(
			logits.view(-1, logits.size(-1)), 
			target_ids.view(-1),
			reduction = "sum", 				# sum to compute perplexity correctly
			)

		# count only the non-padding tokens
		n_tokens = (target_ids != -100).sum().item()
		total_loss += loss.item()
		total_tokens += n_tokens

	model.train()
	avg_loss = total_loss / total_tokens
	perplexity = torch.exp(torch.tensor(avg_loss)).item()

	return print(f"val loss: {avg_loss}, val ppl: {perplexity}")

#TODO: add checkpointing and main loop








# dataset = PhonemeDataset(...)
# loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=make_collate_fn(dataset.pad_id))
