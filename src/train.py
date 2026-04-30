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





dataset = PhonemeDataset(...)
loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=make_collate_fn(dataset.pad_id))
