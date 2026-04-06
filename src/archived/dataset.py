"""
dataset.py  –  Streaming IterableDataset for the phoneme language model.

The dataset reads the .jsonl file written by wikipedia_streaming.py.
Each line has the form:
    {"text": "...", "ids": [<sos_id>, 3, 17, 42, ..., <eos_id>]}

The ids already include <sos>/<eos> because IPATokenizer.encode() is called
with add_sos=True, add_eos=True.

For next-token prediction we produce (input, target) pairs by shifting:
    input  = ids[:-1]   →  [<sos>, t1, t2, ..., t_{n-1}]
    target = ids[1:]    →  [t1,   t2, ..., t_{n-1}, <eos>]

Sequences are concatenated into a flat buffer and carved into fixed-length
windows of size `seq_len`. Short final windows are right-padded; pad
positions in the target are set to IGNORE_INDEX (-100) so they don't
contribute to the cross-entropy loss.

Multi-GPU sharding
------------------
IterableDataset + DDP: sharding is done *inside* __iter__ by having each
rank skip all lines except those where (line_index % world_size == rank).
This means no DistributedSampler is needed – just pass rank and world_size
when constructing the dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset

from vocab import IPAVocab

IGNORE_INDEX = -100   # standard PyTorch cross-entropy ignore label


# ---------------------------------------------------------------------------
# Core IterableDataset
# ---------------------------------------------------------------------------

class PhonemeDataset(IterableDataset):
    """
    Yields fixed-length (input_ids, target_ids, attn_mask) windows
    from the JSONL corpus produced by wikipedia_streaming.py.

    Parameters
    ----------
    jsonl_path  : path to the corpus .jsonl file
    vocab       : IPAVocab loaded from the same vocab JSON
    seq_len     : context window length (tokens per training example)
    repeat      : loop over the file indefinitely when True
    rank        : DDP rank of this process (0 for single-GPU)
    world_size  : total DDP processes (1 for single-GPU)
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        vocab: IPAVocab,
        seq_len: int = 512,
        repeat: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        self.jsonl_path = Path(jsonl_path)
        self.vocab = vocab
        self.seq_len = seq_len
        self.repeat = repeat
        self.rank = rank
        self.world_size = world_size

        if not self.jsonl_path.exists():
            raise FileNotFoundError(
                f"Corpus not found: {self.jsonl_path}\n"
                "Run wikipedia_streaming.py first."
            )

    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """
        Yields dicts:
            input_ids  : LongTensor [seq_len]
            target_ids : LongTensor [seq_len]  (IGNORE_INDEX at pad positions)
            attn_mask  : BoolTensor [seq_len]  (True = real token)
        """
        pad_id  = self.vocab.pad_id
        seq_len = self.seq_len
        # We need seq_len+1 tokens to form one (input, target) pair
        window  = seq_len + 1

        buffer: list[int] = []

        while True:          # outer loop: repeat over the file
            with open(self.jsonl_path, encoding="utf-8") as f:
                for line_idx, raw_line in enumerate(f):

                    # --- DDP sharding: rank i takes lines i, i+W, i+2W, ...
                    if line_idx % self.world_size != self.rank:
                        continue

                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        record = json.loads(raw_line)
                        ids: list[int] = record["ids"]
                    except (json.JSONDecodeError, KeyError):
                        continue

                    # ids already has <sos> … <eos> from IPATokenizer.encode()
                    buffer.extend(ids)

                    # Emit complete windows from the buffer
                    while len(buffer) >= window:
                        chunk  = buffer[:window]
                        buffer = buffer[window:]
                        yield _make_sample(chunk, seq_len, pad_id)

            if not self.repeat:
                break

            # Flush leftover tokens from this epoch before looping
            if len(buffer) >= 2:
                padded = buffer + [pad_id] * (window - len(buffer))
                yield _make_sample(padded[:window], seq_len, pad_id)
            buffer = []


def _make_sample(
    chunk: list[int],
    seq_len: int,
    pad_id: int,
) -> dict[str, torch.Tensor]:
    """chunk must have exactly seq_len+1 elements."""
    inp = torch.tensor(chunk[:-1], dtype=torch.long)   # [seq_len]
    tgt = torch.tensor(chunk[1:],  dtype=torch.long)   # [seq_len]

    # Mask pad tokens in target → they won't contribute to the loss
    tgt = tgt.masked_fill(tgt == pad_id, IGNORE_INDEX)

    # True where the input is a real (non-pad) token
    attn_mask = (inp != pad_id)

    return {"input_ids": inp, "target_ids": tgt, "attn_mask": attn_mask}


# ---------------------------------------------------------------------------
# DataLoader factory  (called by train.py)
# ---------------------------------------------------------------------------

def build_dataloader(
    jsonl_path: str | Path,
    vocab: IPAVocab,
    seq_len: int,
    batch_size: int,
    num_workers: int = 4,
    repeat: bool = True,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """
    Returns a ready-to-use DataLoader for DDP training.

    Notes
    -----
    - No DistributedSampler needed: PhonemeDataset handles sharding internally.
    - pin_memory=True gives a meaningful CPU→GPU transfer speedup.
    - persistent_workers=True avoids re-spawning worker processes each epoch.
    """
    dataset = PhonemeDataset(
        jsonl_path=jsonl_path,
        vocab=vocab,
        seq_len=seq_len,
        repeat=repeat,
        rank=rank,
        world_size=world_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
