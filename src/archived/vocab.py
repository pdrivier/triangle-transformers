"""
vocab.py  –  Thin wrapper around IPATokenizer's saved vocabulary JSON.

Your IPATokenizer already handles symbol → id mapping and saves it to a JSON
file (e.g. "phoneme_vocab.json"). This module loads that file and exposes the
constants the model and dataset need (vocab_size, pad_id, bos_id, eos_id) so
every other module has a single source of truth.

Expected vocab JSON format (as saved by IPATokenizer.save_vocabulary):
    {
        "phoneme_to_id": { "<pad>": 0, "<sos>": 1, "<eos>": 2, "p": 3, ... },
        "id_to_phoneme": { "0": "<pad>", "1": "<sos>", ... }   # optional
    }

If your JSON uses different key names, adjust the constants at the top.
"""

from __future__ import annotations

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Adjust these if your IPATokenizer uses different special-token strings
# ---------------------------------------------------------------------------
PHONEME_TO_ID_KEY = "phoneme_to_id"   # top-level key in the saved JSON
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<sos>"    # IPATokenizer uses <sos> (add_sos=True in encode())
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
# ---------------------------------------------------------------------------


class IPAVocab:
    """
    Lightweight vocab object loaded from IPATokenizer's saved JSON.

    Attributes
    ----------
    symbol_to_id : dict[str, int]
    id_to_symbol : dict[int, str]
    vocab_size   : int
    pad_id       : int
    bos_id       : int   (the <sos> id – used as the first decoder input)
    eos_id       : int
    unk_id       : int   (-1 if <unk> is not present)
    """

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Vocab file not found: {path}\n"
                "Run wikipedia_streaming.py first to build & save the vocabulary."
            )

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Support both {"phoneme_to_id": {...}} and a flat {token: id} JSON
        raw = data.get(PHONEME_TO_ID_KEY, data)

        self.symbol_to_id: dict[str, int] = {str(k): int(v) for k, v in raw.items()}
        self.id_to_symbol: dict[int, str] = {v: k for k, v in self.symbol_to_id.items()}
        self.vocab_size = len(self.symbol_to_id)

        self.pad_id = self._require(PAD_TOKEN)
        self.bos_id = self._require(BOS_TOKEN)
        self.eos_id = self._require(EOS_TOKEN)
        self.unk_id = self.symbol_to_id.get(UNK_TOKEN, -1)

    # ------------------------------------------------------------------
    # Public helpers (mostly useful for inference / evaluation)
    # ------------------------------------------------------------------

    def encode(self, symbols: list[str]) -> list[int]:
        """Map a list of IPA symbol strings → integer IDs."""
        unk = self.unk_id if self.unk_id >= 0 else self.pad_id
        return [self.symbol_to_id.get(s, unk) for s in symbols]

    def decode(self, ids: list[int], skip_special: bool = True) -> list[str]:
        """Map integer IDs → IPA symbol strings."""
        special = {self.pad_id, self.bos_id, self.eos_id}
        out = []
        for i in ids:
            if skip_special and i in special:
                continue
            out.append(self.id_to_symbol.get(i, UNK_TOKEN))
        return out

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return (
            f"IPAVocab(vocab_size={self.vocab_size}, "
            f"pad_id={self.pad_id}, bos_id={self.bos_id}, eos_id={self.eos_id})"
        )

    # ------------------------------------------------------------------

    def _require(self, token: str) -> int:
        if token not in self.symbol_to_id:
            raise KeyError(
                f"Special token '{token}' not found in vocab.\n"
                f"Available tokens (first 20): {list(self.symbol_to_id.keys())[:20]}"
            )
        return self.symbol_to_id[token]
