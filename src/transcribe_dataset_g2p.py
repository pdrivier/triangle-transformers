""" 
Grab and download Wikipedia English shards from HuggingFace
"""

import json
import re
import sys
import nltk

from g2p_en import G2p
from multiprocessing import Pool
from pathlib import Path
import numpy as np

from huggingface_hub import list_repo_files, snapshot_download, hf_hub_download

def find_latest_snapshot(repo_path = "wikimedia/wikipedia", repo_type = "dataset"):
	files = list_repo_files(repo_path, repo_type=repo_type)
	en = sorted(f for f in files if f.split("/")[0].endswith(".en"))
	print(len(en), "files")
	print("\n".join(en[:5]))
	print(sorted({f.split("/")[0] for f in en}))  # available English snapshots
	snapshots = {f.split("/")[0] for f in en}
	snapshots = list(snapshots)
	return snapshots

SNAPSHOT = snapshots[0]  # swap for the newest one from the listing above

local_dir = snapshot_download(
	repo_id="wikimedia/wikipedia",
    repo_type="dataset",
    allow_patterns=f"{SNAPSHOT}/*.parquet",
    local_dir="data/raw/wikipedia_raw",
    max_workers=8,
)



IN_DIR = Path("data/raw/wikipedia_raw") / SNAPSHOT
OUT_DIR = Path("data/wikipedia_ipa/ipa_shards"); OUT_DIR.mkdir(parents=True, exist_ok=True)

def process_shard(shard_path):
    out = OUT_DIR / (shard_path.stem + ".npy")
    if out.exists():
        return out.name, "skipped"

    # Import and construct inside the worker so each process has its own model
    from g2p_en import G2p
    g2p = G2p()

    ids = []
    for art_id, title, text in iter_articles(shard_path):
        # TODO: normalize text, split into sentences, transcribe, encode to ids
        pass

    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, np.array(ids, dtype=np.uint8))
    tmp.rename(out)  # atomic rename so a crash never leaves a half-written shard
    return out.name, "done"

if __name__ == "__main__":
    shards = sorted(IN_DIR.glob("*.parquet"))
    with Pool(processes=8) as pool:
        for name, status in pool.imap_unordered(process_shard, shards):
            print(name, status)