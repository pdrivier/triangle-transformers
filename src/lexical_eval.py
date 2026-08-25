"""
lexical_eval.py

Two evaluations across saved checkpoints:

  1. Validation loss curve  -- how does held-out perplexity change over training?
  2. Lexical discrimination -- do real words get higher summed log-prob than
                               length-matched nonwords?

Expected inputs
---------------
- checkpoints/          one or more ckpt_step<N>.pt / ckpt_final.pt files
- stimuli/words.json    [{"ipa": "kæt", "label": "word"}, ...]
                        OR a CSV with columns: ipa, label  (word / nonword)

Outputs
-------
- results/checkpoint_metrics.jsonl   per-checkpoint scores
- results/lexical_eval.png           two-panel plot
"""

import os
import re
import csv
import json
import glob
import math
import statistics
from dataclasses import asdict

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt

from data import PhonemeDataset, make_collate_fn
from model import PhonemeLM
from train import TrainConfig, get_device


# ===== 1. Checkpoint utilities ================================================

def discover_checkpoints(checkpoint_dir: str) -> list[dict]:
    """
    Find all .pt files in checkpoint_dir and sort them by training step.
    Returns a list of {"path": ..., "step": ...} dicts, ascending by step.
    """
    paths = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    checkpoints = []
    for p in paths:
        # extract step from filename: ckpt_step5000.pt -> 5000
        # ckpt_final.pt gets a very large step so it sorts last
        match = re.search(r"step(\d+)", os.path.basename(p))
        step = int(match.group(1)) if match else int(1e9)
        checkpoints.append({"path": p, "step": step})
    checkpoints.sort(key=lambda x: x["step"])
    if not checkpoints:
        raise FileNotFoundError(f"No .pt files found in {checkpoint_dir}")
    return checkpoints


def load_model_for_eval(checkpoint_path: str, device) -> tuple:
    """Load a PhonemeLM from a checkpoint, with dropout disabled."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg_dict = ckpt["config"]

    # TrainConfig uses Optional[List[int]] for boundary_ids etc.,
    # stored as plain lists in asdict() -- safe to splat back in
    cfg = TrainConfig(**cfg_dict)

    dataset = PhonemeDataset(
        data_path=cfg.data_path,
        vocab_path=cfg.vocab_path,
        corpus_path=cfg.corpus_path,
    )

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
        dropout=0.0,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, dataset, cfg


# ===== 2. Validation perplexity ===============================================

@torch.no_grad()
def compute_perplexity(model, data_loader, device) -> dict:
    total_loss = 0.0
    total_tokens = 0
    for input_ids, target_ids, attention_mask in data_loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += (target_ids != -100).sum().item()
    avg_loss = total_loss / max(total_tokens, 1)
    return {"val_loss": avg_loss, "val_ppl": math.exp(avg_loss)}


# ===== 3. Summed log-prob scoring =============================================

def encode_ipa_string(ipa: str, dataset: PhonemeDataset) -> list[int] | None:
    """
    Encode a pre-transcribed IPA string into phoneme ids.

    Expects the IPA string to be a space-separated sequence of tokens
    that exactly match keys in phoneme_to_id, e.g. "k æ t".

    Returns None if any token is unknown (rather than silently scoring
    a sequence polluted with <UNK> ids).
    """
    sos_id = dataset.phoneme_to_id["<SOS>"]
    eos_id = dataset.phoneme_to_id["<EOS>"]
    unk_id = dataset.phoneme_to_id.get("<UNK>", None)

    tokens = ipa.strip().split()
    ids = [sos_id]
    for token in tokens:
        tid = dataset.phoneme_to_id.get(token)
        if tid is None or tid == unk_id:
            return None   # skip items with unknown phonemes
        ids.append(tid)
    ids.append(eos_id)
    return ids


@torch.no_grad()
def score_sequence(model, ids: list[int], device) -> float:
    """
    Compute the summed log-probability of a phoneme sequence under the model.

    score = Σ_t log P(ids[t] | ids[0..t-1])

    This is computed teacher-forced in a single forward pass (efficient),
    NOT autoregressively.

    Returns a float (higher = model thinks sequence is more probable).
    We return the *per-phoneme average* (divide by sequence length - 1)
    so that scores are comparable across words of different lengths.
    Per-phoneme scoring is standard in psycholinguistics (cf. mean surprisal).
    """
    input_ids = torch.tensor([ids[:-1]], device=device)   # (1, T)
    target_ids = torch.tensor([ids[1:]], device=device)    # (1, T)  -- shifted left by one

    logits = model(input_ids)                              # (1, T, vocab_size)
    log_probs = F.log_softmax(logits, dim=-1)              # (1, T, vocab_size)

    # gather the log-prob of each actual target token
    # target_ids: (1, T) -> (1, T, 1) for gather, then squeeze back
    token_log_probs = log_probs.gather(
        dim=-1,
        index=target_ids.unsqueeze(-1)
    ).squeeze(-1)                                          # (1, T)

    # sum over time, average by sequence length (excluding the SOS input position)
    n_tokens = token_log_probs.shape[1]
    return (token_log_probs.sum() / n_tokens).item()


@torch.no_grad()
def score_all_stimuli(model, stimuli: list[dict], dataset: PhonemeDataset, device) -> list[dict]:
    """
    Score every stimulus in the list.

    Each stimulus dict must have keys:
        "ipa"   : space-separated IPA tokens, e.g. "k æ t"
        "label" : "word" or "nonword"

    Returns a copy with "score", "length", and "encoded" added.
    Stimuli whose IPA can't be encoded are dropped (with a warning).
    """
    results = []
    skipped = 0
    for item in stimuli:
        ids = encode_ipa_string(item["ipa"], dataset)
        if ids is None:
            skipped += 1
            continue
        score = score_sequence(model, ids, device)
        results.append({
            **item,
            "score": score,
            "length": len(ids) - 2,   # number of real phonemes (excl. SOS/EOS)
            "encoded": ids,
        })
    if skipped:
        print(f"  Warning: {skipped} stimuli skipped (unknown phonemes in vocab)")
    return results


# ===== 4. Discrimination metrics ==============================================

def compute_discrimination_metrics(scored_stimuli: list[dict]) -> dict:
    """
    Compute word/nonword discrimination metrics from a list of scored stimuli.

    Returns:
        mean_word_score      : mean per-phoneme log-prob for words
        mean_nonword_score   : mean per-phoneme log-prob for nonwords
        gap                  : mean_word_score - mean_nonword_score
        d_prime              : signal-detection d' (standardized gap)
        accuracy             : proportion of length-matched pairs where word > nonword
    """
    words    = [s for s in scored_stimuli if s["label"] == "word"]
    nonwords = [s for s in scored_stimuli if s["label"] == "nonword"]
    shuffle_words = [s for s in scored_stimuli if s["label"] == "shuffled_word"]
    shuffle_nonwords = [s for s in scored_stimuli if s["label"] == "shuffled_nonword"]

    if not words or not nonwords:
        raise ValueError("Need at least one word and one nonword in stimuli.")

    word_scores    = [s["score"] for s in words]
    nonword_scores = [s["score"] for s in nonwords]
    shuffle_words_scores = [s["score"] for s in shuffle_words]
    shuffle_nonwords_scores = [s["score"] for s in shuffle_nonwords]

    mean_w  = sum(word_scores) / len(word_scores)
    mean_nw = sum(nonword_scores) / len(nonword_scores)
    
    def per_shuffle_means(items: list[dict]) -> dict:
        """Group by shuffle_id and return {shuffle_id: mean_score}."""
        groups: dict[object, list[float]] = {}
        for s in items:
            if "shuffle_id" not in s:
                raise ValueError(f"Item missing shuffle_id: {s}")
            groups.setdefault(s["shuffle_id"], []).append(s["score"])
        return {sid: sum(scores) / len(scores) for sid, scores in groups.items()}

    shw_by_shuffle  = per_shuffle_means(shuffle_words)
    shnw_by_shuffle = per_shuffle_means(shuffle_nonwords)

    shw_means  = list(shw_by_shuffle.values())
    shnw_means = list(shnw_by_shuffle.values())

    mean_shw  = sum(shw_means) / len(shw_means)   if shw_means  else float("nan")
    mean_shnw = sum(shnw_means) / len(shnw_means) if shnw_means else float("nan")

    # variability across shuffles (useful as a noise-floor / control-condition spread)
    std_shw  = statistics.stdev(shw_means)  if len(shw_means)  > 1 else 0.0
    std_shnw = statistics.stdev(shnw_means) if len(shnw_means) > 1 else 0.0


    # d-prime: (mean_word - mean_nonword) / pooled_std
    # TODO 08/11/2026: figure out how to set up contrast with the shuffles
    std_w  = statistics.stdev(word_scores)    if len(word_scores)    > 1 else 1e-8
    std_nw = statistics.stdev(nonword_scores) if len(nonword_scores) > 1 else 1e-8
    pooled_std = ((std_w ** 2 + std_nw ** 2) / 2) ** 0.5
    d_prime = (mean_w - mean_nw) / max(pooled_std, 1e-8)

    # d-prime for the shuffled control, computed the same way but across
    # per-shuffle means (so it reflects shuffle-to-shuffle variability, not
    # pooled item-level noise)
    pooled_std_shuffle = ((std_shw ** 2 + std_shnw ** 2) / 2) ** 0.5
    d_prime_shuffle = (
        (mean_shw - mean_shnw) / max(pooled_std_shuffle, 1e-8)
        if shw_means and shnw_means else float("nan")
    )

    # pairwise accuracy: use pair_id if available (AELP paired design),
    # otherwise fall back to sorting by length and pairing greedily
    word_by_pair    = {s["pair_id"]: s for s in words    if "pair_id" in s}
    nonword_by_pair = {s["pair_id"]: s for s in nonwords if "pair_id" in s}
    shared_pairs    = set(word_by_pair) & set(nonword_by_pair)

    if shared_pairs:
        n_correct = sum(
            1 for pid in shared_pairs
            if word_by_pair[pid]["score"] > nonword_by_pair[pid]["score"]
        )
        pairwise_acc = n_correct / len(shared_pairs)
    else:
        words_sorted    = sorted(words,    key=lambda x: x["length"])
        nonwords_sorted = sorted(nonwords, key=lambda x: x["length"])
        n_pairs   = min(len(words_sorted), len(nonwords_sorted))
        n_correct = sum(
            1 for w, nw in zip(words_sorted[:n_pairs], nonwords_sorted[:n_pairs])
            if w["score"] > nw["score"]
        )
        pairwise_acc = n_correct / max(n_pairs, 1)

    return {
        "mean_word_score":    mean_w,
        "mean_nonword_score": mean_nw,
        "mean_shuffle_word_score":    mean_shw,
        "mean_shuffle_nonword_score": mean_shnw,
        "shuffle_word_scores_by_id":    shw_by_shuffle,
        "shuffle_nonword_scores_by_id": shnw_by_shuffle,
        "std_shuffle_word_means":    std_shw,
        "std_shuffle_nonword_means": std_shnw,
        "n_words":            len(words),
        "n_nonwords":         len(nonwords),
        "n_shuffle_words":    len(shuffle_words),
        "n_shuffle_nonwords": len(shuffle_nonwords),
        "n_shuffles_word":    len(shw_by_shuffle),
        "n_shuffles_nonword": len(shnw_by_shuffle),
        "gap":                mean_w - mean_nw,
        "d_prime":            d_prime,
        "d_prime_shuffle":    d_prime_shuffle,
        "pairwise_accuracy":  pairwise_acc,
    }


# ===== 5. Stimuli loading =====================================================

def load_stimuli(path: str) -> list[dict]:
    """
    Load word/nonword stimuli from a JSON or CSV file.

    JSON format: [{"ipa": "k æ t", "label": "word"}, ...]
    CSV format:  two columns, ipa and label, with a header row.

    The "ipa" field must be space-separated phoneme tokens that match
    keys in your phoneme_to_id vocabulary.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            stimuli = json.load(f)
    elif ext == ".csv":
        stimuli = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stimuli.append({"ipa": row["ipa"], "label": row["label"]})
    else:
        raise ValueError(f"Unsupported stimuli format: {ext}. Use .json or .csv")

    # basic validation
    for i, item in enumerate(stimuli):
        if "ipa" not in item or "label" not in item:
            raise ValueError(f"Stimulus {i} missing 'ipa' or 'label' field: {item}")
        if item["label"] not in ("word", "nonword", "shuffled_word", "shuffled_nonword"):
            raise ValueError(f"Stimulus {i} has invalid label '{item['label']}' (must be 'word' or 'nonword' or a shuffled version of each)")

    return stimuli


# ===== 6. Plotting ============================================================
def plot_results(checkpoint_records: list[dict], output_path: str):
    """
    Two-panel plot:
      Left:  validation loss curve across checkpoints
      Right: word/nonword gap (mean score difference) and d' across checkpoints
    """
    steps       = [r["step"]          for r in checkpoint_records]
    val_ppls    = [r["val_ppl"]       for r in checkpoint_records]
    gaps        = [r["gap"]           for r in checkpoint_records]
    d_primes    = [r["d_prime"]       for r in checkpoint_records]
    pair_accs   = [r["pairwise_accuracy"] for r in checkpoint_records]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Phoneme LM: Evaluation across checkpoints", fontsize=13)
    # --- Panel 1: Validation perplexity ---
    ax = axes[0]
    ax.plot(steps, val_ppls, "o-", color="#2166ac", linewidth=2, markersize=6)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation perplexity")
    ax.set_title("Held-out perplexity")
    ax.grid(True, alpha=0.3)
    # --- Panel 2: Word/nonword score gap ---
    ax = axes[1]
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.plot(steps, gaps, "o-", color="#d6604d", linewidth=2, markersize=6, label="score gap (word − nonword)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean log-prob difference")
    ax.set_title("Lexical discrimination (gap)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    # --- Panel 3: d' and pairwise accuracy ---
    ax = axes[2]
    ax2 = ax.twinx()
    l1, = ax.plot(steps, d_primes,  "o-", color="#4dac26", linewidth=2, markersize=6, label="d′")
    l2, = ax2.plot(steps, pair_accs, "s--", color="#b8860b", linewidth=2, markersize=6, label="pairwise acc")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax2.axhline(0.5, color="#b8860b", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Training step")
    ax.set_ylabel("d′")
    ax2.set_ylabel("Pairwise accuracy")
    ax.set_title("Discrimination sensitivity")
    ax.legend(handles=[l1, l2], fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved -> {output_path}")


def plot_shuffle_histograms(
    checkpoint_records: list[dict],
    output_path: str,
    checkpoint_index: int = -1,   # which checkpoint's record to visualize; -1 = last (final/most-trained)
):
    """
    Two-panel plot for a single checkpoint:
      Left:  histogram of per-shuffle mean scores for shuffled words,
             with vertical lines marking the real word mean and the
             overall shuffled-word mean.
      Right: same, for shuffled nonwords vs. real nonwords.

    Relies on `shuffle_word_scores_by_id` / `shuffle_nonword_scores_by_id`
    (dicts of {shuffle_id: mean_score}) produced by
    compute_discrimination_metrics.
    """
    if not checkpoint_records:
        raise ValueError("checkpoint_records is empty.")
    rec = checkpoint_records[checkpoint_index]
    step = rec["step"]

    shw_by_id  = rec.get("shuffle_word_scores_by_id", {})
    shnw_by_id = rec.get("shuffle_nonword_scores_by_id", {})
    if not shw_by_id or not shnw_by_id:
        raise ValueError(
            f"Checkpoint record for step {step} is missing shuffle score "
            "breakdowns (shuffle_word_scores_by_id / shuffle_nonword_scores_by_id)."
        )

    shw_means  = list(shw_by_id.values())
    shnw_means = list(shnw_by_id.values())

    mean_w   = rec["mean_word_score"]
    mean_nw  = rec["mean_nonword_score"]
    mean_shw  = rec["mean_shuffle_word_score"]
    mean_shnw = rec["mean_shuffle_nonword_score"]
    
    dist_shw = list(rec["shuffle_word_scores_by_id"].values())
    dist_shnw = list(rec["shuffle_nonword_scores_by_id"].values())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    fig.suptitle(f"Shuffle-control distributions (step {step})", fontsize=13)

    # --- Panel 1: shuffled words ---
    ax = axes[0]
    ax.hist(dist_shw, bins=min(10, len(dist_shw)), color="#4393c3",
             edgecolor="white", alpha=0.85)
    ax.axvline(mean_w, color="#2166ac", linewidth=2.2, linestyle="-",
               label=f"real word mean ({mean_w:.3f})")
    ax.axvline(mean_shw, color="#4393c3", linewidth=2.2, linestyle="--",
               label=f"shuffle mean ({mean_shw:.3f})")
    ax.set_xlabel("Mean log-prob per shuffle")
    ax.set_ylabel("Count (shuffles)")
    ax.set_title(f"Words vs. shuffled words (n={len(shw_means)} shuffles)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: shuffled nonwords ---
    ax = axes[1]
    ax.hist(dist_shnw, bins=min(10, len(dist_shnw)), color="#f4a582",
             edgecolor="white", alpha=0.85)
    ax.axvline(mean_nw, color="#d6604d", linewidth=2.2, linestyle="-",
               label=f"real nonword mean ({mean_nw:.3f})")
    ax.axvline(mean_shnw, color="#f4a582", linewidth=2.2, linestyle="--",
               label=f"shuffle mean ({mean_shnw:.3f})")
    ax.set_xlabel("Mean log-prob per shuffle")
    ax.set_ylabel("Count (shuffles)")
    ax.set_title(f"Nonwords vs. shuffled nonwords (n={len(shnw_means)} shuffles)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Shuffle histogram plot saved -> {output_path}")

# ===== 7. Main ================================================================
def main(
    checkpoint_dir: str = "checkpoints/",
    stimuli_path: str   = "stimuli/words_and_shuffles.json",
    output_dir: str     = "results/",
    val_corpus_n: int   = 500,     # number of dataset sequences to use for val perplexity
):
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()

    # --- Load stimuli once (shared across all checkpoints) --------------------
    print(f"Loading stimuli from {stimuli_path}")
    stimuli = load_stimuli(stimuli_path)
    print(f"  {len(stimuli)} stimuli loaded "
          f"({sum(1 for s in stimuli if s['label'] == 'word')} words, "
          f"{sum(1 for s in stimuli if s['label'] == 'nonword')} nonwords), "
          f"{sum(1 for s in stimuli if s['label'] == 'shuffled_word')} shuffle_words, "
          f"{sum(1 for s in stimuli if s['label'] == 'shuffled_nonword')} shuffle_nonwords")

    # --- Discover checkpoints -------------------------------------------------
    checkpoints = discover_checkpoints(checkpoint_dir)
    print(f"\nFound {len(checkpoints)} checkpoints:")
    for ck in checkpoints:
        print(f"  step {ck['step']:>8}  {ck['path']}")

    # --- Evaluate each checkpoint ---------------------------------------------
    all_records = []
    for ck in checkpoints:
        print(f"\n{'='*55}")
        print(f"Evaluating checkpoint: step {ck['step']}  ({ck['path']})")
        model, dataset, cfg = load_model_for_eval(ck["path"], device)

        # -- Validation perplexity (on a subset of the dataset for speed) ------
        from torch.utils.data import DataLoader, Subset
        import random
        indices = random.sample(range(len(dataset)), min(val_corpus_n, len(dataset)))
        subset  = Subset(dataset, indices)
        collate = make_collate_fn(dataset.pad_id)
        val_loader = DataLoader(subset, batch_size=32, shuffle=False, collate_fn=collate)
        ppl_metrics = compute_perplexity(model, val_loader, device)
        print(f"  val_loss={ppl_metrics['val_loss']:.4f}  val_ppl={ppl_metrics['val_ppl']:.2f}")

        # -- Lexical scoring ---------------------------------------------------
        scored = score_all_stimuli(model, stimuli, dataset, device)
        disc   = compute_discrimination_metrics(scored)
        print(f"  word mean score:             {disc['mean_word_score']:.4f}")
        print(f"  nonword mean score:          {disc['mean_nonword_score']:.4f}")
        print(f"  shuffled word mean score:    {disc['mean_shuffle_word_score']:.4f}"
              f"  (n_shuffles={disc['n_shuffles_word']})")
        print(f"  shuffled nonword mean score: {disc['mean_shuffle_nonword_score']:.4f}"
              f"  (n_shuffles={disc['n_shuffles_nonword']})")
        print(f"  gap:                {disc['gap']:.4f}")
        print(f"  d′:                 {disc['d_prime']:.4f}")
        print(f"  d′ (shuffle ctrl):  {disc['d_prime_shuffle']:.4f}")
        print(f"  pairwise accuracy:  {disc['pairwise_accuracy']:.4f}")

        record = {
            "step":      ck["step"],
            "path":      ck["path"],
            **ppl_metrics,
            **disc,
        }
        all_records.append(record)

        # free GPU memory between checkpoints
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --- Save raw results as jsonl --------------------------------------------
    # Note: shuffle_word_scores_by_id / shuffle_nonword_scores_by_id are dicts
    # keyed by shuffle_id, which json.dumps handles fine as long as keys are
    # str/int/float. If your shuffle_ids aren't natively JSON-safe keys,
    # stringify them here.
    results_path = os.path.join(output_dir, "checkpoint_metrics.jsonl")
    with open(results_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")
    print(f"\nMetrics saved -> {results_path}")

    # --- Plot -----------------------------------------------------------------
    plot_path = os.path.join(output_dir, "lexical_eval.png")
    plot_results(all_records, plot_path)

    # --- Print final summary table --------------------------------------------
    print(f"\n{'step':>10}  {'val_ppl':>10}  {'gap':>8}  {'d_prime':>8}  {'d_shuf':>8}  {'pair_acc':>9}")
    print("-" * 66)
    for r in all_records:
        print(f"{r['step']:>10}  {r['val_ppl']:>10.2f}  {r['gap']:>8.4f}  "
              f"{r['d_prime']:>8.4f}  {r['d_prime_shuffle']:>8.4f}  {r['pairwise_accuracy']:>9.4f}")

    # --- Plot -----------------------------------------------------------------
    plot_path = os.path.join(output_dir, "lexical_eval.png")
    plot_results(all_records, plot_path)

    shuffle_hist_path = os.path.join(output_dir, "shuffle_histograms.png")
    plot_shuffle_histograms(all_records, shuffle_hist_path)  # defaults to last checkpoint

if __name__ == "__main__":
    main(
        checkpoint_dir = "checkpoints/",
        stimuli_path   = "stimuli/words_and_shuffles.json",
        output_dir     = "results/",
        val_corpus_n   = 500,
    )