"""
transcribe_stimuli.py

One-time preprocessing script: reads the AELP-derived CSV (which already
contains IPA transcriptions in ipa_words and ipa_nonwords columns), segments
each IPA string into vocab-matched tokens using the same _split_ipa logic
that built the training corpus, and writes stimuli/words.json for lexical_eval.py.

Expected input CSV columns (at minimum):
    word_us      orthographic word
    word_length  orthographic length
    n_phon       phoneme count for word
    nonword      orthographic nonword
    n_phon_nw    phoneme count for nonword
    ipa_words    raw IPA string for word,    e.g. "kæt"
    ipa_nonwords raw IPA string for nonword, e.g. "blɪk"

Output JSON format (stimuli/words.json):
    [
      {"word": "cat",   "ipa": "k æ t",   "label": "word",    "length": 3, "pair_id": 0},
      {"word": "blick", "ipa": "b l ɪ k", "label": "nonword", "length": 4, "pair_id": 0},
      ...
    ]

pair_id links each word to its length-matched nonword, which lets
lexical_eval.py compute pairwise accuracy on truly matched pairs.
"""

import os
import csv
import json

from normalization.ipa_pipeline import IPATokenizer


# ===== Configuration ==========================================================

INPUT_CSV   = "stimuli/aelp_transcribed.csv"   # your CSV with ipa_words / ipa_nonwords
OUTPUT_JSON = "stimuli/words.json"
VOCAB_PATH  = "data/vocab/phoneme_vocab.json"


# ===== Helpers ================================================================

def segment_and_check(ipa_raw: str, tokenizer: IPATokenizer, label: str, word: str):
    """
    Segment a raw IPA string into tokens using _split_ipa and check vocab coverage.
    Returns (ipa_spaced, tokens, oov_list) or (None, None, None) on failure.
    """
    ipa_raw = ipa_raw.strip()
    if not ipa_raw:
        print(f"  Warning: empty IPA string for {label} '{word}', skipping pair")
        return None, None, None

    tokens = tokenizer._split_ipa(ipa_raw)
    if not tokens:
        print(f"  Warning: _split_ipa returned no tokens for {label} '{word}' "
              f"(ipa: '{ipa_raw}'), skipping pair")
        return None, None, None

    oov = [t for t in tokens if t not in tokenizer.phoneme_to_id]
    return " ".join(tokens), tokens, oov


# ===== Main ===================================================================

def main():
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    # Load tokenizer vocab (just for _split_ipa and coverage checking)
    tokenizer = IPATokenizer(language="en-us")
    tokenizer.load_vocabulary(VOCAB_PATH)
    print(f"Vocab loaded: {tokenizer.vocab_size} tokens")

    # Read the CSV
    rows = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"Loaded {len(rows)} rows from {INPUT_CSV}")

    # Process each paired row
    output = []
    skipped = 0
    oov_count = 0

    for pair_id, row in enumerate(rows):
        word_orth   = row["word_us"].strip()
        nw_orth     = row["nonword"].strip()
        ipa_word_raw = row["ipa_words"].strip()
        ipa_nw_raw   = row["ipa_nonwords"].strip()

        # --- Segment word ------------------------------------------------
        word_spaced, word_tokens, word_oov = segment_and_check(
            ipa_word_raw, tokenizer, "word", word_orth
        )
        if word_spaced is None:
            skipped += 1
            continue

        # --- Segment nonword ---------------------------------------------
        nw_spaced, nw_tokens, nw_oov = segment_and_check(
            ipa_nw_raw, tokenizer, "nonword", nw_orth
        )
        if nw_spaced is None:
            skipped += 1
            continue

        # --- OOV warnings (pair still included; lexical_eval drops at score time) ---
        if word_oov:
            print(f"  OOV in word '{word_orth}': {word_oov}  (ipa: '{ipa_word_raw}')")
            oov_count += 1
        if nw_oov:
            print(f"  OOV in nonword '{nw_orth}': {nw_oov}  (ipa: '{ipa_nw_raw}')")
            oov_count += 1

        # --- Append both halves of the pair ------------------------------
        output.append({
            "word":     word_orth,
            "ipa":      word_spaced,      # space-separated tokens for encode_ipa_string
            "label":    "word",
            "length":   len(word_tokens),
            "pair_id":  pair_id,          # links word to its matched nonword
            "n_phon":   row.get("n_phon", ""),
        })
        output.append({
            "word":     nw_orth,
            "ipa":      nw_spaced,
            "label":    "nonword",
            "length":   len(nw_tokens),
            "pair_id":  pair_id,
            "n_phon":   row.get("n_phon_nw", ""),
        })

    # --- Summary -----------------------------------------------------------
    n_pairs = len(output) // 2
    print(f"\nTranscription complete:")
    print(f"  {n_pairs} pairs processed, {skipped} pairs skipped")
    if oov_count:
        print(f"  {oov_count} items have OOV tokens -- these will be dropped at eval time")
        print(f"  Tip: check stress markers (ˈ ˌ ː) -- common OOV cause")

    # --- Sample output -----------------------------------------------------
    print("\nSample pairs:")
    for i in range(0, min(8, len(output)), 2):
        w  = output[i]
        nw = output[i+1]
        print(f"  pair {w['pair_id']:>3}  "
              f"word: {w['word']:12s} -> {w['ipa']}")
        print(f"          "
              f"  nw: {nw['word']:12s} -> {nw['ipa']}")

    # --- Length match check ------------------------------------------------
    word_lens = [r["length"] for r in output if r["label"] == "word"]
    nw_lens   = [r["length"] for r in output if r["label"] == "nonword"]
    diffs = [abs(w - nw) for w, nw in zip(word_lens, nw_lens)]
    print(f"\nLength matching (word vs nonword phoneme count):")
    print(f"  Mean absolute difference: {sum(diffs)/max(len(diffs),1):.2f} phonemes")
    print(f"  Perfectly matched pairs:  {sum(1 for d in diffs if d == 0)}/{len(diffs)}")

    # --- Write output -------------------------------------------------------
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nOutput written to {OUTPUT_JSON}  ({len(output)} total items)")


if __name__ == "__main__":
    main()
