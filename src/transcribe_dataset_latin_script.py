# Attempt to transcribe the HuggingFace Wikipedia corpus

"""
wikipedia_streaming.py
Streams Wikipedia articles and feeds them into the existing IPATokenizer pipeline.
"""

import json
import re
import sys
import nltk
from itertools import islice
from typing import Iterator, List, Optional
from tqdm import tqdm

from datasets import load_dataset
from phonemizer import phonemize
from phonemizer.separator import Separator

from normalization.ipa_pipeline import IPATokenizer

# --- Handle segmentation faults ----------------------------------------------
# import faulthandler

# # Print traceback upon segmentation fault
# faulthandler.enable()


# --- Text cleaning -----------------------------------------------------------

def clean_wikipedia_text(text: str) -> str:
    """
    Light normalization pass specifically for Wikipedia article text.
    Handles noise that espeak/phonemizer won't handle gracefully.
    """
    text = re.sub(r'-\n', '', text)                # rejoins words that got split across lines
    text = re.sub(r'\n+', ' ', text)               # replaces out newline symbols with spaces
    text = re.sub(r'\[[\d\w]+\]', '', text)        # citations [1], [note 3]
    text = re.sub(r'\{\{.*?\}\}', '', text)        # template remnants
    text = re.sub(r'={2,}.*?={2,}', '', text)      # section headers == Foo ==
    text = re.sub(r'https?://\S+', '', text)       # URLs
    text = re.sub(r'\(\s*\)', '', text)            # empty parentheticals ()
    text = re.sub(r'\s+', ' ', text)               # collapse whitespace
    return text.strip()


# --- Language / script filtering ---------------------------------------------

# Even though wikimedia/wikipedia "20231101.en" is nominally English-only,
# articles (especially linguistics, history, and geography articles) frequently
# embed inline quotations in other scripts/languages. nltk.sent_tokenize doesn't
# know or care about language, so these get yielded as "sentences" and silently
# produce garbled IPA when run through an en-us espeak backend.

_LATIN_LETTER_RE = re.compile(r'[A-Za-z]')
_NON_LATIN_ALPHA_RE = re.compile(
    r'[\u0400-\u04FF'   # Cyrillic
    r'\u0370-\u03FF'    # Greek
    r'\u0590-\u05FF'    # Hebrew
    r'\u0600-\u06FF'    # Arabic
    r'\u4E00-\u9FFF'    # CJK Unified Ideographs
    r'\u3040-\u30FF'    # Hiragana / Katakana
    r'\u0900-\u097F'    # Devanagari
    r'\uAC00-\uD7AF'    # Hangul
    r']'
)


def is_mostly_latin(text: str, threshold: float = 0.9) -> bool:
    """
    Returns True if at least `threshold` fraction of alphabetic characters
    in the text are Latin-script. Cheap, fast first-pass filter that catches
    sentences in obviously non-Latin scripts (Cyrillic, Greek, CJK, etc).
    """
    latin = len(_LATIN_LETTER_RE.findall(text))
    non_latin = len(_NON_LATIN_ALPHA_RE.findall(text))
    total_alpha = latin + non_latin
    if total_alpha == 0:
        return False  # no recognizable letters at all -> not a usable sentence
    return (latin / total_alpha) >= threshold


# --- Sentence streaming ------------------------------------------------------

def stream_wikipedia_sentences(
    language: str = "en",
    min_length: int = 30,
    max_length: int = 400,
    buffer_articles: int = 100,
    latin_threshold: float = 0.9,
) -> Iterator[str]:
    """
    Yields cleaned, sentence-tokenized strings from the Wikipedia HuggingFace dataset.

    Args:
        language: Wikipedia language code (default 'en')
        min_length: Minimum character length to keep a sentence
        max_length: Maximum character length (longer sentences are skipped)
        buffer_articles: Number of articles to buffer before yielding sentences
        latin_threshold: Minimum fraction of alphabetic characters that must be
            Latin-script for a sentence to be kept. Filters out inline foreign-
            language quotations (Cyrillic, Greek, CJK, etc.) that slip through
            despite the "en" dataset config.
    """
    nltk.download('punkt_tab', quiet=True)   # Punkt identifies sentence boundaries in English
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )
    for article in dataset:
        cleaned = clean_wikipedia_text(article["text"])
        sentences = nltk.sent_tokenize(cleaned)
        for sent in sentences:
            sent = sent.strip()
            # filter strings by length to avoid grabbing weird text sequences
            # (too short: garbled snippets; too long: tables, lists etc)
            if not (min_length <= len(sent) <= max_length):
                continue
            # filter out sentences that are mostly non-Latin script (foreign-language
            # quotations embedded in English articles -- these produce garbled IPA
            # when phonemized with an en-us backend)
            if not is_mostly_latin(sent, threshold=latin_threshold):
                continue
            yield sent


# --- Main streaming pipeline -------------------------------------------------

def stream_to_ipa_corpus(
    output_path: str,
    start_sentence: int = 0,
    max_sentences: Optional[int] = None,
    batch_size: int = 256,
    vocab_warmup_sentences: int = 0,  # zero to skip vocab building
    save_vocab_path: str = "test_phoneme_vocab.json",
):
    """
    Streams Wikipedia, transcribes to IPA, and writes output using IPATokenizer.

    Args:
        output_path: Path for the output .jsonl corpus file
        start_sentence: How many qualifying sentences to skip before writing.
                        Lets you pick up where a previous chunk left off so
                        chunks don't overlap.
        max_sentences: Cap on sentences written this run (None = to end of corpus)
        batch_size: Sentences per phonemization batch
        vocab_warmup_sentences: How many sentences to use for vocab building
                                before writing the corpus. Set to 0 to skip
                                and load an existing vocab instead.
        save_vocab_path: Where to save/load the vocabulary JSON
    """
    tokenizer = IPATokenizer(language='en-us')
    sentence_stream = stream_wikipedia_sentences()

    # Pick up where the previous run left off, then take the next chunk.
    if start_sentence or max_sentences is not None:
        stop = None if max_sentences is None else start_sentence + max_sentences
        sentence_stream = islice(sentence_stream, start_sentence, stop)

    # --- Phase 1: vocab warmup -----------------------------------------------
    # Build vocab over an initial slice so phoneme_to_id is populated
    # before we start writing encoded sequences.
    if vocab_warmup_sentences > 0:
        print(f"Building vocabulary from first {vocab_warmup_sentences} sentences...")
        warmup_sentences, sentence_stream = _tee_n(sentence_stream, vocab_warmup_sentences)
        vocab_stats = tokenizer.build_vocabulary(warmup_sentences, phonemize)
        print(f"Vocab built: {vocab_stats['vocab_size']} tokens "
              f"({vocab_stats['num_phonemes']} phonemes)")
        tokenizer.save_vocabulary(save_vocab_path)
    else:
        print(f"Loading existing vocabulary from {save_vocab_path}")
        tokenizer.load_vocabulary(save_vocab_path)

    # --- Phase 2: stream + transcribe + write --------------------------------
    print(f"Streaming Wikipedia -> IPA corpus to {output_path} ...")
    total_written = 0
    failed = 0
    with open(output_path, 'w', encoding='utf-8') as out_f:
        batch: List[str] = []
        for sentence in tqdm(sentence_stream):
            batch.append(sentence)
            if len(batch) < batch_size:
                continue
            total_written, failed = _process_and_write_batch(
                batch, tokenizer, out_f, total_written, failed
            )
            batch = []
            if total_written % 1_000 == 0:
                print(f"  {total_written} sentences written...")
        # flush remaining
        if batch:
            total_written, failed = _process_and_write_batch(
                batch, tokenizer, out_f, total_written, failed
            )
    print(f"Done. {total_written} sentences written, {failed} failed.")
    return total_written, failed


def _process_and_write_batch(batch, tokenizer, out_f, total_written, failed):
    """Phonemize a batch and write each result as a JSONL line."""
    for sentence in batch:
        try:
            ids = tokenizer.encode(sentence, phonemize, add_sos=True, add_eos=True)
            out_f.write(json.dumps({
                "text": sentence,
                "ids": ids,
            }) + "\n")
            total_written += 1
        except Exception as e:
            print(f"  Warning: failed on sentence: {e}")
            failed += 1
    return total_written, failed


def _tee_n(iterator: Iterator, n: int):
    """
    Consume the first n items from an iterator as a list,
    and return (list, remainder_iterator).
    """
    head = list(islice(iterator, n))
    return head, iterator


# --- Entry point -------------------------------------------------------------

if __name__ == "__main__":
    CHUNK = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    CHUNK_SIZE = 170_000
    stream_to_ipa_corpus(
        output_path=f"data/raw/wikipedia_ipa_chunk{CHUNK:02d}.jsonl",
        start_sentence=CHUNK * CHUNK_SIZE,
        max_sentences=CHUNK_SIZE,
        batch_size=256,
        vocab_warmup_sentences=0,  # make this nonzero to train the vocab if you don't have one already
        save_vocab_path="data/vocab/phoneme_vocab.json",
    )