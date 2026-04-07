# Attempt to transcribe the HuggingFace Wikipedia corpus

"""
wikipedia_streaming.py
Streams Wikipedia articles and feeds them into the existing IPATokenizer pipeline.
"""

import json
import re
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
    
    # Try to catch utf encoding errors resulting from mis-formatted strings early
    # TODO: check to see if the normalization process is messing up: 
    # e.g. text = unicodedata.normalize('NFKC', text) in the text_normalizer.py script keeping this 
    # program from recognizing the strings, and that's why i ended up adding this ignore errors
    # line, but this line might in turn be preventing the program from tokenizing periods and 
    # words like `hierarchy` appropriately????
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    # TODO: clean up newline markers as well? or maybe not, but devote a vocabulary item/tag for them? 
    text = re.sub(r'-\n', '', text) 			  # rejoins words that got split across lines
    text = re.sub(r'\n+', ' ', text)			  # replaces out newline symbols with spaces
    text = re.sub(r'\[[\d\w]+\]', '', text)       # citations [1], [note 3]
    text = re.sub(r'\{\{.*?\}\}', '', text)        # template remnants
    text = re.sub(r'={2,}.*?={2,}', '', text)      # section headers == Foo ==
    text = re.sub(r'https?://\S+', '', text)       # URLs
    text = re.sub(r'\(\s*\)', '', text)            # empty parentheticals ()
    text = re.sub(r'\s+', ' ', text)               # collapse whitespace
    return text.strip()


# --- Sentence streaming ------------------------------------------------------

def stream_wikipedia_sentences(
    language: str = "en",
    min_length: int = 30,
    max_length: int = 400,
    buffer_articles: int = 100,
) -> Iterator[str]:
    """
    Yields cleaned, sentence-tokenized strings from the Wikipedia HuggingFace dataset.

    Args:
        language: Wikipedia language code (default 'en')
        min_length: Minimum character length to keep a sentence
        max_length: Maximum character length (longer sentences are skipped)
        buffer_articles: Number of articles to buffer before yielding sentences
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
    		# filter strings by length to avoid grabbing weird text sequences (too short: garbled snippets; too long: tables, lists etc)
    		if min_length <= len(sent) <= max_length:
    			yield sent


# --- Main streaming pipeline -------------------------------------------------

def stream_to_ipa_corpus(
    output_path: str,
    max_sentences: Optional[int] = None,
    batch_size: int = 256,
    vocab_warmup_sentences: int = 200,
    save_vocab_path: str = "test_phoneme_vocab.json",
):
    """
    Streams Wikipedia, transcribes to IPA, and writes output using IPATokenizer.

    Args:
        output_path: Path for the output .jsonl corpus file
        max_sentences: Cap on total sentences (None = full Wikipedia)
        batch_size: Sentences per phonemization batch
        vocab_warmup_sentences: How many sentences to use for vocab building
                                before writing the corpus. Set to 0 to skip
                                and load an existing vocab instead.
        save_vocab_path: Where to save/load the vocabulary JSON
    """
    tokenizer = IPATokenizer(language='en-us')
    sentence_stream = stream_wikipedia_sentences()
    if max_sentences:
    	sentence_stream = islice(sentence_stream, max_sentences)
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
    # TODO: issues start emerging here now (although, still present above occasionally, so there must be "bad" sentences?)
    # NOTE: issues don't come up at all if I run on the home desktop?
    print(f"Streaming Wikipedia → IPA corpus to {output_path} ...")
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
	stream_to_ipa_corpus(
		output_path="data/raw/wikipedia_ipa_50000.jsonl",
		max_sentences=50_000,       # 500_000 is good starting size for LM pretraining
		batch_size=256,
		vocab_warmup_sentences=0,  # make this nonzero to train the vocab if you don't have one already
		save_vocab_path="data/vocab/phoneme_vocab.json",
		)
