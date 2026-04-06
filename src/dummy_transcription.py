# This is a test script to try out IPA phoneme-tokenization for a sample corpus.

import os
from normalization.ipa_pipeline import IPATokenizer
from phonemizer import phonemize

# Initialize
tokenizer = IPATokenizer(language='en-us')

corpus = tokenizer.load_corpus_from_file(os.path.join('normalization/', 'train_corpus.txt'))


# for text in corpus: 
# 	ipa_tokens = tokenizer.process_text(text, phonemize)
# 	print(f"IPA tokens: {ipa_tokens}")

# Build vocabulary
vocab_stats = tokenizer.build_vocabulary(corpus, phonemize)
print(f"Vocabulary size: {vocab_stats['vocab_size']}")
print(f"Unique phonemes: {vocab_stats['num_phonemes']}")

# Save vocabulary
tokenizer.save_vocabulary('phoneme_vocab.json')

# Generate and save IPA corpus file
stats = tokenizer.generate_ipa_corpus(
    corpus=corpus,
    output_path='sample_corpus_ipa.txt',
    phonemize_fn=phonemize,
    format='text'  # or 'json' for structured output
)



