# IPA Phoneme Tokenization Pipeline

A complete pipeline for converting text to IPA (International Phonetic Alphabet) phoneme-level tokens, designed for training autoregressive transformer models.

## Features

✅ **Text Normalization**
- Handles punctuation as separate tokens (periods, commas, quotes, etc.)
- Expands numbers to words (0-999)
- Expands common abbreviations (Dr., Mr., St., etc.)
- Preserves contractions (can't, won't, etc.)
- Unicode normalization

✅ **IPA Phonemization**
- Integration with `phonemizer` library
- Support for multiple languages
- Handles multi-character phonemes and diacritics
- Stress markers preserved

✅ **Vocabulary Management**
- Special tokens (PAD, UNK, SOS, EOS, SPACE)
- Punctuation tokens preserved
- Phoneme-to-ID mapping
- Save/load vocabulary

## Installation

### Required
```bash
pip install phonemizer
```

### System Dependencies

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install espeak-ng
```

**macOS:**
```bash
brew install espeak-ng #make sure its chip compatibility matches that of the Anaconda/Python distribution 
```

**Windows:**
Download and install espeak-ng from: https://github.com/espeak-ng/espeak-ng/releases

## File Structure

```
text_normalizer.py       # Core text normalization (no dependencies)
ipa_pipeline.py          # Complete pipeline with mock phonemizer
production_pipeline.py   # Production pipeline with real phonemizer
phoneme_vocab.json      # Example vocabulary file (generated)
README.md               # This file
```

## Quick Start

### 1. Basic Text Normalization (No dependencies)

```python
from text_normalizer import TextNormalizer

normalizer = TextNormalizer(preserve_case=False)
text = "Hello, world! I can't believe it's 2024."
tokens = normalizer.normalize(text)

print(tokens)
# ['hello', '<comma>', 'world', '<exclaim>', 'i', "can't", 
#  'believe', "it's", 'two', 'thousand', 'and', 'twenty', 
#  'four', '<period>']
```

### 2. Complete Pipeline (Requires phonemizer)

```python
from production_pipeline import ProductionIPATokenizer

# Initialize
tokenizer = ProductionIPATokenizer(language='en-us', backend='espeak')

# Process a single text
text = "Hello, world!"
result = tokenizer.process_corpus([text], show_progress=False)[0]

print(result)
# ['h', 'ə', 'l', 'oʊ', '<SPACE>', '<comma>', 
#  'w', 'ɝ', 'l', 'd', '<SPACE>', '<exclaim>']
```

### 3. Process a Corpus and Build Vocabulary

```python
from ipa_pipeline import IPATokenizer

# Initialize
tokenizer = IPATokenizer(language='en-us')

# Your corpus
corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "Natural language processing is fascinating.",
    # ... more texts
]

# Build vocabulary (with real phonemizer)
def phonemize_word(word):
    from phonemizer import phonemize
    return phonemize(word, language='en-us', backend='espeak', strip=True)

vocab_stats = tokenizer.build_vocabulary(corpus, phonemize_word)

print(f"Vocabulary size: {vocab_stats['vocab_size']}")
print(f"Unique phonemes: {vocab_stats['num_phonemes']}")

# Save vocabulary
tokenizer.save_vocabulary('my_vocab.json')
```

### 4. Encode Text for Model Training

```python
# Encode text to IDs
text = "Hello, world!"
ids = tokenizer.encode(text, phonemize_word, add_sos=True, add_eos=True)

print(f"Token IDs: {ids}")
# [2, 24, 25, 26, 27, 4, 28, 29, 30, 31, 4, 32, 3]

# Decode back
decoded = tokenizer.decode(ids)
print(f"Decoded: {decoded}")
```

## Supported Languages

The pipeline supports any language supported by espeak-ng, including:

- English (US): `en-us`
- English (GB): `en-gb`
- French: `fr-fr`
- German: `de`
- Spanish: `es`
- Italian: `it`
- Portuguese: `pt`
- Russian: `ru`
- Chinese: `cmn`
- Japanese: `ja`
- And 100+ more...

Change language when initializing:
```python
tokenizer = ProductionIPATokenizer(language='fr-fr')
```

## Punctuation Tokens

The pipeline preserves the following punctuation as special tokens:

| Punctuation | Token |
|------------|-------|
| . | `<PERIOD>` |
| , | `<COMMA>` |
| ! | `<EXCLAIM>` |
| ? | `<QUESTION>` |
| ; | `<SEMICOLON>` |
| : | `<COLON>` |
| ... | `<ELLIPSIS>` |
| " | `<QUOTE>` |
| - | `<DASH>` |
| — | `<EMDASH>` |
| ( ) | `<LPAREN>` `<RPAREN>` |
| [ ] | `<LBRACKET>` `<RBRACKET>` |

## Special Tokens

| Token | ID | Purpose |
|-------|----|----|
| `<PAD>` | 0 | Padding |
| `<UNK>` | 1 | Unknown |
| `<SOS>` | 2 | Start of sequence |
| `<EOS>` | 3 | End of sequence |
| `<SPACE>` | 4 | Word boundary |

## Customization

### Custom Number Handling

For numbers > 999, modify `_expand_numbers()` in `text_normalizer.py`:

```python
def _expand_numbers(self, text: str) -> str:
    # Add support for thousands, millions, etc.
    pass
```

### Custom Abbreviations

Add to the `abbreviations` dict in `_expand_abbreviations()`:

```python
abbreviations = {
    r'\bMr\.': 'Mister',
    r'\bCustom\.': 'Custom expansion',
    # Add your own...
}
```

### Custom Phoneme Splitting

Override `_split_ipa_detailed()` for language-specific phoneme handling:

```python
def _split_ipa_detailed(self, ipa_string: str) -> List[str]:
    # Custom logic for your language
    pass
```

## Example: Processing a Book

```python
# Load a text file
with open('my_book.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Split into sentences/paragraphs
sentences = text.split('\n')

# Process
tokenizer = ProductionIPATokenizer(language='en-us')
results = tokenizer.process_corpus(sentences, show_progress=True)

# Build vocabulary from results
# (flatten all tokens)
all_tokens = [token for sent in results for token in sent]

# Save for training
import json
with open('processed_data.json', 'w') as f:
    json.dump(results, f)
```

## For Transformer Training

Your typical workflow:

1. **Preprocess corpus** → Tokenize to IPA
2. **Build vocabulary** → Create phoneme-to-ID mapping
3. **Create dataset** → Convert texts to ID sequences
4. **Collate batches** → Pad sequences, create attention masks
5. **Train model** → Feed to transformer

Example dataset class:

```python
import torch
from torch.utils.data import Dataset

class IPADataset(Dataset):
    def __init__(self, texts, tokenizer, phonemize_fn, max_length=512):
        self.tokenizer = tokenizer
        self.phonemize_fn = phonemize_fn
        self.max_length = max_length
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        ids = self.tokenizer.encode(
            text, 
            self.phonemize_fn,
            add_sos=True,
            add_eos=True
        )
        
        # Truncate if needed
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
        
        return torch.tensor(ids, dtype=torch.long)
```

## Troubleshooting

### "phonemizer not installed"
```bash
pip install phonemizer
```

### "espeak-ng not found"
Install system dependency (see Installation section)

### "Could not phonemize word"
- Check language code is correct
- Some words may not be in espeak-ng's dictionary
- Check espeak-ng is installed correctly: `espeak-ng --version`

### Memory issues with large corpus
Process in batches:

```python
batch_size = 1000
for i in range(0, len(corpus), batch_size):
    batch = corpus[i:i+batch_size]
    results = tokenizer.process_corpus(batch)
    # Save results incrementally
```

## Performance Tips

- **Batch processing**: Process multiple texts at once
- **Caching**: Cache phonemized words to avoid re-computation
- **Parallel processing**: Use multiprocessing for large corpora
- **Vocabulary pruning**: Remove rare phonemes if needed

## License

This code is provided as-is for educational and research purposes.

## Contributing

Feel free to extend this pipeline:
- Add support for more languages
- Improve phoneme splitting logic
- Add caching mechanisms
- Implement parallel processing
- Add more test cases
