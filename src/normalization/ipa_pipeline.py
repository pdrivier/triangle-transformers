"""
Complete Text-to-IPA Pipeline
Demonstrates integration of text normalization with phonemization
"""

import re

from normalization.text_normalizer import TextNormalizer
from typing import List, Dict
from tqdm import tqdm



class IPATokenizer:
    """Complete pipeline for converting text to IPA phoneme tokens."""
    
    def __init__(self, language='en-us', preserve_case=True):
        """
        Initialize the IPA tokenizer.
        
        Args:
            language: Language code for phonemization (e.g., 'en-us', 'en-gb')
            preserve_case: Whether to preserve text casing
        """
        self.normalizer = TextNormalizer(preserve_case=preserve_case)
        self.language = language
        
        # Special tokens
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<SOS>': 2,  # Start of sequence
            '<EOS>': 3,  # End of sequence
            '<SPACE>': 4,  # Word boundary
        }
        
        # Will be populated during vocabulary building
        self.phoneme_to_id = {}
        self.id_to_phoneme = {}
        self.vocab_size = 0
        

    def load_corpus_from_file(self, file_path: str, encoding='utf-8') -> List[str]:
        """
        Load corpus from a text file.
        
        Args:
            file_path: Path to the corpus text file
            encoding: Text encoding (default: utf-8)
            
        Returns:
            List of text strings (one per line)
        """
        corpus = []
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                for line in f:
                    # Strip whitespace and skip empty lines
                    text = line.strip()
                    if text:
                        corpus.append(text)
            
            print(f"Loaded {len(corpus)} texts from {file_path}")
            return corpus
            
        except FileNotFoundError:
            print(f"Error: File not found: {file_path}")
            return []
        except Exception as e:
            print(f"Error loading corpus: {e}")
            return []

    def process_text(self, text: str, phonemize_fn=None) -> List[str]:
        """
        Process text through normalization and phonemization (batched).
        
        Args:
            text: Input text string
            phonemize_fn: Function that takes a LIST of words and returns LIST of IPA strings
                         If None, returns normalized tokens without phonemization
        
        Returns:
            List of tokens (phonemes for words, special tokens for punctuation)
        """
        # Step 1: Normalize text
        normalized_tokens = self.normalizer.normalize(text)
        
        # Step 2: Handle non-phonemized case
        if phonemize_fn is None:
            return normalized_tokens
        
        # Step 3: Separate special tokens from words
        words_to_phonemize = []
        token_map = []  # Track positions and types
        
        for token in normalized_tokens:
            if token.startswith('<') and token.endswith('>'):
                # Special token - keep as-is
                token_map.append(('special', token))
            else:
                # Regular word - mark for phonemization
                token_map.append(('word', len(words_to_phonemize)))
                words_to_phonemize.append(token)
        
        # Step 4: Batch phonemize all words at once
        if words_to_phonemize:
            ipa_results = phonemize_fn(words_to_phonemize)
        else:
            ipa_results = []
        
        # Step 5: Reconstruct output with phonemes and special tokens
        output_tokens = []
        for token_type, token_data in token_map:
            if token_type == 'special':
                # Keep special token as-is
                output_tokens.append(token_data)
            else:
                # Get phonemized result for this word
                word_idx = token_data
                ipa = ipa_results[word_idx] if word_idx < len(ipa_results) else ""
                
                if ipa:
                    # Split IPA into individual phonemes
                    phonemes = self._split_ipa(ipa)
                    output_tokens.extend(phonemes)
                    # Add word boundary marker
                    output_tokens.append('<SPACE>')
                else:
                    # Unknown word
                    output_tokens.append('<UNK>')
        
        return output_tokens

    # def phonemize_words_batch(self, words: List[str]) -> List[str]:
    #     """
    #     Convert multiple words to IPA using phonemizer (batch mode).
        
    #     Args:
    #         words: List of words to phonemize
            
    #     Returns:
    #         List of IPA transcription strings (one per word)
    #     """
    #     if not self.phonemizer_available or not words:
    #         return [""] * len(words)
        
    #     try:
    #         # Join words with a unique separator that won't appear in text
    #         separator = " | "
    #         combined_text = separator.join(words)
            
    #         # Phonemize the combined text
    #         ipa_combined = self.phonemize(
    #             combined_text,
    #             language=self.language,
    #             backend=self.backend,
    #             strip=True,
    #             preserve_punctuation=False,
    #             with_stress=True
    #         )
            
    #         # Split back into individual words
    #         ipa_words = ipa_combined.split(separator)
            
    #         # Ensure we have the right number of results
    #         if len(ipa_words) != len(words):
    #             # Fallback: pad with empty strings
    #             ipa_words.extend([""] * (len(words) - len(ipa_words)))
            
    #         return [ipa.strip() for ipa in ipa_words]
            
    #     except Exception as e:
    #         print(f"Warning: Could not phonemize words: {e}")
    #         return [""] * len(words)
    
    def _split_ipa(self, ipa_string: str) -> List[str]:
        """
        Split IPA string into individual phoneme tokens.
        This is a simplified version - you may need more sophisticated parsing.
        
        Args:
            ipa_string: IPA transcription string
            
        Returns:
            List of individual phonemes
        """
        # Remove spaces
        ipa_string = ipa_string.strip()
        
        # This is a basic character-by-character split
        # For production, you'd want to handle:
        # - Diacritics (combining characters)
        # - Multi-character phonemes (affricates, etc.)
        # - Stress markers
        
        phonemes = []
        i = 0
        while i < len(ipa_string):
            char = ipa_string[i]
            
            # Skip whitespace
            if char.isspace():
                i += 1
                continue
            
            # Check for combining diacritics (Unicode combining marks)
            phoneme = char
            i += 1
            while i < len(ipa_string) and self._is_combining_mark(ipa_string[i]):
                phoneme += ipa_string[i]
                i += 1
            
            phonemes.append(phoneme)
        
        return phonemes
    
    def _is_combining_mark(self, char: str) -> bool:
        """Check if character is a Unicode combining mark."""
        import unicodedata
        return unicodedata.category(char) in ('Mn', 'Mc', 'Me')
    
    def build_vocabulary(self, corpus: List[str], phonemize_fn=None) -> Dict:
        """
        Build phoneme vocabulary from a corpus.
        
        Args:
            corpus: List of text strings
            phonemize_fn: Function to phonemize words
            
        Returns:
            Dictionary with vocabulary statistics
        """
        # Start with special tokens (already uppercase)
        self.phoneme_to_id = self.special_tokens.copy()
        
        # Add punctuation tokens (normalize to uppercase)
        punct_tokens = self.normalizer.get_punctuation_vocab()
        seen_special = set(self.phoneme_to_id.keys())  # Track what we've added
        
        for token in punct_tokens:
            # Normalize special tokens to uppercase
            if token.startswith('<') and token.endswith('>'):
                normalized_token = token.upper()
            else:
                normalized_token = token
            
            # Only add if we haven't seen this token (case-insensitive for special tokens)
            if normalized_token not in seen_special:
                self.phoneme_to_id[normalized_token] = len(self.phoneme_to_id)
                seen_special.add(normalized_token)
        
        # Collect all phonemes from corpus
        phoneme_counts = {}
        for text in tqdm(corpus):
            tokens = self.process_text(text, phonemize_fn)
            for token in tokens:
                # Normalize special tokens to uppercase
                if token.startswith('<') and token.endswith('>'):
                    token = token.upper()
                
                # Skip if already in vocab
                if token in self.phoneme_to_id:
                    continue
                
                # Count phonemes (not special tokens)
                if not (token.startswith('<') and token.endswith('>')):
                    phoneme_counts[token] = phoneme_counts.get(token, 0) + 1
        
        # Add phonemes to vocabulary (sorted by frequency)
        sorted_phonemes = sorted(phoneme_counts.items(), 
                                key=lambda x: x[1], 
                                reverse=True)
        
        for phoneme, count in sorted_phonemes:
            if phoneme not in self.phoneme_to_id:
                self.phoneme_to_id[phoneme] = len(self.phoneme_to_id)
        
        # Create reverse mapping
        self.id_to_phoneme = {v: k for k, v in self.phoneme_to_id.items()}
        self.vocab_size = len(self.phoneme_to_id)
        
        return {
            'vocab_size': self.vocab_size,
            'num_phonemes': len(sorted_phonemes),
            'num_special_tokens': len(seen_special),
            'num_punct_tokens': len([t for t in seen_special if t in punct_tokens or t.upper() in punct_tokens]),
            'most_common': sorted_phonemes[:20]
        }
        
    def encode(self, text: str, phonemize_fn=None, 
               add_sos=False, add_eos=False) -> List[int]:
        """
        Encode text to phoneme IDs.
        
        Args:
            text: Input text
            phonemize_fn: Phonemization function
            add_sos: Add start-of-sequence token
            add_eos: Add end-of-sequence token
            
        Returns:
            List of phoneme IDs
        """
        tokens = self.process_text(text, phonemize_fn)
        
        ids = []
        if add_sos:
            ids.append(self.special_tokens['<SOS>'])
        
        for token in tokens:
            token_id = self.phoneme_to_id.get(token, self.special_tokens['<UNK>'])
            ids.append(token_id)
        
        if add_eos:
            ids.append(self.special_tokens['<EOS>'])
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode phoneme IDs back to tokens.
        
        Args:
            ids: List of phoneme IDs
            
        Returns:
            String representation
        """
        tokens = [self.id_to_phoneme.get(id, '<UNK>') for id in ids]
        return ' '.join(tokens)

    def generate_ipa_corpus(self, corpus: List[str], output_path: str, phonemize_fn=None, 
                       format: str = 'text') -> Dict:
        """
        Generate a corpus file in IPA format for autoregressive language modeling.
        
        Args:
            corpus: List of text strings to convert
            output_path: Path to save the output file
            phonemize_fn: Function to phonemize words (batch)
            format: Output format - 'text' (one sequence per line) or 'json' (structured)
            
        Returns:
            Dictionary with generation statistics
        """
        ipa_sequences = []
        total_tokens = 0
        failed_count = 0
        
        print(f"Generating IPA corpus from {len(corpus)} texts...")
        
        for idx, text in enumerate(corpus):
            try:
                # Process text to get IPA tokens
                tokens = self.process_text(text, phonemize_fn)
                
                # Normalize special tokens to uppercase
                normalized_tokens = []
                for token in tokens:
                    if token.startswith('<') and token.endswith('>'):
                        normalized_tokens.append(token.upper())
                    else:
                        normalized_tokens.append(token)
                
                # Add SOS and EOS tokens
                sequence = ['<SOS>'] + normalized_tokens + ['<EOS>']
                ipa_sequences.append(sequence)
                total_tokens += len(sequence)
                
                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1}/{len(corpus)} texts...")
                    
            except Exception as e:
                print(f"Warning: Failed to process text {idx}: {e}")
                failed_count += 1
                continue
        
        # Write to file based on format
        if format == 'text':
            self._write_text_format(output_path, ipa_sequences)
        elif format == 'json':
            self._write_json_format(output_path, ipa_sequences)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'text' or 'json'")
        
        stats = {
            'total_sequences': len(ipa_sequences),
            'total_tokens': total_tokens,
            'avg_tokens_per_sequence': total_tokens / len(ipa_sequences) if ipa_sequences else 0,
            'failed_count': failed_count,
            'output_path': output_path
        }
        
        print(f"\nCorpus generation complete!")
        print(f"Total sequences: {stats['total_sequences']}")
        print(f"Total tokens: {stats['total_tokens']}")
        print(f"Average tokens per sequence: {stats['avg_tokens_per_sequence']:.2f}")
        print(f"Output saved to: {output_path}")
        
        return stats


    def _write_text_format(self, output_path: str, sequences: List[List[str]]):
        """Write corpus in plain text format (one sequence per line, space-separated)."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for sequence in sequences:
                # Join tokens with spaces
                line = ' '.join(sequence)
                f.write(line + '\n')


    def _write_json_format(self, output_path: str, sequences: List[List[str]]):
        """Write corpus in JSON format with metadata."""
        import json
        
        corpus_data = {
            'metadata': {
                'vocab_size': self.vocab_size,
                'language': self.language,
                'num_sequences': len(sequences)
            },
            'vocabulary': self.phoneme_to_id,
            'sequences': sequences
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(corpus_data, f, ensure_ascii=False, indent=2)

    
    def save_vocabulary(self, filepath: str):
        """Save vocabulary to file."""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'phoneme_to_id': self.phoneme_to_id,
                'vocab_size': self.vocab_size,
                'language': self.language
            }, f, ensure_ascii=False, indent=2)
    
    def load_vocabulary(self, filepath: str):
        """Load vocabulary from file."""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.phoneme_to_id = data['phoneme_to_id']
            self.vocab_size = data['vocab_size']
            self.language = data.get('language', 'en-us')
            self.id_to_phoneme = {int(v): k for k, v in self.phoneme_to_id.items()}


# Example usage with mock phonemization
if __name__ == "__main__":
    
    from phonemizer import phonemize
    from ipa_pipeline import IPATokenizer

    
    # Initialize tokenizer
    tokenizer = IPATokenizer(language='en-us')
    
    # Example texts
    corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "Natural language processing is fascinating.",
    "In the immediate blinding agony of the aftermath, he remembered thinking, unaccountably, about the beasts of a certain zodiac.",
    "But it wasn’t the one whose organization he’d once pledged allegiance to, whose many namesakes he’d piloted or otherwise controlled.",
    "He supposed, when it came down to it, that it wouldn’t be completely out of character for one such as he to turn coat once more, cozy up and play false with a new menagerie of forms and entities, seeking meaning where it had surely run as dry as anywhere else he’d sojourned.",
    "Is that what the little prince of old had been in search of as he traipsed through vacuum? Meaning? And, finding himself disappointed as planet after planet promised large and delivered small, he’d perhaps devised his own crash landing on Earth, thinking to collapse himself out of existence, but found instead (or, in addition?) a pilot slowly dying of thirst and exposure while attempting to repair the irreparable; a fox offering wisdom; and a serpent, comfort.",
    # ... more texts
]
    
    print("Text-to-IPA Pipeline Demo")
    print("=" * 60)
    
    # Process each text
    for text in corpus:
        print(f"\nOriginal: {text}")
        
        # Step 1: Normalized tokens
        # normalized = tokenizer.normalizer.normalize(text)
        # print(f"Normalized: {normalized}")
        
        # Step 2: IPA tokens
        ipa_tokens = tokenizer.process_text(text, phonemize)
        print(f"IPA tokens: {ipa_tokens}")
    
    # Build vocabulary
    print("\n" + "=" * 60)
    print("Building vocabulary...")
    vocab_stats = tokenizer.build_vocabulary(corpus, phonemize)
    print(f"\nVocabulary Statistics:")
    print(f"  Total vocab size: {vocab_stats['vocab_size']}")
    print(f"  Unique phonemes: {vocab_stats['num_phonemes']}")
    print(f"  Special tokens: {vocab_stats['num_special_tokens']}")
    print(f"  Punctuation tokens: {vocab_stats['num_punct_tokens']}")
    
    # Encode/decode example
    print("\n" + "=" * 60)
    print("Encoding/Decoding Example:")
    test_text = "Hello, world!"
    encoded = tokenizer.encode(test_text, phonemize, add_sos=True, add_eos=True)
    decoded = tokenizer.decode(encoded)
    print(f"Original: {test_text}")
    print(f"Encoded IDs: {encoded}")
    print(f"Decoded: {decoded}")
    
    # Save vocabulary
    tokenizer.save_vocabulary('phoneme_vocab.json')
    print("\nVocabulary saved to phoneme_vocab.json")
