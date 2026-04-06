"""
Production Pipeline with Real Phonemizer Integration

Installation required:
    pip install phonemizer
    
System requirements:
    - espeak-ng (Linux: apt-get install espeak-ng)
    - Festival (optional)
"""

from text_normalizer import TextNormalizer
from typing import List, Optional
import re


class ProductionIPATokenizer:
    """Production-ready IPA tokenizer with real phonemizer."""
    
    def __init__(self, language='en-us', backend='espeak'):
        """
        Initialize with real phonemizer.
        
        Args:
            language: Language code (e.g., 'en-us', 'en-gb', 'fr-fr', 'de')
            backend: Phonemizer backend ('espeak', 'festival', 'segments')
        """
        self.normalizer = TextNormalizer(preserve_case=False)
        self.language = language
        self.backend = backend
        
        # Try to import phonemizer
        try:
            from phonemizer import phonemize
            from phonemizer.backend import EspeakBackend
            self.phonemize = phonemize
            self.phonemizer_available = True
            
            # Initialize backend to check it's working
            if backend == 'espeak':
                self.backend_obj = EspeakBackend(language, preserve_punctuation=False)
                print(f"✓ Phonemizer initialized with {backend} for {language}")
        except ImportError:
            self.phonemizer_available = False
            print("⚠ Warning: phonemizer not installed. Install with: pip install phonemizer")
        except Exception as e:
            self.phonemizer_available = False
            print(f"⚠ Warning: Could not initialize phonemizer: {e}")


def phonemize_corpus(self, corpus: list[str]) -> list[str]:
    """
    Convert multiple texts to IPA efficiently.
    
    Args:
        corpus: List of text strings to phonemize
        
    Returns:
        List of IPA transcription strings
    """
    if not self.phonemizer_available:
        return [""] * len(corpus)
    
    try:
        # Phonemize all texts in a single batch - most efficient
        ipa_results = self.phonemize(
            corpus,
            language=self.language,
            backend=self.backend,
            strip=True,
            preserve_punctuation=True,
            with_stress=True
        )
        return [ipa.strip() for ipa in ipa_results]
    except Exception as e:
        print(f"Warning: Could not phonemize corpus: {e}")
        return [""] * len(corpus)
    
    def phonemize_word(self, word: str) -> str: 
        """
        Convert a single word to IPA using phonemizer.
        
        Args:
            word: Single word to phonemize
            
        Returns:
            IPA transcription string
        """
        if not self.phonemizer_available:
            return ""
        
        try:
            # Phonemize the word
            ipa = self.phonemize(
                word,
                language=self.language,
                backend=self.backend,
                strip=True,
                preserve_punctuation=False,
                with_stress=True  # Keep stress markers
            )
            return ipa.strip()
        except Exception as e:
            print(f"Warning: Could not phonemize '{word}': {e}")
            return ""
    
    def process_corpus(self, texts: List[str], 
                      show_progress: bool = True) -> List[List[str]]:
        """
        Process a corpus of texts to IPA tokens.
        
        Args:
            texts: List of text strings
            show_progress: Whether to show progress
            
        Returns:
            List of token lists (one per text)
        """
        results = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            # Normalize text
            normalized_tokens = self.normalizer.normalize(text)
            
            # Convert to IPA
            ipa_tokens = []
            for token in normalized_tokens:
                # Check if punctuation token
                if token.startswith('<') and token.endswith('>'):
                    ipa_tokens.append(token)
                else:
                    # Phonemize the word
                    ipa = self.phonemize_word(token)
                    if ipa:
                        # Split into phonemes
                        phonemes = self._split_ipa_detailed(ipa)
                        ipa_tokens.extend(phonemes)
                        ipa_tokens.append('<SPACE>')
            
            results.append(ipa_tokens)
            
            if show_progress and (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{total} texts...")
        
        return results
    
    def _split_ipa_detailed(self, ipa_string: str) -> List[str]:
        """
        Split IPA into phonemes, handling multi-char phonemes and diacritics.
        
        This handles:
        - Combining diacritics (stress, length, nasalization, etc.)
        - Affricates (tʃ, dʒ)
        - Long vowels (aː)
        """
        phonemes = []
        i = 0
        
        # Common multi-character phonemes (affricates, etc.)
        multi_char = ['tʃ', 'dʒ', 'ts', 'dz', 'pf', 'ks']
        
        while i < len(ipa_string):
            # Skip whitespace
            if ipa_string[i].isspace():
                i += 1
                continue
            
            # Check for multi-character phonemes
            matched = False
            for mc in multi_char:
                if ipa_string[i:i+len(mc)] == mc:
                    phoneme = mc
                    i += len(mc)
                    matched = True
                    break
            
            if not matched:
                # Single character phoneme
                phoneme = ipa_string[i]
                i += 1
            
            # Collect any combining marks or modifiers
            while i < len(ipa_string):
                char = ipa_string[i]
                # Check if it's a diacritic or modifier
                if self._is_diacritic(char):
                    phoneme += char
                    i += 1
                else:
                    break
            
            phonemes.append(phoneme)
        
        return phonemes
    
    def _is_diacritic(self, char: str) -> bool:
        """Check if character is an IPA diacritic or modifier."""
        import unicodedata
        
        # Combining marks
        if unicodedata.category(char) in ('Mn', 'Mc', 'Me'):
            return True
        
        # Common IPA modifiers and diacritics
        diacritics = {
            'ː',  # Length mark
            'ˑ',  # Half-length
            'ˈ',  # Primary stress
            'ˌ',  # Secondary stress
            '̃',   # Nasalization
            '̥',   # Voiceless
            '̩',   # Syllabic
            'ʰ',  # Aspirated
            'ʷ',  # Labialized
            'ʲ',  # Palatalized
            '̚',   # No audible release
        }
        
        return char in diacritics


def main():
    """Example usage with real phonemizer."""
    
    print("Production IPA Tokenization Pipeline")
    print("=" * 70)
    
    # Initialize tokenizer
    tokenizer = ProductionIPATokenizer(language='en-us', backend='espeak')
    
    if not tokenizer.phonemizer_available:
        print("\nUsing mock phonemization (install phonemizer for real IPA)")
        print("Install with: pip install phonemizer")
        print("System dependency: espeak-ng")
        return
    
    # Example texts
    test_texts = [
        "Hello, world! How are you today?",
        "The quick brown fox jumps over the lazy dog.",
        "I can't believe it's already 2024!",
        "Natural language processing is fascinating.",
    ]
    
    print("\n" + "=" * 70)
    print("Processing Example Texts")
    print("=" * 70)
    
    # Process texts
    for text in test_texts[:2]:  # Just first 2 for demo
        print(f"\nOriginal text:")
        print(f"  {text}")
        
        # Get normalized tokens
        normalized = tokenizer.normalizer.normalize(text)
        print(f"\nNormalized tokens:")
        print(f"  {' | '.join(normalized)}")
        
        # Get IPA phonemes
        ipa_result = tokenizer.process_corpus([text], show_progress=False)[0]
        print(f"\nIPA phoneme tokens:")
        print(f"  {' '.join(ipa_result)}")
        
        # Show phoneme count
        phoneme_count = len([t for t in ipa_result if not t.startswith('<')])
        special_count = len([t for t in ipa_result if t.startswith('<')])
        print(f"\nToken counts:")
        print(f"  Phonemes: {phoneme_count}")
        print(f"  Special tokens: {special_count}")
        print(f"  Total: {len(ipa_result)}")
    
    print("\n" + "=" * 70)
    print("\nTo process a large corpus:")
    print("  1. Load your text files")
    print("  2. Call tokenizer.process_corpus(texts)")
    print("  3. Build vocabulary from results")
    print("  4. Save vocabulary and tokenized data")
    print("=" * 70)


if __name__ == "__main__":
    main()
