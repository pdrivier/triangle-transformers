"""
Text Normalization Pipeline for IPA Phoneme Tokenization
Handles punctuation extraction, number expansion, and text cleaning
while preserving punctuation as separate tokens.
"""

import re
from typing import List, Tuple
import unicodedata



class TextNormalizer:
    """Normalize text for phoneme-level tokenization while preserving punctuation."""
    
    def __init__(self, preserve_case=True):
        """
        Initialize the normalizer.
        
        Args:
            preserve_case: If True, keeps original casing. If False, lowercases.
        """
        self.preserve_case = preserve_case
        
        # Define punctuation tokens to preserve
        self.punctuation_tokens = {
            '.': '<PERIOD>',
            ',': '<COMMA>',
            '!': '<EXCLAIM>',
            '?': '<QUESTION>',
            ';': '<SEMICOLON>',
            ':': '<COLON>',
            '-': '<DASH>',
            '—': '<EMDASH>',
            '–': '<ENDASH>',
            '...': '<ELLIPSIS>',
            '…': '<ELLIPSIS>',
            '"': '<QUOTE>',
            '"': '<QUOTE>',
            '"': '<QUOTE>',
            "'": '<APOSTROPHE>',
            ''': '<APOSTROPHE>',
            ''': '<APOSTROPHE>',
            '(': '<LPAREN>',
            ')': '<RPAREN>',
            '[': '<LBRACKET>',
            ']': '<RBRACKET>',
            '{': '<LBRACE>',
            '}': '<RBRACE>',
        }
        
        # Number words for expansion
        self.ones = ['zero', 'one', 'two', 'three', 'four', 'five', 
                     'six', 'seven', 'eight', 'nine']
        self.teens = ['ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 
                      'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
        self.tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 
                     'sixty', 'seventy', 'eighty', 'ninety']
        
    def normalize(self, text: str) -> List[str]:
        """
        Main normalization pipeline.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens (words and punctuation markers)
        """
        # Step 1: Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Step 2: Handle ellipsis before other punctuation
        text = re.sub(r'\.{3}', ' <ELLIPSIS> ', text)
        text = text.replace('…', ' <ELLIPSIS> ')
        
        # Step 3: Separate punctuation with spaces (except apostrophes in contractions)
        text = self._separate_punctuation(text)

        # Step 4: Handle common abbreviations
        text = self._expand_abbreviations(text)

        # Step 5: Expand currency symbols to words
        text = self._expand_currency_signs(text)
        
        # Step 6: Expand numbers to words
        text = self._expand_numbers(text)

        # Step 7: Case normalization
        if not self.preserve_case:
            text = text.lower()
        
        # Step 8: Split into tokens
        tokens = text.split()
        
        # Step 9: Filter empty tokens
        tokens = [t for t in tokens if t.strip()]
        
        return tokens
    
    def _separate_punctuation(self, text: str) -> str:
        """Separate punctuation marks with spaces while handling contractions."""
        # Handle contractions - preserve apostrophes in words
        # Match word boundaries around apostrophes
        text = re.sub(r"(\w)'(\w)", r"\1<TEMP_APOS>\2", text)
        
        # Replace punctuation with spaced versions
        apostrophe_chars = ["'", ''', ''']
        for punct, token in self.punctuation_tokens.items():
            if punct not in apostrophe_chars:  # Skip apostrophes for now
                text = text.replace(punct, f' {token} ')
        
        # Handle standalone apostrophes/quotes (not in contractions)
        text = re.sub(r"['\''']", ' <APOSTROPHE> ', text)
        
        # Restore apostrophes in contractions
        text = text.replace('<TEMP_APOS>', "'")

        # And then tag them explicitly
        for punct, token in self.punctuation_tokens.items():
            text = text.replace(punct, f' {token} ')
        
        return text
    
    def _expand_numbers(self, text: str) -> str:
        """Expand numbers to words (handles 0-999)."""
        def number_to_words(match):
            num = int(match.group())
            if num < 10:
                return self.ones[num]
            elif num < 20:
                return self.teens[num - 10]
            elif num < 100:
                tens_digit = num // 10
                ones_digit = num % 10
                if ones_digit == 0:
                    return self.tens[tens_digit]
                return f"{self.tens[tens_digit]} {self.ones[ones_digit]}"
            elif num < 1000:
                hundreds = num // 100
                remainder = num % 100
                result = f"{self.ones[hundreds]} hundred"
                if remainder > 0:
                    result += f" and {self._expand_number_helper(remainder)}"
                return result
            return match.group()  # Return as-is if too large
        
        # Match standalone numbers
        text = re.sub(r'\b\d+\b', number_to_words, text)
        return text
    
    def _expand_number_helper(self, num: int) -> str:
        """Helper for number expansion (for recursive calls)."""
        if num < 10:
            return self.ones[num]
        elif num < 20:
            return self.teens[num - 10]
        else:
            tens_digit = num // 10
            ones_digit = num % 10
            if ones_digit == 0:
                return self.tens[tens_digit]
            return f"{self.tens[tens_digit]} {self.ones[ones_digit]}"

    def _expand_currency_signs(self, text: str) -> str:
        """Handle common currency signs by replacing them with word equivalents."""
        
        currencies = {
            r'\$': "dollars",           # don't worry about plurals
            r'\€': "euros",
            r'\￥': "yen",
            r'\¢': "cents",
        }

        for sym, expansion in currencies.items():
            text = re.sub(sym + r'(\d+(?:\.\d+)?)',
                r'\1' + ' ' + expansion, 
                text
                )

        return text

    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations."""
        abbreviations = {
            r'\bMr\s<PERIOD>': 'Mister',
            r'\bMrs\s<PERIOD>': 'Missus',
            r'\bMs\s<PERIOD>': 'Miss',
            r'\bDr\s<PERIOD>': 'Doctor',
            r'\bRd\s<PERIOD>': 'Road',
            r'\bBlvd\s<PERIOD>': 'Boulevard',
            r'\bJr\s<PERIOD>': 'Junior',
            r'\bSr\s<PERIOD>': 'Senior',
            r'\bCo\s<PERIOD>': 'Company',
            r'\bLtd\s<PERIOD>': 'Limited', 
        }

        
        for abbr, expansion in abbreviations.items():
            text = re.sub(abbr, expansion, text, flags=re.IGNORECASE)

        # Context-sensitive expansions — order matters: specific case first, fallback second.
        # "St." before a capitalized word → Saint  (e.g. "St. Bernadette", "St. Patrick's Day")
        text = re.sub(r'\bSt\s<PERIOD>\s+(?=[A-Z])', 'Saint ', text)
        # "St." anywhere else → Street  (e.g. "Newbury St.", "123 St.")
        text = re.sub(r'\bSt\s<PERIOD>', 'Street', text, flags=re.IGNORECASE)
        
        return text
    
    def get_punctuation_vocab(self) -> List[str]:
        """Return list of all punctuation tokens."""
        return list(set(self.punctuation_tokens.values()))


# Example usage and testing
if __name__ == "__main__":
    normalizer = TextNormalizer(preserve_case=True)
    
    # Test cases
    test_texts = [
        "Hello, world! How are you?",
        "I can't believe it's 2024... Amazing!",
        "Dr. Smith lives on 123 Main St.",
        'She said, "This is wonderful."',
        "The price is $42, but I have 10 dollars.",
        "What?! No way—that's incredible!!!",
        "It's a beautiful day, isn't it?",
        "St. Bernadette Cathedral is lovely, isn't it?",
        "Mrs. Robinson went ahead and paid...",
        "He went to NYC with Michael and co.",
        "As a historically left-wing movement, this reading of anarchism is placed on the farthest left of the political spectrum, usually described as the libertarian wing of the socialist movement (libertarian socialism).",
        'Anarchism is a political philosophy and movement that is skeptical of all justifications for authority and seeks to abolish the institutions it claims maintain unnecessary coercion and hierarchy, typically including nation-states, and capitalism.'
    ]
    
    print("Text Normalization Examples:")
    print("=" * 60)
    
    for text in test_texts:
        tokens = normalizer.normalize(text)
        print(f"\nOriginal: {text}")
        print(f"Tokens:   {tokens}")
    
    print("\n" + "=" * 60)
    print("\nPunctuation Vocabulary:")
    print(normalizer.get_punctuation_vocab())
