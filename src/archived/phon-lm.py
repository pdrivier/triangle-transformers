# attempting to create a phoneme-level autoregressive lm


import os
os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = '/opt/homebrew/Cellar/espeak-ng/1.52.0/lib/libespeak-ng.1.dylib'
os.environ['PHONEMIZER_ESPEAK_PATH'] = '/opt/homebrew/bin/espeak-ng'

from phonemizer import phonemize

text = "Hello world"
ipa = phonemize(text, language='en-us', backend='espeak')