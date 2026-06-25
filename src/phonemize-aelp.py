"""
transcribe the words and nonwords from the Auditory English Lexicon Project: 
https://inetapps.nus.edu.sg/aelp/generate

outputs the aelp dataframe with new appended columns with the corresponding IPA transcriptions
"""

import os
import pandas as pd

from phonemizer import phonemize
from tqdm import tqdm


# Load the aelp dataframe
filepath = "data/raw/"
filename = "aelp_data.csv"

df_orig = pd.read_csv(os.path.join(filepath,filename))

words = df_orig[["word_us"]].values.tolist()
words = [w[0] for w in words]

nonwords = df_orig[["nonword"]].values.tolist()
nonwords = [nw[0] for nw in nonwords]

ipa_words = phonemize(
	words,
	language='en-us',
    backend='espeak',
    strip=True,
    preserve_punctuation=True,
    njobs=4)

ipa_nonwords = phonemize(
	nonwords,
	language='en-us',
	backend='espeak',
	strip=True,
	preserve_punctuation=True,
	njobs=4)


df_orig["ipa_words"] = ipa_words 
df_orig["ipa_nonwords"] = ipa_nonwords

savename = "aelp_phonemized.csv"
df_orig.to_csv(os.path.join(filepath,savename))



