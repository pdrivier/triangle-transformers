"""
transcribe the words and nonwords from the Auditory English Lexicon Project into IPA notation: 
https://inetapps.nus.edu.sg/aelp/generate

writes a stimuli/words.json ready for lexical_eval.py
"""

import os
import random

import pandas as pd


from phonemizer import phonemize
from tqdm import tqdm


# Load the aelp dataframe
filepath = "stimuli/"
filename = "aelp.csv"

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

savename = "aelp_transcribed.csv"
df_orig.to_csv(os.path.join(filepath,savename))



