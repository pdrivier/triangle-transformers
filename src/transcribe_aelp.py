"""
transcribe the words and nonwords from the Auditory English Lexicon Project into IPA notation: 
https://inetapps.nus.edu.sg/aelp/generate

writes a stimuli/words.json ready for lexical_eval.py
"""
import json
import os
import random

import numpy as np
import pandas as pd


from phonemizer import phonemize
from tqdm import tqdm

def shuffle_string(s):
	return ''.join(random.sample(s,len(s)))


# Load the aelp dataframe
filepath = "stimuli/"
filename = "aelp.csv"

df_orig = pd.read_csv(os.path.join(filepath,filename))

words = df_orig[["word_us"]].values.tolist()
words = [w[0] for w in words]

nonwords = df_orig[["nonword"]].values.tolist()
nonwords = [nw[0] for nw in nonwords]

# TODO 08/11/2026: for some reason phonemize not working anymore on lab mac, so annoying! breaks so frequently
# Transcribe each word and its matched nonword and store the results in the same dataframe
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

# Create a random baseline by randomly shuffling the IPA word and nonword transcriptions and storing 
# them in their own columns
k_shuffles = 5
for i in range(k_shuffles):
	df_orig[f"ipa_words_shuffle_{i}"] = df_tmp["ipa_words"].apply(shuffle_string)
	df_orig[f"ipa_nonwords_shuffle_{i}"] = df_tmp["ipa_nonwords"].apply(shuffle_string)

# Store as .csv to quickly scan it and make sure the shuffles look right
savename = "aelp_with_shuffles_transcribed.csv"
df_orig.to_csv(os.path.join(filepath,savename))

# Convert to .json format for lexical evaluation pipeline
records = []

for idx, row in df_orig.iterrows():
    pair_id = row.get('pair_id', idx)  # use existing pair_id column if present, else row index
    # --- original word ---
    ipa = row['ipa_words']
    n_phon = len(ipa.split())
    records.append({
        "word": row['word_us'],
        "ipa": ipa,
        "label": "word",
        "length": len(row['word_us']),
        "pair_id": pair_id,
        "n_phon": str(n_phon),
        "shuffle_id": None
    })
    # --- original nonword ---
    ipa = row['ipa_nonwords']
    n_phon = len(ipa.split())
    records.append({
        "word": row['nonword'],
        "ipa": ipa,
        "label": "nonword",
        "length": len(row['nonword']),
        "pair_id": pair_id,
        "n_phon": str(n_phon),
        "shuffle_id": None
    })
    # --- shuffled words (1-5) ---
    for i in range(1, k_shuffles):
        ipa = row[f'ipa_words_shuffle_{i}']
        n_phon = len(ipa.split())
        records.append({
            "word": row['word_us'],
            "ipa": ipa,
            "label": "shuffled_word",
            "length": len(row['word_us']),
            "pair_id": pair_id,
            "n_phon": str(n_phon),
            "shuffle_id": i
        })
    # --- shuffled nonwords (1-5) ---
    for i in range(1, k_shuffles):
        ipa = row[f'ipa_nonwords_shuffle_{i}']
        n_phon = len(ipa.split())
        records.append({
            "word": row['nonword'],
            "ipa": ipa,
            "label": "shuffled_nonword",
            "length": len(row['nonword']),
            "pair_id": pair_id,
            "n_phon": str(n_phon),
            "shuffle_id": i
        })

filesavename = "words_and_shuffles.json"
with open(os.path.join(filepath,filesavename), 'w', encoding='utf-8') as f:
    json.dump(records, f, ensure_ascii=False, indent=2)




