import json

records = []

for idx, row in df.iterrows():
    pair_id = row.get('pair_id', idx)  # use existing pair_id column if present, else row index

    # --- original word ---
    ipa = row['ipa_word']
    n_phon = len(ipa.split())
    records.append({
        "word": row['word'],
        "ipa": ipa,
        "label": "word",
        "length": len(row['word']),
        "pair_id": pair_id,
        "n_phon": str(n_phon),
        "shuffle_id": None
    })

    # --- original nonword ---
    ipa = row['ipa_nonword']
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
    for i in range(1, 6):
        ipa = row[f'ipa_word_shuffled_{i}']
        n_phon = len(ipa.split())
        records.append({
            "word": row['word'],
            "ipa": ipa,
            "label": "shuffled_word",
            "length": len(row['word']),
            "pair_id": pair_id,
            "n_phon": str(n_phon),
            "shuffle_id": i
        })

    # --- shuffled nonwords (1-5) ---
    for i in range(1, 6):
        ipa = row[f'ipa_nonword_shuffled_{i}']
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

with open('output.json', 'w', encoding='utf-8') as f:
    json.dump(records, f, ensure_ascii=False, indent=2)