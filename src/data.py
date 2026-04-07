# Handles data-related processes



## Write dataset class 
#	- reads .jsonl vocab file, builds the phoneme_to_id map
#   - reads the transcribed wikipedia sequences
#   - prepends the special BOS and EOS tokens
#   - produces (input_ids, target_ids) pairs, where the target is the input shifted left by one
#   - collate_fn handles padding to a fixed sequence length within a batch, with attention mask

#   unit test: 
#   - load a few batches, manually inspect them by printing raw phoneme ids, decode them back to ipa using id_to_phoneme
#   - make sure the shift is correct!


import json
import os
import torch

from pprint import pprint
from transcribe_dataset import stream_to_ipa_corpus


# ----- Dataset Class ------------------------------------------------

class PhonemeDataset:
	"""Pipeline for reading vocab file and prepares the dataset for the model to consume."""

	def __init__(self, data_path = "data/", vocab_path="vocab/phoneme_vocab.json", corpus_path="raw/wikipedia_ipa_50000.jsonl"): 
		"""
		Initialize the Dataset Class.

		Args: 

		"""

		# Load the vocabulary and phoneme-id mappings
		self.vocab_path = os.path.join(data_path, vocab_path)
		with open(self.vocab_path, "r") as file:
			self.vocab = json.load(file)
		#print(self.vocab)
		self.phoneme_to_id = self.vocab["phoneme_to_id"]
		self.id_to_phoneme = {v:k for k, v in self.phoneme_to_id.items()}

		# Grab the PAD id
		self.pad_id = self.phoneme_to_id["<PAD>"]

		# Load the ipa transcribed sequences
		self.corpus_path = os.path.join(data_path, corpus_path)
		self.sequences = []
		#self.debug_text = []		# TODO: remove later, this is just for debugging/checking the transcriptions
		with open(self.corpus_path, "r", encoding="utf-8") as file:
			for line_number, line in enumerate(file, start=1):
				if line_number < 10: ### TODO: remove for runtime, this is just for testing
					try: 
						record = json.loads(line.strip())
						#self.debug_text.append(record["text"])   # TODO: remove after debug
						self.sequences.append(record["ids"])
						print(f"Line {line_number}: {self.sequences}")
					except json.JSONDecodeError as e:
						print(f"Error parsing line {line_number}: {line.strip()}")


	def __len__(self):
		# tells PyTorch how many samples are in the dataset, 
		# required for PyTorch DataLoader to
		# (1) know how many indices it can legally pass into __getitem__
		# (2) how to split the data into batches
		# (3) when one full pass through the dataset (an epoch) is complete
		return len(self.sequences)

	def __getitem__(self,idx):
		ids = self.sequences[idx]
		input_ids = ids[:-1]
		target_ids = ids[1:]
		return input_ids, target_ids



# Always best to define collate_fn outside the Dataset class, to avoid issues when DataLoader attempts to pickle
# it to pass it to multiple workers in a multithreaded context
def make_collate_fn(pad_id):
	def collate_fn(batch):
		"""Produces uniform tensor shapes so the batch can be stacked; makes sure the loss function avoids
		incorporating padding tokens (with the -100 index), and uses attention mask so the model avoids 
		attending to padded positions

		# batch is list of (input_ids, target_ids) tuples produced by __getitem__ for each index in this batch
		""" 
		
		input_seqs, target_seqs = zip(*batch)
		max_len = max(len(s) for s in input_seqs) #finds the longest sequence in this batch
		padded_inputs = []
		padded_targets = []
		attention_masks = []
		for inp, tgt in zip(input_seqs, target_seqs):
			pad_len = max_len - len(inp)
			padded_inputs.append(inp + [pad_id] * pad_len) # pad input_ids with PAD token id
			padded_targets.append(tgt + [-100] * pad_len)  # -100 gets ignored by pytorch, to avoid PAD contributing to loss
			attention_masks.append([1] * len(inp) + [0] * pad_len) # 1 for real tokens, 0 for padding

		return(
			torch.tensor(padded_inputs),  #(B, T)
			torch.tensor(padded_targets), #(B, T)
			torch.tensor(attention_masks) #(B, T)
			)
	return collate_fn

if __name__ == "__main__":
	from torch.utils.data import DataLoader

	#------ Load dataset --------------------------------------
	dataset = PhonemeDataset()
	print(f"Dataset loaded: {len(dataset)} sequences\n")

	#------ Instantiate DataLoader with collate_fn ------------
	loader = DataLoader(
		dataset,
		batch_size=4,
		shuffle=False, # TODO: remove for runtime, keep for reproducibility during debugging
		collate_fn = make_collate_fn(dataset.pad_id)
		)

	#------ Inspect a few batches -----------------------------
	for batch_idx, (input_ids, target_ids, attention_mask) in enumerate(loader): 
		print(f"{'='*60}")
		print(f"BATCH {batch_idx}")
		print(f"  input_ids shape:      {input_ids.shape}")
		print(f"  target_ids shape:     {target_ids.shape}")
		print(f"  attention_mask shape: {attention_mask.shape}")

		#------ Decode and print each sequence in the batch ---
		for seq_idx in range(input_ids.shape[0]):
				inp = input_ids[seq_idx].tolist()
				tgt = target_ids[seq_idx].tolist()
				mask = attention_mask[seq_idx].tolist()

				# strip padding to make output readable
				real_len = sum(mask)
				inp_real = inp[:real_len]
				tgt_real = tgt[:real_len]

				# decode to IPA, marking the -100 padding in targets as <PAD>
				inp_decoded = [dataset.id_to_phoneme.get(i, "<UNK>") for i in inp_real]
				tgt_decoded = [dataset.id_to_phoneme.get(i, "<PAD>") if i != -100 else "<PAD>" for i in tgt_real]

				print(f"\n Sequence {seq_idx}:")
				print(f"      input  (IPA): {' '.join(inp_decoded)}")
				print(f"      target (IPA): {' '.join(tgt_decoded)}")

				# -- Verify the shift -------------------------------
				# input[1:] and target[:-1] should be identical
				# since input = [<SOS>, p1, p2, ..., pn]
				# and target = [p1, p2, ..., pn, <EOS>]
				shift_ok = inp_real[1:] == tgt_real[:-1]
				print(f"     shift correct: {shift_ok}")
				if not shift_ok: 
					# find where it breaks
					for i, (a,b) in enumerate(zip(inp_real[1:], tgt_real[:-1])):
						if a != b:
							print(f"     first mismatch at position {i}: "
								f"input={dataset.id_to_phoneme.get(a)} "
								f"target-{dataset.id_to_phoneme.get(b)}")
							break

				# -- Verify that SOS and EOS are in correct place ----
				sos_id = dataset.phoneme_to_id["<SOS>"]
				eos_id = dataset.phoneme_to_id["<EOS>"]
				print(f"    starts with SOS: {inp_real[0]==sos_id}")
				print(f"    ends with EOS: {tgt_real[-1]==eos_id}")

		if batch_idx >= 2:
				break
				

	print(f"\n{'='*60}")
	print("Inspection complete!")



