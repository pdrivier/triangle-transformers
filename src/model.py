

import torch
import torch.nn as nn

# ===== Phoneme Embedding Block ========================================


class PhonemeEmbedding(nn.Module): 
	def __init__(self, vocab_size, d_model, max_seq_len, pad_id):
		super().__init__()

		self.phoneme_embeddings = nn.Embedding(
			num_embeddings = vocab_size,
			embedding_dim = d_model,
			padding_idx = pad_id)

		self.position_embeddings = nn.Embedding(
			num_embeddings = max_seq_len,
			embedding_dim = d_model
			)

	def forward(self, input_ids):
		# input_ids shape: (B, T)

		B, T = input_ids.shape

		# build position indices, up to max_seq_len - 1 (because of zero-indexing), for every seq in batch
		positions = torch.arange(T, device = input_ids.device).unsqueeze(0).expand(B, T)
		# shape: (B, T)

		phoneme_emb = self.phoneme_embeddings(input_ids)   # (B, T, D)
		position_emb = self.position_embeddings(positions) # (B, T, D)

		return phoneme_emb + position_emb 				   # (B, T, D)

# if __name__ == "__main__": 
# 	vocab_size = 64   # TODO: use real vocab size later, this just for testing
# 	d_model = 256	
# 	max_seq_len = 512
# 	pad_id = 0
# 	B, T = 4, 32      # shape of small fake test batch

# 	embedding_block = PhonemeEmbedding(vocab_size, d_model, max_seq_len, pad_id)

# 	fake_input = torch.randint(0, vocab_size, (B, T))
# 	output = embedding_block(fake_input)

# 	print(f"Input shape: {fake_input.shape}")    # should be: (4, 32)
# 	print(f"Output shape: {output.shape}")		 # should be: (4, 32, 256)
# 	assert output.shape == (B, T, d_model), "Shape mismatch!"
# 	print("Embedding block text passed.")


# ===== Phon-Level Transformer Block   ========================================
# For first-pass phoneme contextualization, before any downsampling
# this might end up needing to be a larger block of heads, to learn GOOD phoneme representations?

class CausalTransformerBlock(nn.Module):
	def __init__(self, d_model, num_heads, ffn_dim, dropout=0.1):
		super().__init__()

		#----- self-attention -------
		self.attention = nn.MultiheadAttention(
			embed_dim = d_model,
			num_heads = num_heads,
			dropout = dropout,
			batch_first = True # expects (B, T, D) rather than (T, B, D)
			)

		##----- feed-forward network -----
		self.ffn = nn.Sequential(
			nn.Linear(d_model, ffn_dim),
			nn.GELU(),
			nn.Linear(ffn_dim,d_model)
			)

		#----- layer norms (pre-norm form) -----
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)

		#----- dropout -----
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		# x shape: (B, T, D)
		B, T, D = x.shape

		# generate the causal mask: upper triangle is -inf, diagonal, and below is 0
		# shape: (T, T)
		causal_mask = nn.Transformer.generate_square_subsequent_mask(
			T, device = x.device)


		#----- causal self-attention with residual -----
		residual = x 
		x = self.norm1(x)
		x, _ = self.attention(
			query = x,
			key = x,
			value = x,
			attn_mask = causal_mask 
			)
		x = self.dropout(x)
		x = x + residual     # incorporate residual connection

		#----- feed-forward network with residual -----
		residual = x 
		x = self.norm2(x)
		x = self.ffn(x)
		x = self.dropout(x)
		x = x + residual     # incorporate residual connection

		return x             # expect (B, T, D)

# if __name__ == "__main__":
# 	d_model = 256
# 	num_heads = 8
# 	ffn_dim = 4 * d_model
# 	B, T = 2, 16

# 	block = CausalTransformerBlock(d_model, num_heads, ffn_dim)
# 	block.eval()

# 	x = torch.randn(B, T, d_model)
# 	output_original = block(x).detach().clone()

# 	# causality test: modify position t=8, verify that positions < 8 are unaffected!
# 	x_modified = x.clone()
# 	x_modified[:, 8, :] = torch.randn(d_model)
# 	output_modified = block(x_modified).detach().clone()

# 	causality_ok = torch.allclose(
# 		output_original[:, :8, :],
# 		output_modified[:, :8, :],
# 		atol = 1e-6)

# 	print(f"Causality test passed: {causality_ok}")
# 	print(f"Output shape: {output_original.shape}")	# expect (2, 16, 256)



# ===== Causal Convolutional Block ========================================
# Not sure I actually want this block, since I use the smaller Transformer above to provide 
# the first-pass contextualization. 

# class CausalConvBlock(nn.Module):
# 	def __init__(self, d_model, kernel_size):
# 		super().__init__()
# 		self.kernel_size = kernel_size
# 		# padding = kernel_size - 1 on the left, 0 on the right
# 		# to ensure that output position t only sees input from positions <= t
# 		self.padding = kernel_size - 1
# 		self.conv = nn.Conv1d(
# 			in_channels = d_model,
# 			out_channels = d_model,
# 			kernel_size = kernel_size,
# 			padding = 0   # apply padding manually in the forward pass to control number of zeros on each side and ensure causal conv
# 			)
# 		self.norm = nn.LayerNorm(d_model) # standard additions, stabilize training
# 		self.activation = nn.GELU()       # add nonlinearity (avoids collapsing into single linear operation)

# 	def forward(self, x):
# 		# x: shape (B, T, D)

# 		# transpose to (B, D, T) for Conv1D
# 		x = x.transpose(1,2)

# 		# pad (kernel_size - 1) zeros on the left, 0 on the right
# 		x = torch.nn.functional.pad(x, (self.padding, 0))

# 		# convolution-output is still (B, D, T)
# 		x = self.conv(x)

# 		# transpose into (B, T, D) for rest of model
# 		x = x.transpose(1,2)

# 		# normalize, and activate
# 		x = self.norm(x)
# 		x = self.activation(x)

# 		return x     # (B, T, D)

# if __name__ == "__main__":
# 	import torch

# 	d_model = 256
# 	kernel_size = 3
# 	B, T = 2, 16

# 	block = CausalConvBlock(d_model, kernel_size)
# 	block.eval()   # disables stochastic behavior

# 	x = torch.randn(B, T, d_model)
# 	output_original = block(x).detach().clone()

# 	# modify position t=8 and check that positions < 8 remain unchanged
# 	x_modified = x.clone()
# 	x_modified[:, 8, :] = torch.randn(d_model)
# 	output_modified = block(x_modified).detach().clone()

# 	# positions 0-7 should be unaffected
# 	causality_ok = torch.allclose(
# 		output_original[:, :8, :],
# 		output_modified[:, :8, :],
# 		atol = 1e-6
# 		)
# 	print(f"Causality test passed: {causality_ok}")

# 	# positions 8 onwards are allowed to differ
# 	print(f"Output shape: {output_original.shape}")   # expect (2, 16, 256)

# ===== Downsampling Convolutions =============================================

class BoundaryAwarePooling(nn.Module):
	def __init__(self, d_model, space_id, boundary_ids=None):
			super().__init__()
			self.space_id = space_id
			# all token ids that should trigger a boundary but avoid being pooled
			self.boundary_ids = set(boundary_ids) if boundary_ids else {space_id}
			# learned projection applied after pooling
			# lets the model transform the raw mean-pooled vector
			self.projection = nn.Linear(d_model, d_model)
			self.norm = nn.LayerNorm(d_model)

			# TODO: Consider replacing nn.Linear with nn.Sequential(linear, gelu, linear)

	def forward(self, x, input_ids):
		# x shape: 			(B, T, D)
		# input_ids shape:  (B, T)
		B, T, D = x.shape
		word_sequences = []

		for b in range(B):
			ids = input_ids[b].tolist()
			segments = []
			current_segment = []     # accumulates phoneme hidden states

			for t in range(T):
				if ids[t] in self.boundary_ids:
					# emit whatever we've accumulated so far
					if current_segment:
						stacked = torch.stack(current_segment, dim=0)  # (seg_len, D)
						segments.append(stacked.mean(dim=0))		   # (D,)
						current_segment = []
					# boundary token itself is never pooled - just skip it!
				else:
					current_segment.append(x[b, t, :])

			# flush any remaining phonemes at end of sequence
			if current_segment:
				stacked = torch.stack(current_segment, dim=0)
				segments.append(stacked.mean(dim=0))

			word_sequences.append(torch.stack(segments, dim=0))        # (W, D)

		# --- pad to max word count in this batch ----------------------------------
		max_words = max(ws.shape[0] for ws in word_sequences)
		padded = torch.zeros(B, max_words, D, device=x.device)
		word_mask = torch.zeros(B, max_words, device=x.device)

		for b, word_seq in enumerate(word_sequences):
			W = word_seq.shape[0]
			padded[b, :W, :] = word_seq
			word_mask[b, :W] = 1.0         # 1 for real words, 0 for padding

		out = self.projection(padded)
		out = self.norm(out)

		return out, word_mask              # expect (B, W, D), (B, W)


# if __name__ == "__main__":
# 	# simulate small vocab with space_id = 5, and other dummy ids
# 	space_id = 5
# 	sos_id = 2
# 	eos_id = 3
# 	comma_id = 6
# 	boundary_ids = [space_id, sos_id, eos_id, comma_id]
# 	d_model = 256
	

# 	downsampler = BoundaryAwareDownConv(d_model, space_id, boundary_ids)
# 	downsampler.eval()

# 	# set up dummy input_ids with spaces at known positions
# 	# sequence 0: [<SOS>, p, p, <SPACE>, p, p, <COMMA>, p, p, <EOS>] -> 3 words
# 	# sequence 1: [<SOS>, p, p, <SPACE>, p, p, p, <COMMA>, p, p, <SPACE>, p, p, <EOS>] -> 4 words
# 	input_ids = torch.tensor([
# 		[2, 1, 1, 5, 1, 1, 6, 1, 1, 5, 1, 3],
# 		])
# 	x = torch.randn(1, 12, d_model)

# 	out, mask = downsampler(x, input_ids)

# 	print(f"Input shape:      {x.shape}")     # (2, 12, 256)
# 	print(f"Output shape:     {out.shape}")   # (2, 4, 256) -- padded to max words
# 	print(f"Word mask:\n{mask}")              # 1s for real words, 0s for padding

# 	# verify that sequences have expected number of words: seq 0 -> 4
# 	assert mask[0].sum().item() == 4, "Sequence 0 should have 4 words"
# 	print("Boundary downsampling test passed.")

# ===== Word-Level Transformer =============================================
class WordTransformerBlock(nn.Module):
	def __init__(self, d_model, num_heads, ffn_dim, dropout=0.1):
		super().__init__()

		#----- self-attention -------
		self.attention = nn.MultiheadAttention(
			embed_dim = d_model,
			num_heads = num_heads,
			dropout = dropout,
			batch_first = True # expects (B, T, D) rather than (T, B, D)
			)

		##----- feed-forward network -----
		self.ffn = nn.Sequential(
			nn.Linear(d_model, ffn_dim),
			nn.GELU(),
			nn.Linear(ffn_dim,d_model)
			)

		#----- layer norms (pre-norm form) -----
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)

		#----- dropout -----
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, word_mask):
		# x shape: (B, W, D)
		# word_mask shape: (B, W)  # 1 for real, 0 for padding
		B, W, D = x.shape

		# --- causal mask: avoid attending to future positions ------
		causal_mask = nn.Transformer.generate_square_subsequent_mask(
			W, device = x.device
			)

		# --- padding mask: avoid attending to padded word positions ---
		# nn.MultiheadAttention expects True where positions should be IGNORED
		# word_mask has 1 for real and 0 for padding, so we need to invert it
		padding_mask = (word_mask == 0)   # (B, W) now True at padding positions
		padding_mask = padding_mask.to(torch.float)  # switch type to match attn mask, otherwise, get warning


		# ---- self-attention with residual --------------
		residual = x 
		x = self.norm1(x)
		x, _ = self.attention(
			query = x,
			key = x,
			value = x, 
			attn_mask=causal_mask,
			key_padding_mask = padding_mask,							
			)
		x = self.dropout(x)
		x = x + residual

		# ---- feed forward with residual -----------------
		residual = x
		x = self.norm2(x)
		x = self.ffn(x)
		x = self.dropout(x)
		x = x + residual

		return x    # expect (B, W, D)

if __name__ == "__main__":
	d_model = 256
	num_heads = 8
	ffn_dim = 1024
	B = 2

	# simulate two sequences with different word counts
	# seq 0 with 5 words, seq 1 with 3 words
	# padded to max_words = 5
	max_words = 5 
	real_words = [5, 3]

	block = WordTransformerBlock(d_model, num_heads, ffn_dim)
	block.eval()

	# ---- simulate word embeddings and mask ---------------
	x = torch.randn(B, max_words, d_model)

	word_mask = torch.zeros(B, max_words)
	for b, n in enumerate(real_words):
		word_mask[b, :n] = 1.0

	print(f"Input shape:      {x.shape}")   # expect (2, 5, 256)
	print(f"Word mask:\n{word_mask}")

	# ---- basic forward pass ------------------------------
	out = block(x, word_mask)
	print(f"Output shape:     {out.shape}") # expect (2, 5, 256)
	assert out.shape == (B, max_words, d_model), "Shape mismatch!"

	# ---- causality test ----------------------------------
	# modify word position t = 2, verify that positions 0 and 1 are unaffected
	x_modified = x.clone()
	x_modified[:, 2, :] = torch.randn(d_model)
	out_modified = block(x_modified, word_mask)

	causality_ok = torch.allclose(
		out[:, :2, :],
		out_modified[:, :2, :],
		atol=1e-6
		)
	print(f"Causality test passed: {causality_ok}")

	# ----- padding isolation test -------------------------
	# the padded positions in seq 1 (pos 3 and 4) shouldn't influence
	# the real word positiions (0, 1, 2) in this seq
	# verify this by randomizing the padding positions and checking that 
	# the real positions remain intact
	x_noise_in_padding = x.clone()
	x_noise_in_padding[1, 3:, :] = torch.randn(2, d_model)   # corrupt padding
	out_noise = block(x_noise_in_padding, word_mask)

	padding_ok = torch.allclose(
		out[1, :3, :],
		out_noise[1, :3, :],
		atol=1e-6
		)

	print(f"Padding isolation test passed: {padding_ok}")
	print("WordTransformerBlock tests complete.")

# ====== Boundary Aware Upsampling ========================

class BoundaryAwareSplitting(nn.Module):
	def __init__(self, space_id, boundary_ids=None):
		... # TODO: continue here

	def forward(self, x, input_ids, word_mask):
		... # TODO: needs filling out


