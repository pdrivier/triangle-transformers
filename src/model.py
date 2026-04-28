

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

class BoundaryAwarePooler(nn.Module):
	def __init__(self, d_model, space_id, 
		boundary_ids=None,     # triggers split, then disappears
		passthrough_ids=None): # kept as own word-level token for attention mechanism to learn
			super().__init__()
			self.space_id = space_id
			# all token ids that should trigger a boundary but avoid being pooled
			self.boundary_ids = set(boundary_ids) if boundary_ids else {space_id}
			self.passthrough_ids = set(passthrough_ids) if passthrough_ids else set()
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
				if ids[t] in self.passthrough_ids:
					# flush current phoneme segment first
					if current_segment:
						stacked = torch.stack(current_segment, dim=0)  # (seg_len, D)
						segments.append(stacked.mean(dim=0))		   # (D,)
						current_segment = []
					# then keep this token as its own word-level vector
					segments.append(x[b, t, :])

				elif ids[t] in self.boundary_ids:
					# flush current segment, then disappear
					if current_segment:
						stacked = torch.stack(current_segment, dim = 0)
						segments.append(stacked.mean(dim=0))
						current_segment = []
						# TODO: agh, can't tell if I want to keep or discard these! 
						# will discard for now, but remove comment to keep
						# segments.append(x[b, t, :])

				else:
					# regular phoneme, accumulate
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


if __name__ == "__main__":
	# simulate small vocab with space_id = 5, and other dummy ids
	space_id = 5
	sos_id = 2
	eos_id = 3
	comma_id = 6
	boundary_ids = [space_id, sos_id, eos_id, comma_id]
	d_model = 256
	

	downsampler = BoundaryAwarePooler(d_model, space_id, boundary_ids)
	downsampler.eval()

	# set up dummy input_ids with spaces at known positions
	# sequence 0: [<SOS>, p, p, <SPACE>, p, p, <COMMA>, p, p, <EOS>] -> 3 words
	# sequence 1: [<SOS>, p, p, <SPACE>, p, p, p, <COMMA>, p, p, <SPACE>, p, p, <EOS>] -> 4 words
	input_ids = torch.tensor([
		[2, 1, 1, 5, 1, 1, 6, 1, 1, 5, 1, 3],
		])
	x = torch.randn(1, 12, d_model)

	out, mask = downsampler(x, input_ids)

	print(f"Input shape:      {x.shape}")     # (2, 12, 256)
	print(f"Output shape:     {out.shape}")   # (2, 4, 256) -- padded to max words
	print(f"Word mask:\n{mask}")              # 1s for real words, 0s for padding

	# verify that sequences have expected number of words: seq 0 -> 4
	assert mask[0].sum().item() == 4, "Sequence 0 should have 4 words"
	print("Boundary downsampling test passed.")

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

class BoundaryAwareSplitter(nn.Module):
	def __init__(self, d_model, boundary_ids):
		super().__init__()
		self.boundary_ids = set(boundary_ids)
		self.projection = nn.Linear(d_model, d_model)
		self.norm = nn.LayerNorm(d_model)

	def forward(self, word_embeddings, phoneme_skip, input_ids):
		# word_embeddings: (B, W, D)
		# phoneme_skip: (B, T, D)
		# input_ids: (B, T)

		B, T, D = phoneme_skip.shape

		out = torch.zeros(B, T, D, device = word_embeddings.device)

		for b in range(B):
			ids = input_ids[b].tolist()
			word_idx = 0
			prev_was_boundary = True # so first phoneme correctly starts word 0

			for t in range(T):
				if ids[t] in self.boundary_ids:
					# boundary token: can grab directly from skip connection
					# TODO: ALTHOUgh, think about whether you want to grab the word-contextualized 
					# version instead of the phoneme-contextualized version?
					out[b, t, :] = phoneme_skip[b, t, :]
					if not prev_was_boundary: 
						# just finished a word, so advance to next word embedding
						word_idx += 1
					prev_was_boundary = True

				else: 
					# real phoneme: word context + fine-grained phoneme identity
					out[b, t, :] = word_embeddings[b, word_idx, :] + phoneme_skip[b, t, :]
					prev_was_boundary = False

		# learned projection to let model blend the two sources
		out = self.projection(out)
		out = self.norm(out)

		return out  # (B, T, D)

if __name__ == "__main__":
	d_model = 256
	space_id = 5
	sos_id = 2
	eos_id = 3
	boundary_ids = [space_id, sos_id, eos_id]

	splitter = BoundaryAwareSplitter(d_model, boundary_ids)
	splitter.eval()

	# [SOS, p, p, SPACE, p, p, p, EOS]
	# word 0 = positions 1, 2 - word 1 = positions 4,5,6
	input_ids = torch.tensor([[2,1,1,5,1,1,1,3]])
	B, T = input_ids.shape

	# 2 words in the word sequence
	word_embeddings = torch.randn(1,2,d_model)
	phoneme_skip = torch.randn(1, T, d_model)


	out = splitter(word_embeddings, phoneme_skip, input_ids)

	print(f"Input shape:       {input_ids.shape}")       # (1, 8)
	print(f"Word embeddings:   {word_embeddings.shape}") # (1,2,256)
	print(f"Output shape:      {out.shape}")			 # (1, 8, 256)
	assert out.shape == (B, T, d_model), "Shape mismatch!"

	# --- verify boundary positions came purely from skip -------------
	# at boundary positions, out = projection(skip), so changing
	# word_embeddings should not affect those positions
	word_embeddings_modified = word_embeddings.clone()
	word_embeddings_modified[:, :, :] = torch.randn(d_model)
	out_modified = splitter(word_embeddings_modified, phoneme_skip, input_ids)

	boundary_positions = [0, 3, 7] # SOS, SPACE, EOS
	for pos in boundary_positions:
		boundary_ok = torch.allclose(
			out[0, pos, :], out_modified[0, pos, :], atol=1e-6
			)
		print(f"Boundary isolation at position {pos}: {boundary_ok}")

	# --- verify phoneme positions reflect word embeddings -------------
	# changing word_embeddings SHOULD affect phoneme positions
	phoneme_positions = [1, 2, 4, 5, 6]
	for pos in phoneme_positions: 
		phoneme_changed = not torch.allclose(
			out[0, pos, :], out_modified[0, pos, :], atol=1e-6
			)

		print(f"Phoneme position {pos} reflects word embedding: {phoneme_changed}")

	print("BoundaryAwareSplitter tests complete.")


class PhonemeLM(nn.Module):
	def __init__(self, vocab_size, d_model, num_heads, ffn_dim,
		max_seq_len, max_word_len, pad_id, space_id,
		boundary_ids, passthrough_ids, dropout=0.1):
	super().__init__()

	# --- phoneme level --------------------------------------------------
	self.embedding = PhonemeEmbedding(
		vocab_size, d_model, max_seq_len, pad_id
		)
	self.early_transformer = CausalTransformerBlock(
		d_model, num_heads, ffn_dim, dropout)

	# --- downsampling --------------------------------------------------
	self.pooler = BoundaryAwarePooler(
		d_model, space_id, boundary_ids, passthrough_ids
		)

	# --- word level  ----------------------------------------------------
	self.word_position_embeddings = nn.Embedding(max_word_len, d_model)
	self.word_transformer = WordTransformerBlock(
		d_model, num_heads, ffn_dim, dropout)

	# --- upsampling -----------------------------------------------------
	self.splitter = BoundaryAwareSplitter(d_model, boundary_ids)
	self.phoneme_position_embeddings = nn.Embedding(max_seq_len, d_model)

	# --- output head ---------------------------------------------------
	self.output_norm = nn.LayerNorm(d_model)
	self.unembedding = nn.Linear(d_model, vocab_size)


	def forward(self, input_ids):
		# input_ids shape: (B, T)
		B, T = input_ids.shape

		# --- phoneme level ---------------------------------------------
		x = self.embedding(input_ids)						  # (B, T, D)
		phoneme_skip = self.early_transformer(x)			  # (B, T, D)


		# --- downsample --------------------------------------------------
		word_embeddings, word_mask = self.pooler(
			phoneme_skip, input_ids
			)										  # (B, W, D), (B, W)

		# --- word positional embeddings ---------------------------------
		B, W, D = word_embeddings.shape
		word_positions = torch.arange(W, device=input_ids.device).unsqueeze(0).expand(B,W)
		word_embeddings = word_embeddings + self.word_position_embeddings(word_positions)

		# --- word level transformer -------------------------------------
		word_embeddings = self.word_transformer(
			word_embeddings, word_mask
			)												   # (B, W, D)

		# --- upsample --------------------------------------------------
		x = self.splitter(
			word_embeddings, phoneme_skip, input_ids
			)												   # (B, T, D)

		# --- reintroduce phoneme positional embeddings ------------------
		positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
		x = x + self.phoneme_position_embeddings(positions)

		# --- output head --------------------------------------------------
		x = self.output_norm(x)
		logits = self.unembedding(x)				   # (B, T, vocab_size)


		return logits

		












		
