

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


# ===== Single Transformer Block   ========================================
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

class BoundaryAwareDownConv(nn.Module):
	def __init__(self, d_model, space_id):
			super().__init__()
			self.space_id = space_id
			# learned projection applied after pooling
			# lets the model transform the raw mean-pooled vector
			self.projection = nn.Linear(d_model, d_model)
			self.norm = nn.LayerNorm(d_model)

	def forward(self, x, input_ids):
		# x shape: 			(B, D, T)
		# input_ids shape:  (B, T)
		B, D, T = x.shape

		word_sequences = []

		for b in range(B): 
			# find all positions with <SPACE> tokens in this sequence
			space_positions = (input_ids[b] == self.space_id).nonzero(as_tuple=True)[0].tolist()

			segments = []
			prev = 0 # start of current word segment

			for pos in space_positions:
				if pos > prev:
					# pool all phonemes from prev up to (but not including) the space
					# causality: we emit the word vector AT the space position, which 
					# means we consider only phonemes we've already observed
					segment = x[b, prev:pos, :]			# (seg_len, D)
					pooled = segment.mean(dim=0)		# (D,)
					segments.append(pooled)

				prev = pos + 1







