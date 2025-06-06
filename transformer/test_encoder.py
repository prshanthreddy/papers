import copy
from model import MultiHeadAttention, FeedFoward, Encoder,EncoderBlock, InputEmbedding, PositionalEncoding
import torch 
import torch.nn as nn


# Configuration
vocab_size = 10000
d_model = 512
d_ff = 2048
num_heads = 8
dropout = 0.1
num_layers = 6
seq_len = 20
batch_size = 2

# Build reusable components
self_attn = MultiHeadAttention(d_model, num_heads, dropout)
ff = FeedFoward(d_model, d_ff, dropout)

# Build encoder layers
encoder_layers = nn.ModuleList([
    EncoderBlock(copy.deepcopy(self_attn), copy.deepcopy(ff), dropout) for _ in range(num_layers)
])

# Final encoder
encoder = Encoder(encoder_layers)

# Input embedding + position encoding
embed = InputEmbedding(d_model, vocab_size)
posenc = PositionalEncoding(d_model)

# Dummy input
x = torch.randint(0, vocab_size, (batch_size, seq_len))
mask = torch.ones(batch_size, 1, 1, seq_len).bool()

# Forward pass
x_embed = posenc(embed(x))  # shape: (batch, seq_len, d_model)
encoded = encoder(x_embed, mask)  # shape: (batch, seq_len, d_model)

print(f"Encoded output shape: {encoded.shape}")
