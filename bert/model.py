import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self,d_model,vocab_size, max_len=512,type_vocab_size=2):
        super().__init__()
        self.d_model=d_model
        self.max_len=max_len
        self.vocab_size=vocab_size
        self.type_vocab_size=type_vocab_size

        self.token_embeddings=nn.Embedding(vocab_size,d_model)
        self.sentence_embeddings=nn.Embedding(type_vocab_size,d_model)
        self.postion_embeddings=nn.Embedding(max_len,d_model)

        self.layer_norm=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(0.1)

    def forward(self,input_ids, token_type_ids):
        seq_len=input_ids.size(1)
        position_ids=torch.arange(seq_len,dtype=torch.long, device=input_ids.device)
        position_ids=position_ids.unsqueeze(0).expand_as(input_ids)

        token_embeds=self.token_embeddings(input_ids)
        segment_embeds=self.sentence_embeddings(token_type_ids)
        position_embeds=self.postion_embeddings(position_ids)

        embeddings= token_embeds+segment_embeds+position_embeds
        return self.dropout(self.layer_norm(embeddings))
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        # Final output projection
        self.out_proj = nn.Linear(d_model, d_model)


    def forward(self, x,mask=None):
        batch_size,seq_len,_=x.size()
            
            # Linear projections: (batch, seq_len, d_model) -> (batch, n_heads, seq_len, head_dim)
        Q = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch, heads, seq_len, seq_len)

        ## THis is for the causal language modeling which uses mask to hide the future words from the context
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)  # (batch, heads, seq_len, seq_len)
        context = torch.matmul(attn_weights, V)       # (batch, heads, seq_len, head_dim)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.out_proj(context)
    

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()  # BERT uses GELU
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear2(self.activation(self.linear1(x))))

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention + residual + norm
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feedforward + residual + norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x
    


class BERTEncoder(nn.Module):
    def __init__(self, num_layers, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)  # final normalization

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class BERTModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_heads=12, d_ff=3072, num_layers=12, max_len=512, type_vocab_size=2):
        super().__init__()
        self.embeddings = InputEmbeddings(d_model, vocab_size, max_len, type_vocab_size)
        self.encoder = BERTEncoder(num_layers, d_model, n_heads, d_ff)

        # MLM Head
        self.mlm_head = nn.Linear(d_model, vocab_size)

        # Optional: Tie weights to input token embeddings
        self.mlm_head.weight = self.embeddings.token_embeddings.weight

    def forward(self, input_ids, token_type_ids, mask=None):
        x = self.embeddings(input_ids, token_type_ids)  # (B, S, D)
        x = self.encoder(x, mask)                       # (B, S, D)
        logits = self.mlm_head(x)                       # (B, S, V)
        return logits


## Example usage
if __name__ == "__main__":
    model = BERTModel(vocab_size=30522, d_model=768, n_heads=12, d_ff=3072, num_layers=12)
    input_ids = torch.randint(0, 30522, (2, 128))  # Batch size of 2, sequence length of 128
    token_type_ids = torch.zeros_like(input_ids)   # Assuming single sentence input
    outputs = model(input_ids, token_type_ids)
    print(outputs.shape)  # Should be (2, 128, 30522)
    # Example output shape: (batch_size, sequence_length, vocab_size)
    print("Model initialized successfully.")    
