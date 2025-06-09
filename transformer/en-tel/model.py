import torch 
import torch.nn as nn
import math
class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len)
        """
        # x = self.embedding(x)
        x = self.embedding(x) * math.sqrt(self.d_model)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        """
        Args:
            d_model: The dimension of the model.
            max_len: The maximum length of the input sequences.
            dropout: The dropout rate.
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)
        # Create the matrix of positional encodings usinhg the max_len and d_model
        # The positional encoding is a matrix of shape (max_len, d_model)
        # where each row corresponds to a position and each column corresponds to a dimension
        # The encoding is defined as:
        # PE(pos, 2i) = sin(pos / (10000 ** (2i / d_model)))
        # PE(pos, 2i+1) = cos(pos / (10000 ** (2i / d_model)))
        positional_encoding=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len,dtype=float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        ## Applying it to the even terms
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)


        positional_encoding=positional_encoding.unsqueeze(0) #(1, max_len, d_model)

        ## Saving as a registered buffer to make sure to save them with the model but not as a learned parameter
        self.register_buffer('positional_encoding',positional_encoding)

    def forward(self, x):
        x = x + self.positional_encoding[:, :x.size(1), :].detach()
        return self.dropout(x)

## Layer Normalization to make sure that the input to the transformer is normalized
class LayerNorm(nn.Module):
    def __init__(self,eps: float =10**-6)-> None:
        super().__init__()
        self.eps =eps
        self.alpha=nn.Parameter(torch.ones(1)) ## Multiplicative 
        self.beta=nn.Parameter(torch.ones(1)) ## Additive 

    def forward(self,x):
        mean=x.mean(dim=-1, keepdim=True)
        std=x.std(dim=-1, keepdim=True)
        return self.alpha *(x-mean)/(std+self.eps)+self.beta
    

## Feed Forward Layer
class FeedForward(nn.Module):
    def __init__(self,d_model: int, d_ff: int,dropout: float):
        super().__init__()
        self.linear_1=nn.Linear(d_model,d_ff)
        self.dropout=nn.Dropout(dropout)
        self.linear_2=nn.Linear(d_ff,d_model)

    def forward(self,x):
        # (batch, max_len,d_model) -->(batch, max_len,d_ff) --> (betch, max_len,d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
## Multihead attention
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model: int, h:int, droupout: float):
        super().__init__()
        self.d_model=d_model
        self.h=h
        ##Divide d_model by h
        assert d_model%h==0, "d_model is not divisible by h"
        self.d_k=d_model//h

        self.w_q=nn.Linear(d_model,d_model) # w_q
        self.w_k=nn.Linear(d_model, d_model) # w_k
        self.w_v=nn.Linear(d_model, d_model) # w_v
        self.w_o=nn.Linear(d_model, d_model) #w_o
        self.dropout=nn.Dropout(droupout)

    @staticmethod
    def attention(query, key, value, mask,dropout:nn.Dropout):
        d_k= query.size(-1)
        attention_scores=(query@key.transpose(-2,-1))/math.sqrt(d_k) 
        if mask is not None:
            attention_scores=attention_scores.masked_fill(mask==0,float('-inf'))
        attention_probs=attention_scores.softmax(dim=-1)   # (batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_probs=dropout(attention_probs)
        # (batch, h, seq_len, seq_len) @ (batch, h, seq_len, d_k) --> (batch, h, seq_len, d_k)
        x=attention_probs@value
        return x, attention_probs 

    def forward(self, q, k, v,mask):
        query=self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key=self.w_k(k)
        value=self.w_v(v)
        # Reshape the query, key and value to (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query=query.view(query.shape[0],query.shape[1],self.h, self.d_k).transpose(1,2)
        key=key.view(key.shape[0],key.shape[1],self.h, self.d_k).transpose(1,2)
        value=value.view(value.shape[0],value.shape[1],self.h, self.d_k).transpose(1,2)

        # Calculate the attention scores
        x, self.attention_scores=MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        # Reshape the output to (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.size(0), -1, self.d_model)
        # Apply the output linear layer
        x=self.w_o(x)
        # Return the output
        return x


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm()

    def forward(self, x, sublayer):
        return self.layer_norm(x + self.dropout(sublayer(x)))

    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttention, feed_forward_block: FeedForward, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([
            ResidualConnection(dropout),
            ResidualConnection(dropout)
        ])
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        mask: (batch_size, 1, 1, seq_len)
        """
        x=self.residual_connections[0](x,lambda x: self.self_attention(x,x,x,mask))
        x=self.residual_connections[1](x,self.feed_forward_block)
        return x
    

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList)-> None:
        super().__init__()
        self.layers=layers
        self.norm =LayerNorm()

    def forward(self, x, mask):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)
    

class DecoderBlock(nn.Module):
    def __init__(self,self_attention: MultiHeadAttention, cross_attention: MultiHeadAttention, feed_forward_block: FeedForward, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([
            ResidualConnection(dropout) for _ in range(3)
        ])

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        encoder_output: (batch_size, src_seq_len, d_model)
        src_mask: (batch_size, 1, 1, src_seq_len)
        tgt_mask: (batch_size, 1, 1, seq_len)
        """
        x=self.residual_connections[0](x,lambda x: self.self_attention(x,x,x,tgt_mask))
        x=self.residual_connections[1](x,lambda x: self.cross_attention(x,encoder_output,encoder_output,src_mask))
        x=self.residual_connections[2](x,self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        encoder_output: (batch_size, src_seq_len, d_model)
        src_mask: (batch_size, 1, 1, src_seq_len)
        tgt_mask: (batch_size, 1, 1, seq_len)
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    


class Projector(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        """
        return torch.log_softmax(self.linear(x), dim=-1)  # (batch_size, seq_len, vocab_size)
    
import copy
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbedding, tgt_embedding: InputEmbedding, src_pos_encoding: PositionalEncoding, tgt_pos_encoding: PositionalEncoding, projector: Projector):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_pos_encoding = src_pos_encoding
        self.tgt_pos_encoding = tgt_pos_encoding
        self.projector = projector

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        src: (batch_size, src_seq_len)
        src_mask: (batch_size, 1, 1, src_seq_len)
        """
        x = self.src_embedding(src)
        x = self.src_pos_encoding(x)
        x = self.encoder(x, src_mask)
        return x  # (batch_size, src_seq_len, d_model)

    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        tgt: (batch_size, tgt_seq_len)
        encoder_output: (batch_size, src_seq_len, d_model)
        src_mask: (batch_size, 1, 1, src_seq_len)
        tgt_mask: (batch_size, 1, 1, tgt_seq_len)
        """
        tgt = tgt.to(torch.long)  # ✅ Convert to correct dtype
        x = self.tgt_embedding(tgt)
        x = self.tgt_pos_encoding(x)
        x = self.decoder(x, encoder_output, src_mask, tgt_mask)
        return x  # (batch_size, tgt_seq_len, d_model)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, tgt_seq_len, d_model)
        """
        return self.projector(x)
    def forward(self, src, tgt, src_mask, tgt_mask):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        return self.project(decoder_output)
    
        
def build_transformer(
    d_model: int, 
    src_vocab_size: int,
    tgt_vocab_size: int,
    n_heads: int,
    src_seq_len: int,
    tgt_seq_len: int,
    n_layers: int,
    d_ff: int = 2048,
    dropout: float = 0.1
) -> Transformer:
    """Builds a Transformer model with the specified parameters.
    Args:
        d_model (int): The dimension of the model.
        src_vocab_size (int): The vocabulary size of the source language.
        tgt_vocab_size (int): The vocabulary size of the target language.
        n_heads (int): The number of attention heads.
        scr_seq_len (int): The maximum length of the source sequences.
        tgt_seq_len (int): The maximum length of the target sequences.
        n_layers (int): The number of layers in the encoder and decoder.
        d_ff (int, optional): The dimension of the feed-forward layer. Defaults to 2048.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
    Returns:
        Transformer: The constructed Transformer model.
    """
    src_embedding = InputEmbedding(d_model, src_vocab_size)
    tgt_embedding = InputEmbedding(d_model, tgt_vocab_size)
    src_pos_encoding = PositionalEncoding(d_model, max_len=src_seq_len, dropout=dropout)
    tgt_pos_encoding = PositionalEncoding(d_model, max_len=tgt_seq_len, dropout=dropout)


    self_attention = MultiHeadAttention(d_model, n_heads, dropout)
    cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
    feed_forward_block = FeedForward(d_model, d_ff, dropout)

    encoder_layers = nn.ModuleList([
    EncoderBlock(copy.deepcopy(self_attention), copy.deepcopy(feed_forward_block), dropout)
    for _ in range(n_layers)
    ])
    decoder_layers = nn.ModuleList([
        DecoderBlock(copy.deepcopy(self_attention), copy.deepcopy(cross_attention), copy.deepcopy(feed_forward_block), dropout)
        for _ in range(n_layers)
    ])
    encoder = Encoder(encoder_layers)
    decoder = Decoder(decoder_layers)

    projector = Projector(d_model, tgt_vocab_size)

    transformer= Transformer(
        encoder=encoder,
        decoder=decoder,
        src_embedding=src_embedding,
        tgt_embedding=tgt_embedding,
        src_pos_encoding=src_pos_encoding,
        tgt_pos_encoding=tgt_pos_encoding,
        projector=projector
    )

    for p in transformer.parameters(): # Initialize the parameters of the transformer model
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer