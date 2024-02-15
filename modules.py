import math

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super(InputEmbedding, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        res = self.embedding(x) * math.sqrt(self.d_model)

        return res


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        self.pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (
                    -math.log(10000.0) / d_model)
        )

        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', self.pe.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        x = self.dropout(x)

        return x


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-12):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x * self.alpha + self.bias

        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, dmodel: int, d_ff: int, dropout: float):
        super(FeedForwardBlock, self).__init__()
        self.linear1 = nn.Linear(dmodel, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, dmodel)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)  # query
        self.W_k = nn.Linear(d_model, d_model)  # key
        self.W_v = nn.Linear(d_model, d_model)  # value
        self.W_o = nn.Linear(d_model, d_model)  # output

    def split_heads(self, x: torch.Tensor):
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        x = x.transpose(1, 2)

        return x  # (batch_size, num_heads, seq_len, d_k)

    def combine_heads(self, x: torch.Tensor):
        batch_size, _, seq_len, d_k = x.size()
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, seq_len, self.d_model)

        return x  # (batch_size, seq_len, d_model)

    @staticmethod
    def attention(Q, K, V, mask=None):
        d_k = Q.shape[-1]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)

        return output

    def forward(self, query, key, value, mask=None):
        query = self.split_heads(self.W_q(query))
        key = self.split_heads(self.W_k(key))
        value = self.split_heads(self.W_v(value))

        attention_output = self.attention(query, key, value, mask)
        output = self.W_o(self.combine_heads(attention_output))

        return output


class EncoderBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            self_attention_block: MultiHeadAttention,
            feed_forward_block: FeedForwardBlock,
            dropout: float
    ):
        super(EncoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_out = self.self_attention_block(
            query=x, key=x, value=x, mask=mask
        )
        x = x + self.dropout(attn_out)
        x = self.norm_1(x)
        ff_out = self.feed_forward_block(x)
        x = x + self.dropout(ff_out)
        x = self.norm_2(x)

        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super(Encoder, self).__init__()
        self.layers = layers
        self.layer_norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        x = self.layer_norm(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            self_attention_block: MultiHeadAttention,
            cross_attention_block: MultiHeadAttention,
            feed_forward_block: FeedForwardBlock,
            dropout: float
    ):
        super(DecoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attention_block(
            query=x, key=x, value=x, mask=tgt_mask
        )
        x = x + self.dropout(attn_output)
        x = self.norm_1(x)

        attn_output = self.cross_attention_block(
            query=x, key=enc_output, value=enc_output, mask=src_mask
        )
        x = x + self.dropout(attn_output)
        x = self.norm_2(x)

        ff_output = self.feed_forward_block(x)
        x = x + self.dropout(ff_output)
        x = self.norm_3(x)

        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super(Decoder, self).__init__()
        self.layers = layers
        self.layer_norm = LayerNormalization()

    def forward(self, enc_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)

        x = self.layer_norm(x)

        return x
