import torch

from collections import OrderedDict
from torch import nn

import torch.nn.functional as F




def sine_cosine_pe(batch):
    
    _, max_len, emb_dim = batch.shape
    
    pos_emb = torch.zeros((max_len, emb_dim))
    positions = torch.arange(max_len).unsqueeze(1)
    i2 = torch.arange(0, emb_dim, 2)
    denominator = 10000**(i2/emb_dim)
    pos_emb[:, 0::2] = torch.sin(positions / denominator)
    pos_emb[:, 1::2] = torch.cos(positions / denominator)

    return pos_emb


class Attention(nn.Module):
    
    def __init__(self, dim, future_mask=True):
        
        super().__init__()
        
        self.dim = dim
        self.future_mask = future_mask
        
        self.query_w = nn.Linear(self.dim, self.dim)
        self.key_w = nn.Linear(self.dim, self.dim)
        self.value_w = nn.Linear(self.dim, self.dim)
        
    def masking(self, matrix, borders):

        batch_size, rows, cols = matrix.shape
        indices = torch.arange(cols) * torch.ones(rows).unsqueeze(1)
        indices = indices.repeat(batch_size, 1, 1)
        mask = indices >= borders
        matrix[mask] = -float('inf')
                
        return matrix
        
        
    def forward(self, batch, lengths):
        
        queries = self.query_w(batch)
        keys = self.key_w(batch)
        values = self.value_w(batch)
        attention_matrix = torch.bmm(queries, torch.transpose(keys, 1, 2)) / torch.sqrt(torch.tensor(self.dim))
        
        if self.future_mask:
            _, seq_lengths, _ = batch.shape
            present = torch.arange(seq_lengths).unsqueeze(1)
            attention_matrix = self.masking(attention_matrix, present+1)
        attention_matrix = self.masking(attention_matrix, lengths.repeat(seq_lengths, seq_lengths, 1).permute(2, 0, 1))
        
        attention_matrix = F.softmax(attention_matrix, dim=2)
        
        weighted_vectors = torch.bmm(attention_matrix, values)
        
        return weighted_vectors


class MultiHeadAttention(nn.Module):
    
    def __init__(self, emb_dim, n_heads=4, future_mask=True):
        
        assert(emb_dim % n_heads == 0)
        
        super().__init__()
        
        self.d_k = emb_dim // n_heads
        self.dim = emb_dim
        self.n_heads = n_heads
        self.future_mask = future_mask
        
        self.query_w = nn.Linear(self.dim, self.d_k*self.n_heads)
        self.key_w = nn.Linear(self.dim, self.d_k*self.n_heads)
        self.value_w = nn.Linear(self.dim, self.d_k*self.n_heads)
        self.w_0 = nn.Linear(self.d_k*self.n_heads, self.dim)
        
    def masking(self, matrix, borders):

        batch_size, heads, rows, cols = matrix.shape
        indices = torch.arange(cols) * torch.ones(rows).unsqueeze(1)
        indices = indices.repeat(batch_size, heads, 1, 1)
        mask = indices >= borders
        matrix[mask] = -float('inf')
                
        return matrix
    
    def head_reshape(self, x, head_shape):
        
        return x.reshape((*head_shape, self.n_heads, self.d_k))
        
    def forward(self, batch, lengths):
        
        head_shape = batch.shape[:-1]
        
        queries = self.query_w(batch)
        keys = self.key_w(batch)
        values = self.value_w(batch)
        
        queries = self.head_reshape(queries, head_shape)
        keys = self.head_reshape(keys, head_shape)
        values = self.head_reshape(values, head_shape)
                
        attention_matrix = torch.einsum('bshd,bqhd->bhsq', queries, keys)
        
        if self.future_mask:
            _, seq_lengths, _ = batch.shape
            present = torch.arange(seq_lengths).unsqueeze(1)
            attention_matrix = self.masking(attention_matrix, present+1)
        attention_matrix = self.masking(attention_matrix, lengths.repeat(self.n_heads, seq_lengths, seq_lengths, 1).permute(3, 0, 1, 2)) #это в принципе не обязательно
        
        
        attention_matrix = F.softmax(attention_matrix, dim=3)
                                                
        weighted_vectors = torch.einsum('bhss,bshd->bshd', attention_matrix, values)                               
        
        concated = weighted_vectors.reshape(*head_shape, -1)
        
        return self.w_0(concated)


class FeedForward(nn.Module):
    
    def __init__(self, emb_dim, hidden_dim=1024, dropout=0.1):
        
        super().__init__()
        
        self.lin_1 = nn.Linear(emb_dim, hidden_dim, bias=True)
        self.lin_2 = nn.Linear(hidden_dim, emb_dim, bias=True)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, batch):
        
        batch = self.lin_1(batch)
        batch = self.act(batch)
        batch = self.lin_2(self.dropout(batch))
        
        return batch


class LayerNorm(nn.Module):
    
    def __init__(self, hidden_dim, epsilon=0.001):
    
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))
        self.epsilon = torch.tensor(epsilon)
    
    def forward(self, batch):
        
        b, l, c = batch.shape
        var, mean = torch.var_mean(batch, dim=2, unbiased=False)
        var, mean = var.unsqueeze(2), mean.unsqueeze(2)

        batch_normed = (batch - mean) / torch.sqrt(var + self.epsilon)
        ln = self.gamma * batch_normed + self.beta
    
        return ln



class DecoderBlock(nn.Module):
    
    def __init__(self, emb_dim, n_heads=4, ff_dim=1024):
        
        super().__init__()
        
        self.attention_block = MultiHeadAttention(emb_dim, n_heads=n_heads)
        self.layer_norm_1 = LayerNorm(emb_dim)
        self.feedforward_block = FeedForward(emb_dim, ff_dim)
        self.layer_norm_2 = LayerNorm(emb_dim)
        
    def create_pad_mask(self, batch, lengths):
        batch_size, seq, dim = batch.shape
        indices = torch.arange(seq) * torch.ones(dim).unsqueeze(1)
        indices = indices.repeat(batch_size, 1, 1).permute(0, 2, 1)
        mask = indices >= lengths.repeat(seq, dim, 1).permute(2, 0, 1)
        pad_mask = torch.ones(batch.shape)
        pad_mask[mask] = 0
        
        return pad_mask
    
    def forward(self, input_merged):
        batch, lengths = input_merged
        pad_mask = self.create_pad_mask(batch, lengths)
        batch = batch*pad_mask
        after_attention = self.attention_block(batch, lengths)
        batch = batch + after_attention
        batch = self.layer_norm_1(batch)
        after_feedforward = self.feedforward_block(batch)
        batch = batch + after_feedforward
        batch = self.layer_norm_2(batch)
        
        return batch, lengths


class ChataboxModel(nn.Module):
    
    def __init__(self, vocab_size, emb_dim, n_layers, n_heads=4):
        
        super().__init__()
        
        self.emb_dim = emb_dim
        
        self.emb_layer = nn.Embedding(vocab_size, self.emb_dim)
        self.pos_emb = sine_cosine_pe
        
        decoders = OrderedDict()
        for i in range(n_layers):
            decoder = DecoderBlock(self.emb_dim)
            decoders[str(i)] = decoder
        self.decoders = nn.Sequential(decoders)
        
        self.lin = nn.Linear(self.emb_dim, vocab_size)
        
    def forward(self, batch, lengths):
        
        batch = self.emb_layer(batch) * torch.sqrt(torch.tensor(self.emb_dim))
        batch += self.pos_emb(batch)
        
        
        batch, _ = self.decoders((batch, lengths))
        
        batch = self.lin(batch)
        probs = batch
        
        return probs