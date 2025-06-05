import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, encoding_type='sinusoidal'):
        super().__init__()
        self.encoding_type = encoding_type
        
        if encoding_type == 'sinusoidal':
            # 原始的正弦位置编码
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
        elif encoding_type == 'learned':
            # 学习型位置编码
            self.pe = nn.Parameter(torch.randn(1, max_len, d_model))
        elif encoding_type == 'relative':
            # 相对位置编码（简化版）
            self.relative_pe = nn.Parameter(torch.randn(max_len * 2 - 1, d_model))
            
    def forward(self, x):
        if self.encoding_type == 'sinusoidal':
            return x + self.pe[:, :x.size(1)]
        elif self.encoding_type == 'learned':
            return x + self.pe[:, :x.size(1)]
        elif self.encoding_type == 'relative':
            # 简化的相对位置编码实现
            seq_len = x.size(1)
            pe = self.relative_pe[seq_len-1:seq_len-1+seq_len]
            return x + pe.unsqueeze(0)
        else:  # 'none'
            return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, v)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(output)

def get_activation(activation_type):
    """获取指定的激活函数"""
    if activation_type == 'relu':
        return nn.ReLU()
    elif activation_type == 'gelu':
        return nn.GELU()
    elif activation_type == 'swish':
        return nn.SiLU()  # SiLU 是 Swish 的 PyTorch 实现
    else:
        return nn.ReLU()  # 默认使用 ReLU

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, activation='relu', 
                 norm_position='post', use_layer_norm=True):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm_position = norm_position
        self.use_layer_norm = use_layer_norm
        
        if use_layer_norm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        if self.norm_position == 'pre' and self.use_layer_norm:
            # 前置归一化
            attn_output = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)
            x = x + self.dropout(attn_output)
            ff_output = self.feed_forward(self.norm2(x))
            return x + self.dropout(ff_output)
        else:
            # 后置归一化或无归一化
            attn_output = self.attention(x, x, x, mask)
            if self.use_layer_norm:
                x = self.norm1(x + self.dropout(attn_output))
            else:
                x = x + self.dropout(attn_output)
            
            ff_output = self.feed_forward(x)
            if self.use_layer_norm:
                return self.norm2(x + self.dropout(ff_output))
            else:
                return x + self.dropout(ff_output)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, activation='relu',
                 norm_position='post', use_layer_norm=True):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.norm_position = norm_position
        self.use_layer_norm = use_layer_norm
        
        if use_layer_norm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, trg_mask=None):
        if self.norm_position == 'pre' and self.use_layer_norm:
            # 前置归一化
            self_attn_output = self.self_attention(self.norm1(x), self.norm1(x), self.norm1(x), trg_mask)
            x = x + self.dropout(self_attn_output)
            cross_attn_output = self.cross_attention(self.norm2(x), enc_output, enc_output, src_mask)
            x = x + self.dropout(cross_attn_output)
            ff_output = self.feed_forward(self.norm3(x))
            return x + self.dropout(ff_output)
        else:
            # 后置归一化或无归一化
            self_attn_output = self.self_attention(x, x, x, trg_mask)
            if self.use_layer_norm:
                x = self.norm1(x + self.dropout(self_attn_output))
            else:
                x = x + self.dropout(self_attn_output)
            
            cross_attn_output = self.cross_attention(x, enc_output, enc_output, src_mask)
            if self.use_layer_norm:
                x = self.norm2(x + self.dropout(cross_attn_output))
            else:
                x = x + self.dropout(cross_attn_output)
            
            ff_output = self.feed_forward(x)
            if self.use_layer_norm:
                return self.norm3(x + self.dropout(ff_output))
            else:
                return x + self.dropout(ff_output)

class TransformerMT(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model=256, num_heads=8,
                 num_encoder_layers=3, num_decoder_layers=3, d_ff=512, dropout=0.1, pad_idx=0,
                 use_positional_encoding=True, use_multihead_attention=True, 
                 activation='relu', position_encoding_type='sinusoidal',
                 norm_position='post', use_layer_norm=True):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.use_positional_encoding = use_positional_encoding
        self.use_multihead_attention = use_multihead_attention
        
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model, padding_idx=pad_idx)
        
        # 只有在启用位置编码时才创建位置编码
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model, encoding_type=position_encoding_type)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads if use_multihead_attention else 1, d_ff, dropout,
                        activation=activation, norm_position=norm_position, use_layer_norm=use_layer_norm)
            for _ in range(num_encoder_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads if use_multihead_attention else 1, d_ff, dropout,
                        activation=activation, norm_position=norm_position, use_layer_norm=use_layer_norm)
            for _ in range(num_decoder_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def make_src_mask(self, src):
        # (batch_size, 1, 1, src_len)
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_trg_mask(self, trg):
        # (batch_size, 1, trg_len, trg_len)
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)
        
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(
            torch.ones((trg_len, trg_len), device=trg.device)
        ).bool()
        
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
    
    def encode(self, src):
        src_mask = self.make_src_mask(src)
        
        # Apply embedding
        src_embedded = self.src_embedding(src)
        
        # Apply positional encoding if enabled
        if self.use_positional_encoding:
            src_embedded = self.positional_encoding(src_embedded)
            
        src_embedded = self.dropout(src_embedded)
        
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
            
        return enc_output, src_mask
    
    def decode(self, trg, enc_output, src_mask):
        trg_mask = self.make_trg_mask(trg)
        
        # Apply embedding
        trg_embedded = self.trg_embedding(trg)
        
        # Apply positional encoding if enabled
        if self.use_positional_encoding:
            trg_embedded = self.positional_encoding(trg_embedded)
            
        trg_embedded = self.dropout(trg_embedded)
        
        dec_output = trg_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, trg_mask)
            
        output = self.fc_out(dec_output)
        return output
    
    def forward(self, src, trg):
        enc_output, src_mask = self.encode(src)
        output = self.decode(trg, enc_output, src_mask)
        return output
