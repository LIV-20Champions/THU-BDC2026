#!/usr/bin/env python3
"""
Transformer模型实现 - Encoder-only架构
用于时间序列预测，用Linear层替换Embedding层
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 模型参数配置
d_model = 512  # 线性层输入维度
d_ff = 2048    # 前向传播隐藏层维度
d_k = d_v = 64  # K(=Q), V的维度
n_layers = 6   # encoder层数
n_heads = 8    # Multi-Head Attention头数
feature = 6    # 输入特征维度

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

def get_attn_pad_mask(seq_q, seq_k):
    """获取注意力掩码"""
    batch_size, len_q = seq_q.size(0), seq_q.size(1)
    batch_size, len_k = seq_k.size(0), seq_k.size(1)
    # 创建布尔类型的掩码，这里我们不使用padding掩码，所以全部设为False
    pad_attn_mask = torch.zeros(batch_size, len_k, dtype=torch.bool)
    return pad_attn_mask.unsqueeze(1)

class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    
    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
    
    def forward(self, input_Q, input_K, input_V, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)
        
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)
        
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1,2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        return F.layer_norm(output + residual, [d_model]), attn

class PoswiseFeedForwardNet(nn.Module):
    """位置前馈网络"""
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
    
    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return F.layer_norm(output + residual, [d_model])

class EncoderLayer(nn.Module):
    """编码器层"""
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
    
    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class Encoder(nn.Module):
    """Transformer Encoder - 只用编码器部分"""
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Linear(feature, d_model)  # 用Linear替换Embedding
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
    
    def forward(self, enc_inputs):
        # enc_inputs: [batch_size, src_len, feature]
        enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs)  # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
        
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        
        return enc_outputs, enc_self_attns

class Transformer(nn.Module):
    """完整的Transformer模型 - Encoder-only"""
    def __init__(self):
        super(Transformer, self).__init__()
        self.Encoder = Encoder()
        self.projection = nn.Linear(d_model, 1, bias=False)
    
    def forward(self, enc_inputs):
        # enc_inputs: [batch_size, src_len, feature]
        enc_outputs, enc_self_attns = self.Encoder(enc_inputs)  # enc_outputs: [batch_size, src_len, d_model]
        dec_logits = self.projection(enc_outputs)  # dec_logits: [batch_size, src_len, 1]
        dec_logits = dec_logits.mean(dim=1)  # 将每个时间步的预测结果取平均，得到 [batch_size, 1]
        return dec_logits.squeeze(-1), enc_self_attns  # 输出 [batch_size]

if __name__ == "__main__":
    # 测试模型
    print("测试Transformer模型...")
    
    # 创建测试数据
    batch_size, seq_len, feature = 32, 5, 6
    test_input = torch.randn(batch_size, seq_len, feature)
    
    # 创建模型
    model = Transformer()
    
    # 前向传播
    output, attns = model(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力层数: {len(attns)}")
    print(f"每层注意力形状: {attns[0].shape if attns else 'None'}")
    
    # 测试参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")
    
    print("模型测试完成!")