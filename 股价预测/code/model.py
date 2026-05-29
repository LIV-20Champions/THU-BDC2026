import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CrossStockAttention(nn.Module):
    """股票间交互注意力模块"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super(CrossStockAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, stock_features, stock_mask=None):
        key_padding_mask = None
        if stock_mask is not None:
            key_padding_mask = ~stock_mask.bool()

        attended, _ = self.cross_attention(
            stock_features,
            stock_features,
            stock_features,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        output = self.norm(stock_features + self.dropout(attended))

        if stock_mask is not None:
            output = output * stock_mask.unsqueeze(-1).to(output.dtype)
        return output


class FeatureAttention(nn.Module):
    """时间维特征注意力聚合"""
    def __init__(self, d_model, dropout=0.1, activation='tanh'):
        super(FeatureAttention, self).__init__()
        activation_name = str(activation).lower()
        if activation_name == 'gelu':
            activation_layer = nn.GELU()
        elif activation_name == 'relu':
            activation_layer = nn.ReLU()
        else:
            activation_layer = nn.Tanh()

        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            activation_layer,
            nn.Linear(d_model // 2, 1),
            nn.Softmax(dim=1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch*num_stocks, seq_len, d_model]
        attention_weights = self.attention(x)         # [batch*num_stocks, seq_len, 1]
        attended = torch.sum(x * attention_weights, dim=1)  # [batch*num_stocks, d_model]
        return self.dropout(attended)


class StockTransformer(nn.Module):
    def __init__(self, input_dim, config, num_stocks, emb_dim=16):
        super(StockTransformer, self).__init__()
        self.model_type = "RankingTransformer"
        self.config = config
        self.num_stocks = num_stocks
        self.use_stock_embedding = bool(config.get("use_stock_embedding", True))
        self.stock_emb_dim = int(config.get("stock_emb_dim", emb_dim))

        numeric_input_dim = input_dim - 1 if self.use_stock_embedding else input_dim
        if numeric_input_dim <= 0:
            raise ValueError(f"numeric_input_dim 非法: {numeric_input_dim}")

        # 数值特征投影
        self.input_proj = nn.Linear(numeric_input_dim, config["d_model"])

        # 股票ID embedding
        if self.use_stock_embedding:
            self.stock_embedding = nn.Embedding(num_stocks, self.stock_emb_dim)
            self.stock_emb_proj = nn.Sequential(
                nn.Linear(self.stock_emb_dim, config["d_model"]),
                nn.LayerNorm(config["d_model"]),
                nn.Dropout(config["dropout"]),
            )

        self.pos_encoder = PositionalEncoding(config["d_model"], config["dropout"], config["sequence_length"])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["d_model"],
            nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config["num_layers"])

        self.feature_attention = FeatureAttention(
            config["d_model"],
            config["dropout"],
            activation=config.get("feature_attention_activation", "gelu"),
        )
        self.cross_stock_attention = CrossStockAttention(config["d_model"], config["nhead"], config["dropout"])

        self.ranking_layers = nn.Sequential(
            nn.Linear(config["d_model"], config["d_model"]),
            nn.LayerNorm(config["d_model"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["d_model"], config["d_model"] // 2),
            nn.LayerNorm(config["d_model"] // 2),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
        )

        self.score_head = nn.Sequential(
            nn.Linear(config["d_model"] // 2, config["d_model"] // 4),
            nn.ReLU(),
            nn.Dropout(config["dropout"] * 0.5),
            nn.Linear(config["d_model"] // 4, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, src, stock_indices=None, stock_mask=None):
        """
        src: [batch, num_stocks, seq_len, feature_dim]
        stock_indices: [batch, num_stocks]，每个股票对应的离散ID
        stock_mask: [batch, num_stocks]，1 表示有效股票，0 表示 padding
        """
        batch_size, num_stocks, seq_len, feature_dim = src.size()

        if self.use_stock_embedding:
            if stock_indices is None:
                raise ValueError("启用 use_stock_embedding 时，stock_indices 不能为空")
            if stock_indices.size(0) != batch_size or stock_indices.size(1) != num_stocks:
                raise ValueError(
                    f"stock_indices 形状不匹配，期望 {[batch_size, num_stocks]}，实际 {list(stock_indices.shape)}"
                )
            # 第 0 列是 instrument 数值ID，不再作为连续特征输入
            numeric_src = src[..., 1:]
        else:
            numeric_src = src

        src_reshaped = numeric_src.reshape(batch_size * num_stocks, seq_len, -1)
        src_proj = self.input_proj(src_reshaped)  # [B*N, L, d_model]

        if self.use_stock_embedding:
            stock_indices = stock_indices.long()
            stock_emb = self.stock_embedding(stock_indices)          # [B, N, emb_dim]
            stock_emb = self.stock_emb_proj(stock_emb)              # [B, N, d_model]
            stock_emb = stock_emb.reshape(batch_size * num_stocks, -1).unsqueeze(1)  # [B*N, 1, d_model]
            src_proj = src_proj + stock_emb

        src_proj = self.pos_encoder(src_proj)

        temporal_features = self.temporal_encoder(src_proj)         # [B*N, L, d_model]
        aggregated_features = self.feature_attention(temporal_features)  # [B*N, d_model]

        stock_features = aggregated_features.reshape(batch_size, num_stocks, -1)  # [B, N, d_model]
        interactive_features = self.cross_stock_attention(stock_features, stock_mask=stock_mask)  # [B, N, d_model]

        interactive_features = interactive_features.reshape(batch_size * num_stocks, -1)
        ranking_features = self.ranking_layers(interactive_features)
        scores = self.score_head(ranking_features)

        output = scores.reshape(batch_size, num_stocks)
        return output
