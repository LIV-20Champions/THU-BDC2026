import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
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
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor


class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super(LayerScale, self).__init__()
        self.scale = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x):
        return x * self.scale


class GatedLinearUnit(nn.Module):
    """GLU: Gate(x) = sigmoid(W_g * x) * (W * x)"""
    def __init__(self, input_size, output_size):
        super(GatedLinearUnit, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.gate = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x) * torch.sigmoid(self.gate(x))


class AddNorm(nn.Module):
    def __init__(self, dimension):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(dimension)

    def forward(self, x, skip):
        return self.norm(x + skip)


class GatedResidualNetwork(nn.Module):
    """GRN with optional context injection, following TFT architecture.

    GRN(x, c) = LayerNorm(x + GLU(eta2 + W_c * c))
    where eta2 = Dense2(ELU(Dense1(x)))

    If context is None, the context path is omitted.
    """
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1, context_size=None):
        super(GatedResidualNetwork, self).__init__()
        self.input_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
        self.context_proj = nn.Linear(context_size, hidden_size) if context_size is not None else None
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.glu = GatedLinearUnit(hidden_size, output_size)
        self.add_norm = AddNorm(output_size)
        self.dropout = nn.Dropout(dropout)
        self.skip_proj = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()

    def forward(self, x, context=None):
        skip = self.skip_proj(x)
        h = self.input_proj(x)
        if context is not None and self.context_proj is not None:
            h = h + self.context_proj(context)
        h = self.dropout(F.elu(self.dense1(h)))
        h = self.dense2(h)
        h = self.dropout(self.glu(h))
        return self.add_norm(h, skip)


class VariableSelectionNetwork(nn.Module):
    """Group-wise VSN: partitions features into groups, each group processed
    by a shared GRN, then softmax-weighted combination.

    Total features F are split into G groups. Each group j has:
      - GRN_j for processing group-level features
      - A weight network that computes attention over groups based on
        flattened feature context

    Output: weighted sum of all group GRN outputs.
    """
    def __init__(self, num_features, d_model, num_groups=10, hidden_size=64,
                 dropout=0.1, temperature=1.0):
        super(VariableSelectionNetwork, self).__init__()
        self.num_groups = num_groups
        self.d_model = d_model
        self.temperature = temperature

        features_per_group = max(1, num_features // num_groups)
        group_sizes = [features_per_group] * num_groups
        remainder = num_features - features_per_group * num_groups
        for i in range(remainder):
            group_sizes[i] += 1

        self.group_sizes = group_sizes
        self.group_starts = [0]
        for size in group_sizes[:-1]:
            self.group_starts.append(self.group_starts[-1] + size)

        # per-group GRN: input=group_size, hidden=hidden_size, output=d_model
        self.group_grns = nn.ModuleList([
            GatedResidualNetwork(sz, hidden_size, d_model, dropout=dropout)
            for sz in group_sizes
        ])

        # weight network: flatten all features → GRN → softmax over groups
        weight_grn_hidden = hidden_size * 2
        self.weight_grn = GatedResidualNetwork(
            num_features, weight_grn_hidden, num_groups, dropout=dropout
        )

    def forward(self, x):
        # x: [B*N, L, F] for temporal input, or [B*N, F] for static input
        is_3d = x.dim() == 3
        if is_3d:
            static_context = x.mean(dim=1)
        else:
            static_context = x

        # compute group weights from static context
        raw_weights = self.weight_grn(static_context)  # [B*N, num_groups]
        weights = F.softmax(raw_weights / self.temperature, dim=-1)

        # process each group
        outputs = []
        for i in range(self.num_groups):
            start = self.group_starts[i]
            end = start + self.group_sizes[i]
            group_features = x[..., start:end]
            if is_3d:
                B, L, G = group_features.shape
                group_features_flat = group_features.reshape(B * L, G)
                grn_out = self.group_grns[i](group_features_flat)
                grn_out = grn_out.reshape(B, L, self.d_model)
                outputs.append(grn_out * weights[:, i:i+1].unsqueeze(1))
            else:
                grn_out = self.group_grns[i](group_features)
                outputs.append(grn_out * weights[:, i:i+1])

        if is_3d:
            result = torch.stack(outputs, dim=0).sum(dim=0)  # [B*N, L, D]
        else:
            result = torch.stack(outputs, dim=0).sum(dim=0)
        return result


class FeatureAttention(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super(FeatureAttention, self).__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cls_pos = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        cls_pos = self.cls_pos.expand(batch_size, -1, -1)
        cls_tokens = cls_tokens + cls_pos
        x_with_cls = torch.cat([cls_tokens, x], dim=1)
        x_with_cls = self.norm(x_with_cls)
        attended, _ = self.attention(x_with_cls, x_with_cls, x_with_cls, need_weights=False)
        cls_output = attended[:, 0, :]
        return self.dropout(cls_output)


class CrossStockAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, layer_scale_init=1e-5, drop_path=0.0):
        super(CrossStockAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_scale = LayerScale(d_model, init_value=layer_scale_init)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, stock_features, stock_mask=None):
        key_padding_mask = None
        if stock_mask is not None:
            key_padding_mask = ~stock_mask.bool()

        attended, _ = self.cross_attention(
            stock_features, stock_features, stock_features,
            key_padding_mask=key_padding_mask, need_weights=False,
        )
        attended = self.dropout(attended)
        if stock_mask is not None:
            attended = attended * stock_mask.unsqueeze(-1).to(attended.dtype)
        output = self.norm(stock_features + self.drop_path(self.layer_scale(attended)))

        if stock_mask is not None:
            output = output * stock_mask.unsqueeze(-1).to(output.dtype)
        return output


class MultiScaleTemporalEncoder(nn.Module):
    """Three-branch multi-scale temporal encoder:

    Short branch: patchify into 5-day patches, 2-layer Transformer
    Medium branch: segment into 15-day segments, 2-layer Transformer
    Long branch: full 60-day sequence, 2-layer Transformer
    Fusion: learnable linear projection of concatenated outputs
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout,
                 short_patch=5, medium_segment=15,
                 short_layers=2, medium_layers=2, long_layers=2,
                 fusion_mode='linear'):
        super(MultiScaleTemporalEncoder, self).__init__()
        self.d_model = d_model
        self.short_patch = short_patch
        self.medium_segment = medium_segment
        self.fusion_mode = fusion_mode

        def _make_encoder(num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=num_layers)

        # patch projection layers
        self.short_proj = nn.Sequential(
            nn.Conv1d(d_model * short_patch, d_model, kernel_size=1),
        )
        self.medium_proj = nn.Sequential(
            nn.Conv1d(d_model * medium_segment, d_model, kernel_size=1),
        )

        self.short_encoder = _make_encoder(short_layers)
        self.medium_encoder = _make_encoder(medium_layers)
        self.long_encoder = _make_encoder(long_layers)

        self.short_attn = FeatureAttention(d_model, nhead=nhead, dropout=dropout)
        self.medium_attn = FeatureAttention(d_model, nhead=nhead, dropout=dropout)
        self.long_attn = FeatureAttention(d_model, nhead=nhead, dropout=dropout)

        self.short_fallback = nn.Parameter(torch.zeros(1, d_model))
        self.medium_fallback = nn.Parameter(torch.zeros(1, d_model))
        nn.init.normal_(self.short_fallback, mean=0.0, std=0.02)
        nn.init.normal_(self.medium_fallback, mean=0.0, std=0.02)

        if fusion_mode == 'linear':
            self.fusion = nn.Sequential(
                nn.Linear(d_model * 3, d_model),
                nn.LayerNorm(d_model),
            )
        elif fusion_mode == 'gated':
            self.fusion_gate = nn.Sequential(
                nn.Linear(d_model * 3, d_model * 3),
                nn.Sigmoid(),
            )
            self.fusion_proj = nn.Linear(d_model * 3, d_model)
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

    def forward(self, x):
        # x: [B*N, L, D]
        B, L, D = x.size()

        # short branch: patchify
        num_patches_short = L // self.short_patch
        if num_patches_short > 0:
            short_trim = L - L % self.short_patch
            short_x = x[:, :short_trim, :]
            short_patches = short_x.reshape(B, num_patches_short, self.short_patch * D)
            short_patches = short_patches.permute(0, 2, 1)  # [B, patch*D, num_patches]
            short_out = self.short_proj(short_patches).permute(0, 2, 1)  # [B, num_patches, D]
            short_encoded = self.short_encoder(short_out)
            short_feat = self.short_attn(short_encoded)
        else:
            short_feat = self.short_fallback.expand(B, -1)

        # medium branch: segment
        num_segments_med = L // self.medium_segment
        if num_segments_med > 0:
            med_trim = L - L % self.medium_segment
            med_x = x[:, :med_trim, :]
            med_segments = med_x.reshape(B, num_segments_med, self.medium_segment * D)
            med_segments = med_segments.permute(0, 2, 1)
            med_out = self.medium_proj(med_segments).permute(0, 2, 1)
            med_encoded = self.medium_encoder(med_out)
            med_feat = self.medium_attn(med_encoded)
        else:
            med_feat = self.medium_fallback.expand(B, -1)

        # long branch: full sequence
        long_encoded = self.long_encoder(x)
        long_feat = self.long_attn(long_encoded)

        # fusion
        concat = torch.cat([short_feat, med_feat, long_feat], dim=-1)
        if self.fusion_mode == 'linear':
            fused = self.fusion(concat)
        else:
            gate = self.fusion_gate(concat)
            fused = self.fusion_proj(gate * concat)

        return fused


class CNNFeatureExtractor(nn.Module):
    """1D CNN feature extractor inspired by AlphaNet.

    Applies temporal convolutions to raw stock features to learn
    new time-series patterns.  Output concat'd with original features
    before the Transformer encoder.
    """
    def __init__(self, in_channels, out_channels=32, kernel_sizes=None):
        super(CNNFeatureExtractor, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 3]
        mid = 64

        self.conv1 = nn.Conv1d(in_channels, mid, kernel_sizes[0], padding=kernel_sizes[0]//2)
        self.bn1 = nn.BatchNorm1d(mid)
        self.conv2 = nn.Conv1d(mid, mid, kernel_sizes[1], padding=kernel_sizes[1]//2)
        self.bn2 = nn.BatchNorm1d(mid)
        self.conv3 = nn.Conv1d(mid, out_channels, kernel_sizes[2], padding=kernel_sizes[2]//2)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x: [B*N, L, F] -> transpose to [B*N, F, L] for Conv1d
        x_t = x.transpose(1, 2)
        x_t = self.activation(self.bn1(self.conv1(x_t)))
        x_t = self.activation(self.bn2(self.conv2(x_t)))
        x_t = self.activation(self.bn3(self.conv3(x_t)))
        # Back to [B*N, L, out_channels]
        return x_t.transpose(1, 2)


class StockTransformer(nn.Module):
    def __init__(self, input_dim, config, num_stocks, emb_dim=16):
        super(StockTransformer, self).__init__()
        self.config = config
        self.num_stocks = num_stocks
        self.use_stock_embedding = bool(config.get("use_stock_embedding", True))
        self.stock_emb_dim = int(config.get("stock_emb_dim", emb_dim))
        self.use_vsn = bool(config.get("use_vsn", False))
        self.use_multi_scale = bool(config.get("use_multi_scale", False))
        self.use_gradient_checkpointing = bool(config.get("use_gradient_checkpointing", False))

        numeric_input_dim = input_dim - 1 if self.use_stock_embedding else input_dim
        if numeric_input_dim <= 0:
            raise ValueError(f"numeric_input_dim illegal: {numeric_input_dim}")

        d_model = config["d_model"]
        dropout = config["dropout"]

        # feature projection: VSN or Linear
        if self.use_vsn:
            vsn_groups = int(config.get("vsn_num_groups", 10))
            vsn_hidden = int(config.get("vsn_hidden_size", 64))
            vsn_temp = float(config.get("vsn_temperature", 1.0))
            vsn_dropout = float(config.get("grn_dropout", dropout))
            self.input_vsn = VariableSelectionNetwork(
                num_features=numeric_input_dim,
                d_model=d_model,
                num_groups=vsn_groups,
                hidden_size=vsn_hidden,
                dropout=vsn_dropout,
                temperature=vsn_temp,
            )
            self.input_proj = None
        else:
            self.input_vsn = None
            self.input_proj = nn.Linear(numeric_input_dim, d_model)

        # CNN feature extractor (AlphaNet-style, parallel path to TA-Lib)
        self.use_cnn_features = bool(config.get("use_cnn_features", False))
        if self.use_cnn_features:
            cnn_out_dim = int(config.get("cnn_feature_dim", 32))
            self.cnn_extractor = CNNFeatureExtractor(
                in_channels=numeric_input_dim,
                out_channels=cnn_out_dim,
            )
            self.cnn_proj = nn.Sequential(
                nn.Linear(cnn_out_dim, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
            )
            self.fusion_proj = nn.Linear(d_model * 2, d_model)
        else:
            self.cnn_extractor = None

        # stock embedding
        if self.use_stock_embedding:
            self.stock_embedding = nn.Embedding(num_stocks, self.stock_emb_dim)
            self.stock_emb_proj = nn.Sequential(
                nn.Linear(self.stock_emb_dim, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
            )

        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=5000)

        # temporal encoding: multi-scale or single Transformer
        if self.use_multi_scale:
            self.temporal_encoder = MultiScaleTemporalEncoder(
                d_model=d_model,
                nhead=config["nhead"],
                dim_feedforward=config["dim_feedforward"],
                dropout=dropout,
                short_patch=int(config.get("ms_short_patch", 5)),
                medium_segment=int(config.get("ms_medium_segment", 15)),
                short_layers=int(config.get("ms_short_layers", 2)),
                medium_layers=int(config.get("ms_medium_layers", 2)),
                long_layers=int(config.get("ms_long_layers", 2)),
                fusion_mode=config.get("ms_fusion_mode", "linear"),
            )
            # when multi_scale is used, the temporal encoder outputs [B*N, D] directly
            # (FeatureAttention is built into each branch)
            self._ms_feature_attn = None
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=config["nhead"],
                dim_feedforward=config["dim_feedforward"],
                dropout=dropout, batch_first=True, norm_first=True,
            )
            self.temporal_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=config["num_layers"]
            )
            self._ms_feature_attn = FeatureAttention(
                d_model, nhead=config["nhead"], dropout=dropout
            )

        # GRU dual path
        self.use_gru_dual_path = bool(config.get("use_gru_dual_path", False))
        if self.use_gru_dual_path:
            gru_hidden = int(config.get("gru_hidden_dim", 128))
            gru_layers = int(config.get("gru_num_layers", 1))
            self.gru_encoder = nn.GRU(
                input_size=d_model, hidden_size=gru_hidden,
                num_layers=gru_layers, batch_first=True,
                dropout=dropout if gru_layers > 1 else 0, bidirectional=True,
            )
            fusion_in_dim = d_model + gru_hidden * 2
            self.fusion_gate = nn.Sequential(
                nn.Linear(fusion_in_dim, d_model), nn.Sigmoid(),
            )
            self.fusion_proj = nn.Linear(gru_hidden * 2, d_model)
            self.fusion_norm = nn.LayerNorm(d_model)
            self.fusion_dropout = nn.Dropout(dropout)

        # cross-stock attention layers
        csa_layers = int(config.get("cross_stock_layers", 1))
        drop_path_rate = float(config.get("drop_path_rate", 0.0))
        layer_scale_init = float(config.get("layer_scale_init", 1e-5))

        dp_rates = [drop_path_rate * i / max(csa_layers - 1, 1) for i in range(csa_layers)]
        self.cross_stock_layers = nn.ModuleList([
            CrossStockAttention(
                d_model, config["nhead"], dropout,
                layer_scale_init=layer_scale_init, drop_path=dp_rates[i],
            )
            for i in range(csa_layers)
        ])

        # feature interaction
        self.use_feature_interaction = bool(config.get("use_feature_interaction", False))
        if self.use_feature_interaction:
            interaction_hidden = int(config.get("interaction_hidden_dim", 64))
            self.interaction = nn.Sequential(
                nn.Linear(d_model, interaction_hidden),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(interaction_hidden, d_model),
            )
            self.interaction_norm = nn.LayerNorm(d_model)

        self.ranking_layers = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.score_head = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 4, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.TransformerEncoder, nn.TransformerEncoderLayer,
                                   nn.MultiheadAttention)):
                continue
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _cross_stock_to_scores(self, aggregated_features, batch_size, num_stocks, stock_mask):
        stock_features = aggregated_features.reshape(batch_size, num_stocks, -1)
        for csa_layer in self.cross_stock_layers:
            stock_features = csa_layer(stock_features, stock_mask=stock_mask)

        if self.use_feature_interaction:
            residual = stock_features
            stock_features_flat = stock_features.reshape(batch_size * num_stocks, -1)
            stock_features_flat = self.interaction(stock_features_flat)
            stock_features = stock_features_flat.reshape(batch_size, num_stocks, -1)
            stock_features = self.interaction_norm(residual + stock_features)

        interactive_features = stock_features.reshape(batch_size * num_stocks, -1)
        ranking_features = self.ranking_layers(interactive_features)
        scores = self.score_head(ranking_features)
        return scores.reshape(batch_size, num_stocks)

    def forward(self, src, stock_indices=None, stock_mask=None):
        batch_size, num_stocks, seq_len, feature_dim = src.size()

        if self.use_stock_embedding:
            if stock_indices is None:
                raise ValueError("use_stock_embedding requires stock_indices")
            if stock_indices.size(0) != batch_size or stock_indices.size(1) != num_stocks:
                raise ValueError(
                    f"stock_indices shape mismatch: expected {[batch_size, num_stocks]}, "
                    f"got {list(stock_indices.shape)}"
                )
            numeric_src = src[..., 1:]
            numeric_input_dim = feature_dim - 1
        else:
            numeric_src = src
            numeric_input_dim = feature_dim

        src_reshaped = numeric_src.reshape(batch_size * num_stocks, seq_len, numeric_input_dim)

        # feature projection: VSN or Linear
        if self.use_vsn:
            if self.use_gradient_checkpointing and self.training:
                src_proj = cp.checkpoint(self.input_vsn, src_reshaped, use_reentrant=False)
            else:
                src_proj = self.input_vsn(src_reshaped)
        else:
            src_proj = self.input_proj(src_reshaped)

        # CNN feature extraction (parallel path, concat then fuse)
        if self.use_cnn_features:
            src_cnn = self.cnn_extractor(src_reshaped)
            src_cnn_proj = self.cnn_proj(src_cnn)
            src_proj = self.fusion_proj(torch.cat([src_proj, src_cnn_proj], dim=-1))

        # stock embedding
        if self.use_stock_embedding:
            stock_indices = stock_indices.long()
            safe_indices = stock_indices.clamp(min=0)
            stock_emb = self.stock_embedding(safe_indices)
            stock_emb = self.stock_emb_proj(stock_emb)
            stock_emb = stock_emb.reshape(batch_size * num_stocks, -1).unsqueeze(1)
            if stock_mask is not None:
                mask_flat = stock_mask.reshape(batch_size * num_stocks, 1, 1).to(stock_emb.dtype)
                stock_emb = stock_emb * mask_flat
            src_proj = src_proj + stock_emb

        src_proj = self.pos_encoder(src_proj)

        # temporal encoding: multi-scale or single Transformer
        if self.use_multi_scale:
            if self.use_gradient_checkpointing and self.training:
                aggregated_features = cp.checkpoint(self.temporal_encoder, src_proj, use_reentrant=False)
            else:
                aggregated_features = self.temporal_encoder(src_proj)
            return self._cross_stock_to_scores(aggregated_features, batch_size, num_stocks, stock_mask)

        # single Transformer path
        if self.use_gradient_checkpointing and self.training:
            temporal_features = cp.checkpoint(self.temporal_encoder, src_proj, use_reentrant=False)
        else:
            temporal_features = self.temporal_encoder(src_proj)

        if self.use_gru_dual_path:
            gru_out, _ = self.gru_encoder(src_proj)
            fused = torch.cat([temporal_features, gru_out], dim=-1)
            gate = self.fusion_gate(fused)
            temporal_features = gate * temporal_features + (1 - gate) * self.fusion_proj(gru_out)
            temporal_features = self.fusion_norm(temporal_features)
            temporal_features = self.fusion_dropout(temporal_features)

        if self.use_gradient_checkpointing and self.training:
            aggregated_features = cp.checkpoint(self._ms_feature_attn, temporal_features, use_reentrant=False)
        else:
            aggregated_features = self._ms_feature_attn(temporal_features)
        return self._cross_stock_to_scores(aggregated_features, batch_size, num_stocks, stock_mask)