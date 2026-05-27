import sys
sys.path.insert(0, 'code/src')
from model import StockTransformer

config_baseline = {
    'd_model': 128, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 256,
    'dropout': 0.35, 'use_stock_embedding': True, 'stock_emb_dim': 16,
    'use_vsn': False, 'use_multi_scale': False, 'use_gru_dual_path': False,
    'use_feature_interaction': False, 'use_cross_sectional_features': False,
    'use_gradient_checkpointing': False, 'cross_stock_layers': 2,
    'layer_scale_init': 1e-5, 'drop_path_rate': 0.1,
}

# Original: 39 features, effective input dim = 39
m1 = StockTransformer(input_dim=39, config=config_baseline, num_stocks=300)
p1 = sum(p.numel() for p in m1.parameters() if p.requires_grad)
print(f"Original (input_dim=39): {p1:,} params")

# Phase 1 v3: 31 features, effective input dim = 31
m2 = StockTransformer(input_dim=31, config=config_baseline, num_stocks=300)
p2 = sum(p.numel() for p in m2.parameters() if p.requires_grad)
print(f"Phase 1 v3 (input_dim=31): {p2:,} params")
