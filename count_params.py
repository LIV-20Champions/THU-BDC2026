import sys
sys.path.insert(0, '/home/vega/c4/THU-BDC2026/code/src')

from config import config, feature_columns_map, get_eff_input_dim
from model import StockTransformer

num_features = len(feature_columns_map['39'])
input_dim = get_eff_input_dim(num_features)
print(f"feature_columns_map['39'] length: {num_features}")
print(f"get_eff_input_dim({num_features}): {input_dim}")

model = StockTransformer(input_dim=input_dim, config=config, num_stocks=300)

total = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total:,}")

components = {
    'input_proj': model.input_proj,
    'stock_embedding': model.stock_embedding,
    'stock_emb_proj': model.stock_emb_proj,
    'pos_encoder': model.pos_encoder,
    'temporal_encoder': model.temporal_encoder,
    '_ms_feature_attn': model._ms_feature_attn,
    'cross_stock_layers': model.cross_stock_layers,
    'interaction': model.interaction,
    'interaction_norm': model.interaction_norm,
    'ranking_layers': model.ranking_layers,
    'score_head': model.score_head,
}

component_total = 0
print(f"\n{'Component':<25} {'Params':>12}")
print('-' * 39)
for name, mod in components.items():
    if mod is None:
        print(f"{name:<25} {'None':>12}")
        continue
    count = sum(p.numel() for p in mod.parameters())
    component_total += count
    print(f"{name:<25} {count:>12,}")

print('-' * 39)
print(f"{'Component sum':<25} {component_total:>12,}")
print(f"{'Total':<25} {total:>12,}")
print(f"{'Diff (untracked)':<25} {total - component_total:>12,}")
