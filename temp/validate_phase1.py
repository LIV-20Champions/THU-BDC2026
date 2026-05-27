import sys
sys.path.insert(0, 'code/src')
from config import config
print(f"d_model={config['d_model']}, layers={config['num_layers']}, dropout={config['dropout']}")
print(f"mixup={config['use_mixup']}, label_alpha={config['label_alpha']}, val_months={config['val_months']}")
print(f"top_k={config['selected_top_k_features']}, wd={config['weight_decay']}, dim_ff={config['dim_feedforward']}")
print("config OK")
