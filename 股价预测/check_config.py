import sys; sys.path.insert(0,"code/src"); from config import config
print("Config OK")
print(f"feature_num={config['feature_num']}, seq={config['sequence_length']}")
print(f"num_epochs={config['num_epochs']}, ensemble={config['ensemble_size']}")
print(f"mse_aux_weight={config['mse_aux_weight']}, dropout={config['dropout']}")
print(f"use_per_stock_normalize={config['use_per_stock_normalize']}")
print(f"val_months={config['val_months']}")
print(f"output_dir={config['output_dir']}")