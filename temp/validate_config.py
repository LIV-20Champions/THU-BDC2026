import sys; sys.path.insert(0,'code/src')
from config import config
for k in ['use_sam','use_vsn','use_feature_interaction','use_mse_aux_loss','use_cosine_restarts']:
    print(f'{k}={config[k]}')
print(f'd_model={config["d_model"]}, dropout={config["dropout"]}, label_alpha={config["label_alpha"]}')
print(f'val_months={config["val_months"]}, ensemble_size={config["ensemble_size"]}')
print(f'output_dir={config["output_dir"]}')
print(f'sam_rho={config["sam_rho"]}, restart_T0={config["cosine_restart_T0"]}')
print(f'grad_accum={config["gradient_accumulation_steps"]}')
print('Config OK')
