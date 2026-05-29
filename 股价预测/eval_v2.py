import sys, os, json
sys.path.insert(0, 'code/src')
from config import config

# Setup ensemble
model_dirs = []
for i in range(3):
    d = f'./model/v2_csrank/model_{i}'
    if os.path.exists(os.path.join(d, 'best_model.pth')):
        model_dirs.append(d)
print(f'Found {len(model_dirs)} models')

if not model_dirs:
    print('No models found')
    sys.exit(1)

config['ensemble_model_dirs'] = model_dirs
config['output_dir'] = './model/v2_csrank'

meta = {'ensemble_size': len(model_dirs), 'seeds': [42 + i*7 for i in range(len(model_dirs))]}
os.makedirs('./model/v2_csrank', exist_ok=True)
with open('./model/v2_csrank/ensemble_config.json', 'w') as f:
    json.dump(meta, f)

import subprocess
# Predict
r = subprocess.run([sys.executable, 'code/src/predict.py'], capture_output=True, text=True, timeout=120)
print('PREDICT:', r.stdout[-300:])
if r.stderr:
    print('STDERR:', r.stderr[-200:])

# Score
r2 = subprocess.run([sys.executable, 'test/score_self.py'], capture_output=True, text=True, timeout=60)
print(r2.stdout.strip())
