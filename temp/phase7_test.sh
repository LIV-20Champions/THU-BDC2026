#!/bin/bash
source .venv/bin/activate
for seed in 42 7 123; do
  echo "=== Seed $seed (158+39, 8 epochs, no val) ==="
  python code/src/train.py --seed $seed --output_dir ./model/phase7_s${seed} --num_epochs_override 8 2>&1 | grep "训练完成"
  python3 -c "
import json
with open('model/phase7_s${seed}/config.json') as f:
    c = json.load(f)
c['predict_rank_alpha'] = 0.2
with open('model/phase7_s${seed}/config.json', 'w') as f:
    json.dump(c, f, indent=4)
" 2>/dev/null
  python3 -c "
import sys; sys.path.insert(0, 'code/src')
from config import config
config['output_dir'] = './model/phase7_s${seed}'
from predict import main
" 2>/dev/null
  # Use sed to override output_dir for predict
  cp code/src/config.py code/src/config.py.bak
  sed -i \"s|'output_dir': '[^']*'|'output_dir': './model/phase7_s${seed}'|\" code/src/config.py
  python code/src/predict.py 2>&1 | grep -A6 stock_id
  python test/score_self.py 2>&1 | grep "加权收益率"
  cp code/src/config.py.bak code/src/config.py
  echo ""
done