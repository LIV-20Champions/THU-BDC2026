#!/bin/bash
source .venv/bin/activate
for seed in 42 7 123 99 2025; do
  echo "=== Seed $seed (15 epochs) ==="
  python code/src/train.py --seed $seed --output_dir ./model/phase5_s${seed} --num_epochs_override 15 2>&1 | grep "训练完成"
  # Set predict alpha and score
  python3 -c "import json; c=json.load(open('model/phase5_s${seed}/config.json')); c['predict_rank_alpha']=0.2; json.dump(c, open('model/phase5_s${seed}/config.json','w'), indent=4)" 2>/dev/null
  sed -i "s|'output_dir': '[^']*'|'output_dir': './model/phase5_s${seed}'|" code/src/config.py
  python code/src/predict.py 2>&1 | grep -A6 stock_id
  python test/score_self.py 2>&1 | grep "加权收益率"
  echo ""
done
