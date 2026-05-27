#!/bin/bash
source .venv/bin/activate
for s in 42 7 123 99 2025; do
  python3 -c "import json; c=json.load(open('model/phase4_warm0_s${s}/config.json')); c['predict_rank_alpha']=0.2; json.dump(c, open('model/phase4_warm0_s${s}/config.json','w'), indent=4)" 2>/dev/null
  sed -i "s|'output_dir': '[^']*'|'output_dir': './model/phase4_warm0_s${s}'|" code/src/config.py
  out=$(python code/src/predict.py 2>&1 | grep -A6 stock_id)
  score=$(python test/score_self.py 2>&1 | grep "加权收益率")
  echo "=== Seed $s ==="
  echo "$out"
  echo "$score"
  echo ""
done