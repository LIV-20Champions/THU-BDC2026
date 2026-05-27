#!/bin/bash
source .venv/bin/activate
for s in 7 123; do
  python3 -c "import json; c=json.load(open('model/phase3_final_s${s}/config.json')); c['predict_rank_alpha']=0.2; json.dump(c, open('model/phase3_final_s${s}/config.json','w'), indent=4)"
done
for s in 42 7 123; do
  sed -i "s|'output_dir': './model/[^']*'|'output_dir': './model/phase3_final_s${s}'|" code/src/config.py
  echo "=== Seed $s ==="
  python code/src/predict.py 2>&1 | grep -A6 stock_id
  python test/score_self.py 2>&1 | grep "加权收益率"
done
sed -i "s|'output_dir': './model/[^']*'|'output_dir': './model/phase3_final_s42'|" code/src/config.py
