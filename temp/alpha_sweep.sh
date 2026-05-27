#!/bin/bash
source .venv/bin/activate
for alpha in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1.0 1.5; do
  python3 -c "import json; c=json.load(open('model/phase2_loss64_s42/config.json')); c['predict_rank_alpha']=$alpha; json.dump(c, open('model/phase2_loss64_s42/config.json','w'), indent=4, ensure_ascii=False)"
  python code/src/predict.py 2>&1 | tail -1
  python test/score_self.py 2>&1 | grep "加权收益率"
  echo "alpha=$alpha done"
done
