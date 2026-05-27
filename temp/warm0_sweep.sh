#!/bin/bash
source .venv/bin/activate
for seed in 42 7 123 99 2025; do
  echo "=== Seed $seed ==="
  python code/src/train.py --seed $seed --output_dir ./model/phase4_warm0_s${seed} --num_epochs_override 2 2>&1 | grep "Best"
done