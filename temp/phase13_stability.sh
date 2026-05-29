#!/bin/bash
source .venv/bin/activate

echo "=== Phase 13 Original Dataset (3 runs) ==="
cp data/stock_data_orig_backup.csv data/stock_data.csv
python data/split_train_test.py 2>&1 | tail -1
for run in 1 2 3; do
  rm -rf model/phase13_nozscore output/result.csv temp/tmp.csv
  python code/src/train.py --num_epochs_override 30 2>&1 | tail -1
  python code/src/predict.py 2>&1 | tail -1
  python test/score_self.py 2>&1 | tail -1
  cp temp/tmp.csv "temp/phase13_orig_run${run}.csv"
  echo "Run $run saved"
done

echo ""
echo "=== Phase 13 April Dataset (3 runs) ==="
cp data/stock_data_April.csv data/stock_data.csv
python data/split_train_test.py 2>&1 | tail -1
for run in 1 2 3; do
  rm -rf model/phase13_nozscore output/result.csv temp/tmp.csv
  python code/src/train.py --num_epochs_override 30 2>&1 | tail -1
  python code/src/predict.py 2>&1 | tail -1
  python test/score_self.py 2>&1 | tail -1
  cp temp/tmp.csv "temp/phase13_april_run${run}.csv"
  echo "Run $run saved"
done

cp data/stock_data_orig_backup.csv data/stock_data.csv
python data/split_train_test.py 2>&1 | tail -1

echo ""
echo "=== Results ==="
for f in temp/phase13_orig_run*.csv; do
  score=$(python3 -c "import pandas as pd; print(pd.read_csv('$f')['Final Score'].iloc[0])")
  echo "$f: $score"
done
for f in temp/phase13_april_run*.csv; do
  score=$(python3 -c "import pandas as pd; print(pd.read_csv('$f')['Final Score'].iloc[0])")
  echo "$f: $score"
done
