"""Run classmate's full training + predict pipeline and measure time.

Usage: python 股价预测/run_classmate.py [original|april]
"""
import sys, os, time, shutil

# Use classmate's code directory
classmate_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code')
sys.path.insert(0, classmate_dir)

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Which dataset
dataset = sys.argv[1] if len(sys.argv) > 1 else 'original'

# Prepare data
if dataset == 'april':
    print("=== Using April dataset ===")
    import subprocess
    subprocess.run([sys.executable, 'data/split_train_test.py'],
                   env={**os.environ, 'STOCK_DATA': 'data/stock_data_April.csv'},
                   timeout=60)
else:
    print("=== Using original dataset ===")

# Import classmate's code
from config import config
print(f"Config: feature_num={config['feature_num']}, seq={config['sequence_length']}")
print(f"Batch: {config['batch_size']}, epochs={config['num_epochs']}")
print(f"Val months: {config['val_months']}, d_model={config['d_model']}")
print(f"Output dir: {config['output_dir']}")

# Clean output
output_dir = config['output_dir']
if os.path.exists(output_dir):
    print(f"Removing existing output: {output_dir}")
    shutil.rmtree(output_dir)

# Train
import train as classmate_train
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

start = time.time()
best_score = classmate_train.main()
train_time = time.time() - start
print(f"\nTraining completed in {train_time:.0f}s ({train_time/60:.1f}min)")
print(f"Best val score: {best_score:.6f}")

# Predict
print("\n=== Running predict ===")
import predict as classmate_predict
start = time.time()

# Reset config after training modifications
import importlib
importlib.reload(__import__('config', fromlist=['config']))
from config import config as fresh_config

# Need to reload predict too since it imports config
classmate_predict.config = fresh_config
classmate_predict.main()
predict_time = time.time() - start
print(f"Prediction completed in {predict_time:.0f}s")

# Score
print("\n=== Running score_self ===")
import subprocess
r = subprocess.run([sys.executable, 'test/score_self.py'],
                   capture_output=True, text=True, timeout=60)
print(r.stdout.strip())
print(f"\n=== Total time: {train_time + predict_time:.0f}s ({(train_time + predict_time)/60:.1f}min) ===")
