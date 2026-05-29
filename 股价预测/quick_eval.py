import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))
from config import config

# Quick score test with available models
config['output_dir'] = './model/_l3_seq15_noval'

# Check for available models
import glob
model_dirs = sorted(glob.glob('./model/_l3_seq15_noval/model_*'))
print(f"Available models: {len(model_dirs)}")
for md in model_dirs:
    pth = os.path.join(md, 'best_model.pth')
    if os.path.exists(pth):
        print(f"  {md}: exists ({os.path.getsize(pth)} bytes)")
    else:
        print(f"  {md}: MISSING")

# Run predict
config['ensemble_model_dirs'] = model_dirs
print(f"\nRunning predict with {len(model_dirs)} models...")

import subprocess
result = subprocess.run(
    [sys.executable, 'code/src/predict.py'],
    capture_output=True, text=True, timeout=120, cwd=os.path.dirname(os.path.abspath(__file__)) + '/..'
)
print(result.stdout[-500:])
if result.stderr:
    print("STDERR:", result.stderr[-200:])

# Score
print("\nRunning score_self...")
result2 = subprocess.run(
    [sys.executable, 'test/score_self.py'],
    capture_output=True, text=True, timeout=60, cwd=os.path.dirname(os.path.abspath(__file__)) + '/..'
)
print(result2.stdout[-200:])
