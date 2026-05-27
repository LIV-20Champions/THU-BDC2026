"""Test Phase 7: 158+39 features, val_months=0, 8 epochs."""
import subprocess, sys, json, os

seeds = [42, 7, 123]
for seed in seeds:
    print(f"\n=== Seed {seed} (158+39, 8 epochs, no val) ===")
    # Train
    r = subprocess.run([sys.executable, 'code/src/train.py', '--seed', str(seed),
                       '--output_dir', f'./model/phase7_s{seed}',
                       '--num_epochs_override', '8'],
                      capture_output=True, text=True)
    for line in r.stdout.split('\n'):
        if '训练完成' in line:
            print(line)

    # Set predict alpha
    cfg_path = f'model/phase7_s{seed}/config.json'
    with open(cfg_path) as f:
        c = json.load(f)
    c['predict_rank_alpha'] = 0.2
    with open(cfg_path, 'w') as f:
        json.dump(c, f, indent=4)

    # Update config.py output_dir
    with open('code/src/config.py', 'r') as f:
        cfg_text = f.read()
    import re
    cfg_text = re.sub(r"'output_dir':\s*'[^']*'", f"'output_dir': './model/phase7_s{seed}'", cfg_text)
    with open('code/src/config.py', 'w') as f:
        f.write(cfg_text)

    # Predict
    r = subprocess.run([sys.executable, 'code/src/predict.py'],
                      capture_output=True, text=True)
    for line in r.stdout.split('\n'):
        if 'stock_id' in line or any(s in line for s in ['0.2406', '0.2095']):
            continue
        if 'stock_id' in line:
            print(line.strip())
    # Print top stocks
    lines = r.stdout.split('\n')
    for i, line in enumerate(lines):
        if 'stock_id' in line:
            for j in range(i, min(i+7, len(lines))):
                print(lines[j].strip())

    # Score
    r = subprocess.run([sys.executable, 'test/score_self.py'],
                      capture_output=True, text=True)
    for line in r.stdout.split('\n'):
        if '加权收益率' in line:
            print(line)

# Restore config
with open('code/src/config.py', 'r') as f:
    cfg_text = f.read()
cfg_text = re.sub(r"'output_dir':\s*'[^']*'", "'output_dir': './model/phase7_s42'", cfg_text)
with open('code/src/config.py', 'w') as f:
    f.write(cfg_text)
print("\nDone!")
