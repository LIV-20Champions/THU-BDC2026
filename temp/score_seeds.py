import subprocess, sys, json, re

for seed in [42, 7, 123]:
    cfg_path = f'model/phase7_s{seed}/config.json'
    with open(cfg_path) as f: c = json.load(f)
    c['predict_rank_alpha'] = 0.2
    with open(cfg_path, 'w') as f: json.dump(c, f, indent=4)

    with open('code/src/config.py', 'r') as f: t = f.read()
    t = re.sub(r"'output_dir':\s*'[^']*'", f"'output_dir': './model/phase7_s{seed}'", t)
    with open('code/src/config.py', 'w') as f: f.write(t)

    r = subprocess.run([sys.executable, 'code/src/predict.py'], capture_output=True, text=True)
    print(f"\n=== Seed {seed} ===")
    for line in r.stdout.split('\n'):
        if 'stock_id' in line or any(c.isdigit() for c in line[:6] if len(line.strip()) > 5):
            print(line.strip())

    r = subprocess.run([sys.executable, 'test/score_self.py'], capture_output=True, text=True)
    for line in r.stdout.split('\n'):
        if '加权收益率' in line: print(line)
