import glob, pandas as pd
for f in sorted(glob.glob('temp/phase13_*run*.csv')):
    score = pd.read_csv(f)['Final Score'].iloc[0]
    print(f'{f}: {score}')
