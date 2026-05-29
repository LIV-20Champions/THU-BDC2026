import pandas as pd
for f in ['data/stock_data.csv', 'data/stock_data_orig_backup.csv']:
    df = pd.read_csv(f)
    dates = pd.to_datetime(df['日期'])
    print(f'{f}: {dates.min().date()} ~ {dates.max().date()}, rows={len(df)}')
