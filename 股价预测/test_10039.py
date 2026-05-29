import sys, os
sys.path.insert(0, 'code/src')
from utils import engineer_features_100plus39
import pandas as pd

df = pd.DataFrame({
    '股票代码': ['000001']*100,
    '日期': pd.date_range('2024-01-01', periods=100),
    '开盘': [10.0+i*0.1 for i in range(100)],
    '收盘': [10.5+i*0.1 for i in range(100)],
    '最高': [11.0+i*0.1 for i in range(100)],
    '最低': [9.5+i*0.1 for i in range(100)],
    '成交量': [1e6]*100,
    '成交额': [1e7]*100,
    '振幅': [0.1]*100,
    '涨跌额': [0.5]*100,
    '换手率': [0.01]*100,
    '涨跌幅': [0.05]*100,
})
result = engineer_features_100plus39(df)
print(f'Features: {len(df.columns)} → {len(result.columns)}')
