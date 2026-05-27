import sys
sys.path.insert(0, 'code/src')
from config import config
print(f"alpha={config['predict_rank_alpha']}")
print(f"weights={config['predict_rank_weights']}")
