import sys,os,json,numpy as np,torch,pandas as pd
sys.path.insert(0,'code/src')
from config import config, get_eff_input_dim
from model import StockTransformer
from predict import preprocess_predict_data, build_inference_sequences

df=pd.read_csv('data/stock_data.csv',dtype={'股票代码':str})
df['股票代码']=df['股票代码'].astype(str).str.zfill(6)
ids=sorted(df['股票代码'].unique()); s2i={s:i for i,s in enumerate(ids)}
p,fcols=preprocess_predict_data(df,s2i)
latest=p['日期'].max(); stocks=sorted(p['股票代码'].unique())
seqs,seq_stocks=build_inference_sequences(p,fcols,config['sequence_length'],stocks,latest)
from utils import per_stock_sliding_zscore
for i in range(len(seqs)): seqs[i]=per_stock_sliding_zscore(seqs[i].copy(),60,clip_range=5.0)
st=torch.from_numpy(seqs).float().unsqueeze(0)
si=torch.tensor([[s2i.get(s,0) for s in seq_stocks]])
md='model/phase12_test/model_0'
if os.path.exists(md):
    cfg_p=os.path.join(md,'config.json')
    with open(cfg_p) as f: sc=json.load(f)
    for k,v in sc.items():
        if k in config: config[k]=v
    config['use_cnn_features']=False
    eff=get_eff_input_dim(len(fcols))
    m=StockTransformer(input_dim=eff,config=config,num_stocks=len(ids))
    m.load_state_dict(torch.load(os.path.join(md,'best_model.pth'),map_location='cpu'))
    m.eval()
    with torch.no_grad(): scores=m(st,stock_indices=si).squeeze().numpy()
    r=np.argsort(scores)[::-1]
    print(f'Phase12 test: std={scores.std():.4f} top5={[seq_stocks[i] for i in r[:5]]}')
else:
    print('Phase12 model not found')
