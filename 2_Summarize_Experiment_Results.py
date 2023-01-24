import os 
import pandas as pd


i=0
for dataset in os.listdir('./Results'):
    if os.path.isdir(f'./Results/{dataset}'):
        for file in os.listdir(f'./Results/{dataset}'):
            if file.startswith('sum'):
                if i == 0: 
                    df=pd.read_csv(f'./Results/{dataset}/{file}')
                    i=i+1
                else: 
                    df= pd.concat([df,pd.read_csv(f'./Results/{dataset}/{file}') ])

df.to_csv('./Results/summary.csv')
