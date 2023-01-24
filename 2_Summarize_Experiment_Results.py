import os 
import pandas as pd


i=0
j=0
for dataset in os.listdir('./Results'):
    if os.path.isdir(f'./Results/{dataset}'):
        for file in os.listdir(f'./Results/{dataset}'):
            if file.startswith('sum'):
                if i == 0: 
                    df=pd.read_csv(f'./Results/{dataset}/{file}')
                    i=i+1
                else: 
                    df= pd.concat([df,pd.read_csv(f'./Results/{dataset}/{file}') ])
            elif file.startswith('Res'):
                if j == 0: 
                    df_res=pd.read_csv(f'./Results/{dataset}/{file}')
                    j=j+1
                else: 
                    df_res= pd.concat([df_res,pd.read_csv(f'./Results/{dataset}/{file}') ])

df.to_csv('./Results/summary.csv')
df_res.to_csv('./Results/results.csv')
