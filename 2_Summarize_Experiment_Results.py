import os 
import pandas as pd

import numpy as np
i=0
j=0
length_complete =0
valid=0
smo=0
smr=0
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
                print(file)
                if j == 0: 
                    df_res=pd.read_csv(f'./Results/{dataset}/{file}')
                    valid+=len(df_res)
                    length_complete+=250
                    smo+=len(np.where(df_res['semantic']==1))
                    print(df_res.columns)
                    smr+=len(np.where(df_res['correct_relationships']==1.0))

                    j=j+1
                else: 
                    df_res= pd.concat([df_res,pd.read_csv(f'./Results/{dataset}/{file}') ])
                    valid+=len(df_res)
                    length_complete+=250
                    smo+=len(np.where(df_res['semantic']==1))
                    print(df_res.columns)
                    smr+=len(np.where(df_res['correct_relationships']==1.0))

df.to_csv('./Results/summary.csv')
df_res.to_csv('./Results/results.csv')
pd.DataFrame([[valid, length_complete, smo, smr]], columns=['valid', 'length_complete', 'smo', 'smr']).to_csv('./Results/Venn.csv')