import os 
import pandas as pd

import numpy as np
i=0
j=0
length_complete =0
valid=0
smo=0
smr=0
smoandsmr =0
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
                    smr+=len(np.where(df_res['correct_relationships']==1.0))

                    j=j+1
                else: 
                    df_res= pd.concat([df_res,pd.read_csv(f'./Results/{dataset}/{file}') ])
                    length_complete+=250
valid+=len(df_res)
smo+=len(df_res[df_res['semantic']==1 ])
smr+=len(df_res[df_res['correct_relationships']==1.0 ])
one=df_res[df_res['semantic']==1 ]
smoandsmr += len(one[one['correct_relationships']==1.0 ])


#smr+=len(np.where(df_res['correct_relationships']==1 ))
#smo+=len(np.where(df_res['semantic']==1.0 ))

df.to_csv('./Results/summary.csv')
df_res.to_csv('./Results/results.csv')
pd.DataFrame([[valid, length_complete, smo, smr,smoandsmr]], columns=['valid', 'length_complete', 'smo', 'smr','smosmr']).to_csv('./Results/Venn.csv')