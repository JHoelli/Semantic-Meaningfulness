from traceback import print_tb
from IPython.display import display
import os
import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


import warnings
warnings.filterwarnings('ignore')

from carla.data.causal_model import CausalModel
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import numpy as np 
# Load Data 

from carla.data.catalog import CsvCatalog
X_display,y_display = shap.datasets.adult(display=True) # human readable feature values
print(X_display.columns)
#X_display=X_display.fillna(0)
categorial = ['Workclass', 'Education-Num', 'Marital Status', 'Occupation',
       'Relationship', 'Race', 'Sex', 'Country']#['Country','Workclass','Marital Status', 'Occupation','Relationship', 'Race', 'Sex']
for a in categorial:
    enc= OrdinalEncoder()
    enc.fit(np.array(X_display[a].values).reshape(-1, 1))
    temp= enc.transform(np.array(X_display[a].values).reshape(-1, 1))
   
    X_display[a]= temp.reshape(-1)

dataframe=X_display
dataframe['target']=y_display
dataframe.to_csv('census.csv')
print(dataframe['target'])
continuous = dataframe.drop(columns=['target']).columns

dataset = CsvCatalog(file_path="census.csv",
                     continuous=continuous,
                     categorical=[],
                     immutables=[],
                     encoding_method=None,
                     target='target')

scm = CausalModel("adult")
#dataset = scm.generate_dataset(100)
from carla.models.catalog import MLModelCatalog


training_params = {"lr": 0.01, "epochs": 10, "batch_size": 16, "hidden_size": [18, 9, 2]}

ml_model = MLModelCatalog(
    dataset, model_type="ann", load_online=False, backend="pytorch"
)
ml_model.train(
    learning_rate=training_params["lr"],
    epochs=training_params["epochs"],
    batch_size=training_params["batch_size"],
    hidden_size=training_params["hidden_size"],
    force_train=True
)
pickle.dump(ml_model,open('./models/census.pkl','wb'))
#from carla.models.negative_instances import predict_negative_instances
#from carla.recourse_methods.catalog.causal_recourse import (
#    CausalRecourse,
#    constraints,
#    samplers,
#)

# get factuals
#factuals = predict_negative_instances(ml_model, dataset.df)
#test_factual = factuals.iloc[:1]

#hyperparams = {
#    "optimization_approach": "brute_force",
#    "num_samples": 2,
#    "scm": scm,
#    "constraint_handle": constraints.point_constraint,
#    "sampler_handle": samplers.sample_true_m0,
#}
#cfs = CausalRecourse(ml_model, hyperparams).get_counterfactuals(test_factual)

#display(cfs)

#print(cfs)