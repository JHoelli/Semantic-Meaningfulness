from traceback import print_tb
from IPython.display import display
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


import warnings
warnings.filterwarnings('ignore')

from carla.data.causal_model import CausalModel
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from carla.models.catalog import MLModelCatalog 
# Load Data 


from carla.data.catalog import CsvCatalog
X_display,y_display = shap.datasets.nhanesi(display=False)
print(y_display)
dataframe=X_display
dataframe['target']=y_display
dataframe.to_csv('nutrition.csv')
continuous = dataframe.columns
#TODO
#colum_Mapping={
#'sex_isFemale':'x1',
#'age':'x2',
#'blodd_pr':'x3', 
#'sbp':'x4',
#'pulse_pressure':'x5',
#'inflamation','x6',
#'povery','x7',
#'inflamation': 'x8',



#}

dataset = CsvCatalog(file_path="nutrition.csv",
                     continuous=continuous,
                     categorical=[],
                     immutables=[],
                     target='target')

scm = CausalModel("nutrition")
#dataset = scm.generate_dataset(100)
#from carla.models.catalog import MLModelCatalog
#from carla.data.catalog.online_catalog import CSVCatalog
#categorial = ['Country','Workclass','Marital Status', 'Occupation','Relationship', 'Race', 'Sex']

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
from carla.models.negative_instances import predict_negative_instances
from carla.recourse_methods.catalog.causal_recourse import (
    CausalRecourse,
    constraints,
    samplers,
)

# get factuals
factuals = predict_negative_instances(ml_model, dataset.df)
test_factual = factuals.iloc[:1]

hyperparams = {
    "optimization_approach": "brute_force",
    "num_samples": 2,
    "scm": scm,
    "constraint_handle": constraints.point_constraint,
    "sampler_handle": samplers.sample_true_m0,
}
cfs = CausalRecourse(ml_model, hyperparams).get_counterfactuals(test_factual)

display(cfs)

print(cfs)