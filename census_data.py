from traceback import print_tb
from IPython.display import display

import warnings
warnings.filterwarnings('ignore')

from carla.data.causal_model import CausalModel
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# Load Data 
X,y = shap.datasets.adult()
X_display,y_display = shap.datasets.adult(display=True) # human readable feature values

#TODO this needs to be chnaged --> Ordinal Encoding !
categorial = []
for a in categorial:
    enc= OrdinalEncoder()
    X_display[a]= enc.fit_transform(X_display[a].reshape(1,-1))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

scm = CausalModel("census")
dataset = scm.generate_dataset(10000)
from carla.models.catalog import MLModelCatalog


training_params = {"lr": 0.01, "epochs": 10, "batch_size": 16, "hidden_size": [18, 9, 3]}

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
test_factual = factuals.iloc[:5]

hyperparams = {
    "optimization_approach": "brute_force",
    "num_samples": 10,
    "scm": scm,
    "constraint_handle": constraints.point_constraint,
    "sampler_handle": samplers.sample_true_m0,
}
cfs = CausalRecourse(ml_model, hyperparams).get_counterfactuals(test_factual)

display(cfs)

print(cfs)