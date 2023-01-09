# set to CPU 
import os

os.environ["CUDA_VISIBLE_DEVICES"]=""

import pandas as pd
import warnings
import Semantic_Maningfullness
warnings.filterwarnings('ignore')
from carla.data.causal_model import CausalModel
import numpy as np 
import torch
import random
import carla.evaluation.catalog as evaluation_catalog
from carla.data.catalog import CsvCatalog
from carla import Benchmark
from Semantic_Maningfullness import Sematic

#SEED Setting
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

import pandas as pd

# generate data
scm = CausalModel("credit")
dataset = scm.generate_dataset(10000, False)

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
# get factuals
factuals = predict_negative_instances(ml_model, dataset.df)
test_factual_with_labels = factuals.iloc[:10].reset_index(drop=True)
test_factual=test_factual_with_labels.copy()

print(test_factual)


from carla.recourse_methods.catalog.causal_recourse import (
    CausalRecourse,
    constraints,
    samplers,
)
hyperparams = {
    "optimization_approach": "brute_force",
    "num_samples": 10,
    "scm": scm,
    "constraint_handle": constraints.point_constraint,
    "sampler_handle": samplers.sample_true_m0,
}


## structural counterfactual (SCF)
causal_recourse = CausalRecourse(ml_model, hyperparams)
#output = cfs.reset_index(drop=True) - test_factual.loc[:,~test_factual.columns.isin(['label'])].reset_index(drop=True)


benchmark_wachter = Benchmark(ml_model, causal_recourse, test_factual)


scm_output=CausalModel("credit_output")

mapping_dict={ 
      'u1': 'x1',
      'u2': 'x2',
    'u3': 'x3',
      }

# now you can decide if you want to run all measurements
# or just specific ones.
evaluation_measures = [
    #evaluation_catalog.YNN(benchmark.mlmodel, {"y": 5, "cf_label": 1}),
    #evaluation_catalog.Distance(benchmark.mlmodel),
    #evaluation_catalog.SuccessRate(),
    #evaluation_catalog.Redundancy(benchmark.mlmodel, {"cf_label": 1}),
    #evaluation_catalog.ConstraintViolation(benchmark_wachter.mlmodel),
    #evaluation_catalog.AvgTime({"time": benchmark_wachter.timer}),
    Sematic(ml_model,scm_output,mapping_dict),    
]

# now run all implemented measurements and create a
# DataFrame which consists of all results
results = benchmark_wachter.run_benchmark(evaluation_measures)
