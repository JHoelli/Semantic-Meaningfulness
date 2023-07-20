# Semantic Meaningfulness: Evaluating counterfactual approaches for real world plausibility

This repository contains the code for our paper "Semantic Meaningfulness: Evaluating counterfactual approaches for real world plausibility". We benchmark the semantic meaningfulness of known Counterfactual Approaches. 

This repository consits of two branches: 
- main (the current branch): with a minimal pip-installable version of the metric 
- experiment: the original code from the experiments in the paper

For a step to step guide on how to run the eperiments, we refer the reader to ./experiments/Readme.md.

## Install:
Due to its dependency on [CARLA](https://github.com/carla-recourse/CARLA) our code works best with python 3.7.
Clone this repository and install via pip. 

```shell
pip install . 
```

## Usage with the CARLA - Counterfactual And Recourse Library :
For details on the usage refer to [1_Benchmarking_CARLA.ipynb](1_Benchmarking_CARLA.ipynb).

### Import the Packages and Place a reference to the models provided by out paper
```python
import Semantic_Meaningfulness 
import carla
carla.data.causal_model=Semantic_Meaningfulness.carla_adaptions.causal_model
```

### Get Data 
```python
# generate data
scm = carla.data.causal_model.CausalModel("sanity-3-lin")
dataset = scm.generate_dataset(10000, False)

```

### Train a Model 
```python
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


```

### Define Recourse: 


'''Causal Recourse Model '''
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

recourse=CausalRecourse(ml_model, hyperparams)




### Get Full SCM 
```python
scm_output=carla.data.causal_model.CausalModel("sanity-3-lin-output")
```

### Run Benchmark 
```python 
from Semantic_Meaningfulness.Semantic_Meaningfulness import Semantic
benchmark = Benchmark(ml_model, recourse, test_factual)


evaluation_measures = [
    Semantic(ml_model, causal_graph_full=scm_output,causal_graph_small=scm),    
]


results = benchmark.run_benchmark(evaluation_measures)

```
## Stand Alone Usage

Find Details in [2_Standalone.ipynb](2_Standalone.ipynb).

```python 

import Semantic_Meaningfulness 
import carla
carla.data.causal_model=Semantic_Meaningfulness.carla_adaptions.causal_model

causal_graph_small = carla.data.causal_model.CausalModel("sanity-3-lin")
causal_graph_full = carla.data.causal_model.CausalModel("sanity-3-lin-output")
metric=Semantic(ml_model, causal_graph_full, causal_graph_small)
metric.evaluate(factuals, counterfactuals)

```
# Citation
If you use this work please consider citing it : 
```
@inproceedings{H{\"o}llig2023Sem, 
organization={Springer}
year = {2023}, 
booktitle={1st International Conference on eXplainable Artificial Intelligence (xAI 2023)},
author = {Jacqueline H{\"o}llig and Aniek F. Markus and Jef de Slegte and Prachi Bagave}, 
title = {Semantic Meaningfulness: Evaluating Counterfactual Approaches for Real World Plausibility and Feasibility} 
} 

```
If you use it in combination with CARLA or a specific Counterfactual Method, please consider citing those works too. 
