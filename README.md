# Semantic Meaningfulness: Evaluating counterfactual approaches for real world plausibility

This repository contains the code for our paper "Semantic Meaningfulness: Evaluating counterfactual approaches for real world plausibility". We benchmark the semantic meaningfulness of known Counterfactual Approaches. 
The implementation in here allows to replicate the experiments, and the benchmarking of (new) counterfactual approaches. For easy usage of out metric, refer to Benchmarking_Output_Causal.ipynb.

This repository consits of two branches: 
- main (the current branch): with a minimal pip-installable version of the metric 
- experiment: the original code from the experiments in the paper

For a step to step guide on how to run the eperiments, we refer the reader to ./experiments/Readme.md.

## Install:
Clone this repository and install via pip. 

```shell
pip install . 
```

## Usage:
For the usage refer to Benchmarking_Output_Causal.ipynb.

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
# Citation
