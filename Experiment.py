from carla import Benchmark
import carla.evaluation.catalog as evaluation_catalog
from carla.data.catalog import OnlineCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
import carla.recourse_methods.catalog as recourse_catalog
import shap 
from sklearn.preprocessing import OrdinalEncoder
from carla.data.causal_model import CausalModel
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import numpy as np 
from carla.data.catalog.online_catalog import OnlineCatalog
import pickle
import warnings
warnings.filterwarnings("ignore")

'''Basic Part'''
dataset = OnlineCatalog('adult')


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
import carla.recourse_methods.catalog as recourse_catalog

'''Models'''

factuals = predict_negative_instances(ml_model, dataset.df)
test_factual = factuals.iloc[:100]
pickle.dump(test_factual,open('./Resuls_Wachter_factuals.pkl','wb'))


hyperparams = {"loss_type": "BCE", "binary_cat_features": True}

recourse_method = recourse_catalog.Wachter(ml_model, hyperparams)
df_cfs = recourse_method.get_counterfactuals(test_factual)
pickle.dump(df_cfs,open('./Resuls_Wachter_CF.pkl','wb'))

'''Benchmarking'''
#first initialize the benchmarking class by passing
# black-box-model, recourse method, and factuals into it
benchmark = Benchmark(ml_model, recourse_method, factuals)

# now you can decide if you want to run all measurements
# or just specific ones.
evaluation_measures = [
    evaluation_catalog.YNN(benchmark.mlmodel, {"y": 5, "cf_label": 1}),
    evaluation_catalog.Distance(benchmark.mlmodel),
    evaluation_catalog.SuccessRate(),
    evaluation_catalog.Redundancy(benchmark.mlmodel, {"cf_label": 1}),
    evaluation_catalog.ConstraintViolation(benchmark.mlmodel),
    evaluation_catalog.AvgTime({"time": benchmark.timer}),
]

# now run all implemented measurements and create a
# DataFrame which consists of all results
results = benchmark.run_benchmark(evaluation_measures)

