import argparse
import importlib
import Semantic_Meaningfulness_v2
from Semantic_Meaningfulness_v2 import Sematic
importlib.reload(Semantic_Meaningfulness_v2)
import carla.recourse_methods.catalog as recourse_catalog
from carla.data.causal_model import CausalModel
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
from carla.data.catalog import CsvCatalog
#from carla.recourse_methods.catalog import recourse_catalog
#from carla.recourse_methods.catalog.roar.model import Roar
from carla.recourse_methods.catalog.causal_recourse import (
    CausalRecourse,
    constraints,
    samplers,
)
#from carla.data.causal_model.synthethic_data import SCMDataset
from carla.recourse_methods import GrowingSpheres
from carla import Benchmark
import numpy as np 
import pandas as pd
import torch
import random
import os 
import pickle
os.environ["CUDA_VISIBLE_DEVICES"]=""
import warnings
warnings.filterwarnings('ignore')


def wachter(ml_model,scm, name, data):
    '''
    Calls Wachter Recourse. 
    Attributes: 
        ml_model caral.XXX : Classifier
        name str: name of Model

    Return: 
        carla.recourse...
    '''
    hyperparams = {"loss_type": "BCE"}
    return recourse_catalog.wachter.model.Wachter(ml_model, hyperparams)


def causal_recourse(ml_model,scm,name, data):
    '''
    Calls causal recourse. 
    Attributes: 
        ml_model caral.XXX : Classifier
        name str: name of Model

    Return: 
        carla.recourse...
    '''
    hyperparams = {
    "optimization_approach": "brute_force",
    "num_samples": 10, #used to be 10
    "scm": scm,
    "constraint_handle": constraints.point_constraint,
    "sampler_handle": samplers.sample_true_m0,
    }
    causal_recourse = CausalRecourse(ml_model, hyperparams)
    return causal_recourse

def growingspheres(model,scm,name, data):
    '''
    Calls growingspheres. 
    Attributes: 
        ml_model caral.XXX : Classifier
        name str: name of Model

    Return: 
        carla.recourse...
    '''
    return GrowingSpheres(model)

def focus(model,scm,name, data):
    '''
    Calls focous, only works with XGBoost Backend. 
    Attributes: 
        ml_model caral.XXX : Classifier
        name str: name of Model

    Return: 
        carla.recourse...
    '''
    hyperparams = {
    "optimizer": "adam",
    "lr": 0.001,
    "n_class": 2,
    "n_iter": 1000,
    "sigma": 1.0,
    "temperature": 1.0,
    "distance_weight": 0.01,
    "distance_func": "l1",
    }

    return recourse_catalog.FOCUS(model, hyperparams)

def cchvae(mlmodel,scm, name, data):
    '''
    #TODO This has to be tested
    '''
    hyperparams = {
    "data_name": name,
    "n_search_samples": 100,
    "p_norm": 1,
    "step": 0.1,
    "max_iter": 1000,
    "clamp": True,
    "binary_cat_features": True,
    "vae_params": {
        "layers": [len(ml_model.feature_input_order), 512, 256, 8],
        "train": True,
        "lambda_reg": 1e-6,
        "epochs": 5,
        "lr": 1e-3,
        "batch_size": 32,
        },
        }

    cchvae = recourse_catalog.CCHVAE(mlmodel, hyperparams)
    return cchvae
# Actionable Recourse (AR)
def actionable_recourse(mlmodel,scm, name, data):
    '''
    AR does not always find a counterfactual example. The probability of finding one rises for a high size of flip set.
    https://carla-counterfactual-and-recourse-library.readthedocs.io/en/latest/recourse.html#module-recourse_methods.catalog.actionable_recourse.model
    '''
    return recourse_catalog.ActionableRecourse(mlmodel,hyperparams=None)
    
# Counterfactual Latent Uncertainty Explanations (CLUE)
def Clue(mlmodel, scm, name, data):
    return recourse_catalog.Clue(data, mlmodel, hyperparams=None)

def Dice(mlmodel, scm, name, data):
    return recourse_catalog.Dice(ml_model, hyperparams=None)

def FeatureTweak(mlmodel, scm, name, data):
    return recourse_catalog.FeatureTweak(mlmodel)

def Face(mlmodel, scm, name, data):
    '''
    Currently Ignored, needs Immutable Features to be set !
    '''
    return recourse_catalog.Face(mlmodel, hyperparams={"mode":"knn","fraction":0.5})

def Cruds(mlmodel, scm, name, data):
    #TODO PARAms
    return recourse_catalog.crud.model.CRUD(mlmodel, hyperparams={"data_name":name,"vae_params": {
            "layers": [data.df.shape[-1]-1,64,2],
            "train": True,
            "epochs": 5,
            "lr": 1e-3,
            "batch_size": 32,
        },})


#def Roar(mlmodel, scm, name, data):
#TODO not found in library
#    return Roar(mlmodel)

#def Revise(mlmodel, scm, name, data):
     #TODO WHere ? --Throws errie
#    return recourse_catalog.Revise(mlmodel, data)




  


def linear(dataset, name):
    '''
    Linear Model.
    Attributes: 
        dataset carla.XXX : data to train on 
        name str: dataset name 
    Returns: 
        carla.XXX
    '''
    training_params = {"lr": 0.01, "epochs": 10, "batch_size": 16, "hidden_size": [18, 9, 3]}
    ml_model = MLModelCatalog(
    dataset, model_type="linear", load_online=False, backend="pytorch"
    )
    ml_model.train(
        learning_rate=training_params["lr"],
        epochs=training_params["epochs"],
        batch_size=training_params["batch_size"],
        hidden_size=training_params["hidden_size"],
        force_train=True
    )

    return ml_model

def forest(dataset, name):
    '''
    TODO Test
    '''
    ml_model = MLModelCatalog(dataset, "forest", backend="sklearn", load_online=False)
    ml_model.train(max_depth=2, n_estimators=5, force_train=True)
    return ml_model


def MLP(dataset, name):
    '''
    Load and return MLP. 
    Attributes: 
        dataset carla.XXX : data to train on 
        name str: dataset name 
    Returns: 
        carla.XXX
    '''
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
    return ml_model

def data(name, not_causal=True):
    '''
    Load and return Toy Dataset.
    Attribute: 
        name str: Name of the Dataset. 

    Returns: 
        (dataset, scm, scm_output): returns Dataset, Structural Causal Model and Structural Causal Model with output Layer.
    '''
    scm = CausalModel(f"{name}")
    scm_output = CausalModel(f"{name}-output")
    if not os.path.isdir(f'./data/{name}'):
        os.mkdir(f'./data/{name}')
    if not os.path.isfile(f'./data/{name}/{name}.csv'):
        print('TRUE 1 ')
        # generate data
        dataset = scm.generate_dataset(10000, False)
        print(f'./data/{name}/{name}.csv')
        dataset.df.to_csv(f'./data/{name}/{name}.csv', index=False)
        #pickle.dump(dataset.train_raw, open(f'./data/{name}/{name}_train_raw.pkl','wb'))
        #pickle.dump(dataset.test_raw, open(f'./data/{name}/{name}_test_raw.pkl','wb'))
        #pickle.dump(dataset.raw, open(f'./data/{name}/{name}_raw.pkl','wb'))
        #pickle.dump(dataset.noise, open(f'./data/{name}/{name}_noise.pkl','wb'))
        if not_causal:
            dataset = pd.read_csv(f'./data/{name}/{name}.csv')
            #TODO Better way for defining continous varaibles ?
            continuous_wachter = dataset.drop(columns=['label']).columns
            dataset = CsvCatalog(file_path=f'./data/{name}/{name}.csv',
                     continuous=continuous_wachter,
                     categorical=[],
                     immutables=[],
                     target='label',
                     scaling_method='Identity')
    else: 
        if not_causal:
            dataset = pd.read_csv(f'./data/{name}/{name}.csv')
            #TODO Better way for defining continous varaibles ?
            continuous_wachter = dataset.drop(columns=['label']).columns
            dataset = CsvCatalog(file_path=f'./data/{name}/{name}.csv',
                     continuous=continuous_wachter,
                     categorical=[],
                     immutables=[],
                     target='label',
                     scaling_method='Identity')
        else: 

            dataset = pd.read_csv(f'./data/{name}/{name}.csv')
            continuous_wachter = dataset.drop(columns=['label']).columns
            #TODO Does Scaling Method Idendity make sense ? 
            dataset = CsvCatalog(file_path=f'./data/{name}/{name}.csv',
                     continuous=continuous_wachter,
                     categorical=[],
                     immutables=[],
                     target='label',
                     scaling_method='Identity')
    
    return dataset, scm , scm_output

if __name__ =='__main__':
    #import 1_Experiment
    #SEED Setting
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    parser = argparse.ArgumentParser(
                    prog = 'ProgramName',
                    description = 'What the program does',
                    epilog = 'Text at the bottom of help')
    
    parser.add_argument('data') 
    parser.add_argument('model')
    parser.add_argument('CF')
    parser.add_argument('n_eval', type=int,  default=1)
    parser.add_argument('semantic_measure', default='both')
    #parser.add_argument('save_path', default='./Results/') 
    args = parser.parse_args() 
    #print(locals()["MLP"]())

    print(f'Parameters : {args.data}, {args.model}, {args.CF}. {args.n_eval}, {args.semantic_measure}')   
    not_causal=True
    if 'causal' in f'{args.CF}':
        not_causal=False
    #Load Dataset    
    dataset, scm, scm_output=data(args.data, not_causal)
    ml_model= locals()[f"{args.model}"](dataset,args.data)

    # get factuals
    factuals = predict_negative_instances(ml_model, dataset.df)
    test_factual_with_labels = factuals.iloc[:args.n_eval].reset_index(drop=True)
    test_factual=test_factual_with_labels.copy()

    #Recourse Method
    recourse= locals()[f"{args.CF}"](ml_model,scm,args.data,dataset)
    
    # Benchmarking
    benchmark_wachter = Benchmark(ml_model, recourse, test_factual)
    evaluation_measures = [
    Sematic(ml_model, causal_graph_full=scm_output,causal_graph_small=scm),    
    ]

    results = benchmark_wachter.run_benchmark(evaluation_measures)
    if not os.path.isdir(f'./Results/{args.data}'):
        os.mkdir(f'./Results/{args.data}')
    results.to_csv(f'./Results/{args.data}/Results_{args.model}_{args.CF}.csv')

    summary=pd.DataFrame([])
    summary['model']=[f'{args.model}']
    summary['dataset']=[f'{args.data}']
    summary['CF']=[f'{args.CF}']
    summary['semantic_mean']=np.mean(results['semantic'])
    summary['semantic_std']=np.std(results['semantic'])
    summary['relationship_mean']=np.mean(results['correct_relationships'])
    summary['relationship_std']=np.std(results['correct_relationships'])
    summary.to_csv(f'./Results/{args.data}/summary_{args.model}_{args.CF}.csv')

    #TODO save THIS
    #mean= np.mean(results)
    #std= np.std(results)

    #print(f'Semantic results {mean} +/- {std}')
    #print(f'Parameters FINAL : {args.data}, {args.model}, {args.CF}. {args.n_eval}, {args.semantic_measure}')   



