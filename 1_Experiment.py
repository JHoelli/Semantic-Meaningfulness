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
from carla.evaluation.catalog.success_rate import SuccessRate
#from carla.recourse_methods.catalog import recourse_catalog
#from carla.recourse_methods.catalog.roar.model import Roar
from carla.recourse_methods.catalog.causal_recourse import (
    CausalRecourse,
    constraints,
    samplers,
)
#from carla.data.causal_model.synthethic_data import SCMDataset
from carla.recourse_methods import GrowingSpheres
from causal_recourse_do_calculus.model import CausalRecourse_DoCalculus
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

    hyperparams = {"loss_type": "BCE",
    "lr":0.01,
    "norm":1,
    "lambda_param":0.01
     }
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
    HAS NO HYPERPARAMS
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

def do_calculus(mlmodel,scm, name, data):
    import causal_recourse_do_calculus.constraints_do_calculus as constraints_do
    import causal_recourse_do_calculus.samplers as samplers
    from causal_recourse_do_calculus.action_set import get_discretized_action_sets
    from causal_recourse_do_calculus.cost import action_set_cost
    hyperparams= {
        "optimization_approach": "brute_force",
        "num_samples": 10,
        "scm": scm,
        "constraint_handle_do": constraints_do.point_constraint,
        "sampler_handle": samplers.sample_true_m0,
    }

    return CausalRecourse_DoCalculus(mlmodel,hyperparams)

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
    ''' We use the default hyperparameters from [ 3], which are set as a function of the data set
    dimension d. Performing hyperparameter search did not yield results that were improving distances
    while keeping the same success rate'''
    return recourse_catalog.Clue(data, mlmodel, hyperparams=None)

def Dice(ml_model, scm, name, data):
    '''
    Since DICE is able to compute a set of counterfactuals for a given instance, we only chose
to generate one CE per input instance. We use a grid search for the proximity and diversity
weights. -> #TODO Better way
    '''
    return recourse_catalog.Dice(ml_model, hyperparams=None)

def FeatureTweak(mlmodel, scm, name, data):
    '''
    eps=0.1
    '''
    return recourse_catalog.FeatureTweak(mlmodel)



def Cruds(mlmodel, scm, name, data):
    '''
    Parameters from Cruds Paper
    '''
    return recourse_catalog.crud.model.CRUD(mlmodel, hyperparams={"data_name":name, "optimizer": "ADAM","vae_params": {
            "layers": [data.df.shape[-1]-1,64,1],
            "train": True,
            "epochs": 500,
            "lr": 1e-3,
            "batch_size": 32,
        },})

#def Face(mlmodel, scm, name, data):
#    '''
#    Currently Ignored, needs Immutable Features to be set !
#    '''
#    return recourse_catalog.Face(mlmodel, hyperparams={"mode":"knn","fraction":0.5})
#def Roar(mlmodel, scm, name, data):
#TODO not found in library
#    return Roar(mlmodel)

#def Revise(mlmodel, scm, name, data):
     #TODO WHere ? --Throws errie
#    return recourse_catalog.Revise(mlmodel, data)




  


def linear(dataset, name,hyperparams,i):
    '''
    Linear Model.
    Attributes: 
        dataset carla.XXX : data to train on 
        name str: dataset name 
    Returns: 
        carla.XXX
    '''
    training_params = hyperparams#{"lr": 0.01, "epochs": 10, "batch_size": 16, "hidden_size": [18, 9, 3]}
    ml_model = MLModelCatalog(
    dataset, model_type="linear", load_online=False, backend="pytorch"
    )
    if os.path.isfile(f'./Results/Model/Linear_{name}.pth'):
        model=torch.load(f'./Results/Model/Linear_{name}.pth')
        ml_model._model=model
    else:
        ml_model.train(
        learning_rate=training_params["lr"],
        epochs=training_params["epochs"],
        batch_size=training_params["batch_size"],
        hidden_size=training_params["hidden_size"],
        force_train=True
        )
        torch.save(ml_model.raw_model,f'./Results/Model/Linear_{name}.pth')

    return ml_model

def forest(dataset, name,hyperparams,i):
    '''
    TODO Test
    '''
    ml_model = MLModelCatalog(dataset, "forest", backend="sklearn", load_online=False)
    if os.path.isfile(f'./Results/Model/Forest_{name}.pth'):
        model=torch.load(f'./Results/Model/Forest_{name}.pth')
        ml_model._model=model
    else:
        ml_model.train(max_depth=2, n_estimators=5, force_train=True)
        torch.save(ml_model.raw_model,f'./Results/Model/Forest_{name}.pth')
    return ml_model


def MLP(dataset, name,hyperparams,i):
    '''
    Load and return MLP. 
    Attributes: 
        dataset carla.XXX : data to train on 
        name str: dataset name 
    Returns: 
        carla.XXX
    '''
    
    training_params =hyperparams #{"lr": 0.01, "epochs": 10, "batch_size": 16, "hidden_size": [18, 9, 3]}
    #if name=='economic':
    #     training_params = {"lr": 0.002, "epochs": 10, "batch_size": 1024, "hidden_size": [18, 9, 3],' num_of_classes':2}
    print

    ml_model = MLModelCatalog(
    dataset, model_type="ann", load_online=False, backend="pytorch"
    )
    if os.path.isfile(f'./Results/Model/MLP_{name}{i}.pth'):
        model=torch.load(f'./Results/Model/MLP_{name}{i}.pth')
        ml_model._model=model
    else:
        ml_model.train(
        learning_rate=training_params["lr"],
        epochs=training_params["epochs"],
        batch_size=training_params["batch_size"],
        hidden_size=training_params["hidden_size"],
        force_train=True
        )

        torch.save(ml_model.raw_model,f'./Results/Model/MLP_{name}{i}.pth')
    return ml_model

def data(name, not_causal=True, scaler='Identity'):
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
                     scaling_method=scaler)
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
                     scaling_method=scaler)
        else: 

            dataset = pd.read_csv(f'./data/{name}/{name}.csv')
            continuous_wachter = dataset.drop(columns=['label']).columns
            #TODO Does Scaling Method Idendity make sense ? 
            dataset = CsvCatalog(file_path=f'./data/{name}/{name}.csv',
                     continuous=continuous_wachter,
                     categorical=[],
                     immutables=[],
                     target='label',
                     scaling_method=scaler)
    
    return dataset, scm , scm_output

if __name__ =='__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    i=None

    hyperparams={
        #0: {"lr": 0.1, "epochs": 10, "batch_size": 16, "hidden_size": [18, 9, 3]},
        #1: {"lr": 0.01, "epochs": 10, "batch_size": 16, "hidden_size": [18, 9, 3]},
        0: {"lr": 0.001, "epochs": 10, "batch_size": 16, "hidden_size": [18, 9, 3],' num_of_classes':2}


    }
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
    scaler='Identity'
    if 'causal' in f'{args.CF}':
        not_causal=False
    if 'economic' in f'{args.data}':
        scaler='MinMax'
    #Load Dataset    
    dataset, scm, scm_output=data(args.data, not_causal,scaler)
    if args.model!='MLP':
        print('NoHyper')
        hyperparams={0:''}
    for hyper in hyperparams.keys():
        ml_model= locals()[f"{args.model}"](dataset,args.data, hyperparams[hyper],hyper)
        i=hyper
        # get factuals
        #factuals = predict_negative_instances(ml_model, dataset.df)
        test_factual_with_labels = dataset.df.iloc[:args.n_eval].reset_index(drop=True)
        test_factual=test_factual_with_labels.copy()

        #Recourse Method
        recourse= locals()[f"{args.CF}"](ml_model,scm,args.data,dataset)
    
        # Benchmarking
        benchmark_wachter = Benchmark(ml_model, recourse, test_factual)
        evaluation_measures = [
        Sematic(ml_model, causal_graph_full=scm_output,causal_graph_small=scm),    
        SuccessRate()

        ]

        results = benchmark_wachter.run_benchmark(evaluation_measures)
        if not os.path.isdir(f'./Results/{args.data}'):
            os.mkdir(f'./Results/{args.data}')
        results['model']=np.repeat(f'{args.model}{i}', len(results.index))
        results['CF']=np.repeat(args.CF, len(results.index))
        results['dataset']=np.repeat(args.data, len(results.index))
        results.to_csv(f'./Results/{args.data}/Results_{args.model}{hyper}_{args.CF}.csv')

        summary=pd.DataFrame([])
        summary['model']=[f'{args.model}{i}']
        summary['dataset']=[f'{args.data}']
        summary['CF']=[f'{args.CF}']
        summary['semantic_mean']=np.mean(results['semantic'])
        summary['semantic_std']=np.std(results['semantic'])
        summary['relationship_mean']=np.mean(results['correct_relationships'])
        summary['relationship_std']=np.std(results['correct_relationships'])
        summary.to_csv(f'./Results/{args.data}/summary_{args.model}{hyper}_{args.CF}.csv')




