import argparse
import importlib
import Semantic_Meaningfulness_v2
from Semantic_Meaningfulness_v2 import Sematic
importlib.reload(Semantic_Meaningfulness_v2)
from carla.data.causal_model import CausalModel
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
from carla.recourse_methods.catalog.causal_recourse import (
    CausalRecourse,
    constraints,
    samplers,
)
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


def wachter():
    pass

def causal_recourse(ml_model,name):
    hyperparams = {
    "optimization_approach": "brute_force",
    "num_samples": 10, #used to be 10
    "scm": scm,
    "constraint_handle": constraints.point_constraint,
    "sampler_handle": samplers.sample_true_m0,
    }
    causal_recourse = CausalRecourse(ml_model, hyperparams)
    return causal_recourse

def logistic_regression():
    pass

def MLP(dataset, name):
    '''
    Load and return MLP. 
    #TODO How ro save those ? 
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

def data(name):
    '''
    Load and return Toy Dataset.
    #TODO Train Test / Split
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
        dataset.df.to_csv(f'./data/{name}/{name}.csv')
        #pickle.dump(dataset, open(f'./data/{name}/{name}.pkl','wb'))
    else: 
        dataset = pd.read_csv(f'./data/{name}/{name}.csv')
        #dataset = pickle.load(open(f'./data/{name}/{name}.pkl','rb'))
    return dataset, scm , scm_output

if __name__ =='__main__':
    import Experiment
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

    #Load Dataset    
    dataset, scm, scm_output=data(args.data)
    ml_model= locals()[f"{args.model}"](dataset,args.data)

    # get factuals
    factuals = predict_negative_instances(ml_model, dataset.df)
    test_factual_with_labels = factuals.iloc[:args.n_eval].reset_index(drop=True)
    test_factual=test_factual_with_labels.copy()

    #Recourse Method
    recourse= locals()[f"{args.CF}"](ml_model,args.data)
    
    # Benchmarking
    benchmark_wachter = Benchmark(ml_model, recourse, test_factual)
    evaluation_measures = [
    Sematic(ml_model, causal_graph_full=scm_output,causal_graph_small=scm),    
    ]

    results = benchmark_wachter.run_benchmark(evaluation_measures)
    if not os.path.isdir(f'./Results/{args.data}'):
        os.mkdir(f'./Results/{args.data}')
    results.to_csv(f'./Results/{args.data}/Results_{args.data}.csv')

    summary=pd.DataFrame([])
    summary['model']=[f'{args.model}']
    summary['dataset']=[f'{args.data}']
    summary['CF']=[f'{args.CF}']
    summary['semantic_mean']=np.mean(results['semantic'])
    summary['semantic_std']=np.std(results['semantic'])
    summary['relationship_mean']=np.mean(results['correct_relationships'])
    summary['relationship_std']=np.std(results['correct_relationships'])
    summary.to_csv(f'./Results/{args.data}/summary_{args.data}_{args.model}_{args.CF}.csv')

    #TODO save THIS
    #mean= np.mean(results)
    #std= np.std(results)

    #print(f'Semantic results {mean} +/- {std}')
    #print(f'Parameters FINAL : {args.data}, {args.model}, {args.CF}. {args.n_eval}, {args.semantic_measure}')   



