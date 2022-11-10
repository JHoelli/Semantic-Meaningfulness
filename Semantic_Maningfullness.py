from cmath import nan
from telnetlib import SE
from carla.data import causal_model
from carla.evaluation import remove_nans
from carla.evaluation.api import Evaluation
import numpy as np 
import pandas as pd 

def get_pred_from_causal(scm, values, cf_label, mapping_dict, threshold):
    '''
    Infers the prediction from the causal model. Sets Exogenous variable to counterfactual value.
    Attributes: 
        scm: structural causal model 
        values: The counterfactual 
        cf_label: The counterfactual label 
        mapping_dict: variable name mapping betwenn CF and causal model
    Returns Label
    '''
    values['target']=cf_label
    #print(values)
    def _get_noise_string(node):
        def _get_node_id(node):
            return node[1:]
        if not node[0] == "x":
            raise ValueError
        return "u" + _get_node_id(node)
    print('exogenous')
    for node in scm.get_topological_ordering("exogenous"):
        print(node)
    exogenous_variables = np.concatenate(
        [
            #np.array(scm.noise_distributions[node].sample(1)).reshape(-1, 1)
            np.array(values[mapping_dict[node]]).reshape(-1, 1)
            for node in scm.get_topological_ordering("exogenous")
        ],
        axis=1,
    )
    exogenous_variables = pd.DataFrame(
        exogenous_variables, columns=scm.get_topological_ordering("exogenous")
    )

    endogenous_variables =exogenous_variables.copy() # np.array(values[mapping_dict[node]]).reshape(-1, 1)
    endogenous_variables = endogenous_variables.rename(
        columns=dict(
            zip(
                scm.get_topological_ordering("exogenous"),
                scm.get_topological_ordering("endogenous"),
            )
        )
    )
    print('finish exogenous')
    # used later to make sure parents are populated when computing children
    endogenous_variables.loc[:] = np.nan
    print('endogenous')
    for node in scm.get_topological_ordering("endogenous"):
        # print(node)
        parents = scm.get_parents(node)
        if endogenous_variables.loc[:, list(parents)].isnull().values.any():
            raise ValueError(
                "parents in endogenous_variables should already be occupied"
            )
        #print(_get_noise_string(node))
        endogenous_variables[node] = scm.structural_equations_np[node](
            exogenous_variables[_get_noise_string(node)],
            *[endogenous_variables[p] for p in parents],
        )
    print('finish endogenous')
    # fix a hyperplane
    w = np.ones((endogenous_variables.shape[1], 1))
    # get the average scale of (w^T)*X, this depends on the scale of the data
    scale = 2.5 / np.mean(np.abs(np.dot(endogenous_variables, w)))
    predictions = 1 / (1 + np.exp(-scale * np.dot(endogenous_variables, w)))
    #print('predictions', predictions)

    uniform_rv = threshold
    #uniform_rv = np.random.rand(endogenous_variables.shape[0], 1)
    labels = int(uniform_rv < predictions)
    return labels

def get_pred_from_causal_v2(scm, values, cf_label, mapping_dict, threshold):
    '''
    Infers the prediction from the causal model. This is implemented accoring to Karimi et al .: 
    Assumption: Couterfactual returns a valued for every endogenous variable ! 
    Assumption: Output Node value is not contained in CF!
    #TODO does this Assumption make sense?
    Attributes: 
        scm: structural causal model 
        values: The counterfactual 
        cf_label: The counterfactual label 
        mapping_dict: variable name mapping betwenn CF and causal model
    Returns Label
    '''

    endogenous_variables = values
    for node in scm.get_topological_ordering("endogenous"):
        parents = scm.get_parents(node)
        if node not in values:
            '''Assumption Output Node value is not contained in CF '''
            output_node = node
            value=scm.structural_equations_np[node](
                *[endogenous_variables[p] for p in parents],
            )
            print('probability',value)
            endogenous_variables[node]= value[0] 
    
    predictions= endogenous_variables[output_node]
    uniform_rv = threshold
    labels = int(uniform_rv < predictions)
    
    return labels


class Sematic(Evaluation):
    """
    Semnatic Evaluation Metric.
    Attributes: 
        ml_model: Machine Learning Model
        causal_graph: ground truth causal graph
        mapping_dict: name mapping
    Returns: Consistency
    """

    def __init__(self, ml_model, causal_graph, mapping_dict, threshold=0.5):
        self.ml_model=ml_model
        self.causal_graph=causal_graph
        self.mapping_dict=mapping_dict
        self.threshold=threshold
    def get_evaluation(self,factuals: np.ndarray, counterfactuals: np.ndarray):
        # generate data
        cf_label = self.ml_model.predict(np.array(counterfactuals.values).reshape(-1, counterfactuals.values.shape[-1]))
        print('cflabel', cf_label)

        factuals_label = self.ml_model.predict(np.array(factuals.values).reshape(-1, factuals.values.shape[-1]))
        print('factuals_label', factuals_label)

        #threshold = np.random.rand(1, 1)
        #print('threshold', threshold)

        if cf_label[0][0] > self.threshold: # > 0.5: 
            cf_label=1
        else:
            cf_label=0
        causal_label = get_pred_from_causal_v2(self.causal_graph, counterfactuals, cf_label, self.mapping_dict, self.threshold)
        if cf_label == causal_label:
            return pd.DataFrame([[1]], columns=["semantic"])
        else: 
            return pd.DataFrame([[0]], columns=["semantic"])