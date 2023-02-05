from cmath import nan
from telnetlib import SE
from carla.data import causal_model
from carla.evaluation import remove_nans
from carla.evaluation.api import Evaluation
import numpy as np 
import pandas as pd 

def get_abduction_noise(
    node: str, parents, structural_equation, factual_instance: dict
):
    return factual_instance[node] - structural_equation(
        0, *[factual_instance[p] for p in parents]
    )


def relationship_check(scm, values, precision = 4):
    '''
    TODO 
    * Detailling
    * Significance
    * What to do if first level is inccorect ? use intervened value ?  
    '''

    # Default see all relationships as met    
    relationships= np.ones_like(values).reshape(1,-1)
    relationships=pd.DataFrame(relationships, columns=scm.get_topological_ordering("endogenous"))
    print(relationships)
    endogenous_variables = values
    print(endogenous_variables)
    num_relationship_tested = 0
    for node in scm.get_topological_ordering("endogenous"):
        print(node)
        parents = scm.get_parents(node)
        print(parents)
        #TODO check relationships here
        if len(parents)!=0:
            structural_equation = scm.structural_equations_np[node]

            predicted_noise = get_abduction_noise(
                node, parents, structural_equation, values
            )

            noise = np.array(predicted_noise)
            node_sample = structural_equation(noise, *[values[p] for p in parents])
            value = node_sample
            print(value)
            #value=scm.structural_equations_np[node](0,
            #    *[endogenous_variables[p] for p in parents],
            #)   
            #print('Calculated Value',value)
            #print('CF Value', endogenous_variables[node])
            if round(value,precision) != round(endogenous_variables[node],precision):
                relationships[node]=0 
        num_relationship_tested += 1
    return [np.sum(relationships.iloc[0])/num_relationship_tested]

def get_pred_from_causal(scm, values, cf_label, threshold):
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
                *[endogenous_variables[p] for p in parents]
            )
            print('probability',value)
            #TODO ELIMINATE THIS
            try:
                endogenous_variables[node]= value[0] 
            except:
                endogenous_variables[node]= value 
    
    predictions= endogenous_variables[output_node]
    uniform_rv = threshold
    labels = int(uniform_rv < predictions)
    print('labels', labels)
    
    return labels


class Sematic(Evaluation):
    """
    Semantic Evaluation Metrics.
    Relationship and Outputwise. Useable for full structural causal models and partly structural causal models.
    Attributes: 
        ml_model: Machine Learning Model
        causal_graph: ground truth causal graph
        mapping_dict: name mapping
    Returns: Consistency
    """

    def __init__(self, ml_model, causal_graph_full=None,causal_graph_small=None,  threshold=0.5 ):
        self.ml_model=ml_model
        self.causal_graph_full=causal_graph_full
        self.threshold=threshold
        self. causal_graph_small= causal_graph_small
    def get_evaluation(self,factuals, counterfactuals):
        try:
            #print('try')
            factuals= factuals.loc[:,~factuals.columns.isin(['label'])]
        except: 
            #print('except')
            pass
        counterfactuals.dropna(axis=0, how='all', inplace=True)
        counterfactuals = counterfactuals.reset_index(drop=True)
        cf_label = self.ml_model.predict(np.array(counterfactuals.values).reshape(-1, counterfactuals.values.shape[-1]))
        semantic=[]
        if self.causal_graph_full is not None and self.causal_graph_small is None:
            for a in counterfactuals.index:
                causal_label = get_pred_from_causal(self.causal_graph_full, counterfactuals.iloc[a], cf_label[a], self.threshold)

                if cf_label[a] == causal_label:
                    semantic.append([1])
                else:
                    semantic.append([0])
            return pd.DataFrame(semantic, columns=["semantic"])
        elif self.causal_graph_full is not None and self.causal_graph_small is not None:
            num=[]
            for a in counterfactuals.index:
                causal_label = get_pred_from_causal(self.causal_graph_full, counterfactuals.iloc[a], cf_label[a], self.threshold)
                num.append(relationship_check(scm=self.causal_graph_small, values=counterfactuals.iloc[a]))
                print('causal_label', causal_label)
                print('cf_label', cf_label[a])
                if hasattr(cf_label[a], "__len__"):
                    if round(cf_label[a][0]) == causal_label:
                        semantic.append([1])
                    else:
                        semantic.append([0])
                else: 
                    if round(cf_label[a]) == causal_label:
                        semantic.append([1])
                    else:
                        semantic.append([0])
            print(semantic)
            df=pd.DataFrame([])
            df['semantic']=np.array(semantic).reshape(-1)
            df["correct_relationships"]=np.array(num).reshape(-1)

            #pd.DataFrame(semantic, columns=['semantic']).to_csv('test.csv')
            return df #pd.DataFrame(np.vstack([np.array(semantic).reshape(-1),np.array(num).reshape(-1)]).reshape(-1,2), columns=["semantic","correct_relationships"])

        else: 
            num=[]
            for a in counterfactuals.index:
                num.append(relationship_check(scm=self.causal_graph_small, values=counterfactuals.iloc[a]))
            return pd.DataFrame(num, columns=["correct_relationships"])
                

        
     