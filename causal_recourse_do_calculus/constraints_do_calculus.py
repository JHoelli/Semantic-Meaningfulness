from causal_recourse_do_calculus.sampler import Sampler
import numpy as np

def point_constraint(scm, factual_instance, action_set, sampling_handle, mlmodel):
    """
    Check if perturbed factual instance is a counterfactual.

    Parameters
    ----------
    scm: StructuralCausalModel
        Needed to create new samples.
    factual_instance: pd.Series
        Contains a single factual instance, where each element corresponds to a feature.
    action_set: dict
        Contains perturbation of features.
    sampling_handle: function
        Function that control the sampling.
    mlmodel: MLModelCatalog
        The classifier.

    Returns
    -------
    bool
    """

    # if action set is empty, return false as we don't flip the label with a factual instance
    #print('DO CALCULUS CONSTRAINTS')
    if not bool(action_set):
        #TODO Flexibilise
        return False, [[None, None, None]]
    #Samples dataset from Original Instance
    #print('Action Set', action_set)
    print('Original', factual_instance)
    sampler = Sampler(scm)
    cf_instance = sampler.sample(1, factual_instance, action_set, sampling_handle)
    print('cf Inatance from Constraint', cf_instance)
    prediction = mlmodel.predict(cf_instance)
    #print('prediction from DL Model', prediction)
    #print('Tuple', (prediction.round() == 1, cf_instance))
    return prediction.round() == 1, cf_instance
