import itertools
from typing import Dict

import numpy as np
import pandas as pd
from sklearn import preprocessing

from carla.recourse_methods.processing import merge_default_parameters
import causal_recourse_do_calculus.constraints_do_calculus as constraints_do
import causal_recourse_do_calculus.samplers as samplers
from causal_recourse_do_calculus.action_set import get_discretized_action_sets
from causal_recourse_do_calculus.cost import action_set_cost


def _series_plus_dict(x: pd.Series, y: dict):
    """Helper function to implemention addition for a Series object and a dict with overlapping keys

    Parameters
    ----------
    x: pd.Series
    y: dict

    Returns
    -------
    pd.Series analogous to x + y
    """
    y = pd.Series(y)

    result = x + y
    nan_cols = result.index[result.isna()].tolist()
    result = result.drop(index=nan_cols)
    result = pd.concat([result, x[nan_cols]])

    return result


# https://stackoverflow.com/a/1482316/2759976
def powerset(iterable):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )


class CausalRecourse_DoCalculus():
    """
    Implementation of causal recourse [1].

    Parameters
    ----------
    mlmodel : carla.model.MLModel
        Black-Box-Model
    checked_hyperparams : dict
        Dictionary containing hyperparameters. See Notes below to see its content.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.

        * "optimization_approach": {"brute_force", "gradient_descent"}
            Choose which optimization approach to use.
        * "num_samples": int
        * "scm": CausalModel
            Class that contains the structural causal model, and has some useful helper methods.
        * "constraint_handle": method
            Method that returns a boolean, true if constraint is met.
        * "sampler_handle": method
            Method used to sample.

        .. [1] Karimi, A. H., Schölkopf, B., & Valera, I. (2021, March). Algorithmic recourse: from counterfactual
        explanations to interventions. In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and
        Transparency (pp. 353-362).
    """

    _DEFAULT_HYPERPARAMS = {
        "optimization_approach": "brute_force",
        "num_samples": 10,
        "scm": None,
        "constraint_handle_do": constraints_do.point_constraint,
        "sampler_handle": samplers.sample_true_m0,
    }

    def __init__(self, mlmodel, hyperparams: Dict):
        #print('Initialization Starting')
        print('DO CALCULUS')
        supported_backends = ["tensorflow", "pytorch"]
        if mlmodel.backend not in supported_backends:
            raise ValueError(
                f"{mlmodel.backend} is not in supported backends {supported_backends}"
            )

        self._mlmodel = mlmodel
        self._dataset = mlmodel.data

        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )

        self._optimization_approach = checked_hyperparams["optimization_approach"]
        self._num_samples = checked_hyperparams["num_samples"]
        self._scm = checked_hyperparams["scm"]

        self._constraint_handle_do = checked_hyperparams["constraint_handle_do"]
        self._sampler_handle = checked_hyperparams["sampler_handle"]
        #print('Initialization Finished')

    def get_intervenable_nodes(self) -> dict:
        intervenable_nodes = {
            "continuous": np.setdiff1d(
                self._dataset.continuous, self._dataset.immutables
            ),
            "categorical": np.setdiff1d(
                self._dataset.categorical, self._dataset.immutables
            ),
        }
        return intervenable_nodes

    def _get_original_df(self):
        return self._dataset.df

    def _get_range_values(self):
        data_df = self._get_original_df()
        min_values = data_df.min()
        max_values = data_df.max()
        return min_values, max_values

    def _get_mean_values(self):
        data_df = self._get_original_df()
        mean_values = data_df.mean()
        return mean_values

    def compute_optimal_action_set(
        self, factual_instance, constraint_handle, sampling_handle
    ):
        #print(1)
        intervenables_nodes = self.get_intervenable_nodes()
        #print(2)
        min_values, max_values = self._get_range_values()
        #print(3)
        mean_values = self._get_mean_values()
        #print(4)
        min_cost = np.infty
        min_action_set = {}
        min_cf=[]
        #print(5)
        if self._optimization_approach == "brute_force":

            valid_action_sets = get_discretized_action_sets(
                intervenables_nodes, min_values, max_values, mean_values
            )
            #print(6)
            # we need to make sure that actions don't go out of bounds [0, 1]
            if isinstance(self._dataset.scaler, preprocessing.MinMaxScaler):
                out_of_bounds_idx = []
                for i, action_set in enumerate(valid_action_sets):
                    #print(7)
                    instance = _series_plus_dict(factual_instance, action_set)
                    if not np.all((1 > instance.values) & (instance.values > 0)):
                        out_of_bounds_idx.append(i)
                valid_action_sets = [
                    action_set
                    for i, action_set in enumerate(valid_action_sets)
                    if i not in set(out_of_bounds_idx)
                ]

            for action_set in valid_action_sets:
                #print(8)
                tup = constraint_handle(self._scm, factual_instance,  action_set,sampling_handle, self._mlmodel,)
                #print('tuple in costs',tup)
                bo, cf_instance=tup
                #print(cf_instance)
                if bo :
                    #print(9)
                    cost = action_set_cost(
                        factual_instance, action_set, max_values - min_values
                    )
                    if cost < min_cost:
                        min_cost = cost
                        min_action_set = action_set
                        min_cf=cf_instance

        elif self._optimization_approach == "gradient_descent":
            raise NotImplementedError
        else:
            raise ValueError("optimization approach not recognized")

        # #print("MIN COST", min_cost, "ACTION SET", min_action_set)

        return min_action_set, min_cost, min_cf
    def get_counterfactuals(self, factuals: pd.DataFrame):
        #print('BUUUUT WHY ')
        factual_df = factuals.drop(columns=self._dataset.target)

        cfs = []
        # actions = []
        for index, factual_instance in factual_df.iterrows():

            #print('FACTUAL INSTANCE NO', index)
            #print('index',index)
            min_action_set, _, cf = self.compute_optimal_action_set(
                factual_instance, self._constraint_handle_do, self._sampler_handle
            )
            #print('min_action_set', min_action_set)
            #print('computation finished')
            #cf = _series_plus_dict(factual_instance, min_action_set)
            # min_action_set["cost"] = min_cost
            # actions.append(min_action_set)
            #print('CF TO Factual instacne No', index)
            print('CF', cf)
            try:
                cfs.append(cf.iloc[0])
            except: 
                cfs.append(cf)

            print('CFS',cfs)

        # convert to dataframe
        cfs = pd.DataFrame(cfs)
        #print('Fisish get CF')
        # action_df = pd.DataFrame(actions)
        return cfs
