from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from carla.data.api import Data


def _get_noise_string(node):
    if not node[0] == "x":
        raise ValueError
    return "u" + _get_node_id(node)


def _get_signal_string(node):
    if not node[0] == "u":
        raise ValueError
    return "x" + _get_node_id(node)


def _get_node_id(node):
    return node[1:]


def _add_noise(signal, noise):
    nodes = [_get_node_id(node) for node in noise.columns]
    for node in nodes:
        signal["x" + node] = signal["x" + node] + noise["u" + node]
    return signal


def _create_synthetic_data(scm, num_samples,fuzzy=False):
    """
    Generate synthetic data.

    Parameters
    ----------
    scm : CausalModel
        Structural causal model
    num_samples : int
        Number of samples in the dataset

    """

    exogenous_variables = np.concatenate(
        [
            np.array(scm.noise_distributions[node].sample(num_samples)).reshape(-1, 1)
            for node in scm.get_topological_ordering("exogenous")
        ],
        axis=1,
    )
    exogenous_variables = pd.DataFrame(
        exogenous_variables, columns=scm.get_topological_ordering("exogenous")
    )

    endogenous_variables = exogenous_variables.copy()
    endogenous_variables = endogenous_variables.rename(
        columns=dict(
            zip(
                scm.get_topological_ordering("exogenous"),
                scm.get_topological_ordering("endogenous"),
            )
        )
    )
    leng=0
    # used later to make sure parents are populated when computing children
    endogenous_variables.loc[:] = np.nan
    for node in scm.get_topological_ordering("endogenous"):
        parents = scm.get_parents(node)
        if endogenous_variables.loc[:, list(parents)].isnull().values.any():
            raise ValueError(
                "parents in endogenous_variables should already be occupied"
            )
        endogenous_variables[node] = scm.structural_equations_np[node](
            exogenous_variables[_get_noise_string(node)],
            *[endogenous_variables[p] for p in parents],
        )
        leng+=1

    # fix a hyperplane
    print(leng)
    if leng==3:
        w = np.ones((endogenous_variables.shape[1], 1))
        # get the average scale of (w^T)*X, this depends on the scale of the data
        scale = 2.5 / np.mean(np.abs(np.dot(endogenous_variables, w)))
        predictions = 1 / (1 + np.exp(-scale * np.dot(endogenous_variables, w)))
    elif leng==7:
        #w = np.ones((endogenous_variables.shape[0], 1))
        #predictions= 1 / (1 + np.exp(-0.3/(-np.dot(endogenous_variables['x4'],w)-np.dot(endogenous_variables['x5'],w)+np.dot(endogenous_variables['x6'],w)+np.dot(endogenous_variables['x7'],w)+np.dot(endogenous_variables['x6'],w)*np.dot(endogenous_variables['x7'],w)) ) )
        print(endogenous_variables['x4'].shape)
        predictions= 1 / (1 + np.exp(-0.3/(-endogenous_variables['x4']-endogenous_variables['x5']+endogenous_variables['x6']+endogenous_variables['x7']+endogenous_variables['x6']*endogenous_variables['x7']) ) )
    elif leng==11:
        value=  0.538 *endogenous_variables['x7'] +0.426*endogenous_variables['x8']+0.826*endogenous_variables['x12']+ 0.293*endogenous_variables['x2']+0.527 *endogenous_variables['x11']+ 0.169 *endogenous_variables['x4']+0.411*endogenous_variables['x1']
        predictions = value > 500000.000000
        predictions=predictions.astype(int)
    elif leng==8:
        predictions=-0.21*endogenous_variables['x2']+-0.59 * endogenous_variables['x1'] + 0.03 *endogenous_variables['x8']- 0.04* endogenous_variables['x7']+0.02*endogenous_variables['x5']+ 0.1*endogenous_variables['x4']
        predictions = predictions > -6.5
        predictions=predictions.astype(int)
    #if not 0.20 < np.std(predictions) < 0.42:
    #    raise ValueError(f"std of labels is strange: {np.std(predictions)}")
    if fuzzy:
        uniform_rv = np.random.rand(endogenous_variables.shape[0], 1)
        #print('randsom',uniform_rv.shape)
    else: 
        uniform_rv = np.ones((endogenous_variables.shape[0], 1))*0.5
    print('threshold',uniform_rv.shape)
    print('predictions',predictions)
    print('Meadian',np.median(predictions,axis=0))
        #print('threshold',uniform_rv)
    if type(predictions)==pd.DataFrame or type(predictions)==pd.Series:
        predictions=predictions.values.reshape(-1,1)
    labels = uniform_rv < predictions
    labels = pd.DataFrame(data=labels, columns={"label"})

    df_endogenous = pd.concat([labels, endogenous_variables], axis=1).astype("float64")
    df_exogenous = pd.concat([exogenous_variables], axis=1).astype("float64")
    return df_endogenous, df_exogenous


class ScmDataset(Data):
    """
    Generate a dataset from structural equations

    Parameters
    ----------
    scm : CausalModel
        Structural causal model
    size : int
        Number of samples in the dataset
    """

    def __init__(
        self,
        scm,
        size: int,
        fuzzy: bool, 
    ):
        # TODO setup normalization with generate_dataset in CausalModel class
        self.scm = scm
        self.name = scm.scm_class
        raw, noise = _create_synthetic_data(scm, num_samples=size,fuzzy=fuzzy)

        train_raw, test_raw = train_test_split(raw)
        train_noise = noise.iloc[train_raw.index]
        test_noise = noise.iloc[test_raw.index]

        self._df = raw
        self._df_train = train_raw
        self._df_test = test_raw

        self._noise = noise
        self._noise_train = train_noise
        self._noise_test = test_noise

        self._identity_encoding = True
        self.encoder = None
        self.scaler = None

    @property
    def categorical(self) -> List[str]:
        return self.scm._categorical

    @property
    def continuous(self) -> List[str]:
        return self.scm._continuous

    @property
    def immutables(self) -> List[str]:
        return self.scm._immutables

    @property
    def target(self) -> str:
        return "label"

    @property
    def categorical_noise(self) -> List[str]:
        """
        Provides the column names of the categorical data.

        Returns
        -------
        List[str]
        """
        return self.scm._categorical_noise

    @property
    def continuous_noise(self) -> List[str]:
        """
        Provides the column names of the continuous data.

        Returns
        -------
        List[str]
        """
        return self.scm._continuous_noise

    @property
    def df(self) -> pd.DataFrame:
        return self._df.copy()

    @property
    def df_train(self) -> pd.DataFrame:
        return self._df_train.copy()

    @property
    def df_test(self) -> pd.DataFrame:
        return self._df_test.copy()

    @property
    def noise(self) -> pd.DataFrame:
        return self._noise.copy()

    @property
    def noise_train(self) -> pd.DataFrame:
        return self._noise_train.copy()

    @property
    def noise_test(self) -> pd.DataFrame:
        return self._noise_test.copy()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO add normalization support
        return df.copy()

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO add normalization support
        return df.copy()
