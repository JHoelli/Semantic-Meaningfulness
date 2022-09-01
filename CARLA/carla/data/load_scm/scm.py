"""
Based on https://github.com/amirhk/recourse/blob/master/loadSCM.py
"""
from carla.data.load_scm.distributions import MixtureOfGaussians, Normal

def census_model():
    print('Census Model Initiate')
    structural_equations_np = {
        "x1": lambda n_samples: n_samples,
        "x2": lambda n_samples: n_samples,
        "x3": lambda n_samples: n_samples,
        "x4": lambda n_samples, x1: 0.02 * x1 + n_samples,
        "x5": lambda n_samples, x1: 0.01* x1 + n_samples,
        "x6": lambda n_samples, x1, x2, x3: -0.01 * x1+ 0.03 * x2 - 0.01*x3 + n_samples,
        "x7": lambda n_samples: n_samples,
        "x8": lambda n_samples: n_samples,
        "x9": lambda n_samples, x4,x5, x6, x3, x7, x8: 0.05 * x4 + 0.03 * x6+ 0.04 * x3-0.04*x7-0.02 * x8 + n_samples,

    }
    print('Structural Equation Finished')
    structural_equations_ts = structural_equations_np
    noises_distributions = {
        "u1": MixtureOfGaussians([0.5, 0.5], [-2, +1], [1.5, 1]),
        "u2": Normal(0, 1),
        "u3": Normal(0, 1),
        "u4": Normal(0, 1),
        "u5": Normal(0, 1),
        "u6": Normal(0, 1),
        "u7": Normal(0, 1),
        "u8": Normal(0, 1),
        "u9": Normal(0, 1),
    }
    print('Noise Distribution Finished')
    continuous = list(structural_equations_np.keys()) + list(
        noises_distributions.keys()
    )
    categorical =[]# ['x8', 'relationship', 'x2', 'sex']
    immutables =[]# ['x3','sex']
    print((structural_equations_np,
        structural_equations_ts,
        noises_distributions,
        continuous,
        categorical,
        immutables,))
    return (
        structural_equations_np,
        structural_equations_ts,
        noises_distributions,
        continuous,
        categorical,
        immutables,
    )


def sanity_3_lin():
    structural_equations_np = {
        "x1": lambda n_samples: n_samples,
        "x2": lambda n_samples, x1: -x1 + n_samples,
        "x3": lambda n_samples, x1, x2: 0.5 * (0.1 * x1 + 0.5 * x2) + n_samples,
    }
    structural_equations_ts = structural_equations_np
    noises_distributions = {
        "u1": MixtureOfGaussians([0.5, 0.5], [-2, +1], [1.5, 1]),
        "u2": Normal(0, 1),
        "u3": Normal(0, 1),
    }
    continuous = list(structural_equations_np.keys()) + list(
        noises_distributions.keys()
    )
    categorical = []
    immutables = []

    return (
        structural_equations_np,
        structural_equations_ts,
        noises_distributions,
        continuous,
        categorical,
        immutables,
    )


