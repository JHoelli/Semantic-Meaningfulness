"""
Based on https://github.com/amirhk/recourse/blob/master/loadSCM.py
"""
from carla.data.load_scm.distributions import Bernoulli, MixtureOfGaussians, Normal,Uniform, Bernoulli

def adult_model():
    #Taken from https://github.com/amirhk/recourse/blob/master/loadSCM.py

    print('Adult Model Initiate')
    #structural_equations_np = {
    #  'x1': lambda n_samples,:                                    n_samples,  # A_sex
    #  'x2': lambda n_samples,:                                    n_samples,  # C_age
    #  'x3': lambda n_samples,:                                    n_samples,  # C_nationality
     # 'x4': lambda n_samples, x1, x2, x3,:                        n_samples,  # M_marital_status
     # 'x5': lambda n_samples, x1, x2, x3,:                        n_samples,  # L_education_level / real-valued
     # 'x6': lambda n_samples, x1, x2, x3, x4, x5:                 n_samples,  # R_working_class
     # 'x7': lambda n_samples, x1, x2, x3, x4, x5:                 n_samples,  # R_occupation
     # 'x8': lambda n_samples, x1, x2, x3, x4, x5:                 n_samples,  # R_hours_per_week
    #}
    #structural_equations_ts = structural_equations_np
    #noises_distributions = {
    #  'u1': Bernoulli(0.5, '-11'),
    #  'u2': Bernoulli(0.5, '-11'),
    #  'u3': Bernoulli(0.5, '-11'),
     # 'u4': Normal(0, 1),
     # 'u5': Normal(0, 1),
     # 'u6': Normal(0, 1),
     # 'u7': Normal(0, 1),
     # 'u8': Normal(0, 1),
    #}
    structural_equations_np = {
        "x1": lambda n_samples: n_samples,
       "x2": lambda n_samples: n_samples,
        "x3": lambda n_samples: n_samples,
        "x4": lambda n_samples, x1: 0.02 * x1 + n_samples,
        "x5": lambda n_samples, x1: 0.01* x1 + n_samples,
        "x6": lambda n_samples, x1, x2, x3: -0.01 * x1+ 0.03 * x2 - 0.01*x3 + n_samples,
        "x7": lambda n_samples: n_samples,
        "x8": lambda n_samples: n_samples,
        "x9": lambda n_samples, x4,x5, x6, x3, x7, x8:0*x5+0.05 * x4 + 0.03 * x6+ 0.04 * x3-0.04*x7-0.02 * x8 + n_samples,

    }
    print('Structural Equation Finished')
    structural_equations_ts = structural_equations_np
    noises_distributions = {
       "u1": Uniform(0,1),
        "u2": Uniform(0,41),
        "u3": Uniform(17, 90),
        "u4": Uniform(0, 5),
       "u5": Uniform(1, 99),
       "u6": Uniform(0, 99999),
       "u7": Uniform(0, 15),
       "u8": Uniform(0, 14),
      "u9": Uniform(0, 1),
    }
    print('Noise Distribution Finished')
    continuous = list(structural_equations_np.keys()) + list(
        noises_distributions.keys()
    )
    categorical =[]# ['x8', 'relationship', 'x2', 'sex']
    immutables =[]# ['x3','sex']
    #print((structural_equations_np,
    #    structural_equations_ts,
    #    noises_distributions,
    ##    continuous,
     #   categorical,
     #   immutables,))
    return (
        structural_equations_np,
        structural_equations_ts,
        noises_distributions,
        continuous,
        categorical,
        immutables,
    )


def nutrition_model():
    print('Adult Model Initiate')
    structural_equations_np = {
        #Age
        "x1": lambda n_samples: n_samples,
        #Sex
        "x2": lambda n_samples: n_samples,
        #BloodPressure
        "x3": lambda n_samples,x1: 0.02* x1+ n_samples,
        #SBP
        "x4": lambda n_samples, x3: 0.12 * x3+ n_samples,
        #PulsePresure
        "x5": lambda n_samples, x4: 0.02* x4 + n_samples,
        #Inflamantion
        "x6": lambda n_samples: n_samples,
        #Povertyn index
        "x7": lambda n_samples: n_samples,
        #Sedimation rae
        "x8": lambda n_samples, x7:0.03*x7+ n_samples,
        "x9": lambda n_samples, x1,x2,x4,x5,x7,x8: -0.21*x2+-0.59 * x1 + 0.03 * x8- 0.04* x7+0.02*x5+ 0.1*x4+ n_samples,

    }
    print('Structural Equation Finished')
    structural_equations_ts = structural_equations_np
    noises_distributions = {
        #25
        #Age
        "u1": Normal(25, 1),
        #Sex --> Uniform 
        "u2": Normal(0, 1),
        #Bloodpressure
        "u3": Normal(0, 1),
        #SBP
        "u4": Normal(80, 1),
        #Pulse Pressure
        "u5": Normal(10, 1),
        #Inflamation
        "u6": Normal(0, 1),
        #Poverty
        "u7": Normal(0, 1),
        #Sedimation
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


