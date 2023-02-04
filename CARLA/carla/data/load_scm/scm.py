"""
Based on https://github.com/amirhk/recourse/blob/master/loadSCM.py
"""
from carla.data.load_scm.distributions import Bernoulli, MixtureOfGaussians, Normal,Uniform, Bernoulli, Gamma,Poisson
import numpy as np

def sanity_3_lin_output():
    structural_equations_np = {
        "x1": lambda n_samples: n_samples,
        "x2": lambda n_samples, x1: -x1 + n_samples,
        "x3": lambda n_samples, x1, x2: 0.5 * (0.1 * x1 + 0.5 * x2) + n_samples,
        #TODO Does it make sense to exclude n samples here ? 
        "x4":lambda  x1, x2,x3:1 / (1 + np.exp(-(2.5 / np.mean(np.abs(np.dot([round(x1,8),round(x2,8),round(x3,8)], np.ones((3, 1)))))) * np.dot([round(x1,8),round(x2,8),round(x3,8)], np.ones((3, 1))))), #1/ (1 + np.exp(- 2.5 / np.mean(np.abs(np.dot([x1,x2,x3], np.ones((3, 1))))))* np.dot([x1,x2,x3], np.ones((3, 1)))),
    }
    structural_equations_ts = structural_equations_np
    noises_distributions = {
        "u1": MixtureOfGaussians([0.5, 0.5], [-2, +1], [1.5, 1]),
        "u2": Normal(0, 1),
        "u3": Normal(0, 1),
        "u4": Normal(0, 1),
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

def sanity_3_non_lin_output():
    structural_equations_np = {
        "x1": lambda n_samples: n_samples,
        "x2": lambda n_samples, x1: -1+(3/(1+np.exp(-2*x1))) + n_samples,
        "x3": lambda n_samples, x1, x2: 0.5 * (0.1 * x1 + 0.5 * x2) + n_samples,
        #TODO Does it make sense to exclude n samples here ? 
        "x4":lambda  x1, x2,x3:1 / (1 + np.exp(-(2.5 / np.mean(np.abs(np.dot([round(x1,8),round(x2,8),round(x3,8)], np.ones((3, 1)))))) * np.dot([round(x1,8),round(x2,8),round(x3,8)], np.ones((3, 1))))), #1/ (1 + np.exp(- 2.5 / np.mean(np.abs(np.dot([x1,x2,x3], np.ones((3, 1))))))* np.dot([x1,x2,x3], np.ones((3, 1)))),
    }
    structural_equations_ts = structural_equations_np
    noises_distributions = {
        "u1": MixtureOfGaussians([0.5, 0.5], [-2, +1], [1.5, 1]),
        "u2": Normal(0, 0.1),
        "u3": Normal(0, 1),
        "u4": Normal(0, 1),
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

def sanity_3_non_lin():
    structural_equations_np = {
        "x1": lambda n_samples: n_samples,
        "x2": lambda n_samples, x1: -1+(3/(1+np.exp(-2*x1))) + n_samples,
        "x3": lambda n_samples, x1, x2: 0.5 * (0.1 * x1 + 0.5 * x2) + n_samples,
        
    }
    structural_equations_ts = structural_equations_np
    noises_distributions = {
        "u1": MixtureOfGaussians([0.5, 0.5], [-2, +1], [1.5, 1]),
        "u2": Normal(0, 0.1),
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

def sanity_3_non_add_output():
    structural_equations_np = {
        "x1": lambda n_samples: n_samples,
        "x2": lambda n_samples, x1: np.sign(n_samples) * (x1 ** 2 + n_samples) / 5,
        "x3": lambda n_samples, x1, x2:  -1 * np.sqrt(x1**2 + x2**2) + n_samples,
        #TODO Does it make sense to exclude n samples here ? 
        "x4":lambda  x1, x2,x3:1 / (1 + np.exp(-(2.5 / np.mean(np.abs(np.dot([round(x1,8),round(x2,8),round(x3,8)], np.ones((3, 1)))))) * np.dot([round(x1,8),round(x2,8),round(x3,8)], np.ones((3, 1))))), #1/ (1 + np.exp(- 2.5 / np.mean(np.abs(np.dot([x1,x2,x3], np.ones((3, 1))))))* np.dot([x1,x2,x3], np.ones((3, 1)))),
    }
    structural_equations_ts = structural_equations_np
    noises_distributions = {
        "u1": MixtureOfGaussians([0.5, 0.5], [-2, +1], [1.5, 1]),
        "u2": Normal(0, 0.3),
        "u3": Normal(0, 1),
        "u4": Normal(0, 1),
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

def sanity_3_non_add():
    structural_equations_np = {
        "x1": lambda n_samples: n_samples,
        "x2": lambda n_samples, x1: np.sign(n_samples) * (x1 ** 2 + n_samples) / 5,
        "x3": lambda n_samples, x1, x2:  -1 * np.sqrt(x1**2 + x2**2) + n_samples,
        
    }
    structural_equations_ts = structural_equations_np
    noises_distributions = {
        "u1": MixtureOfGaussians([0.5, 0.5], [-2, +1], [1.5, 1]),
        "u2": Normal(0, 0.3),
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

def german_credit():
    e_0 = -1
    e_G = 0.5
    e_A = 1

    l_0 = 1
    l_A = .01
    l_G = 1

    d_0 = -1
    d_A = .1
    d_G = 2
    d_L = 1

    i_0 = -4
    i_A = .1
    i_G = 2
    i_E = 10
    i_GE = 1

    s_0 = -4
    s_I = 1.5

    structural_equations_np = {
      # Gender
      'x1': lambda n_samples,: n_samples,
      # Age
      'x2': lambda n_samples,: -35 + n_samples,
      # Education
      'x3': lambda n_samples, x1, x2 : -0.5 + (1 + np.exp(-(e_0 + e_G * x1 + e_A * (1 + np.exp(- .1 * (x2)))**(-1) + n_samples)))**(-1),
      # Loan amount
      'x4': lambda n_samples, x1, x2 :  l_0 + l_A * (x2 - 5) * (5 - x2) + l_G * x1 + n_samples,
      # Loan duration
      'x5': lambda n_samples, x1, x2, x4 : d_0 + d_A * x2 + d_G * x1 + d_L * x4 + n_samples,
      # Income
      'x6': lambda n_samples, x1, x2, x3 : i_0 + i_A * (x2 + 35) + i_G * x1 + i_GE * x1 * x3 + n_samples,
      # Savings
      'x7': lambda n_samples, x6 : s_0 + s_I * (x6 > 0) * x6 + n_samples,
    }
    structural_equations_ts = {
      # Gender
      'x1': lambda n_samples,: n_samples,
      # Age
      'x2': lambda n_samples,: -35 + n_samples,
      # Education
      'x3': lambda n_samples, x1, x2 : -0.5 + (1 + np.exp(-(e_0 + e_G * x1 + e_A * (1 + np.exp(- .1 * (x2)))**(-1) + n_samples)))**(-1),
      # Loan amount
      'x4': lambda n_samples, x1, x2 :  l_0 + l_A * (x2 - 5) * (5 - x2) + l_G * x1 + n_samples,
      # Loan duration
      'x5': lambda n_samples, x1, x2, x4 : d_0 + d_A * x2 + d_G * x1 + d_L * x4 + n_samples,
      # Income
      'x6': lambda n_samples, x1, x2, x3 : i_0 + i_A * (x2 + 35) + i_G * x1 + i_GE * x1 * x3 + n_samples,
      # Savings
      'x7': lambda n_samples, x6 : s_0 + s_I * (x6 > 0) * x6 + n_samples,
    }
    noises_distributions = {
      # Gender
      'u1': Bernoulli(0.5),
      # Age
      'u2': Gamma(10, 3.5),
      # Education
      'u3': Normal(0, 0.5**2),
      # Loan amount
      'u4': Normal(0, 2**2),
      # Loan duration
      'u5': Normal(0, 3**2),
      # Income
      'u6': Normal(0, 2**2),
      # Savings
      'u7': Normal(0, 5**2),
    }
    continuous = list(structural_equations_np.keys()) + list(
        noises_distributions.keys()
    )
    categorical =[]
    immutables =[]
    return (
        structural_equations_np,
        structural_equations_ts,
        noises_distributions,
        continuous,
        categorical,
        immutables,
    )


def german_credit_output():
    e_0 = -1
    e_G = 0.5
    e_A = 1

    l_0 = 1
    l_A = .01
    l_G = 1

    d_0 = -1
    d_A = .1
    d_G = 2
    d_L = 1

    i_0 = -4
    i_A = .1
    i_G = 2
    i_E = 10
    i_GE = 1

    s_0 = -4
    s_I = 1.5

    structural_equations_np = {
      # Gender
      'x1': lambda n_samples,: n_samples,
      # Age
      'x2': lambda n_samples,: -35 + n_samples,
      # Education
      'x3': lambda n_samples, x1, x2 : -0.5 + (1 + np.exp(-(e_0 + e_G * x1 + e_A * (1 + np.exp(- .1 * (x2)))**(-1) + n_samples)))**(-1),
      # Loan amount
      'x4': lambda n_samples, x1, x2 :  l_0 + l_A * (x2 - 5) * (5 - x2) + l_G * x1 + n_samples,
      # Loan duration
      'x5': lambda n_samples, x1, x2, x4 : d_0 + d_A * x2 + d_G * x1 + d_L * x4 + n_samples,
      # Income
      'x6': lambda n_samples, x1, x2, x3 : i_0 + i_A * (x2 + 35) + i_G * x1 + i_GE * x1 * x3 + n_samples,
      # Savings
      'x7': lambda n_samples, x6 : s_0 + s_I * (x6 > 0) * x6 + n_samples,
      # Output
      'x8': lambda  x4, x5,x6, x7:1 / (1 + np.exp(-0.3/(-round(x4,8)-round(x5,8)+round(x6,8)+round(x7,8)+round(x6,8)*round(x7,8)) ) ), 
    }
    structural_equations_ts = {
      # Gender
      'x1': lambda n_samples,: n_samples,
      # Age
      'x2': lambda n_samples,: -35 + n_samples,
      # Education
      'x3': lambda n_samples, x1, x2 : -0.5 + (1 + np.exp(-(e_0 + e_G * x1 + e_A * (1 + np.exp(- .1 * (x2)))**(-1) + n_samples)))**(-1),
      # Loan amount
      'x4': lambda n_samples, x1, x2 :  l_0 + l_A * (x2 - 5) * (5 - x2) + l_G * x1 + n_samples,
      # Loan duration
      'x5': lambda n_samples, x1, x2, x4 : d_0 + d_A * x2 + d_G * x1 + d_L * x4 + n_samples,
      # Income
      'x6': lambda n_samples, x1, x2, x3 : i_0 + i_A * (x2 + 35) + i_G * x1 + i_GE * x1 * x3 + n_samples,
      # Savings
      'x7': lambda n_samples, x6 : s_0 + s_I * (x6 > 0) * x6 + n_samples,
       # Output
      'x8': lambda  x4, x5,x6, x7:1 / (1 + np.exp(-0.3/(-round(x4,8)-round(x5,8)+round(x6,8)+round(x7,8)+round(x6,8)*round(x7,8)) ) ), 
    }
    noises_distributions = {
      # Gender
      'u1': Bernoulli(0.5),
      # Age
      'u2': Gamma(10, 3.5),
      # Education
      'u3': Normal(0, 0.5**2),
      # Loan amount
      'u4': Normal(0, 2**2),
      # Loan duration
      'u5': Normal(0, 3**2),
      # Income
      'u6': Normal(0, 2**2),
      # Savings
      'u7': Normal(0, 5**2),
      #Output
      "u8": Normal(0, 1),
    }
    continuous = list(structural_equations_np.keys()) + list(
        noises_distributions.keys()
    )
    categorical =[]
    immutables =[]
    return (
        structural_equations_np,
        structural_equations_ts,
        noises_distributions,
        continuous,
        categorical,
        immutables,
    )

def economic_growth_china_output():
    #FROM https://www.sciencedirect.com/science/article/abs/pii/S0360544221031546 
    structural_equations_np = {
        # Latent Variables
        # Energy source structure 
        "x1": lambda n_samples: n_samples,
        # Informatization level 
        "x2": lambda x4,x3, n_samples: 0.836 * x4+0.464 *x3+ n_samples,#x4 0.023, 6.156
        # Ecological Awarness
        "x3": lambda x4, n_samples: 0.889 *x4+ n_samples,
        #Observed Variables
        # Electrictity Consumption
        "x4": lambda n_samples: n_samples,
        # Economic Growth #TODO THis is still not right ? 
        "x5": lambda x7,x8,x12,x2,x11, x4,x1: int((0.538 *x7 +0.426*x8+0.826*x12+ 0.293*x2+0.527 *x11+ 0.169 *x4+0.411*x1) > 500000.000000),
        # Electrictiy Investment
        "x6": lambda x4,n_samples: 0.898*x4 +n_samples, #0.426*x4+ 7.984
        # Investment other Idustries
        "x7": lambda x6,  n_samples:0.783* x6+ n_samples,# x6 13.023*** 0.783 7.617
        # Employment
        "x8": lambda x4,  n_samples: 0.789 *x4+ n_samples,# x4 .882** 0.789 3.307
        # Development of the secondary industry 
        "x9": lambda x2, x4,n_samples: 0.566*x4+0.561* x2+n_samples, #'x4 3.014,0.566,7.816',x2336.3*** 0.561 9.785
        # Development of the tertiary industry 
        "x10": lambda x2, x4, n_samples:0.537* x4+0.712*x2+ n_samples, # x4 3.768*** 0.537 7.093, x2899.0*** 0.712 8.018
        # Proportion of non-agriculture
        "x11": lambda x9, x10, x7, x2,n_samples:0.731*x9+ 0.612 *x10+0.662*x7+0.605 *x2+ n_samples,# x9 0.234** 0.731 3.183, x10 0.014*** 0.612 8.251
        # Labor productivity
        "x12": lambda x4,n_samples: 0.918* x4 + n_samples,#227*** 0.918 7.648
    
    }
    structural_equations_ts = structural_equations_np
   #TODO
    noises_distributions = {
        #http://www.stats.gov.cn/tjsj/ndsj/2019/indexeh.htm
       "u1": Uniform(0,1),#TODO
        "u2": Uniform(0,41),#TODO
        "u3": Uniform(17, 90),#TODO
        "u4": Uniform(0, 100000),#100 mio kwh
       "u5": Uniform(1, 99),# Outcome not relevant
       "u6": Uniform(0, 99999),#TODO Investment
       "u7": Uniform(0, 15),#TODO Investment
       "u8": Uniform(0, 70),#Done
     "u9": Uniform(0, 2000), #DOne
    "u10": Uniform(0,2000),#Done
    "u11": Uniform(0,100),# DOne
    "u12": Uniform(0,500000),#Done

    }
    continuous = list(structural_equations_np.keys()) + list(
        noises_distributions.keys()
    )
    categorical =[]
    immutables =[]
    return (
        structural_equations_np,
        structural_equations_ts,
        noises_distributions,
        continuous,
        categorical,
        immutables,
    )
def economic_growth_china():
    #FROM https://www.sciencedirect.com/science/article/abs/pii/S0360544221031546 
    #1 Billion = 1.000.000.000
    structural_equations_np = {
        # Latent Variables
        # Energy source structure 
        "x1": lambda n_samples: n_samples,
        # Informatization level 
        "x2": lambda x4,x3, n_samples: 0.836 * x4+0.464 *x3+ n_samples,#x4 0.023, 6.156
        # Ecological Awarness
        "x3": lambda x4, n_samples: 0.889 *x4+ n_samples,
        #Observed Variables
        # Electrictity Consumption
        "x4": lambda n_samples: n_samples,
        # Electrictiy Investment
        "x6": lambda x4,n_samples: 0.898*x4 +n_samples, #0.426*x4+ 7.984
        # Investment other Idustries
        "x7": lambda x6,  n_samples:0.783* x6+ n_samples,# x6 13.023*** 0.783 7.617
        # Employment
        "x8": lambda x4,  n_samples: 0.789 *x4+ n_samples,# x4 .882** 0.789 3.307
        # Development of the secondary industry 
        "x9": lambda x2, x4,n_samples: 0.566*x4+0.561* x2+n_samples, #'x4 3.014,0.566,7.816',x2336.3*** 0.561 9.785
        # Development of the tertiary industry 
        "x10": lambda x2, x4, n_samples:0.537* x4+0.712*x2+ n_samples, # x4 3.768*** 0.537 7.093, x2899.0*** 0.712 8.018
        # Proportion of non-agriculture
        "x11": lambda x9, x10, x7, x2,n_samples:0.731*x9+ 0.612 *x10+0.662*x7+0.605 *x2+ n_samples,# x9 0.234** 0.731 3.183, x10 0.014*** 0.612 8.251
        # Labor productivity
        "x12": lambda x4,n_samples: 0.918* x4 + n_samples,#227*** 0.918 7.648
    
    }
    structural_equations_ts = structural_equations_np
    #http://www.stats.gov.cn/tjsj/ndsj/2019/indexeh.htm
    noises_distributions = {
        #http://www.stats.gov.cn/tjsj/ndsj/2019/indexeh.htm
       "u1": Uniform(0,1),#TODO
        "u2": Uniform(0,41),#TODO
        "u3": Uniform(17, 90),#TODO
        "u4": Uniform(0, 100000),#100 mio kwh
       "u6": Uniform(0, 99999),#TODO Investment
       "u7": Uniform(0, 15),#TODO Investment
       "u8": Uniform(0, 70),#Done
     "u9": Uniform(0, 2000), #DOne
    "u10": Uniform(0,2000),#Done
    "u11": Uniform(0,100),# DOne
    "u12": Uniform(0,500000),#Done

    }
    print('Noise Distribution Finished')
    continuous = list(structural_equations_np.keys()) + list(
        noises_distributions.keys()
    )
    categorical =[]# ['x8', 'relationship', 'x2', 'sex']
    immutables =[]# 
    return (
        structural_equations_np,
        structural_equations_ts,
        noises_distributions,
        continuous,
        categorical,
        immutables,
    )

#def adult_model():
    #Taken from https://github.com/amirhk/recourse/blob/master/loadSCM.py

#    print('Adult Model Initiate')
#    structural_equations_np = {
       # A_sex
#        "x1": lambda n_samples: n_samples,
        # C_age
#       "x2": lambda n_samples: n_samples,
       # C_nationality
#        "x3": lambda n_samples: n_samples,
         # M_marital_status
#        "x4": lambda n_samples, x1: 0.02 * x1 + n_samples,
        # L_education_level / real-valued
#        "x5": lambda n_samples, x1: 0.01* x1 + n_samples,
         # R_working_class
#        "x6": lambda n_samples, x1, x2, x3: -0.01 * x1+ 0.03 * x2 - 0.01*x3 + n_samples,
        # R_occupation
#        "x7": lambda n_samples: n_samples,
        # R_hours_per_week
#        "x8": lambda n_samples: n_samples,
#        "x9": lambda n_samples, x4,x5, x6, x3, x7, x8:0*x5+0.05 * x4 + 0.03 * x6+ 0.04 * x3-0.04*x7-0.02 * x8 + n_samples,

#    }
#    print('Structural Equation Finished')
 #   structural_equations_ts = structural_equations_np
#    noises_distributions = {
 #      "u1": Uniform(0,1),
#        "u2": Uniform(0,41),
#        "u3": Uniform(17, 90),
#        "u4": Uniform(0, 5),
#       "u5": Uniform(1, 99),
#       "u6": Uniform(0, 99999),
#       "u7": Uniform(0, 15),
#       "u8": Uniform(0, 14),
#      "u9": Uniform(0, 1),
#    }
#    print('Noise Distribution Finished')
#    continuous = list(structural_equations_np.keys()) + list(
#        noises_distributions.keys()
#    )
#    categorical =[]# ['x8', 'relationship', 'x2', 'sex']
#    immutables =[]# ['x3','sex']
#    return (
#        structural_equations_np,
#        structural_equations_ts,
#        noises_distributions,
#        continuous,
#        categorical,
#        immutables,
#    )
#def adult_model_output():
    #Taken from https://github.com/amirhk/recourse/blob/master/loadSCM.py

#    print('Adult Model Initiate')
#    structural_equations_np = {
        # A_sex
#        "x1": lambda n_samples: n_samples,
        # C_age
#       "x2": lambda n_samples: n_samples,
       # C_nationality
#        "x3": lambda n_samples: n_samples,
         # M_marital_status
#        "x4": lambda n_samples, x1: 0.02 * x1 + n_samples,
        # L_education_level / real-valued
#        "x5": lambda n_samples, x1: 0.01* x1 + n_samples,
         # R_working_class
#        "x6": lambda n_samples, x1, x2, x3: -0.01 * x1+ 0.03 * x2 - 0.01*x3 + n_samples,
        # R_occupation
#        "x7": lambda n_samples: n_samples,
        # R_hours_per_week
#        "x8": lambda n_samples: n_samples,
#        "x9": lambda n_samples, x4,x5, x6, x3, x7, x8:0*x5+0.05 * x4 + 0.03 * x6+ 0.04 * x3-0.04*x7-0.02 * x8 + n_samples,

#    }
#    print('Structural Equation Finished')
#    structural_equations_ts = structural_equations_np
#    noises_distributions = {
#       "u1": Uniform(0,1),
#        "u2": Uniform(0,41),
#        "u3": Uniform(17, 90),
#        "u4": Uniform(0, 5),
#       "u5": Uniform(1, 99),
#       "u6": Uniform(0, 99999),
#       "u7": Uniform(0, 15),
#       "u8": Uniform(0, 14),
#      "u9": Uniform(0, 1),
#    }
#    print('Noise Distribution Finished')
#    continuous = list(structural_equations_np.keys()) + list(
#        noises_distributions.keys()
#    )
#    categorical =[]# ['x8', 'relationship', 'x2', 'sex']
#    immutables =[]# ['x3','sex']
#    return (
#        structural_equations_np,
#        structural_equations_ts,
#        noises_distributions,
#        continuous,
 #       categorical,
 #       immutables,
 #   )




def nutrition_model_output():

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
        #Res
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
    #print((structural_equations_np,
    #    structural_equations_ts,
    #    noises_distributions,
    #    continuous,
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
    #    continuous,
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


