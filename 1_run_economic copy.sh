#! /bin/bash

# DO- Calculus
#python 1_Experiment.py nutrition linear do_calculus 100 both
#python 1_Experiment.py nutrition MLP do_calculus 100 both


# causal Recourse

#python 1_Experiment.py nutrition MLP causal_recourse 100 both
#python 1_Experiment.py nutrition linear causal_recourse 100 both


# Wachter
python 1_Experiment.py nutrition linear wachter 100 both
python 1_Experiment.py nutrition MLP wachter 100 both



#Spheres
python 1_Experiment.py nutrition linear growingspheres 100 both
python 1_Experiment.py nutrition MLP growingspheres 100 both


#CCHVAE
python 1_Experiment.py nutrition linear cchvae 100 both
python 1_Experiment.py nutrition MLP cchvae 100 both


# CLue

python 1_Experiment.py nutrition linear Clue 100 both
python 1_Experiment.py nutrition MLP Clue 100 both

# Actionable Recourse
python 1_Experiment.py nutrition linear actionable_recourse 100 both
python 1_Experiment.py nutrition MLP actionable_recourse 100 both



# CRUDS
python 1_Experiment.py nutrition linear Cruds 100 both
python 1_Experiment.py nutrition MLP Cruds 100 both

# Focus only works with SK or XGBoost-Backend 
python 1_Experiment.py nutrition forest focus 100 both
python 1_Experiment.py nutrition forest FeatureTweak 100 both


