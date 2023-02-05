#! /bin/bash

# DO- Calculus
#python 1_Experiment.py nutrition linear do_calculus 250 both
#python 1_Experiment.py nutrition MLP do_calculus 250 both


# causal Recourse

#python 1_Experiment.py nutrition MLP causal_recourse 250 both
#python 1_Experiment.py nutrition linear causal_recourse 250 both


# Wachter
python 1_Experiment.py nutrition linear wachter 250 both
python 1_Experiment.py nutrition MLP wachter 250 both



#Spheres
python 1_Experiment.py nutrition linear growingspheres 250 both
python 1_Experiment.py nutrition MLP growingspheres 250 both


#CCHVAE
python 1_Experiment.py nutrition linear cchvae 250 both
python 1_Experiment.py nutrition MLP cchvae 250 both


# CLue

python 1_Experiment.py nutrition linear Clue 250 both
python 1_Experiment.py nutrition MLP Clue 250 both

# Actionable Recourse
python 1_Experiment.py nutrition linear actionable_recourse 250 both
python 1_Experiment.py nutrition MLP actionable_recourse 250 both



# CRUDS
python 1_Experiment.py nutrition linear Cruds 250 both
python 1_Experiment.py nutrition MLP Cruds 250 both

# Focus only works with SK or XGBoost-Backend 
python 1_Experiment.py nutrition forest focus 250 both
python 1_Experiment.py nutrition forest FeatureTweak 250 both


