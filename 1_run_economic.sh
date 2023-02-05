#! /bin/bash

# DO- Calculus
#python 1_Experiment.py economic linear do_calculus 250 both
#python 1_Experiment.py economic MLP do_calculus 250 both


# causal Recourse

#python 1_Experiment.py economic MLP causal_recourse 250 both
#python 1_Experiment.py economic linear causal_recourse 250 both


# Wachter
python 1_Experiment.py economic linear wachter 250 both
python 1_Experiment.py economic MLP wachter 250 both



#Spheres
python 1_Experiment.py economic linear growingspheres 250 both
python 1_Experiment.py economic MLP growingspheres 250 both


#CCHVAE
python 1_Experiment.py economic linear cchvae 250 both
python 1_Experiment.py economic MLP cchvae 250 both


# CLue

python 1_Experiment.py economic linear Clue 250 both
python 1_Experiment.py economic MLP Clue 250 both

# Actionable Recourse
python 1_Experiment.py economic linear actionable_recourse 250 both
python 1_Experiment.py economic MLP actionable_recourse 250 both



# CRUDS
python 1_Experiment.py economic linear Cruds 250 both
python 1_Experiment.py economic MLP Cruds 250 both

# Focus only works with SK or XGBoost-Backend 
python 1_Experiment.py economic forest focus 250 both
python 1_Experiment.py economic forest FeatureTweak 250 both


