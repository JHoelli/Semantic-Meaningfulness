#! /bin/bash

# DO- Calculus
#python 1_Experiment.py economic linear do_calculus 100 both
#python 1_Experiment.py economic MLP do_calculus 100 both


# causal Recourse

#python 1_Experiment.py economic MLP causal_recourse 100 both
#python 1_Experiment.py economic linear causal_recourse 100 both


# Wachter
python 1_Experiment.py economic linear wachter 100 both
python 1_Experiment.py economic MLP wachter 100 both



#Spheres
python 1_Experiment.py economic linear growingspheres 100 both
python 1_Experiment.py economic MLP growingspheres 100 both


#CCHVAE
python 1_Experiment.py economic linear cchvae 100 both
python 1_Experiment.py economic MLP cchvae 100 both


# CLue

python 1_Experiment.py economic linear Clue 100 both
python 1_Experiment.py economic MLP Clue 100 both

# Actionable Recourse
python 1_Experiment.py economic linear actionable_recourse 100 both
python 1_Experiment.py economic MLP actionable_recourse 100 both



# CRUDS
python 1_Experiment.py economic linear Cruds 100 both
python 1_Experiment.py economic MLP Cruds 100 both

# Focus only works with SK or XGBoost-Backend 
python 1_Experiment.py economic forest focus 100 both
python 1_Experiment.py economic forest FeatureTweak 100 both


