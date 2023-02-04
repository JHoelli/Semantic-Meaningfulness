#! /bin/bash

# DO- Calculus
python 1_Experiment.py sanity-3-lin linear do_calculus 100 both
python 1_Experiment.py sanity-3-lin MLP do_calculus 100 both

# causal Recourse

python 1_Experiment.py sanity-3-lin MLP causal_recourse 100 both
python 1_Experiment.py sanity-3-lin linear causal_recourse 100 both

# Wachter
python 1_Experiment.py sanity-3-lin linear wachter 100 both
python 1_Experiment.py sanity-3-lin MLP wachter 100 both
python 1_Experiment.py credit linear wachter 100 both
python 1_Experiment.py credit MLP wachter 100 both

#Spheres
python 1_Experiment.py sanity-3-lin linear growingspheres 100 both
python 1_Experiment.py sanity-3-lin MLP growingspheres 100 both
python 1_Experiment.py credit linear growingspheres 100 both
python 1_Experiment.py credit MLP growingspheres 100 both

#CCHVAE
python 1_Experiment.py sanity-3-lin linear cchvae 100 both
python 1_Experiment.py sanity-3-lin MLP cchvae 100 both
python 1_Experiment.py credit linear cchvae 100 both
python 1_Experiment.py credit MLP cchvae 100 both

# CLue

python 1_Experiment.py sanity-3-lin linear Clue 100 both
python 1_Experiment.py sanity-3-lin MLP Clue 100 both
python 1_Experiment.py credit linear Clue 100 both
python 1_Experiment.py credit MLP Clue 100 both

# Actionable Recourse
python 1_Experiment.py sanity-3-lin linear actionable_recourse 100 both
python 1_Experiment.py sanity-3-lin MLP actionable_recourse 100 both
python 1_Experiment.py credit linear actionable_recourse 100 both
python 1_Experiment.py credit MLP actionable_recourse 100 both

# CRUDS
python 1_Experiment.py sanity-3-lin linear Cruds 100 both
python 1_Experiment.py sanity-3-lin MLP Cruds 100 both
python 1_Experiment.py credit linear Cruds 100 both
python 1_Experiment.py credit MLP Cruds 100 both

# Focus only works with SK or XGBoost-Backend 
python 1_Experiment.py sanity-3-lin forest focus 100 both
python 1_Experiment.py sanity-3-lin forest FeatureTweak 100 both
python 1_Experiment.py credit forest focus 100 both
python 1_Experiment.py credit forest FeatureTweak 100 both
