#! /bin/bash
# causal Recourse

python 1_Experiment.py sanity-3-lin MLP causal_recourse 10 both
python 1_Experiment.py sanity-3-lin linear causal_recourse 10 both

# Wachter
python 1_Experiment.py sanity-3-lin linear wachter 10 both
python 1_Experiment.py sanity-3-lin MLP wachter 10 both
python 1_Experiment.py credit linear wachter 10 both
python 1_Experiment.py credit MLP wachter 10 both

#Spheres
python 1_Experiment.py sanity-3-lin linear growingspheres 10 both
python 1_Experiment.py sanity-3-lin MLP growingspheres 10 both
python 1_Experiment.py credit linear growingspheres 10 both
python 1_Experiment.py credit MLP growingspheres 10 both

#CCHVAE
python 1_Experiment.py sanity-3-lin linear cchvae 10 both
python 1_Experiment.py sanity-3-lin MLP cchvae 10 both
python 1_Experiment.py credit linear cchvae 10 both
python 1_Experiment.py credit MLP cchvae 10 both

# CLue

python 1_Experiment.py sanity-3-lin linear Clue 10 both
python 1_Experiment.py sanity-3-lin MLP Clue 10 both
python 1_Experiment.py credit linear Clue 10 both
python 1_Experiment.py credit MLP Clue 10 both

# Actionable Recourse
python 1_Experiment.py sanity-3-lin linear actionable_recourse 10 both
python 1_Experiment.py sanity-3-lin MLP actionable_recourse 10 both
python 1_Experiment.py credit linear actionable_recourse 10 both
python 1_Experiment.py credit MLP actionable_recourse 10 both

# CRUDS
python 1_Experiment.py sanity-3-lin linear Cruds 10 both
python 1_Experiment.py sanity-3-lin MLP Cruds 10 both
python 1_Experiment.py credit linear Cruds 10 both
python 1_Experiment.py credit MLP Cruds 10 both

# Focus only works with SK or XGBoost-Backend 
python 1_Experiment.py sanity-3-lin forest focus 10 both
python 1_Experiment.py sanity-3-lin forest FeatureTweak 10 both
python 1_Experiment.py credit forest focus 10 both
python 1_Experiment.py credit forest FeatureTweak 10 both
