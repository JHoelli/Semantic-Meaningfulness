#! /bin/bash

# causal Recourse

python 1_Experiment.py sanity-3-lin MLP causal_recourse 1 both
python 1_Experiment.py sanity-3-non-lin MLP causal_recourse  1 both
python 1_Experiment.py sanity-3-non-add MLP causal_recourse 1 both
python 1_Experiment.py sanity-3-lin linear causal_recourse 1 both
python 1_Experiment.py sanity-3-non-lin linear causal_recourse  1 both
python 1_Experiment.py sanity-3-non-add linear causal_recourse 1 both


# Wachter
python 1_Experiment.py sanity-3-lin MLP wachter 1 both
python 1_Experiment.py credit MLP wachter 1 both
python 1_Experiment.py sanity-3-non-lin MLP wachter 1 both
python 1_Experiment.py sanity-3-non-add MLP  wachter 1 both
python 1_Experiment.py economic MLP wachter 1 both
python 1_Experiment.py sanity-3-lin linear wachter 1 both
python 1_Experiment.py credit linear wachter 1 both
python 1_Experiment.py sanity-3-non-lin linear wachter 1 both
python 1_Experiment.py sanity-3-non-add linear  wachter 1 both
python 1_Experiment.py economic linear wachter 1 both
python 1_Experiment.py nutrition linear wachter 1 both
python 1_Experiment.py nutrition MLP wachter 1 both


#Spheres
python 1_Experiment.py sanity-3-lin MLP growingspheres 1 both
python 1_Experiment.py credit MLP growingspheres 1 both
python 1_Experiment.py sanity-3-non-lin MLP growingspheres 1 both
python 1_Experiment.py sanity-3-non-add MLP  growingspheres 1 both
python 1_Experiment.py economic MLP growingspheres 1 both
python 1_Experiment.py sanity-3-lin linear growingspheres 1 both
python 1_Experiment.py credit linear growingspheres 1 both
python 1_Experiment.py sanity-3-non-lin linear growingspheres 1 both
python 1_Experiment.py sanity-3-non-add linear  growingspheres 1 both
python 1_Experiment.py economic linear growingspheres 1 both
python 1_Experiment.py nutrition linear growingspheres 1 both
python 1_Experiment.py nutrition MLP growingspheres 1 both



#CCHVAE
python 1_Experiment.py sanity-3-lin MLP cchvae 1 both
python 1_Experiment.py credit MLP cchvae 1 both
python 1_Experiment.py sanity-3-non-lin MLP cchvae 1 both
python 1_Experiment.py sanity-3-non-add MLP  cchvae 1 both
python 1_Experiment.py economic MLP cchvae 1 both
python 1_Experiment.py sanity-3-lin linear cchvae 1 both
python 1_Experiment.py credit linear cchvae 1 both
python 1_Experiment.py sanity-3-non-lin linear cchvae 1 both
python 1_Experiment.py sanity-3-non-add linear  cchvae 1 both
python 1_Experiment.py economic linear cchvae 1 both
python 1_Experiment.py nutrition linear cchvae 1 both
python 1_Experiment.py nutrition MLP cchvae 1 both

# CLue
python 1_Experiment.py sanity-3-lin MLP Clue 1 both
python 1_Experiment.py credit MLP Clue 1 both
python 1_Experiment.py sanity-3-non-lin MLP Clue 1 both
python 1_Experiment.py sanity-3-non-add MLP Clue 1 both
python 1_Experiment.py economic MLP Clue 1 both
python 1_Experiment.py sanity-3-lin linear Clue 1 both
python 1_Experiment.py credit linear Clue 1 both
python 1_Experiment.py sanity-3-non-lin linear Clue 1 both
python 1_Experiment.py sanity-3-non-add linear Clue 1 both
python 1_Experiment.py economic linear Clue 1 both
python 1_Experiment.py nutrition linear Clue 1 both
python 1_Experiment.py nutrition MLP Clue 1 both


# Actionable Recourse
python 1_Experiment.py sanity-3-lin MLP actionable_recourse 1 both
python 1_Experiment.py credit MLP actionable_recourse 1 both
python 1_Experiment.py sanity-3-non-lin MLP actionable_recourse 1 both
python 1_Experiment.py sanity-3-non-add MLP actionable_recourse 1 both
python 1_Experiment.py economic MLP actionable_recourse 1 both
python 1_Experiment.py sanity-3-lin linear actionable_recourse 1 both
python 1_Experiment.py credit linear actionable_recourse 1 both
python 1_Experiment.py sanity-3-non-lin linear actionable_recourse 1 both
python 1_Experiment.py sanity-3-non-add linear actionable_recourse 1 both
python 1_Experiment.py economic linear actionable_recourse 1 both
python 1_Experiment.py nutrition linear actionable_recourse 1 both
python 1_Experiment.py nutrition MLP actionable_recourse 1 both


# CRUDS
python 1_Experiment.py sanity-3-lin MLP Cruds 1 both
python 1_Experiment.py credit MLP Cruds 1 both
python 1_Experiment.py sanity-3-non-lin MLP Cruds 1 both
python 1_Experiment.py sanity-3-non-add MLP Cruds 1 both
python 1_Experiment.py economic MLP Cruds 1 both
python 1_Experiment.py sanity-3-lin linear Cruds 1 both
python 1_Experiment.py credit linear Cruds 1 both
python 1_Experiment.py sanity-3-non-lin linear Cruds 1 both
python 1_Experiment.py sanity-3-non-add linear Cruds 1 both
python 1_Experiment.py economic linear Cruds 1 both
python 1_Experiment.py nutrition linear Cruds 1 both
python 1_Experiment.py nutrition MLP Cruds 1 both


# FOREST
python 1_Experiment.py sanity-3-lin forest focus 1 both
python 1_Experiment.py sanity-3-lin forest FeatureTweak 1 both

python 1_Experiment.py sanity-3-non-lin forest focus 100 both
python 1_Experiment.py sanity-3-non-lin forest FeatureTweak 100 both

python 1_Experiment.py sanity-3-non-add forest focus 100 both
python 1_Experiment.py sanity-3-non-add forest FeatureTweak 100 both

python 1_Experiment.py credit forest focus 100 both
python 1_Experiment.py credit forest FeatureTweak 100 both

python 1_Experiment.py economic forest focus 100 both
python 1_Experiment.py economic forest FeatureTweak 100 both

python 1_Experiment.py nutrition forest focus 1 both
python 1_Experiment.py nutrition forest FeatureTweak 1 both


# Dice
#python 1_Experiment.py sanity-3-lin MLP Dice 1 both
#python 1_Experiment.py credit MLP Dice 1 both
#python 1_Experiment.py sanity-3-non-lin MLP Dice 1 both
#python 1_Experiment.py sanity-3-non-add MLP Dice 1 both
#python 1_Experiment.py economic MLP Dice 1 both
#python 1_Experiment.py sanity-3-lin linear Dice 1 both
#python 1_Experiment.py credit linear Dice 1 both
#python 1_Experiment.py sanity-3-non-lin linear Dice 1 both
#python 1_Experiment.py sanity-3-non-add linear Dice 1 both
#python 1_Experiment.py economic linear Dice 1 both