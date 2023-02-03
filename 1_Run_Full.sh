#! /bin/bash

# DO- Calculus
#python 1_Experiment.py sanity-3-lin MLP do_calculus 250 both
python 1_Experiment.py sanity-3-non-lin MLP do_calculus  250 both
python 1_Experiment.py sanity-3-non-add MLP do_calculus 250 both
python 1_Experiment.py sanity-3-lin linear do_calculus 250 both
python 1_Experiment.py sanity-3-non-lin linear do_calculus  250 both
python 1_Experiment.py sanity-3-non-add linear do_calculus 250 both



# causal Recourse

python 1_Experiment.py sanity-3-lin MLP causal_recourse 250 both
python 1_Experiment.py sanity-3-non-lin MLP causal_recourse  250 both
python 1_Experiment.py sanity-3-non-add MLP causal_recourse 250 both
python 1_Experiment.py sanity-3-lin linear causal_recourse 250 both
python 1_Experiment.py sanity-3-non-lin linear causal_recourse  250 both
python 1_Experiment.py sanity-3-non-add linear causal_recourse 250 both


# Wachter
python 1_Experiment.py sanity-3-lin MLP wachter 250 both
python 1_Experiment.py credit MLP wachter 250 both
python 1_Experiment.py sanity-3-non-lin MLP wachter 250 both
python 1_Experiment.py sanity-3-non-add MLP  wachter 250 both
python 1_Experiment.py economic MLP wachter 250 both
python 1_Experiment.py sanity-3-lin linear wachter 250 both
python 1_Experiment.py credit linear wachter 250 both
python 1_Experiment.py sanity-3-non-lin linear wachter 250 both
python 1_Experiment.py sanity-3-non-add linear  wachter 250 both
python 1_Experiment.py economic linear wachter 250 both


#Spheres
python 1_Experiment.py sanity-3-lin MLP growingspheres 250 both
python 1_Experiment.py credit MLP growingspheres 250 both
python 1_Experiment.py sanity-3-non-lin MLP growingspheres 250 both
python 1_Experiment.py sanity-3-non-add MLP  growingspheres 250 both
python 1_Experiment.py economic MLP growingspheres 250 both
python 1_Experiment.py sanity-3-lin linear growingspheres 250 both
python 1_Experiment.py credit linear growingspheres 250 both
python 1_Experiment.py sanity-3-non-lin linear growingspheres 250 both
python 1_Experiment.py sanity-3-non-add linear  growingspheres 250 both
python 1_Experiment.py economic linear growingspheres 250 both


#CCHVAE
python 1_Experiment.py sanity-3-lin MLP cchvae 250 both
python 1_Experiment.py credit MLP cchvae 250 both
python 1_Experiment.py sanity-3-non-lin MLP cchvae 250 both
python 1_Experiment.py sanity-3-non-add MLP  cchvae 250 both
python 1_Experiment.py economic MLP cchvae 250 both
python 1_Experiment.py sanity-3-lin linear cchvae 250 both
python 1_Experiment.py credit linear cchvae 250 both
python 1_Experiment.py sanity-3-non-lin linear cchvae 250 both
python 1_Experiment.py sanity-3-non-add linear  cchvae 250 both
python 1_Experiment.py economic linear cchvae 250 both

# CLue
python 1_Experiment.py sanity-3-lin MLP Clue 250 both
python 1_Experiment.py credit MLP Clue 250 both
python 1_Experiment.py sanity-3-non-lin MLP Clue 250 both
python 1_Experiment.py sanity-3-non-add MLP Clue 250 both
python 1_Experiment.py economic MLP Clue 250 both
python 1_Experiment.py sanity-3-lin linear Clue 250 both
python 1_Experiment.py credit linear Clue 250 both
python 1_Experiment.py sanity-3-non-lin linear Clue 250 both
python 1_Experiment.py sanity-3-non-add linear Clue 250 both
python 1_Experiment.py economic linear Clue 250 both


# Actionable Recourse
python 1_Experiment.py sanity-3-lin MLP actionable_recourse 250 both
python 1_Experiment.py credit MLP actionable_recourse 250 both
python 1_Experiment.py sanity-3-non-lin MLP actionable_recourse 250 both
python 1_Experiment.py sanity-3-non-add MLP actionable_recourse 250 both
python 1_Experiment.py economic MLP actionable_recourse 250 both
python 1_Experiment.py sanity-3-lin linear actionable_recourse 250 both
python 1_Experiment.py credit linear actionable_recourse 250 both
python 1_Experiment.py sanity-3-non-lin linear actionable_recourse 250 both
python 1_Experiment.py sanity-3-non-add linear actionable_recourse 250 both
python 1_Experiment.py economic linear actionable_recourse 250 both


# CRUDS
python 1_Experiment.py sanity-3-lin MLP Cruds 250 both
python 1_Experiment.py credit MLP Cruds 250 both
python 1_Experiment.py sanity-3-non-lin MLP Cruds 250 both
python 1_Experiment.py sanity-3-non-add MLP Cruds 250 both
python 1_Experiment.py economic MLP Cruds 250 both
python 1_Experiment.py sanity-3-lin linear Cruds 250 both
python 1_Experiment.py credit linear Cruds 250 both
python 1_Experiment.py sanity-3-non-lin linear Cruds 250 both
python 1_Experiment.py sanity-3-non-add linear Cruds 250 both
python 1_Experiment.py economic linear Cruds 250 both


# FOREST
python 1_Experiment.py sanity-3-lin forest focus 100 both
python 1_Experiment.py sanity-3-lin forest FeatureTweak 100 both

python 1_Experiment.py sanity-3-non-lin forest focus 100 both
python 1_Experiment.py sanity-3-non-lin forest FeatureTweak 100 both

python 1_Experiment.py sanity-3-non-add forest focus 100 both
python 1_Experiment.py sanity-3-non-add forest FeatureTweak 100 both

python 1_Experiment.py credit forest focus 100 both
python 1_Experiment.py credit forest FeatureTweak 100 both

python 1_Experiment.py economic forest focus 100 both
python 1_Experiment.py economic forest FeatureTweak 100 both