#! /bin/bash

# Cruds
python 1_Experiment.py sanity-3-non-lin linear Cruds 250 both
python 1_Experiment.py sanity-3-non-add linear Cruds 250 both
python 1_Experiment.py economic linear Cruds 250 both

# DO- Calculus
python 1_Experiment.py sanity-3-lin linear do_calculus 250 both
python 1_Experiment.py sanity-3-non-lin linear do_calculus  250 both
python 1_Experiment.py sanity-3-non-add linear do_calculus 250 both



# causal Recourse

python 1_Experiment.py sanity-3-lin linear causal_recourse 250 both
python 1_Experiment.py sanity-3-non-lin linear causal_recourse  250 both
python 1_Experiment.py sanity-3-non-add linear causal_recourse 250 both


# Wachter

python 1_Experiment.py sanity-3-lin linear wachter 250 both
python 1_Experiment.py credit linear wachter 250 both
python 1_Experiment.py sanity-3-non-lin linear wachter 250 both
python 1_Experiment.py sanity-3-non-add linear  wachter 250 both
python 1_Experiment.py economic linear wachter 250 both


#Spheres

python 1_Experiment.py sanity-3-lin linear growingspheres 250 both
python 1_Experiment.py credit linear growingspheres 250 both
python 1_Experiment.py sanity-3-non-lin linear growingspheres 250 both
python 1_Experiment.py sanity-3-non-add linear  growingspheres 250 both
python 1_Experiment.py economic linear growingspheres 250 both


#CCHVAE
python 1_Experiment.py sanity-3-lin linear cchvae 250 both
python 1_Experiment.py credit linear cchvae 250 both
python 1_Experiment.py sanity-3-non-lin linear cchvae 250 both
python 1_Experiment.py sanity-3-non-add linear  cchvae 250 both
python 1_Experiment.py economic linear cchvae 250 both

# CLue
python 1_Experiment.py sanity-3-lin linear Clue 250 both
python 1_Experiment.py credit linear Clue 250 both
python 1_Experiment.py sanity-3-non-lin linear Clue 250 both
python 1_Experiment.py sanity-3-non-add linear Clue 250 both
python 1_Experiment.py economic linear Clue 250 both


# Actionable Recourse
python 1_Experiment.py sanity-3-lin linear actionable_recourse 250 both
python 1_Experiment.py credit linear actionable_recourse 250 both
python 1_Experiment.py sanity-3-non-lin linear actionable_recourse 250 both
python 1_Experiment.py sanity-3-non-add linear actionable_recourse 250 both
python 1_Experiment.py economic linear actionable_recourse 250 both


# CRUDS

#python 1_Experiment.py sanity-3-lin linear Cruds 250 both
#python 1_Experiment.py credit linear Cruds 250 both
#python 1_Experiment.py sanity-3-non-lin linear Cruds 250 both
#python 1_Experiment.py sanity-3-non-add linear Cruds 250 both
#python 1_Experiment.py economic linear Cruds 250 both

# Dice
python 1_Experiment.py sanity-3-lin MLP Dice 250 both
python 1_Experiment.py credit MLP Dice 250 both
python 1_Experiment.py sanity-3-non-lin MLP Dice 250 both
python 1_Experiment.py sanity-3-non-add MLP Dice 250 both
python 1_Experiment.py economic MLP Dice 250 both
python 1_Experiment.py sanity-3-lin linear Dice 250 both
python 1_Experiment.py credit linear Dice 250 both
python 1_Experiment.py sanity-3-non-lin linear Dice 250 both
python 1_Experiment.py sanity-3-non-add linear Dice 250 both
python 1_Experiment.py economic linear Dice 250 both