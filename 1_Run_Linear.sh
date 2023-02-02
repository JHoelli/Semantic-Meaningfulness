#! /bin/bash

# DO- Calculus
python 1_Experiment.py sanity-3-lin linear do_calculus 100 both
#python 1_Experiment.py credit linear wachter 100 both
python 1_Experiment.py sanity-3-non-lin linear do_calculus  100 both
python 1_Experiment.py sanity-3-non-add linear do_calculus 100 both
#python 1_Experiment.py economic linear wachter 100 both

# causal Recourse

python 1_Experiment.py sanity-3-lin linear causal_recourse 100 both
python 1_Experiment.py sanity-3-non-lin linear causal_recourse  100 both
python 1_Experiment.py sanity-3-non-add linear causal_recourse 100 both


# Wachter
python 1_Experiment.py sanity-3-lin linear wachter 100 both
python 1_Experiment.py credit linear wachter 100 both
python 1_Experiment.py sanity-3-non-lin linear wachter 100 both
python 1_Experiment.py sanity-3-non-add linear  wachter 100 both
python 1_Experiment.py economic linear wachter 100 both


#Spheres
python 1_Experiment.py sanity-3-lin linear growingspheres 100 both
python 1_Experiment.py credit linear growingspheres 100 both
python 1_Experiment.py sanity-3-non-lin linear growingspheres 100 both
python 1_Experiment.py sanity-3-non-add linear  growingspheres 100 both
python 1_Experiment.py economic linear growingspheres 100 both



#CCHVAE
python 1_Experiment.py sanity-3-lin linear cchvae 100 both
python 1_Experiment.py credit linear cchvae 100 both
python 1_Experiment.py sanity-3-non-lin linear cchvae 100 both
python 1_Experiment.py sanity-3-non-add linear  cchvae 100 both
python 1_Experiment.py economic linear cchvae 100 both

# CLue
python 1_Experiment.py sanity-3-lin linear Clue 100 both
python 1_Experiment.py credit linear Clue 100 both
python 1_Experiment.py sanity-3-non-lin linear Clue 100 both
python 1_Experiment.py sanity-3-non-add linear Clue 100 both
python 1_Experiment.py economic linear Clue 100 both


# Actionable Recourse
python 1_Experiment.py sanity-3-lin linear actionable_recourse 100 both
python 1_Experiment.py credit linear actionable_recourse 100 both
python 1_Experiment.py sanity-3-non-lin linear actionable_recourse 100 both
python 1_Experiment.py sanity-3-non-add linear actionable_recourse 100 both
python 1_Experiment.py economic linear actionable_recourse 100 both

# CRUDS
python 1_Experiment.py sanity-3-lin linear Cruds 100 both
python 1_Experiment.py credit linear Cruds 100 both
python 1_Experiment.py sanity-3-non-lin linear Cruds 100 both
python 1_Experiment.py sanity-3-non-add linear Cruds 100 both
python 1_Experiment.py economic linear Cruds 100 both


