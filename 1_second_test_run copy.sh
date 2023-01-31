#! /bin/bash

# DO- Calculus
python 1_Experiment.py sanity-3-non-lin linear do_calculus 100 both
python 1_Experiment.py sanity-3-non-lin MLP do_calculus 100 both
python 1_Experiment.py sanity-3-non-add linear do_calculus 100 both
python 1_Experiment.py sanity-3-non-add MLP do_calculus 100 both

# causal Recourse

python 1_Experiment.py sanity-3-non-lin MLP causal_recourse 100 both
python 1_Experiment.py sanity-3-non-lin linear causal_recourse 100 both
python 1_Experiment.py sanity-3-non-add MLP causal_recourse 100 both
python 1_Experiment.py sanity-3-non-add linear causal_recourse 100 both

# Wachter
python 1_Experiment.py sanity-3-non-lin linear wachter 100 both
python 1_Experiment.py sanity-3-non-lin MLP wachter 100 both
python 1_Experiment.py sanity-3-non-add linear wachter 100 both
python 1_Experiment.py sanity-3-non-add MLP wachter 100 both


#Spheres
python 1_Experiment.py sanity-3-non-lin linear growingspheres 100 both
python 1_Experiment.py sanity-3-non-lin MLP growingspheres 100 both
python 1_Experiment.py sanity-3-non-add linear growingspheres 100 both
python 1_Experiment.py sanity-3-non-add MLP growingspheres 100 both

#CCHVAE
python 1_Experiment.py sanity-3-non-lin linear cchvae 100 both
python 1_Experiment.py sanity-3-non-lin MLP cchvae 100 both
python 1_Experiment.py sanity-3-non-add linear cchvae 100 both
python 1_Experiment.py sanity-3-non-add MLP cchvae 100 both

# CLue

python 1_Experiment.py sanity-3-non-lin linear Clue 100 both
python 1_Experiment.py sanity-3-non-lin MLP Clue 100 both
python 1_Experiment.py sanity-3-non-add linear Clue 100 both
python 1_Experiment.py sanity-3-non-add MLP Clue 100 both

# Actionable Recourse
python 1_Experiment.py sanity-3-non-lin linear actionable_recourse 100 both
python 1_Experiment.py sanity-3-non-lin MLP actionable_recourse 100 both
python 1_Experiment.py sanity-3-non-add linear actionable_recourse 100 both
python 1_Experiment.py sanity-3-non-add MLP actionable_recourse 100 both


# CRUDS
python 1_Experiment.py sanity-3-non-lin linear Cruds 100 both
python 1_Experiment.py sanity-3-non-lin MLP Cruds 100 both
python 1_Experiment.py sanity-3-non-add linear Cruds 100 both
python 1_Experiment.py sanity-3-non-add MLP Cruds 100 both

# Focus only works with SK or XGBoost-Backend 
python 1_Experiment.py sanity-3-non-lin forest focus 100 both
python 1_Experiment.py sanity-3-non-lin forest FeatureTweak 100 both
python 1_Experiment.py sanity-3-non-add forest focus 100 both
python 1_Experiment.py sanity-3-non-add forest FeatureTweak 100 both

