#! /bin/bash

python 1_Experiment.py sanity-3-lin MLP causal_recourse 10 both
python 1_Experiment.py sanity-3-lin linear causal_recourse 10 both
python 1_Experiment.py sanity-3-lin linear wachter 10 both
python 1_Experiment.py sanity-3-lin MLP wachter 10 both
python 1_Experiment.py sanity-3-lin linear growingspheres 10 both
python 1_Experiment.py sanity-3-lin MLP growingspheres 10 both
python 1_Experiment.py sanity-3-lin MLP cchvae 10 both
python 1_Experiment.py sanity-3-lin MLP Clue 10 both
python 1_Experiment.py sanity-3-lin MLP actionable_recourse 10 both
ython 1_Experiment.py sanity-3-lin MLP Cruds 10 both

# Focus only works with SK or XGBoost-Backend 
python 1_Experiment.py sanity-3-lin forest focus 10 both
python 1_Experiment.py sanity-3-lin forest FeatureTweak 10 both

# Other Dataset

