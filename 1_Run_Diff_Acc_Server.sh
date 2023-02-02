#! /bin/bash


# CRUDS
python 1_Experiment.py sanity-3-lin MLP Cruds 100 both
python 1_Experiment.py credit MLP Cruds 100 both
python 1_Experiment.py sanity-3-non-lin MLP Cruds 100 both
python 1_Experiment.py sanity-3-non-add MLP Cruds 100 both
python 1_Experiment.py economic MLP Cruds 100 both


# Actionable Recourse
python 1_Experiment.py sanity-3-lin MLP actionable_recourse 100 both
python 1_Experiment.py credit MLP actionable_recourse 100 both
python 1_Experiment.py sanity-3-non-lin MLP actionable_recourse 100 both
python 1_Experiment.py sanity-3-non-add MLP actionable_recourse 100 both
python 1_Experiment.py economic MLP actionable_recourse 100 both


# CLue
python 1_Experiment.py sanity-3-lin MLP Clue 100 both
python 1_Experiment.py credit MLP Clue 100 both
python 1_Experiment.py sanity-3-non-lin MLP Clue 100 both
python 1_Experiment.py sanity-3-non-add MLP Clue 100 both
python 1_Experiment.py economic MLP Clue 100 both


#CCHVAE
python 1_Experiment.py sanity-3-lin MLP cchvae 100 both
python 1_Experiment.py credit MLP cchvae 100 both
python 1_Experiment.py sanity-3-non-lin MLP cchvae 100 both
python 1_Experiment.py sanity-3-non-add MLP  cchvae 100 both
python 1_Experiment.py economic MLP cchvae 100 both