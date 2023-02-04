
#! /bin/bash

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