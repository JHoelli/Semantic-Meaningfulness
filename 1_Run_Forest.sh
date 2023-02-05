#! /bin/bash

# FOREST
python 1_Experiment.py sanity-3-lin forest focus 250 both
python 1_Experiment.py sanity-3-lin forest FeatureTweak 250 both

python 1_Experiment.py sanity-3-non-lin forest focus 250 both
python 1_Experiment.py sanity-3-non-lin forest FeatureTweak 250 both

python 1_Experiment.py sanity-3-non-add forest focus 250 both
python 1_Experiment.py sanity-3-non-add forest FeatureTweak 250 both

python 1_Experiment.py credit forest focus 250 both
python 1_Experiment.py credit forest FeatureTweak 250 both

python 1_Experiment.py economic forest focus 250 both
python 1_Experiment.py economic forest FeatureTweak 250 both
