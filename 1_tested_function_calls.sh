#! /bin/bash
python 1_Experiment.py sanity-3-lin MLP causal_recourse 10 both
python 1_Experiment.py sanity-3-lin linear causal_recourse 10 both
python 1_Experiment.py sanity-3-lin linear wachter 10 both
python 1_Experiment.py sanity-3-lin MLP wachter 10 both