# Semantic Meaningfulness: Evaluating Counterfactual Approaches for Real World Plausibility and Feasibility

This repository contains the code for our paper "SSemantic Meaningfulness: Evaluating Counterfactual Approaches for Real World Plausibility and Feasibility". We benchmark the semantic meaningfulness of known Counterfactual Approaches. 
The implementation in here allows to replicate the experiments, and the benchmarking of (new) counterfactual approaches. For easy usage of out metric, refer to Benchmarking_Output_Causal.ipynb.

## Install :
Our Measure is based on the CARLA Framework (https://github.com/carla-recourse/CARLA) and is compatible with commit #
a8e238b. 
Be aware that CARLA currently only works with Python 3.7..
Install everything with: 
```
pip install .
```
Note that the most recent version tested is commit #a8e238b. In case anything is not working with the current version of carla, please roll back to this commit.

## Usage: 
This branch contains the experiment code and the results.
For Usage refer to [master](https://github.com/JHoelli/Semantic-Meaningfulness/tree/master).


Call an Experiment: 

```
    python 1_Experiments.py [dataset] [classifier] [cf] [#samples] both
```
Datasets available: 
    - credit
    - sanity-3-lin
    - sanity-3-non-lin
    - sanity-3-non-add
    - economic
    - linear
Dataset sanity-3-lin was already availabe in CARLA; The remaining Datasets and SCM with Output for the Semantic Meaningfull Measure were added here : ./CARLA/data/load_scm/scm. We indent to make a pull request to CARLA afte acceptance.

Classifiers: 
    - MLP
    - linear
    - forest

cf: 
    - cruds
    - FeatureTweak (only works with forest)
    - Dice
    - Clue
    - actionable_recourse
    - cchvae
    - focus (only works with forest)
    - growingspheres
    - causal_recourse (stuck for Credit!)
    - wachter

## Replicate Results

Run Preliminary Experiment Version: 
```
sh 1_Run_Full.sh 
```
If you want to run via more samples, just change 10 to smth else. Currently runs quiete fast. 

Aggregate Data: 
```
python 2_Summarize_Experiment_Results.py 
```
Checkout Plot and Visualization of the current reults in: ./Results/Analyzing.ipynb
