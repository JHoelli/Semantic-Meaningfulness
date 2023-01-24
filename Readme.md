
Call an Experiment: 

```
    python 1_Experiments.py [dataset] [classifier] [cf] [#samples] both
```
Datasets available: 
    - credit
    - sanity-3-lin

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

Run Preliminary Experiment Version: 
```
sh 1_first_test_run.sh 
```
If you want to run via more samples, just change 10 to smth else. Currently runs quiete fast. 

Aggregate Data: 
```
python 2_Summarize_Experiment_Results.py 
```
Checkout Plot and Visualization of the current reults in: ./Results/Analyzing.ipynb

TODOS: 
- [ ] How to cope with huge number of Parameters of CF Methods --> Preposition: set to original paper parameter
- [ ] What else to logg for evaluation ? Model accuracies and confusion matrix?
- [ ] Why are sematic and relation ship quiete close together ? --> can we conclude that relationship checks are a sufficient approx.? --> How would could we design an experiment?