Tried different combination of encoders on categorical variables.
Also stacked models upto two levels.

Finally, the best model was given by using:
encoding: WOEencoder on all categorical variables
imputation: kNN
stacking: logistic regression over decision-trees, catboost and random-forest models
