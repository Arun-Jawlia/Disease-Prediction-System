# Hyperparameter Tuning Report

## 1. Objective

The objective of hyperparameter tuning is to improve
the baseline classification performance by systematically
evaluating multiple hyperparameter combinations.

## 2. Tuning Methodology

- Algorithm: GridSearchCV
- Cross-validation: 5-fold StratifiedKFold
- Shuffle: True
- Random State: 42
- Primary optimization metric: F1
- Final evaluation: Held-out test set

## 3. Logistic Regression

Parameters evaluated:

- C
- solver

## 4. Decision Tree

Parameters evaluated:

- criterion
- max_depth
- min_samples_split
- min_samples_leaf

## 5. Random Forest

Parameters evaluated:

- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf
- max_features

## 6. Support Vector Machine

Parameters evaluated:

- C
- kernel
- gamma

## 7. K-Nearest Neighbors

Parameters evaluated:

- n_neighbors
- weights
- metric

## 8. Diabetes Results

Insert actual results from tuned_results.csv.

## 9. Heart Disease Results

Insert actual results from tuned_results.csv.

## 10. Liver Disease Results

Insert actual results from tuned_results.csv.

## 11. Baseline vs Tuned Comparison

Compare actual baseline and tuned results.

## 12. Best Hyperparameters

Document the best parameters identified for each model.

## 13. Observations

Discuss improvements, stability,
overfitting indicators, and metric trade-offs.

## 14. Conclusion

The tuned models will be evaluated and compared
in Phase 7 for final model selection.