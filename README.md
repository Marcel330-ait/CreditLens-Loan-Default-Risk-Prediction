# CreditLens 鈥� Loan Default Risk Prediction

CreditLens is a supervised machine learning project for predicting LendingClub loan defaults. It focuses on leakage-aware feature selection, imbalanced classification, model comparison, and interpretable credit-risk analysis.

## Project Highlights

- Started with **42,538 records and 137 raw columns** and retained **39,786 completed loans** labeled as fully paid or charged off.
- Removed all-missing fields, unusable columns, and post-loan outcome variables that could create **data leakage**.
- Prepared 30 borrower and loan features for modeling with a **75/25 stratified train-test split**.
- Compared Logistic Regression, class-weighted Logistic Regression, Random Forest, and Gradient Boosting.
- Increased default recall to **64%** with class-weighted Logistic Regression.
- Achieved the highest saved **ROC-AUC of 0.707** with Gradient Boosting.

## Results

| Model | Default Recall | ROC-AUC | Key Observation |
| --- | ---: | ---: | --- |
| Logistic Regression | <1% | 0.698 | High accuracy but missed almost all defaults |
| Class-weighted Logistic Regression | **64%** | 0.697 | Strongest default detection at the standard threshold |
| Random Forest | <1% | 0.694 | Predicted almost every loan as fully paid |
| Gradient Boosting | <1% | **0.707** | Best ranking performance, but threshold recall remained low |

Although several models reached approximately 86% accuracy, the confusion matrices showed that accuracy was misleading because only about 14% of the modeled loans were defaults. The class-weighted model made the tradeoff visible: substantially higher default recall with more false positives.

## Key Risk Indicators

Gradient Boosting feature importance identified the following as leading predictors in the fitted model:

- Interest rate
- Annual income
- Loan sub-grade
- Loan term
- Loan purpose and revolving utilization

These rankings explain the model's predictive behavior and should not be interpreted as causal effects.

## Workflow

1. Audit the raw LendingClub schema and missingness.
2. Restrict the target to fully paid and charged-off loans.
3. Remove post-origination fields and other leakage-prone variables.
4. Clean percentage, term, employment-length, and credit-history fields.
5. Encode categorical attributes and split the data with stratification.
6. Train baseline and ensemble classifiers.
7. Compare confusion matrices, precision, recall, F1-score, and ROC-AUC.
8. Interpret model feature importance and class-imbalance tradeoffs.

## Technologies

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Google Colab

## Repository Contents

- [`CreditLens.ipynb`](./CreditLens.ipynb) 鈥� data cleaning, leakage controls, modeling, and evaluation

## How to Run

1. Open the notebook in Google Colab or Jupyter Notebook.
2. Run the included data-download cell or place `loan-clean-version.csv` in the working directory.
3. Run all cells in order.

## Limitations

- Default probabilities are based on historical LendingClub loans and may not generalize to other lenders or economic periods.
- Categorical encoding in the current notebook can be improved with a reusable preprocessing pipeline.
- The default 0.5 decision threshold is not optimal for every lending objective; threshold selection should reflect the relative cost of missed defaults and false alarms.
- This project is educational and is not intended for real lending decisions.
