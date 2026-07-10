# Methods

This document summarizes the methodological choices used in the project.

## 1. Input table validation

The toy gene expression workflow expects a table with:

- a `tissue` column containing tissue annotations
- one or more gene-expression feature columns starting with `gene_`

The function `validate_gene_expression_table` checks these minimum structural requirements before the toy workflow continues.

This validation step is intentionally lightweight. It is designed to make the example workflow clearer and more testable without assuming that every possible external gene expression dataset has the same full structure.

## 2. Label inference

The project uses a binary cancer-vs-normal label.

The label is inferred from the text stored in the `tissue` annotation.

A sample is labeled as cancer (`1`) when the tissue annotation contains one of the configured cancer-related keywords:

- `carcinoma`
- `adenocarcinoma`

Otherwise, the sample is labeled as normal (`0`).

This rule is implemented in:

```text
src/gene_expression_cancer_classification/data_preparation.py
```

## 3. Important limitation of the labels

The binary label is rule-based.

It should not be interpreted as an independent clinical diagnosis.

This means that the model is learning from labels inferred from text annotations, not from a separate clinical gold standard.

## 4. Feature selection

Gene-expression features are selected by column name.

The helper `select_gene_feature_columns` returns columns whose names start with:

```text
gene_
```

This keeps the toy workflow explicit and reproducible. Metadata columns such as `sample_id` or `tissue` are not used as model input features.

The helper `create_feature_label_data` separates the selected gene-expression feature columns from the binary `label` column.

## 5. Class balance

The helper `summarize_class_balance` reports how many samples are available for each binary label.

This is useful because accuracy can be misleading when classes are imbalanced. Reporting class balance makes the toy workflow more transparent before model training.

## 6. Data splitting

The function `preprocess_and_split` splits a labeled dataset into:

- training set
- validation set
- test set

The split uses stratification based on the `label` column.

Stratification helps preserve the class balance across the train, validation, and test sets.

The default random seed is:

```text
42
```

Using a fixed random seed helps make the split reproducible.

## 7. Evaluation metrics

The current evaluation utilities report:

- accuracy
- confusion matrix
- precision
- recall
- F1 score

These metrics are useful for a first binary classification workflow.

However, for imbalanced biological datasets, accuracy alone may be misleading. Precision, recall, F1 score, and the confusion matrix should also be inspected.

The evaluation helper uses a fixed binary label order `[0, 1]` for the confusion matrix, where `0` represents normal samples and `1` represents cancer-labeled samples.

Undefined precision, recall, or F1 cases are handled with `zero_division=0`. This avoids warnings when a model does not predict any positive samples and makes the metric behavior explicit and testable.

## 8. Classical models

The runnable examples use simple classical machine-learning models.

The basic example trains one logistic regression model on a tiny artificial dataset.

The toy gene expression workflow uses:

- logistic regression with standard scaling
- a shallow decision tree classifier

These models are intentionally simple. They are included to demonstrate a reproducible comparison workflow rather than to claim biological or clinical performance.

## 9. Model comparison

The helper `build_default_classical_models` creates the default candidate models used in the toy workflow.

The helper `compare_classical_models` trains each candidate model on the same training data and evaluates each model on the same test data.

The returned comparison table includes:

- model name
- accuracy
- precision
- recall
- F1 score

The table is sorted by F1 score by default.

This step makes the workflow more informative than a single-model demonstration because the same evaluation criteria are applied consistently to more than one candidate model.

## 10. Automated tests

The automated tests use small artificial datasets.

This keeps the tests:

- fast
- deterministic
- independent of large external data files
- easy to run on any machine

The tests check both normal behaviour and error cases, including missing required columns, missing gene feature columns, single-class labels, exact metric values, model-comparison behaviour, command-line behaviour, and plotting output.

The full gene expression dataset is therefore not required to run the test suite.
