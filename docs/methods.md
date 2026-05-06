# Methods

This document summarizes the current methodological choices used in the project.

## 1. Label inference

The project currently uses a binary cancer-vs-normal label.

The label is inferred from the text stored in the `tissue` annotation.

A sample is labeled as cancer (`1`) when the tissue annotation contains one of the configured cancer-related keywords:

- `carcinoma`
- `adenocarcinoma`

Otherwise, the sample is labeled as normal (`0`).

This rule is implemented in:

```text
src/gene_expression_cancer_classification/data_preparation.py
```

## 2. Important limitation of the labels

The binary label is rule-based.

It should not be interpreted as an independent clinical diagnosis.

This means that the model is learning from labels inferred from text annotations, not from a separate clinical gold standard.

## 3. Data splitting

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

## 4. Evaluation metrics

The current evaluation utilities report:

- accuracy
- confusion matrix
- precision
- recall
- F1 score

These metrics are useful for a first classification workflow.

However, for imbalanced biological datasets, accuracy alone may be misleading. Precision, recall, F1 score, and the confusion matrix should also be inspected.

The evaluation helper uses a fixed binary label order `[0, 1]` for the confusion matrix, where `0` represents normal samples and `1` represents cancer-labeled samples.

Undefined precision, recall, or F1 cases are handled with `zero_division=0`. This avoids warnings when a model does not predict any positive samples and makes the metric behavior explicit and testable.

## 5. Classical model example

The current runnable script uses a small artificial example with logistic regression.

The purpose of this script is not to provide a final biological model.

Its purpose is to demonstrate that the package can be imported, a model can be trained, predictions can be generated, and evaluation metrics can be printed.

## 6. Automated tests

The automated tests use small artificial datasets.

This keeps the tests:

- fast
- deterministic
- independent of large external data files
- easy to run on any machine

The full gene expression dataset is therefore not required to run the test suite.
