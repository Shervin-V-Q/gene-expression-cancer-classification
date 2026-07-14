# Usage

This document describes how to install the project, run the automated tests, and execute the example workflows.

## 1. Create a virtual environment

From the project root, create a virtual environment.

On Windows:

```bash
py -m venv .venv
.venv\Scripts\activate
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

## 2. Install the project

Install the project and its development dependencies with:

```bash
python -m pip install -r requirements.txt
```

The project dependencies are defined in `pyproject.toml`.

The `requirements.txt` file installs the local package in editable mode, including the development dependencies needed for testing.

Editable installation means that changes to the source files under `src/` are immediately available without reinstalling the package.

## 3. Run the automated tests

Run the test suite from the project root:

```bash
python -m pytest
```

A successful run should report all tests as passed.

The tests use small artificial examples and do not require the full external gene expression dataset.

The tests cover data validation, label creation, feature selection, class-balance summarisation, model evaluation, model-comparison behaviour, command-line execution, configurable CLI options, plotting behaviour, and expected error cases.

## 4. Run the classical model example

A small example command is provided after installing the project:

```bash
gene-cancer-classify example
```

For compatibility, the example script can also be run with:

```bash
python run_classical_models.py
```

The command trains a minimal logistic regression model on a tiny artificial dataset and prints:

- predictions
- accuracy
- confusion matrix
- precision
- recall
- F1 score

## 5. Run the toy gene expression workflow

The repository includes a small toy gene expression dataset:

```text
examples/toy_gene_expression.csv
```

This toy dataset is intentionally small and is used only to demonstrate the package workflow without requiring the full external dataset. It is not intended for biological or clinical conclusions.

Run the toy workflow with the default options:

```bash
gene-cancer-classify train-example
```

This command:

- loads the toy CSV file
- validates that the expected `tissue` column and `gene_` feature columns are present
- creates binary labels from the tissue annotation
- selects columns starting with `gene_` as model features
- reports class balance
- trains a logistic regression model
- compares default classical models using the same evaluation metrics
- prints evaluation metrics and a model comparison table

The output includes dataset information, selected feature columns, class balance, test split size, random seed, predictions, evaluation results, and the model comparison table.

## 6. Configurable CLI options

The `train-example` command accepts optional arguments so that the workflow can be run without editing the source code.

Example:

```bash
gene-cancer-classify train-example --input-path examples/toy_gene_expression.csv --test-size 0.33 --random-state 42 --sort-by f1
```

Available options:

- `--input-path`: path to the input CSV file
- `--test-size`: fraction of samples assigned to the test split
- `--random-state`: random seed used for reproducible splitting and model construction
- `--sort-by`: metric used to sort the model comparison table

The `--sort-by` option accepts:

```text
accuracy
precision
recall
f1
```

For example, to sort the model comparison table by accuracy:

```bash
gene-cancer-classify train-example --sort-by accuracy
```

Invalid test split sizes, such as `--test-size 1.0`, are rejected with a clear error.

## 7. Notes

The full gene expression dataset is not required for installation, testing, or running the toy example.

The full dataset is only needed for reproducing the exploratory analysis described in the notebook and dataset documentation.
