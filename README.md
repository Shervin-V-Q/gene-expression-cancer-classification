# Gene Expression Cancer Classification

A Python project for an educational gene-expression cancer classification workflow.

This repository provides a reproducible workflow for preparing gene expression data, creating rule-based binary cancer labels from tissue annotations, selecting gene-expression feature columns, comparing simple classical machine-learning models, and evaluating classification performance.

The project is educational and exploratory. It is not intended to be used as a clinical diagnostic tool.

## Project features

The repository includes:

- an exploratory notebook for analysis and reporting
- an installable Python package under `src/`
- reusable modules for data validation, label creation, feature selection, class-balance inspection, model training, model comparison, evaluation, plotting, and command-line execution
- automated tests with `pytest`
- a command-line interface for running reproducible example workflows
- a small toy gene expression CSV example for demonstrating the package workflow
- project configuration with `pyproject.toml`
- additional documentation in the `docs/` directory

## Repository structure

```text
gene-expression-cancer-classification/
├── docs/
│   ├── dataset.md
│   ├── methods.md
│   └── usage.md
├── examples/
│   └── toy_gene_expression.csv
├── exploratory_gene_expression_analysis.ipynb
├── pyproject.toml
├── requirements.txt
├── run_classical_models.py
├── src/
│   └── gene_expression_cancer_classification/
│       ├── __init__.py
│       ├── cli.py
│       ├── data_preparation.py
│       ├── evaluation.py
│       ├── models.py
│       └── plotting.py
└── tests/
    ├── test_classical_models.py
    ├── test_cli.py
    ├── test_data_preparation.py
    ├── test_evaluation_utils.py
    ├── test_plotting_utils.py
    └── test_run_classical_models.py
```

## Documentation

Additional documentation is available in the `docs/` directory:

- [`docs/dataset.md`](docs/dataset.md) — dataset availability, expected local data placement, toy dataset information, and label creation
- [`docs/usage.md`](docs/usage.md) — installation, testing, and command-line usage
- [`docs/methods.md`](docs/methods.md) — methodological choices, evaluation metrics, model comparison, and reproducibility notes

## Main modules

- `data_preparation.py` — utilities for validating gene-expression tables, creating binary labels, selecting gene feature columns, creating feature-label data, checking class balance, filtering tissues, defining tissue subsets, and splitting data
- `evaluation.py` — reusable evaluation helpers for binary classification metrics
- `models.py` — helpers for building default classical models, fitting and evaluating models, and comparing multiple models in a metric table
- `plotting.py` — plotting helper for confusion matrix heatmaps
- `cli.py` — command-line interface for running reproducible example workflows
- `run_classical_models.py` — compatibility script for running the small classical model example

## Important limitation

The binary cancer label is inferred from text in the `tissue` annotation. A sample is labeled as cancer when its tissue annotation contains one of the configured cancer-related keywords, such as `carcinoma` or `adenocarcinoma`.

This rule-based labeling step should not be interpreted as an independent clinical diagnosis. More details are provided in [`docs/dataset.md`](docs/dataset.md) and [`docs/methods.md`](docs/methods.md).

## Installation

From the project root, create and activate a virtual environment.

On Windows:

```bash
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The project dependencies are defined in `pyproject.toml`. The `requirements.txt` file installs the local project in editable mode together with the development dependencies.

## Running tests

Run the test suite from the project root with:

```bash
python -m pytest
```

The automated tests use small artificial examples and do not require the full external gene expression dataset.

The tests cover:

- rule-based label creation
- gene-expression table validation
- gene feature column selection
- feature-label data creation
- class-balance summarisation
- tissue filtering and predefined subset creation
- train, validation, and test splitting checks
- exact binary classification metrics
- classical model training and model-comparison behaviour
- command-line interface behaviour
- plotting helper behaviour
- compatibility script execution

## Running the classical model example

Run the small classical model example with:

```bash
gene-cancer-classify example
```

For compatibility, the example script can also be run with:

```bash
python run_classical_models.py
```

This example trains a minimal logistic regression model on a tiny artificial dataset and prints:

- predictions
- accuracy
- confusion matrix
- precision
- recall
- F1 score

## Running the toy gene expression example

A small toy gene expression dataset is provided in:

```text
examples/toy_gene_expression.csv
```

This file is intentionally small and is used only to demonstrate the package workflow without requiring the full external dataset. It is not intended for biological or clinical conclusions.

Run the toy workflow with:

```bash
gene-cancer-classify train-example
```

This command:

- loads the toy CSV file
- validates that the expected tissue and gene-expression columns are present
- creates binary labels from the tissue annotation
- selects columns starting with `gene_` as features
- reports class balance
- trains a logistic regression model
- compares default classical models using the same evaluation metrics
- prints evaluation metrics and a model comparison table

## Notebook

The notebook `exploratory_gene_expression_analysis.ipynb` is kept as an exploratory analysis and reporting layer.

Reusable code is stored in the Python package under `src/gene_expression_cancer_classification/`.

## Use of assistance

Language and documentation drafting tools were used to improve clarity and structure. The project code, design decisions, tests, command-line workflows, and final validation were reviewed and executed by the author.
