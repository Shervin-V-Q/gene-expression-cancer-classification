# Gene Expression Cancer Classification

A small scientific Python project for gene expression based cancer classification.

This repository started from an exploratory notebook and is being gradually reorganized into a cleaner, testable, and reproducible Python project.

## Project goal

The goal of this project is to build and test a small machine learning workflow for classifying cancer-related and normal tissue samples from gene expression data.

The project is educational and exploratory. It is not intended to be used as a clinical diagnostic tool.

## Current project status

At the current stage, the repository includes:

- an exploratory notebook for analysis and reporting
- an installable Python package under `src/`
- reusable modules for data preparation, evaluation, model training, and plotting
- automated tests for the extracted modules
- a small runnable example script for the classical model workflow
- a small toy gene expression CSV example for demonstrating the package workflow
- project configuration with `pyproject.toml`

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

- [`docs/dataset.md`](docs/dataset.md) — dataset availability, expected local data placement, and label creation
- [`docs/usage.md`](docs/usage.md) — installation, testing, and example usage
- [`docs/methods.md`](docs/methods.md) — methodological choices, evaluation metrics, and reproducibility notes

## Main modules

- `cli.py` — command-line interface for running reproducible example workflows
- `data_preparation.py` — utilities for binary label creation, tissue filtering, subset definition, and train/validation/test splitting
- `evaluation.py` — reusable evaluation helpers
- `models.py` — utilities for training and evaluating classical machine learning models
- `plotting.py` — plotting helper for confusion matrix heatmaps
- `run_classical_models.py` — small runnable example script

## Implemented features

- Binary label creation for cancer vs normal tissue samples
- Rule-based cancer label inference from tissue annotations
- Selection of high-count tissue categories
- Filtering datasets by selected tissue types
- Train/validation/test split with stratification
- Predefined tissue subsets for lung, colon, and kidney
- Evaluation with accuracy and confusion matrix
- Classical model evaluation with precision, recall, and F1 score
- Tested plotting utility for confusion matrix heatmaps
- Automated tests with `pytest`

## Important limitation

The binary cancer label is inferred from text in the `tissue` annotation.

A sample is labeled as cancer when its tissue annotation contains one of the configured cancer-related keywords, such as `carcinoma` or `adenocarcinoma`.

This is a rule-based labeling step and should not be interpreted as an independent clinical diagnosis.

More details about data availability, expected local data placement, and label creation are provided in [`docs/dataset.md`](docs/dataset.md).

A summary of the current methodological choices is available in [`docs/methods.md`](docs/methods.md).

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

The `requirements.txt` file installs the project in editable mode using the configuration in `pyproject.toml`.

## Running tests

Run the test suite from the project root with:

```bash
python -m pytest
```

Additional installation and usage instructions are available in [`docs/usage.md`](docs/usage.md).

The current local test suite contains tests for:

- data preparation utilities
- label creation and input validation
- evaluation utilities
- classical model training and evaluation
- plotting utilities
- the runnable example script

## Running the classical model example

A small runnable example is provided in:

```text
run_classical_models.py
```

Run it from the project root with:

```bash
gene-cancer-classify example
```

For compatibility, the example script can also be run with:

```bash
python run_classical_models.py
```

This script runs a minimal classical classification example and prints:

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
- creates binary labels from the tissue annotation
- uses columns starting with `gene_` as features
- trains a logistic regression model
- prints evaluation metrics

## Notebook

The notebook `exploratory_gene_expression_analysis.ipynb` is kept as an exploratory analysis and reporting layer.

Reusable code should live in the Python package under `src/gene_expression_cancer_classification/`, not inside the notebook.

## Development notes

This project follows a gradual refactoring process:

1. start from exploratory analysis,
2. move reusable logic into Python modules,
3. add tests for the extracted functionality,
4. make the project installable,
5. document usage and limitations clearly.

This structure is intended to make the project easier to test, reproduce, and extend.
