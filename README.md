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
- project configuration with `pyproject.toml`

## Repository structure

```text
gene-expression-cancer-classification/
├── exploratory_gene_expression_analysis.ipynb
├── pyproject.toml
├── requirements.txt
├── run_classical_models.py
├── src/
│   └── gene_expression_cancer_classification/
│       ├── __init__.py
│       ├── data_preparation.py
│       ├── evaluation.py
│       ├── models.py
│       └── plotting.py
└── tests/
    ├── test_classical_models.py
    ├── test_data_preparation.py
    ├── test_evaluation_utils.py
    ├── test_plotting_utils.py
    └── test_run_classical_models.py
```

## Main modules

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
python run_classical_models.py
```

This script runs a minimal classical classification example and prints:

- predictions
- accuracy
- confusion matrix
- precision
- recall
- F1 score

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
