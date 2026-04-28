# gene-expression-cancer-classification

A machine learning project for gene expression based cancer and tissue classification.

## Project status
This repository started from an exploratory notebook and is being gradually reorganized into a cleaner and more reproducible Python project.

At the current stage, the repository includes:
- a notebook for exploratory analysis and reporting
- reusable Python modules for data preparation, evaluation, classical model utilities, and plotting
- automated tests for the extracted modules
- a runnable example script for the classical model workflow

## Repository structure
- `exploratory_gene_expression_analysis.ipynb` — main notebook used for analysis and experimentation
- `data_preparation.py` — utilities for label creation, tissue filtering, subset definition, and dataset splitting
- `evaluation_utils.py` — reusable evaluation helpers
- `classical_models.py` — utility for training and evaluating classical machine learning models
- `plotting_utils.py` — reusable plotting helper for confusion matrix heatmaps
- `run_classical_models.py` — runnable example script for the classical model pipeline
- `tests/test_run_classical_models.py` — test for the runnable classical model script
- `tests/test_data_preparation.py` — tests for data preparation functions
- `tests/test_evaluation_utils.py` — tests for evaluation utilities
- `tests/test_classical_models.py` — tests for classical model utilities
- `tests/test_plotting_utils.py` — test for the plotting utility
- `requirements.txt` — project dependencies
- `.gitignore` — ignored local and temporary files

## Current implemented features
- Binary label creation for cancer vs normal tissue samples
- Selection of high-count tissue categories
- Filtering datasets by selected tissue types
- Train/validation/test split utility
- Predefined tissue subsets for lung, colon, and kidney
- Evaluation with accuracy and confusion matrix
- Initial utility for training and evaluating classical ML models
- Tested plotting utility for confusion matrix heatmaps
- Runnable example script for the classical model workflow
- Automated tests for extracted modules

## Installation
Create and activate a virtual environment, then install the dependencies:

```bash
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
```
## Running tests

Run the test suite from the project root with:

```bash
pytest
```

## Running the classical model example

A small runnable example is provided in:

```bash
run_classical_models.py
```
You can run it from the project root with:

```bash
py run_classical_models.py
```
If `py` does not work on your system, use:
```bash
python run_classical_models.py
```

This script runs a minimal classical classification example and prints:

- predictions
- accuracy
- confusion matrix
- precision
- recall
- f1

## Testing status

The extracted modules have been tested locally with pytest.

## Goal

The goal of this project is to analyze gene expression data and build models for tissue and cancer-related classification tasks.

## Ongoing refactoring direction

The long-term goal is to progressively move reusable logic out of the notebook and into separate Python modules, while keeping the notebook as an analysis and presentation layer.



