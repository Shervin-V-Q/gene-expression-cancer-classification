# gene-expression-cancer-classification

A machine learning project for gene expression based cancer and tissue classification.

## Project status
This repository started from an exploratory notebook and is being gradually reorganized into a cleaner and more reproducible Python project.

At the current stage, the repository includes:
- a notebook for exploratory analysis and reporting
- reusable Python modules for data preparation, evaluation, and classical model utilities
- automated tests for the extracted modules

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
