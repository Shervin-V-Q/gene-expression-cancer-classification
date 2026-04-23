# gene-expression-cancer-classification

A machine learning project for gene expression based cancer and tissue classification.

## Project status
This repository currently contains an exploratory notebook together with an initial refactored Python module for data preparation and a first test suite.

## Repository contents
- `exploratory_gene_expression_analysis.ipynb`: exploratory notebook version of the project
- `data_preparation.py`: reusable data preparation utilities
- `tests/test_data_preparation.py`: initial automated tests for the data preparation module
- `requirements.txt`: project dependencies
- `.gitignore`: ignored local and temporary files

## Current implemented features
- Binary label creation for cancer vs normal tissue samples
- Selection of high-count tissue categories
- Filtering datasets by selected tissue types
- Train/validation/test split utility
- Predefined tissue subsets for lung, colon, and kidney
- Initial automated tests for core data preparation functions

## Installation
Create and activate a virtual environment, then install the dependencies:

```bash
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
