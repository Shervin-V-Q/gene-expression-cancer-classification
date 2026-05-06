# Dataset

This project was developed around a gene expression dataset used for exploratory cancer-related classification.

## Data availability

The full dataset is not included in this repository.

The dataset used during development is publicly available on Zenodo:

- Dataset: `Subset of TCGA & GTEx`
- File name: `gtex_with_cancer.csv.gz`
- DOI: `10.5281/zenodo.7828660`
- Expected local path: `data/gtex_with_cancer.csv.gz`

This is intentional because gene expression datasets can be large and should not be stored directly in the GitHub repository unless they are small, public, and clearly licensed for redistribution.

## Expected data location

If a user wants to reproduce the full analysis, the dataset should be downloaded separately and placed in a local `data/` directory.

After downloading the file from Zenodo, place it in the local `data/` directory without changing the file name.

Example local structure:

```text
gene-expression-cancer-classification/
├── data/
│   └── gtex_with_cancer.csv.gz
├── src/
├── tests/
└── README.md
```

The `data/` directory is ignored by Git and should remain local.

## Toy dataset included in the repository

A very small artificial toy dataset is included at:

```text
examples/toy_gene_expression.csv
```

This file is not intended for biological or clinical conclusions.

It is only used to demonstrate the package workflow and to make the project runnable without downloading the full external dataset.

The full analysis still requires the external dataset described above.

## Label creation

The current binary cancer label is inferred from the `tissue` annotation.

A sample is labeled as cancer when the tissue annotation contains one of the configured cancer-related keywords, such as:

- `carcinoma`
- `adenocarcinoma`

This rule is implemented in:

```text
src/gene_expression_cancer_classification/data_preparation.py
```

## Important limitation

The inferred binary label is a rule-based label derived from text annotations.

It should not be interpreted as an independent clinical diagnosis, and this project should not be used as a medical diagnostic tool.

## Reproducibility note

The automated tests in this repository do not require the full dataset.

They use small artificial examples so that the test suite can run quickly and reproducibly on any machine.
