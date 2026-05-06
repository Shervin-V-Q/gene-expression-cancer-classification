# Usage

This document describes how to install the project, run the automated tests, and execute the small classical model example.

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

The `requirements.txt` file installs the package in editable mode using the project configuration in `pyproject.toml`.

Editable installation means that changes to the source files under `src/` are immediately available without reinstalling the package.

## 3. Run the automated tests

Run the test suite from the project root:

```bash
python -m pytest
```

A successful run should report all tests as passed.

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

## 5. Notes

The automated tests do not require the full gene expression dataset.

They use small artificial examples so that the test suite can run quickly and reproducibly on any machine.
