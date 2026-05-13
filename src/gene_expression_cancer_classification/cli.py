import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from gene_expression_cancer_classification.data_preparation import add_binary_label
from gene_expression_cancer_classification.models import (
    train_and_evaluate_classical_model,
)


def run_example() -> None:
    """
    Run a minimal classical machine-learning example.

    Returns
    -------
    None
        The function prints predictions and evaluation metrics to the terminal.

    Notes
    -----
    This example uses a tiny artificial dataset. Its purpose is to demonstrate
    that the package can train a model, generate predictions, and report
    evaluation metrics.
    """
    x_train = [[0], [1], [2], [3]]
    y_train = [0, 0, 1, 1]

    x_test = [[1], [2]]
    y_test = [0, 1]

    model = LogisticRegression()
    results = train_and_evaluate_classical_model(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
    )

    print("Predictions:", results["predictions"])
    print("Accuracy:", results["accuracy"])
    print("Confusion matrix:")
    print(results["confusion_matrix"])
    print("Precision:", results["precision"])
    print("Recall:", results["recall"])
    print("F1:", results["f1"])


def run_train_example() -> None:
    """
    Run the toy gene expression classification workflow.

    Returns
    -------
    None
        The function prints dataset information, predictions, and evaluation
        metrics to the terminal.

    Raises
    ------
    FileNotFoundError
        If the toy dataset cannot be found at
        ``examples/toy_gene_expression.csv``.
    ValueError
        If no columns starting with ``gene_`` are found.

    Notes
    -----
    This command reads the small toy CSV file included in the repository,
    creates rule-based binary labels from the ``tissue`` annotation, uses
    columns starting with ``gene_`` as features, trains a logistic regression
    model, and prints evaluation metrics.

    The toy dataset is intended only to demonstrate the package workflow and
    should not be used for biological or clinical conclusions.
    """
    input_path = Path("examples/toy_gene_expression.csv")

    if not input_path.exists():
        raise FileNotFoundError(
            "Toy dataset not found. Expected file at "
            "'examples/toy_gene_expression.csv'."
        )

    data = pd.read_csv(input_path)

    labeled_data = add_binary_label(data)

    feature_columns = [
        column for column in labeled_data.columns if column.startswith("gene_")
    ]

    if not feature_columns:
        raise ValueError(
            "No gene expression columns found. Expected columns starting with 'gene_'."
        )

    x = labeled_data[feature_columns]
    y = labeled_data["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.33,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression()
    results = train_and_evaluate_classical_model(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
    )

    print("Input file:", input_path)
    print("Number of samples:", len(labeled_data))
    print("Feature columns:", feature_columns)
    print("Predictions:", results["predictions"])
    print("Accuracy:", results["accuracy"])
    print("Confusion matrix:")
    print(results["confusion_matrix"])
    print("Precision:", results["precision"])
    print("Recall:", results["recall"])
    print("F1:", results["f1"])


def main() -> None:
    """
    Run the command-line interface.

    Returns
    -------
    None
        The selected command is executed and results are printed to the
        terminal.

    Notes
    -----
    Available commands are:

    - ``example``: run a minimal artificial classification example
    - ``train-example``: run the toy gene expression CSV workflow
    """
    parser = argparse.ArgumentParser(
        description="Gene expression cancer classification tools."
    )

    parser.add_argument(
        "command",
        choices=["example", "train-example"],
        help="Command to run.",
    )

    args = parser.parse_args()

    if args.command == "example":
        run_example()
    elif args.command == "train-example":
        run_train_example()


if __name__ == "__main__":
    main()
