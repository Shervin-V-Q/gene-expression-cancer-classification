import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from gene_expression_cancer_classification.data_preparation import (
    add_binary_label,
    create_feature_label_data,
    summarize_class_balance,
    validate_gene_expression_table,
)
from gene_expression_cancer_classification.models import (
    build_default_classical_models,
    compare_classical_models,
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
        The function prints dataset information, class balance, single-model
        metrics, and a simple model comparison table to the terminal.

    Raises
    ------
    FileNotFoundError
        If the toy dataset cannot be found at
        ``examples/toy_gene_expression.csv``.
    ValueError
        If the toy dataset does not contain the expected project columns.

    Notes
    -----
    This command reads the small toy CSV file included in the repository,
    validates its structure, creates rule-based binary labels from the
    ``tissue`` annotation, selects gene expression feature columns, trains
    a logistic regression model, compares a small set of classical models,
    and prints evaluation metrics.

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
    validate_gene_expression_table(data)

    labeled_data = add_binary_label(data)
    class_balance = summarize_class_balance(labeled_data)
    x, y = create_feature_label_data(labeled_data)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.33,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(max_iter=1000)
    results = train_and_evaluate_classical_model(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
    )

    default_models = build_default_classical_models(random_state=42)
    comparison = compare_classical_models(
        default_models,
        x_train,
        y_train,
        x_test,
        y_test,
    )

    print("Input file:", input_path)
    print("Number of samples:", len(labeled_data))
    print("Feature columns:", list(x.columns))
    print("Class balance:", class_balance)
    print("Predictions:", results["predictions"])
    print("Accuracy:", results["accuracy"])
    print("Confusion matrix:")
    print(results["confusion_matrix"])
    print("Precision:", results["precision"])
    print("Recall:", results["recall"])
    print("F1:", results["f1"])
    print("Model comparison:")
    print(comparison.to_string(index=False))


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
