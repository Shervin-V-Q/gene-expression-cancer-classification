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


def run_train_example(
    input_path: str | Path = Path("examples/toy_gene_expression.csv"),
    test_size: float = 0.33,
    random_state: int = 42,
    sort_by: str = "f1",
) -> None:
    """
    Run the toy gene expression classification workflow.

    Parameters
    ----------
    input_path
        Path to the CSV file used by the toy workflow.
    test_size
        Fraction of the dataset assigned to the test split.
    random_state
        Random seed used for reproducible splitting and model construction.
    sort_by
        Metric used to sort the model comparison table.

    Returns
    -------
    None
        The function prints dataset information, class balance, single-model
        metrics, and a simple model comparison table to the terminal.

    Raises
    ------
    FileNotFoundError
        If the selected input CSV file cannot be found.
    ValueError
        If the selected dataset does not contain the expected project columns.
    ValueError
        If ``test_size`` is not between 0 and 1.

    Notes
    -----
    This command reads a small CSV file, validates its structure, creates
    rule-based binary labels from the ``tissue`` annotation, selects gene
    expression feature columns, trains a logistic regression model, compares
    a small set of classical models, and prints evaluation metrics.

    The included toy dataset is intended only to demonstrate the package
    workflow and should not be used for biological or clinical conclusions.
    """
    input_path = Path(input_path)

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found. Expected file at '{input_path}'."
        )

    data = pd.read_csv(input_path)
    validate_gene_expression_table(data)

    labeled_data = add_binary_label(data)
    class_balance = summarize_class_balance(labeled_data)
    x, y = create_feature_label_data(labeled_data)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
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

    default_models = build_default_classical_models(random_state=random_state)
    comparison = compare_classical_models(
        default_models,
        x_train,
        y_train,
        x_test,
        y_test,
        sort_by=sort_by,
    )

    print("Input file:", input_path)
    print("Number of samples:", len(labeled_data))
    print("Feature columns:", list(x.columns))
    print("Class balance:", class_balance)
    print("Test size:", test_size)
    print("Random state:", random_state)
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

    The ``train-example`` command accepts optional arguments for the input
    path, test split size, random seed, and model-comparison sorting metric.
    """
    parser = argparse.ArgumentParser(
        description="Gene expression cancer classification tools."
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    subparsers.add_parser(
        "example",
        help="Run a minimal artificial classification example.",
    )

    train_parser = subparsers.add_parser(
        "train-example",
        help="Run the toy gene expression CSV workflow.",
    )

    train_parser.add_argument(
        "--input-path",
        default="examples/toy_gene_expression.csv",
        help="Path to the input CSV file.",
    )

    train_parser.add_argument(
        "--test-size",
        type=float,
        default=0.33,
        help="Fraction of samples assigned to the test split.",
    )

    train_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for reproducible splitting and model construction.",
    )

    train_parser.add_argument(
        "--sort-by",
        choices=["accuracy", "precision", "recall", "f1"],
        default="f1",
        help="Metric used to sort the model comparison table.",
    )

    args = parser.parse_args()

    if args.command == "example":
        run_example()
    elif args.command == "train-example":
        run_train_example(
            input_path=args.input_path,
            test_size=args.test_size,
            random_state=args.random_state,
            sort_by=args.sort_by,
        )


if __name__ == "__main__":
    main()
