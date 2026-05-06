import argparse

from sklearn.linear_model import LogisticRegression

from gene_expression_cancer_classification.models import (
    train_and_evaluate_classical_model,
)


def run_example() -> None:
    """Run a small reproducible classical model example."""
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


def main() -> None:
    """Command-line interface for the project."""
    parser = argparse.ArgumentParser(
        description="Gene expression cancer classification tools."
    )

    parser.add_argument(
        "command",
        choices=["example"],
        help="Command to run.",
    )

    args = parser.parse_args()

    if args.command == "example":
        run_example()


if __name__ == "__main__":
    main()
