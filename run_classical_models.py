from sklearn.linear_model import LogisticRegression

from gene_expression_cancer_classification.models import train_and_evaluate_classical_model


def main():
    x_train = [[0], [1], [2], [3]]
    y_train = [0, 0, 1, 1]

    x_test = [[1], [2]]
    y_test = [0, 1]

    model = LogisticRegression()
    results = train_and_evaluate_classical_model(
        model, x_train, y_train, x_test, y_test
    )

    print("Predictions:", results["predictions"])
    print("Accuracy:", results["accuracy"])
    print("Confusion matrix:")
    print(results["confusion_matrix"])
    print("Precision:", results["precision"])
    print("Recall:", results["recall"])
    print("F1:", results["f1"])


if __name__ == "__main__":
    main()
