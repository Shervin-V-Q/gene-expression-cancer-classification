from sklearn.linear_model import LogisticRegression

from gene_expression_cancer_classification.models import train_and_evaluate_classical_model


def test_train_and_evaluate_classical_model_returns_expected_keys():
    x_train = [[0], [1], [2], [3]]
    y_train = [0, 0, 1, 1]

    x_test = [[1], [2]]
    y_test = [0, 1]

    model = LogisticRegression()
    results = train_and_evaluate_classical_model(
        model, x_train, y_train, x_test, y_test
    )

    assert "predictions" in results
    assert "accuracy" in results
    assert "confusion_matrix" in results


def test_train_and_evaluate_classical_model_returns_prediction_for_each_test_sample():
    x_train = [[0], [1], [2], [3]]
    y_train = [0, 0, 1, 1]

    x_test = [[1], [2]]
    y_test = [0, 1]

    model = LogisticRegression()
    results = train_and_evaluate_classical_model(
        model, x_train, y_train, x_test, y_test
    )

    assert len(results["predictions"]) == len(y_test)


def test_train_and_evaluate_classical_model_returns_extended_metrics():
    x_train = [[0], [1], [2], [3]]
    y_train = [0, 0, 1, 1]

    x_test = [[1], [2]]
    y_test = [0, 1]

    model = LogisticRegression()
    results = train_and_evaluate_classical_model(
        model, x_train, y_train, x_test, y_test
    )

    assert "precision" in results
    assert "recall" in results
    assert "f1" in results
