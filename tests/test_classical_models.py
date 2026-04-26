import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from sklearn.linear_model import LogisticRegression

from classical_models import train_and_evaluate_classical_model


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