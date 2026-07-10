import pytest
from sklearn.linear_model import LogisticRegression

from gene_expression_cancer_classification.models import train_and_evaluate_classical_model


def test_train_and_evaluate_classical_model_returns_exact_results_for_simple_dataset():
    """
    Check that the classical model helper fits a logistic regression model,
    predicts the expected labels, and returns exact evaluation metrics for
    a simple linearly separable dataset.
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

    assert results["predictions"].tolist() == [0, 1]
    assert results["accuracy"] == pytest.approx(1.0)
    assert results["confusion_matrix"].tolist() == [[1, 0], [0, 1]]
    assert results["precision"] == pytest.approx(1.0)
    assert results["recall"] == pytest.approx(1.0)
    assert results["f1"] == pytest.approx(1.0)


def test_train_and_evaluate_classical_model_includes_all_expected_outputs():
    """
    Check that the model helper returns both raw predictions and the complete
    set of evaluation metrics used by the project workflow.
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

    expected_keys = {
        "predictions",
        "accuracy",
        "confusion_matrix",
        "precision",
        "recall",
        "f1",
    }

    assert set(results.keys()) == expected_keys


def test_train_and_evaluate_classical_model_fits_input_model():
    """
    Check that the helper fits the estimator passed by the caller instead of
    creating a hidden model internally.
    """
    x_train = [[0], [1], [2], [3]]
    y_train = [0, 0, 1, 1]

    x_test = [[1], [2]]
    y_test = [0, 1]

    model = LogisticRegression()

    assert not hasattr(model, "classes_")

    train_and_evaluate_classical_model(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
    )

    assert model.classes_.tolist() == [0, 1]
