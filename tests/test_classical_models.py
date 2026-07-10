import pytest
from sklearn.linear_model import LogisticRegression

from gene_expression_cancer_classification.models import (
    build_default_classical_models,
    compare_classical_models,
    train_and_evaluate_classical_model,
)


def test_build_default_classical_models_returns_expected_models():
    """
    Check that the default model builder returns the named estimators used
    by the toy model-comparison workflow.
    """
    models = build_default_classical_models(random_state=42)

    assert set(models.keys()) == {"logistic_regression", "decision_tree"}
    assert hasattr(models["logistic_regression"], "fit")
    assert hasattr(models["logistic_regression"], "predict")
    assert hasattr(models["decision_tree"], "fit")
    assert hasattr(models["decision_tree"], "predict")


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


def test_compare_classical_models_returns_sorted_metric_table():
    """
    Check that model comparison evaluates every candidate model and returns
    a metric table sorted by the selected score.
    """
    x_train = [[0], [1], [2], [3]]
    y_train = [0, 0, 1, 1]

    x_test = [[1], [2]]
    y_test = [0, 1]

    models = build_default_classical_models(random_state=42)
    comparison = compare_classical_models(
        models,
        x_train,
        y_train,
        x_test,
        y_test,
        sort_by="f1",
    )

    assert comparison["model"].tolist() == ["logistic_regression", "decision_tree"]
    assert comparison["accuracy"].tolist() == [1.0, 1.0]
    assert comparison["precision"].tolist() == [1.0, 1.0]
    assert comparison["recall"].tolist() == [1.0, 1.0]
    assert comparison["f1"].tolist() == [1.0, 1.0]


def test_compare_classical_models_rejects_empty_model_dictionary():
    """
    Check that model comparison fails clearly when no candidate models are
    provided.
    """
    x_train = [[0], [1], [2], [3]]
    y_train = [0, 0, 1, 1]
    x_test = [[1], [2]]
    y_test = [0, 1]

    with pytest.raises(ValueError, match="At least one model"):
        compare_classical_models({}, x_train, y_train, x_test, y_test)


def test_compare_classical_models_rejects_unknown_sort_metric():
    """
    Check that model comparison validates the requested sorting metric
    instead of silently returning a misleading ordering.
    """
    x_train = [[0], [1], [2], [3]]
    y_train = [0, 0, 1, 1]
    x_test = [[1], [2]]
    y_test = [0, 1]

    models = build_default_classical_models(random_state=42)

    with pytest.raises(ValueError, match="sort_by"):
        compare_classical_models(
            models,
            x_train,
            y_train,
            x_test,
            y_test,
            sort_by="unsupported_metric",
        )
