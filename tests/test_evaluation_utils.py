import pytest

from gene_expression_cancer_classification.evaluation import evaluate_predictions


def test_evaluate_predictions_returns_expected_metrics():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]

    results = evaluate_predictions(y_true, y_pred)

    assert "accuracy" in results
    assert "confusion_matrix" in results
    assert "precision" in results
    assert "recall" in results
    assert "f1" in results

    assert results["accuracy"] == pytest.approx(0.75)
    assert results["confusion_matrix"].shape == (2, 2)
    assert results["confusion_matrix"].tolist() == [[2, 0], [1, 1]]
    assert results["precision"] == pytest.approx(1.0)
    assert results["recall"] == pytest.approx(0.5)
    assert results["f1"] == pytest.approx(2 / 3)


def test_evaluate_predictions_handles_no_positive_predictions():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 0, 0]

    results = evaluate_predictions(y_true, y_pred)

    assert results["confusion_matrix"].shape == (2, 2)
    assert results["confusion_matrix"].tolist() == [[2, 0], [2, 0]]
    assert results["precision"] == pytest.approx(0.0)
    assert results["recall"] == pytest.approx(0.0)
    assert results["f1"] == pytest.approx(0.0)


def test_evaluate_predictions_returns_2x2_confusion_matrix_for_single_class_input():
    y_true = [0, 0, 0]
    y_pred = [0, 0, 0]

    results = evaluate_predictions(y_true, y_pred)

    assert results["confusion_matrix"].shape == (2, 2)
    assert results["confusion_matrix"].tolist() == [[3, 0], [0, 0]]
