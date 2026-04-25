import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


def test_evaluate_predictions_returns_accuracy_and_confusion_matrix():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]

    results = evaluate_predictions(y_true, y_pred)

    assert "accuracy" in results
    assert "confusion_matrix" in results
    assert results["accuracy"] == 0.75
    assert results["confusion_matrix"].tolist() == [[2, 0], [1, 1]]
