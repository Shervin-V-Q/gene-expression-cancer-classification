from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_predictions(y_true, y_pred):
    """
    Evaluate binary classification predictions.

    Label convention:
    - 0: normal
    - 1: cancer

    Returns
    -------
    dict
        Dictionary containing accuracy, confusion matrix, precision,
        recall, and F1 score.

    Notes
    -----
    The confusion matrix is always computed with labels [0, 1],
    so the returned matrix is always 2x2 even if one class is missing
    from the predictions.

    The zero_division=0 setting avoids undefined metric warnings when
    the model does not predict any positive samples.
    """
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    return results
