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

    Parameters
    ----------
    y_true
        True binary labels using the convention ``0`` = normal and
        ``1`` = cancer-labeled sample.
    y_pred
        Predicted binary labels using the same label convention.

    Returns
    -------
    dict
        Dictionary containing the following evaluation outputs:

        - ``accuracy``: fraction of correctly classified samples
        - ``confusion_matrix``: 2x2 confusion matrix with label order ``[0, 1]``
        - ``precision``: positive-class precision
        - ``recall``: positive-class recall
        - ``f1``: positive-class F1 score

    Notes
    -----
    The confusion matrix is always computed with labels ``[0, 1]``, so the
    returned matrix remains 2x2 even if one class is missing from the
    predictions.

    The ``zero_division=0`` setting avoids undefined metric warnings when
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
