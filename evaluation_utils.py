from sklearn.metrics import accuracy_score, confusion_matrix


def evaluate_predictions(y_true, y_pred):
    """
    Evaluate predictions using accuracy and confusion matrix.

    Returns a dictionary with the main evaluation results.
    """
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    return results
