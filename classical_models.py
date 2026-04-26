from sklearn.metrics import precision_score, recall_score, f1_score

from evaluation_utils import evaluate_predictions


def train_and_evaluate_classical_model(model, x_train, y_train, x_test, y_test):
    """
    Train a classical ML model and evaluate it on the test set.

    Returns a dictionary containing predictions and evaluation metrics.
    """
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    evaluation_results = evaluate_predictions(y_test, predictions)

    return {
        "predictions": predictions,
        "accuracy": evaluation_results["accuracy"],
        "confusion_matrix": evaluation_results["confusion_matrix"],
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
    }
