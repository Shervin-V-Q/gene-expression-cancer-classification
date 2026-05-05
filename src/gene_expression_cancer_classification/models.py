from gene_expression_cancer_classification.evaluation import evaluate_predictions


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
        **evaluation_results,
    }
