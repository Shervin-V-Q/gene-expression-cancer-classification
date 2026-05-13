from gene_expression_cancer_classification.evaluation import evaluate_predictions


def train_and_evaluate_classical_model(model, x_train, y_train, x_test, y_test):
    """
    Train a classical machine-learning model and evaluate it on a test set.

    Parameters
    ----------
    model
        A scikit-learn-like estimator implementing ``fit`` and ``predict``.
    x_train
        Training feature matrix.
    y_train
        Training labels.
    x_test
        Test feature matrix.
    y_test
        Test labels.

    Returns
    -------
    dict
        Dictionary containing model predictions and evaluation metrics.

        The returned dictionary includes:

        - ``predictions``
        - ``accuracy``
        - ``confusion_matrix``
        - ``precision``
        - ``recall``
        - ``f1``

    Notes
    -----
    The input model is fitted in place, following the usual scikit-learn
    estimator behavior.

    The evaluation metrics assume binary labels using the convention
    ``0`` = normal and ``1`` = cancer-labeled sample.
    """
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    evaluation_results = evaluate_predictions(y_test, predictions)

    return {
        "predictions": predictions,
        **evaluation_results,
    }
