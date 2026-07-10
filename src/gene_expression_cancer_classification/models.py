from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from gene_expression_cancer_classification.evaluation import evaluate_predictions


def build_default_classical_models(random_state: int = 42) -> dict[str, Any]:
    """
    Build the default classical models used in the example workflow.

    Parameters
    ----------
    random_state
        Random seed used for reproducible model behaviour when supported by
        the estimator.

    Returns
    -------
    dict[str, Any]
        Dictionary mapping model names to scikit-learn-compatible estimators.

    Notes
    -----
    The returned models are intentionally simple. They are used to demonstrate
    a reproducible comparison workflow rather than to claim biological or
    clinical performance.
    """
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, random_state=random_state),
                ),
            ]
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=3,
            random_state=random_state,
        ),
    }


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


def compare_classical_models(
    models: dict[str, Any],
    x_train,
    y_train,
    x_test,
    y_test,
    sort_by: str = "f1",
) -> pd.DataFrame:
    """
    Train and compare multiple classical machine-learning models.

    Parameters
    ----------
    models
        Dictionary mapping model names to scikit-learn-compatible estimators.
    x_train
        Training feature matrix.
    y_train
        Training labels.
    x_test
        Test feature matrix.
    y_test
        Test labels.
    sort_by
        Metric used to sort the returned comparison table.

    Returns
    -------
    pandas.DataFrame
        Table containing one row per model and the main evaluation metrics.

    Raises
    ------
    ValueError
        If no models are provided.
    ValueError
        If ``sort_by`` is not one of the supported metric columns.

    Notes
    -----
    This helper makes model comparison explicit in the project workflow.
    It is more informative than training a single model because it reports
    the same metrics for each candidate model in a comparable table.
    """
    if not models:
        raise ValueError("At least one model must be provided.")

    rows = []

    for model_name, model in models.items():
        results = train_and_evaluate_classical_model(
            model,
            x_train,
            y_train,
            x_test,
            y_test,
        )

        rows.append(
            {
                "model": model_name,
                "accuracy": results["accuracy"],
                "precision": results["precision"],
                "recall": results["recall"],
                "f1": results["f1"],
            }
        )

    comparison = pd.DataFrame(rows)

    supported_sort_columns = {"accuracy", "precision", "recall", "f1"}

    if sort_by not in supported_sort_columns:
        raise ValueError(
            f"sort_by must be one of {sorted(supported_sort_columns)}."
        )

    comparison = comparison.sort_values(
        by=sort_by,
        ascending=False,
    ).reset_index(drop=True)

    return comparison
