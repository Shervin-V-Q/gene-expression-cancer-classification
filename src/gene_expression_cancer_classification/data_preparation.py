import pandas as pd
from sklearn.model_selection import train_test_split

CANCER_KEYWORDS = (
    "carcinoma",
    "adenocarcinoma",
)


def add_binary_label(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary cancer label inferred from the tissue annotation.

    Parameters
    ----------
    data
        Input dataframe containing a ``tissue`` column.

    Returns
    -------
    pandas.DataFrame
        Copy of the input dataframe with an additional ``label`` column.

    Raises
    ------
    ValueError
        If the input dataframe does not contain a ``tissue`` column.

    Notes
    -----
    Label convention:

    - ``1``: cancer-labeled sample
    - ``0``: normal-labeled sample

    This is a rule-based label inference step. A sample is labeled as cancer
    when its tissue annotation contains one of the keywords listed in
    ``CANCER_KEYWORDS``.

    The inferred label should not be interpreted as an independent clinical
    diagnosis.
    """
    if "tissue" not in data.columns:
        raise ValueError("Input dataframe must contain a 'tissue' column.")

    labeled_data = data.copy()

    labeled_data["label"] = labeled_data["tissue"].apply(
        lambda tissue_name: int(
            any(keyword in tissue_name.lower() for keyword in CANCER_KEYWORDS)
        )
    )

    return labeled_data


def get_high_count_tissues(data: pd.DataFrame, min_count: int = 250) -> list[str]:
    """
    Return tissue names that appear at least ``min_count`` times.

    Parameters
    ----------
    data
        Input dataframe containing a ``tissue`` column.
    min_count
        Minimum number of occurrences required for a tissue type to be returned.

    Returns
    -------
    list[str]
        Tissue names whose frequency is greater than or equal to ``min_count``.

    Notes
    -----
    This helper is used to identify tissue categories with enough samples for
    exploratory analysis.
    """
    tissue_counts = data["tissue"].value_counts()
    high_count_tissues = tissue_counts[tissue_counts >= min_count].index.tolist()
    return high_count_tissues


def filter_by_tissues(data: pd.DataFrame, tissue_types: list[str]) -> pd.DataFrame:
    """
    Return rows whose tissue annotation is included in ``tissue_types``.

    Parameters
    ----------
    data
        Input dataframe containing a ``tissue`` column.
    tissue_types
        Tissue annotations to keep.

    Returns
    -------
    pandas.DataFrame
        Filtered copy of the input dataframe.

    Notes
    -----
    The input dataframe is not modified in place.
    """
    filtered_data = data[data["tissue"].isin(tissue_types)].copy()
    return filtered_data


def preprocess_and_split(
    data: pd.DataFrame,
    test_size: float = 0.2,
    validate_size: float = 0.3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a binary-labeled dataset into train, validation, and test sets.

    Parameters
    ----------
    data
        Input dataframe containing a binary ``label`` column.
    test_size
        Fraction of the full dataset assigned to the test split.
    validate_size
        Fraction of the temporary training set assigned to validation.
    random_state
        Random seed used by scikit-learn's ``train_test_split``.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Train, validation, and test dataframes.

    Raises
    ------
    ValueError
        If the input dataframe does not contain a ``label`` column.
    ValueError
        If the ``label`` column does not contain exactly two classes.

    Notes
    -----
    Stratification is applied using the ``label`` column.

    The split is performed in two steps:

    1. split the full dataset into temporary training data and test data;
    2. split the temporary training data into final training and validation data.
    """
    if "label" not in data.columns:
        raise ValueError("Input dataframe must contain a 'label' column.")

    if data["label"].nunique() != 2:
        raise ValueError("Label column must contain exactly two classes: 0 and 1.")

    train_temp, test = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data["label"],
    )

    train, validate = train_test_split(
        train_temp,
        test_size=validate_size,
        random_state=random_state,
        stratify=train_temp["label"],
    )

    return train, validate, test


def define_subsets(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Create predefined tissue subsets used in the exploratory analysis.

    Parameters
    ----------
    data
        Input dataframe containing a ``tissue`` column.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Dictionary with the keys ``lung``, ``colon``, and ``kidney``.

    Notes
    -----
    The predefined subsets are project-specific and are intended for the
    exploratory workflow. They should not be interpreted as a general-purpose
    biomedical ontology.
    """
    subsets = {
        "lung": filter_by_tissues(
            data,
            [
                "Lung - Non-small cell carcinoma",
                "Lung - Adenocarcinoma",
                "Lung",
            ],
        ),
        "colon": filter_by_tissues(
            data,
            [
                "Colon - Adenocarcinoma",
                "Colon - Mucinous adenocarcinoma",
                "Colon - Serrated adenocarcinoma",
                "Colon - Sigmoid",
                "Colon - Transverse",
            ],
        ),
        "kidney": filter_by_tissues(
            data,
            [
                "Kidney - Clear cell renal cell carcinoma",
                "Kidney - Papillary renal cell carcinoma",
                "Kidney",
            ],
        ),
    }

    return subsets
