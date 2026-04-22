import pandas as pd
from sklearn.model_selection import train_test_split


def add_binary_label(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary label column to the dataframe.

    Label convention:
    - 1: cancer
    - 0: normal

    The label is inferred from the 'tissue' column.
    """
    data = data.copy()

    data["label"] = data["tissue"].apply(
        lambda x: 1 if ("carcinoma" in x.lower() or "adenocarcinoma" in x.lower()) else 0
    )

    return data


def get_high_count_tissues(data: pd.DataFrame, min_count: int = 250) -> list[str]:
    """
    Return tissue names that appear at least `min_count` times.
    """
    tissue_counts = data["tissue"].value_counts()
    high_count_tissues = tissue_counts[tissue_counts >= min_count].index.tolist()
    return high_count_tissues


def filter_by_tissues(data: pd.DataFrame, tissue_types: list[str]) -> pd.DataFrame:
    """
    Return only rows whose tissue is in `tissue_types`.
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

    First split:
    - train_temp / test

    Second split:
    - train / validation

    Stratification is applied using the 'label' column.
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
    Create predefined tissue subsets used in the project.

    Returns a dictionary of named subsets.
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
