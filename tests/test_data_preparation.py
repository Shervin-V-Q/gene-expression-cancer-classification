import pandas as pd
import pytest

from gene_expression_cancer_classification.data_preparation import (
    add_binary_label,
    define_subsets,
    filter_by_tissues,
    get_high_count_tissues,
    preprocess_and_split,
)


def test_add_binary_label_creates_label_column():
    """
    Test that rule-based label inference creates the expected binary
    cancer labels from tissue annotations.
    """
    data = pd.DataFrame(
        {
            "tissue": [
                "Lung - Adenocarcinoma",
                "Lung",
                "Colon - Mucinous adenocarcinoma",
                "Kidney",
            ]
        }
    )

    result = add_binary_label(data)

    assert "label" in result.columns
    assert result["label"].tolist() == [1, 0, 1, 0]


def test_add_binary_label_does_not_modify_input_dataframe():
    """
    Test that label creation returns a copy and does not modify
    the input dataframe in place.
    """
    data = pd.DataFrame({"tissue": ["Lung - Adenocarcinoma", "Lung"]})

    add_binary_label(data)

    assert "label" not in data.columns


def test_add_binary_label_raises_error_when_tissue_column_is_missing():
    """
    Test that label creation raises a clear error when the tissue
    annotation column is missing.
    """
    data = pd.DataFrame({"wrong_column": ["Lung - Adenocarcinoma"]})

    with pytest.raises(ValueError, match="tissue"):
        add_binary_label(data)


def test_get_high_count_tissues_returns_expected_tissues():
    """
    Test that tissue categories meeting the minimum count threshold
    are returned in frequency order.
    """
    data = pd.DataFrame(
        {
            "tissue": [
                "Lung",
                "Lung",
                "Lung",
                "Colon",
                "Colon",
                "Kidney",
            ]
        }
    )

    result = get_high_count_tissues(data, min_count=2)

    assert result == ["Lung", "Colon"]


def test_filter_by_tissues_keeps_only_requested_tissues():
    """
    Test that filtering keeps only rows whose tissue annotation is
    included in the requested tissue list.
    """
    data = pd.DataFrame(
        {
            "tissue": ["Lung", "Colon", "Kidney", "Lung"],
            "value": [1, 2, 3, 4],
        }
    )

    result = filter_by_tissues(data, ["Lung", "Kidney"])

    assert result["tissue"].tolist() == ["Lung", "Kidney", "Lung"]
    assert result["value"].tolist() == [1, 3, 4]


def test_define_subsets_returns_expected_keys():
    """
    Test that predefined exploratory tissue subsets are returned
    with the expected dictionary keys.
    """
    data = pd.DataFrame(
        {
            "tissue": [
                "Lung - Non-small cell carcinoma",
                "Lung - Adenocarcinoma",
                "Lung",
                "Colon - Adenocarcinoma",
                "Colon - Sigmoid",
                "Kidney - Clear cell renal cell carcinoma",
                "Kidney",
            ]
        }
    )

    subsets = define_subsets(data)

    assert "lung" in subsets
    assert "colon" in subsets
    assert "kidney" in subsets


def test_define_subsets_filters_lung_subset_correctly():
    """
    Test that the predefined lung subset keeps the expected lung-related
    normal and cancer tissue annotations.
    """
    data = pd.DataFrame(
        {
            "tissue": [
                "Lung - Non-small cell carcinoma",
                "Lung - Adenocarcinoma",
                "Lung",
                "Colon - Adenocarcinoma",
                "Kidney",
            ]
        }
    )

    subsets = define_subsets(data)

    assert subsets["lung"]["tissue"].tolist() == [
        "Lung - Non-small cell carcinoma",
        "Lung - Adenocarcinoma",
        "Lung",
    ]


def test_preprocess_and_split_raises_error_when_label_is_missing():
    """
    Test that preprocessing raises a clear error when the binary label
    column is missing.
    """
    data = pd.DataFrame(
        {
            "tissue": ["Lung", "Colon", "Kidney"],
            "value": [1, 2, 3],
        }
    )

    with pytest.raises(ValueError, match="label"):
        preprocess_and_split(data)
