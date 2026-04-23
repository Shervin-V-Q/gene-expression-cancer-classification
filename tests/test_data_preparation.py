import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from data_preparation import (
    add_binary_label,
    get_high_count_tissues,
    filter_by_tissues,
    define_subsets,
    preprocess_and_split,
)


def test_add_binary_label_creates_label_column():
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


def test_get_high_count_tissues_returns_expected_tissues():
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
    data = pd.DataFrame(
        {
            "tissue": ["Lung", "Colon", "Kidney"],
            "value": [1, 2, 3],
        }
    )

    with pytest.raises(ValueError, match="label"):
        preprocess_and_split(data)
