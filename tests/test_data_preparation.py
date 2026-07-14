import pandas as pd
import pytest

from gene_expression_cancer_classification.data_preparation import (
    add_binary_label,
    create_feature_label_data,
    define_subsets,
    filter_by_tissues,
    get_high_count_tissues,
    preprocess_and_split,
    select_gene_feature_columns,
    summarize_class_balance,
    validate_gene_expression_table,
)


def test_validate_gene_expression_table_accepts_expected_structure():
    """
    Check that a table with a tissue column and gene expression columns
    satisfies the minimum structure required by the toy workflow.
    """
    data = pd.DataFrame(
        {
            "tissue": ["Lung", "Lung - Adenocarcinoma"],
            "gene_1": [0.1, 2.1],
            "gene_2": [0.2, 2.2],
        }
    )

    validate_gene_expression_table(data)


def test_validate_gene_expression_table_rejects_missing_tissue_column():
    """
    Check that validation fails early when the tissue annotation column
    required for rule-based label creation is missing.
    """
    data = pd.DataFrame(
        {
            "gene_1": [0.1, 2.1],
            "gene_2": [0.2, 2.2],
        }
    )

    with pytest.raises(ValueError, match="tissue"):
        validate_gene_expression_table(data)


def test_validate_gene_expression_table_rejects_missing_gene_columns():
    """
    Check that validation fails when no feature columns using the project
    gene-expression naming convention are available.
    """
    data = pd.DataFrame(
        {
            "tissue": ["Lung", "Lung - Adenocarcinoma"],
            "feature_1": [0.1, 2.1],
        }
    )

    with pytest.raises(ValueError, match="gene_"):
        validate_gene_expression_table(data)


def test_select_gene_feature_columns_returns_only_gene_prefixed_columns():
    """
    Check that feature selection keeps only columns that follow the
    gene-expression prefix convention and preserves their order.
    """
    data = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "tissue": ["Lung", "Lung - Adenocarcinoma"],
            "gene_1": [0.1, 2.1],
            "other_measure": [5.0, 6.0],
            "gene_2": [0.2, 2.2],
        }
    )

    result = select_gene_feature_columns(data)

    assert result == ["gene_1", "gene_2"]


def test_create_feature_label_data_returns_expected_features_and_labels():
    """
    Check that the modelling input helper separates gene-expression
    features from the binary label without including metadata columns.
    """
    data = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3"],
            "tissue": ["Lung", "Colon", "Kidney"],
            "gene_1": [0.1, 0.2, 0.3],
            "gene_2": [1.1, 1.2, 1.3],
            "label": [0, 1, 0],
        }
    )

    x, y = create_feature_label_data(data)

    assert list(x.columns) == ["gene_1", "gene_2"]
    assert x.values.tolist() == [[0.1, 1.1], [0.2, 1.2], [0.3, 1.3]]
    assert y.tolist() == [0, 1, 0]


def test_create_feature_label_data_rejects_missing_label_column():
    """
    Check that feature-label creation fails with a clear error when labels
    have not yet been created.
    """
    data = pd.DataFrame(
        {
            "tissue": ["Lung", "Lung - Adenocarcinoma"],
            "gene_1": [0.1, 2.1],
        }
    )

    with pytest.raises(ValueError, match="label"):
        create_feature_label_data(data)


def test_create_feature_label_data_rejects_missing_gene_columns():
    """
    Check that feature-label creation fails when the table has labels but
    no gene-expression feature columns.
    """
    data = pd.DataFrame(
        {
            "tissue": ["Lung", "Lung - Adenocarcinoma"],
            "label": [0, 1],
            "feature_1": [0.1, 2.1],
        }
    )

    with pytest.raises(ValueError, match="gene expression columns"):
        create_feature_label_data(data)


def test_summarize_class_balance_counts_each_label():
    """
    Check that class balance summarisation returns exact counts for each
    binary label, sorted by label value.
    """
    data = pd.DataFrame({"label": [1, 0, 1, 0, 1]})

    result = summarize_class_balance(data)

    assert result == {0: 2, 1: 3}


def test_summarize_class_balance_rejects_missing_label_column():
    """
    Check that class balance summarisation fails clearly when the label
    column is unavailable.
    """
    data = pd.DataFrame({"tissue": ["Lung", "Colon"]})

    with pytest.raises(ValueError, match="label"):
        summarize_class_balance(data)


def test_add_binary_label_creates_label_column():
    """
    Check that rule-based label inference creates the expected binary
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


def test_add_binary_label_is_case_insensitive():
    """
    Check that cancer keyword matching is case-insensitive, so labels do
    not depend on capitalization in the tissue annotation.
    """
    data = pd.DataFrame(
        {
            "tissue": [
                "LUNG - ADENOCARCINOMA",
                "colon - carcinoma",
                "Kidney",
            ]
        }
    )

    result = add_binary_label(data)

    assert result["label"].tolist() == [1, 1, 0]


def test_add_binary_label_does_not_modify_input_dataframe():
    """
    Check that label creation returns a copy and does not modify the input
    dataframe in place.
    """
    data = pd.DataFrame({"tissue": ["Lung - Adenocarcinoma", "Lung"]})

    add_binary_label(data)

    assert "label" not in data.columns


def test_add_binary_label_raises_error_when_tissue_column_is_missing():
    """
    Check that label creation raises a clear error when the tissue
    annotation column is missing.
    """
    data = pd.DataFrame({"wrong_column": ["Lung - Adenocarcinoma"]})

    with pytest.raises(ValueError, match="tissue"):
        add_binary_label(data)


def test_get_high_count_tissues_returns_expected_tissues():
    """
    Check that tissue categories meeting the minimum count threshold are
    returned in frequency order.
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
    Check that filtering keeps only rows whose tissue annotation is
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


def test_filter_by_tissues_returns_independent_copy():
    """
    Check that filtering returns an independent dataframe copy, so later
    edits to the filtered result do not alter the original dataframe.
    """
    data = pd.DataFrame(
        {
            "tissue": ["Lung", "Colon", "Kidney"],
            "value": [1, 2, 3],
        }
    )

    result = filter_by_tissues(data, ["Lung"])
    result.loc[result.index[0], "value"] = 99

    assert data.loc[0, "value"] == 1


def test_define_subsets_returns_expected_keys():
    """
    Check that predefined exploratory tissue subsets are returned with
    the expected dictionary keys.
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

    assert set(subsets.keys()) == {"lung", "colon", "kidney"}


def test_define_subsets_filters_lung_subset_correctly():
    """
    Check that the predefined lung subset keeps the expected lung-related
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
    Check that preprocessing raises a clear error when the binary label
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


def test_preprocess_and_split_rejects_single_class_labels():
    """
    Check that stratified splitting rejects data with only one class,
    because binary classification requires both labels.
    """
    data = pd.DataFrame(
        {
            "gene_1": [0.1, 0.2, 0.3, 0.4],
            "label": [0, 0, 0, 0],
        }
    )

    with pytest.raises(ValueError, match="exactly two classes"):
        preprocess_and_split(data)


def test_preprocess_and_split_returns_expected_split_sizes_and_labels():
    """
    Check that preprocessing creates train, validation, and test splits
    while preserving both binary classes in each split.
    """
    data = pd.DataFrame(
        {
            "gene_1": list(range(20)),
            "gene_2": list(range(20, 40)),
            "label": [0, 1] * 10,
        }
    )

    train, validate, test = preprocess_and_split(
        data,
        test_size=0.2,
        validate_size=0.25,
        random_state=42,
    )

    assert len(train) == 12
    assert len(validate) == 4
    assert len(test) == 4

    assert set(train["label"]) == {0, 1}
    assert set(validate["label"]) == {0, 1}
    assert set(test["label"]) == {0, 1}

    combined_indices = set(train.index) | set(validate.index) | set(test.index)

    assert combined_indices == set(data.index)
    assert set(train.index).isdisjoint(validate.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(validate.index).isdisjoint(test.index)
