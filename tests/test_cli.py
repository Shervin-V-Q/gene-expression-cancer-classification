import pytest

from gene_expression_cancer_classification.cli import main


def test_cli_example_runs_without_error(monkeypatch, capsys):
    """
    Check that the basic CLI example command runs successfully and prints
    the expected evaluation report for the minimal artificial dataset.
    """
    monkeypatch.setattr("sys.argv", ["gene-cancer-classify", "example"])

    main()

    captured = capsys.readouterr()

    assert "Predictions:" in captured.out
    assert "Accuracy: 1.0" in captured.out
    assert "Confusion matrix:" in captured.out
    assert "Precision: 1.0" in captured.out
    assert "Recall: 1.0" in captured.out
    assert "F1: 1.0" in captured.out


def test_cli_train_example_runs_without_error(monkeypatch, capsys):
    """
    Check that the toy gene expression CLI workflow loads the example CSV,
    reports dataset information, and prints exact evaluation metrics.
    """
    monkeypatch.setattr("sys.argv", ["gene-cancer-classify", "train-example"])

    main()

    captured = capsys.readouterr()

    assert "Input file:" in captured.out
    assert "examples" in captured.out
    assert "toy_gene_expression.csv" in captured.out
    assert "Number of samples: 12" in captured.out
    assert "Feature columns: ['gene_1', 'gene_2', 'gene_3', 'gene_4']" in captured.out
    assert "Class balance: {0: 6, 1: 6}" in captured.out
    assert "Test size: 0.33" in captured.out
    assert "Random state: 42" in captured.out
    assert "Accuracy: 1.0" in captured.out
    assert "Confusion matrix:" in captured.out
    assert "Precision: 1.0" in captured.out
    assert "Recall: 1.0" in captured.out
    assert "F1: 1.0" in captured.out


def test_cli_train_example_prints_model_comparison(monkeypatch, capsys):
    """
    Check that the toy workflow reports a model comparison table, making
    the command more informative than a single-model demonstration.
    """
    monkeypatch.setattr("sys.argv", ["gene-cancer-classify", "train-example"])

    main()

    captured = capsys.readouterr()

    assert "Model comparison:" in captured.out
    assert "logistic_regression" in captured.out
    assert "decision_tree" in captured.out
    assert "accuracy" in captured.out
    assert "precision" in captured.out
    assert "recall" in captured.out
    assert "f1" in captured.out


def test_cli_train_example_accepts_configurable_options(monkeypatch, capsys):
    """
    Check that the toy workflow accepts command-line options for the input
    path, test split, random seed, and model-comparison sorting metric.
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "gene-cancer-classify",
            "train-example",
            "--input-path",
            "examples/toy_gene_expression.csv",
            "--test-size",
            "0.33",
            "--random-state",
            "42",
            "--sort-by",
            "accuracy",
        ],
    )

    main()

    captured = capsys.readouterr()

    assert "Input file:" in captured.out
    assert "toy_gene_expression.csv" in captured.out
    assert "Test size: 0.33" in captured.out
    assert "Random state: 42" in captured.out
    assert "Model comparison:" in captured.out
    assert "logistic_regression" in captured.out
    assert "decision_tree" in captured.out


def test_cli_train_example_rejects_invalid_test_size(monkeypatch):
    """
    Check that the toy workflow rejects invalid test split sizes instead of
    silently running with an impossible configuration.
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "gene-cancer-classify",
            "train-example",
            "--test-size",
            "1.0",
        ],
    )

    with pytest.raises(ValueError, match="test_size"):
        main()


def test_cli_rejects_unknown_command(monkeypatch):
    """
    Check that the command-line interface rejects unsupported commands
    through argparse instead of silently doing nothing.
    """
    monkeypatch.setattr("sys.argv", ["gene-cancer-classify", "unknown-command"])

    with pytest.raises(SystemExit):
        main()
