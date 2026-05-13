from gene_expression_cancer_classification.cli import main


def test_cli_example_runs_without_error(monkeypatch, capsys):
    """
    Test that the basic CLI example command runs and prints the main
    evaluation outputs.
    """
    monkeypatch.setattr("sys.argv", ["gene-cancer-classify", "example"])

    main()

    captured = capsys.readouterr()

    assert "Predictions:" in captured.out
    assert "Accuracy:" in captured.out
    assert "Confusion matrix:" in captured.out
    assert "Precision:" in captured.out
    assert "Recall:" in captured.out
    assert "F1:" in captured.out


def test_cli_train_example_runs_without_error(monkeypatch, capsys):
    """
    Test that the toy gene expression CLI workflow runs and prints
    dataset information and evaluation outputs.
    """
    monkeypatch.setattr("sys.argv", ["gene-cancer-classify", "train-example"])

    main()

    captured = capsys.readouterr()

    assert "Input file:" in captured.out
    assert "Number of samples:" in captured.out
    assert "Feature columns:" in captured.out
    assert "Accuracy:" in captured.out
    assert "Confusion matrix:" in captured.out
    assert "F1:" in captured.out
