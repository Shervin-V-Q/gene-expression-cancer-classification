from gene_expression_cancer_classification.cli import main


def test_cli_example_runs_without_error(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["gene-cancer-classify", "example"])

    main()

    captured = capsys.readouterr()

    assert "Predictions:" in captured.out
    assert "Accuracy:" in captured.out
    assert "Confusion matrix:" in captured.out
    assert "Precision:" in captured.out
    assert "Recall:" in captured.out
    assert "F1:" in captured.out
