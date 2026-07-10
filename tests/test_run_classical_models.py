from run_classical_models import main


def test_run_classical_models_main_prints_expected_report(capsys):
    """
    Check that the compatibility script runs the same minimal example as
    the CLI and prints the expected evaluation report.
    """
    main()
    captured = capsys.readouterr()

    assert "Predictions: [0 1]" in captured.out
    assert "Accuracy: 1.0" in captured.out
    assert "Confusion matrix:" in captured.out
    assert "[[1 0]" in captured.out
    assert "[0 1]]" in captured.out
    assert "Precision: 1.0" in captured.out
    assert "Recall: 1.0" in captured.out
    assert "F1: 1.0" in captured.out
