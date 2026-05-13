from run_classical_models import main


def test_run_classical_models_main_runs_without_error(capsys):
    """
    Test that the compatibility example script runs and prints the main
    evaluation outputs.
    """
    main()
    captured = capsys.readouterr()

    assert "Predictions:" in captured.out
    assert "Accuracy:" in captured.out
    assert "Confusion matrix:" in captured.out
    assert "Precision:" in captured.out
    assert "Recall:" in captured.out
    assert "F1:" in captured.out
