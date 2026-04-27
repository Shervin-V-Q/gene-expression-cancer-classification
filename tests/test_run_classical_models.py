import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from run_classical_models import main


def test_run_classical_models_main_runs_without_error(capsys):
    main()
    captured = capsys.readouterr()

    assert "Predictions:" in captured.out
    assert "Accuracy:" in captured.out
    assert "Confusion matrix:" in captured.out
    assert "Precision:" in captured.out
    assert "Recall:" in captured.out
    assert "F1:" in captured.out
