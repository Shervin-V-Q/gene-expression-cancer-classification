import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from plotting_utils import plot_heatmap


def test_plot_heatmap_sets_labels_and_title():
    cm = np.array([[5, 1], [2, 4]])
    class_names = ["normal", "cancer"]

    fig, ax = plt.subplots()
    plot_heatmap(cm, class_names, ax, "Test Heatmap")

    assert ax.get_title() == "Test Heatmap"
    assert ax.get_xlabel() == "Predicted label"
    assert ax.get_ylabel() == "True label"

    plt.close(fig)
