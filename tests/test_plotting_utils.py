import matplotlib.pyplot as plt
import numpy as np

from gene_expression_cancer_classification.plotting import plot_heatmap


def test_plot_heatmap_sets_labels_and_title():
    """
    Test that the confusion matrix plotting helper sets the expected
    axis labels and title.
    """
    cm = np.array([[5, 1], [2, 4]])
    class_names = ["normal", "cancer"]

    fig, ax = plt.subplots()
    plot_heatmap(cm, class_names, ax, "Test Heatmap")

    assert ax.get_title() == "Test Heatmap"
    assert ax.get_xlabel() == "Predicted label"
    assert ax.get_ylabel() == "True label"

    plt.close(fig)
