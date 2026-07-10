import matplotlib.pyplot as plt
import numpy as np

from gene_expression_cancer_classification.plotting import plot_heatmap


def test_plot_heatmap_sets_labels_title_and_tick_names():
    """
    Check that the confusion matrix plotting helper applies the requested
    title, axis labels, and class names to the provided matplotlib axis.
    """
    cm = np.array([[5, 1], [2, 4]])
    class_names = ["normal", "cancer"]

    fig, ax = plt.subplots()
    plot_heatmap(cm, class_names, ax, "Test Heatmap")

    x_tick_labels = [label.get_text() for label in ax.get_xticklabels()]
    y_tick_labels = [label.get_text() for label in ax.get_yticklabels()]

    assert ax.get_title() == "Test Heatmap"
    assert ax.get_xlabel() == "Predicted label"
    assert ax.get_ylabel() == "True label"
    assert x_tick_labels == class_names
    assert y_tick_labels == class_names

    plt.close(fig)
