import seaborn as sns


def plot_heatmap(cm, class_names, ax, title):
    """
    Plot a confusion matrix heatmap on an existing matplotlib axis.

    Parameters
    ----------
    cm
        Confusion matrix to plot.
    class_names
        Names shown on the x-axis and y-axis tick labels.
    ax
        Matplotlib axis on which the heatmap is drawn.
    title
        Plot title.

    Returns
    -------
    None
        The function modifies the provided axis in place.
    """
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
