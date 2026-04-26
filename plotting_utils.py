import seaborn as sns


def plot_heatmap(cm, class_names, ax, title):
    """
    Plot a confusion matrix heatmap on the provided axis.
    """
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
