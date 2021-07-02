import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, auc


def plot_recall_iou(recalls, thresholds, average_recall=None, name=None, ax=None):

    if ax is None:
        _, ax = plt.subplots()

    label = None
    if average_recall is True:
        average_recall = auc(thresholds, recalls) * 2

    if average_recall is not None and name is not None:
        label = f"{name} (AR = {average_recall:0.2f})"
    elif name is not None:
        label = f"{name}"
    elif average_recall is not None:
        label = f"AR = {average_recall:.2f}"

    ax.plot(thresholds, recalls, label=label)
    ax.set_title("Recall vs IOU")
    ax.set_xlabel("IOU")
    ax.set_ylabel("Recall")
    ax.legend()
    return ax.figure


def plot_confusion_matrix(confusion_matrix, **kwargs):
    return ConfusionMatrixDisplay(confusion_matrix).plot(**kwargs)
