#%%
from functools import wraps
import numpy as np
from sklearn import metrics
import torch
import scipy


def pick(key):
    def inner_function(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            func_args = (
                arg[key] if hasattr(arg, "__getitem__") else arg for arg in args
            )
            return function(*func_args, **kwargs)

        return wrapper

    return inner_function


def _align(y_true, y_pred, names=None):

    assert y_true.shape == y_pred.shape
    if isinstance(names, str):
        names = (names,)

    if y_true.ndim == 1:
        y_true = y_true.view(-1, 1)
        y_pred = y_pred.view(-1, 1)

    y_true = y_true.flatten(1)
    y_pred = y_pred.flatten(1)

    if names is None:
        names = list(range(y_true.size(1)))
    else:
        assert len(names) == y_true.size(1)

    return y_true, y_pred, names


@pick("labels")
def roc_curve(y_true, y_score, *, names=None, **kwargs):

    roc = {}
    y_true, y_score, names = _align(y_true, y_score, names)

    for yt, ys, n in zip(y_true.unbind(-1), y_score.unbind(-1), names):
        roc[n] = metrics.roc_curve(yt, ys, **kwargs)

    return roc


@pick("labels")
def roc_auc_score(y_true, y_score, *, names=None, **kwargs):
    roc_auc = {}

    y_true, y_score, names = _align(y_true, y_score, names)

    for yt, ys, n in zip(y_true.unbind(-1), y_score.unbind(-1), names):
        roc_auc[n] = metrics.roc_auc_score(yt, ys, **kwargs)

    return roc_auc


@pick("labels")
def precision_recall_curve(y_true, y_score, *, names=None, **kwargs):
    pr_curve = {}

    y_true, y_score, names = _align(y_true, y_score, names)

    for yt, ys, n in zip(y_true.unbind(-1), y_score.unbind(-1), names):
        pr_curve[n] = metrics.precision_recall_curve(yt, ys, **kwargs)

    return pr_curve


@pick("labels")
def balanced_accuracy_score(y_true, y_score, *, names=None, threshold=0.5, **kwargs):

    balanced_accuracy = {}

    y_score = (y_score >= threshold).to(int)
    y_true = y_true.to(int)

    y_true, y_score, names = _align(y_true, y_score, names)

    for yt, ys, n in zip(y_true.unbind(-1), y_score.unbind(-1), names):
        balanced_accuracy[n] = metrics.balanced_accuracy_score(yt, ys, **kwargs)

    return balanced_accuracy


@pick("labels")
def confusion_matrix(y_true, y_score, *, names=None, threshold=0.5, **kwargs):
    cm = {}

    y_score = (y_score >= threshold).to(int)
    y_true = y_true.to(int)

    y_true, y_score, names = _align(y_true, y_score, names)

    for yt, ys, n in zip(y_true.unbind(-1), y_score.unbind(-1), names):
        cm[n] = metrics.confusion_matrix(yt, ys, **kwargs)

    return cm


#%%
def recall_iou_curve(ious, num_thresholds=25):
    thresholds = torch.linspace(0.5, 1.0, num_thresholds)
    detections = (ious >= thresholds[:, None]).to(int)

    y_true = torch.ones(len(ious))

    recalls = [metrics.recall_score(y_true, y_pred) for y_pred in detections]
    recalls = torch.as_tensor(recalls)
    return recalls, thresholds


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


def evaluate_ious(ious: torch.Tensor):
    """
    Evaluate IOU Values
    Args:
        ious (torch.Tensor[N,])

    Returns:
        evaluation (dict) with keys
            - min   (torch.Min)
            - max   (torch.Max)
            - ci    (tuple[mean:float, lower:float, upper:float])
                    95% confidence interval
            - recall_curve (dict) with keys
                - iou (torch.Tensor) thresholds
                - recall (torch.Tensor) recall values
            - ar_score (float) average recall score,
                        i.e. 2 x area under recall curve
    """

    evaluation = {}

    evaluation["min"] = ious.min()
    evaluation["max"] = ious.max()

    recall, thresholds = recall_iou_curve(ious)
    mean, lower, upper = mean_confidence_interval(ious)

    evaluation["ci"] = torch.as_tensor((mean, lower, upper))
    evaluation["recall_curve"] = {
        "iou": thresholds,
        "recall": recall,
        "auc": metrics.auc(thresholds, recall) * 2,
    }

    return evaluation


def evaluate_scores(scores: torch.Tensor, targets: torch.Tensor):
    evaluation = {}

    fpr, tpr, thresholds = metrics.roc_curve(targets, scores)
    evaluation["roc_curve"] = {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": metrics.roc_auc_score(targets, scores),
    }

    decision = thresholds[(tpr - fpr).argmax()]
    prec, rec, thresholds = metrics.precision_recall_curve(targets, scores)

    evaluation["pr_curve"] = {
        "precision": prec,
        "recall": rec,
        "thresholds": thresholds,
        "auc@0.5": metrics.precision_score(targets, (scores >= 0.5).long()),
        "auc@db": metrics.precision_score(targets, (scores >= decision).long()),
    }

    evaluation["decision_boundary"] = decision

    predictions_05 = (scores >= 0.5).to(int)
    predictions_db = (scores >= decision).to(int)

    evaluation["confusion_matrix"] = {
        "@0.5": metrics.confusion_matrix(targets, predictions_05),
        "@db": metrics.confusion_matrix(targets, predictions_db),
    }
    evaluation["balanced_accuracy"] = {
        "@0.5": metrics.balanced_accuracy_score(targets, predictions_05),
        "@db": metrics.balanced_accuracy_score(targets, predictions_db),
    }
    return evaluation


def evaluate_boxes(predictions: torch.Tensor, targets: torch.Tensor):
    ratios = predictions / targets
    means = [mean_confidence_interval(each) for each in ratios.unbind(-1)]
    return means
