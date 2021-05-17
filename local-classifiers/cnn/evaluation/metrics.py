import numpy as np
from torch import Tensor
from sklearn.metrics import roc_curve, auc, hamming_loss, accuracy_score, f1_score
import pdb
import torch

CLASSIFICATION_THRESHOLD: float = 0.5  # Best keep it in [0.0, 1.0] range

# def accuracy(out, labels):
#     outputs = np.argmax(out, axis=1)
#     return np.sum(outputs == labels)


def accuracy(y_true: Tensor, y_pred: Tensor):

    outputs = np.argmax(y_pred, axis=1)
    return np.mean(outputs.numpy() == y_true.detach().cpu().numpy())


def accuracy_multilabel(y_true: Tensor, y_pred: Tensor,sigmoid: bool = True):
    if sigmoid:
        y_pred = y_pred.sigmoid()
    outputs = np.argmax(y_pred, axis=1)
    real_vals = np.argmax(y_true, axis=1)
    return np.mean(outputs.numpy() == real_vals.numpy())

# def f1_micro(y_true: Tensor, y_pred: Tensor):
#     y_pred = (y_pred > CLASSIFICATION_THRESHOLD).float()
#     return f1_score(y_true, y_pred, average='micro')

def f1_micro_fn (y_true: Tensor, y_pred: Tensor, eps: float = 1e-9):
    y_pred = torch.ge(y_pred.float(), CLASSIFICATION_THRESHOLD).float()
    y_true = y_true.float()
    tp_l = (y_pred * y_true).sum(0).float()
    fp_l = (y_pred * (1 - y_true)).sum(0).float()
    fn_l = ((1 - y_pred) * y_true).sum(0).float()
    precision_label = tp_l.div(tp_l + fp_l + eps)
    recall_label = tp_l.div(tp_l + fn_l + eps)
    tp = tp_l.sum()
    fp = fp_l.sum()
    fn = fn_l.sum()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1_micro = (precision * recall).div(precision + recall + eps) * 2
    return f1_micro.item(), precision.item(), recall.item()

def f1_micro_absence (y_true: Tensor, y_pred: Tensor, eps: float = 1e-9):
    y_pred = torch.lt(y_pred.float(), CLASSIFICATION_THRESHOLD).float()
    y_true_absence = 1 - y_true.float()
    tp_l = (y_pred * y_true_absence).sum(0).float()
    fp_l = (y_pred * (1 - y_true_absence)).sum(0).float()
    fn_l = ((1 - y_pred) * y_true_absence).sum(0).float()
    precision_label = tp_l.div(tp_l + fp_l + eps)
    recall_label = tp_l.div(tp_l + fn_l + eps)
    tp = tp_l.sum()
    fp = fp_l.sum()
    fn = fn_l.sum()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1_micro = (precision * recall).div(precision + recall + eps) * 2
    return f1_micro.item(), precision.item(), recall.item()

def f1_macro_fn (y_true: Tensor, y_pred: Tensor, eps: float = 1e-9):
    y_pred = torch.ge(y_pred.float(), CLASSIFICATION_THRESHOLD).float()
    y_true = y_true.float()
    tp_l = (y_pred * y_true).sum(0).float()
    fp_l = (y_pred * (1 - y_true)).sum(0).float()
    fn_l = ((1 - y_pred) * y_true).sum(0).float()
    precision_label = tp_l.div(tp_l + fp_l + eps)
    recall_label = tp_l.div(tp_l + fn_l + eps)
    f1_macro = torch.mean((precision_label * recall_label).div(precision_label + recall_label + eps) * 2)
    return f1_macro.item(), torch.mean(precision_label).item(), torch.mean(recall_label).item()

def f1_macro_absence (y_true: Tensor, y_pred: Tensor, eps: float = 1e-9):
    y_pred = torch.lt(y_pred.float(), CLASSIFICATION_THRESHOLD).float()
    y_true_absence = 1 - y_true.float()
    tp_l = (y_pred * y_true_absence).sum(0).float()
    fp_l = (y_pred * (1 - y_true_absence)).sum(0).float()
    fn_l = ((1 - y_pred) * y_true_absence).sum(0).float()
    precision_label = tp_l.div(tp_l + fp_l + eps)
    recall_label = tp_l.div(tp_l + fn_l + eps)
    f1_macro = torch.mean((precision_label * recall_label).div(precision_label + recall_label + eps) * 2)
    return f1_macro.item(), torch.mean(precision_label).item(), torch.mean(recall_label).item()

def f1_score_per_label(y_true: Tensor, y_pred: Tensor, eps: float = 1e-9):
    y_pred = torch.ge(y_pred.float(), CLASSIFICATION_THRESHOLD).float()
    y_true = y_true.float()
    tp_l = (y_pred * y_true).sum(0).float()
    fp_l = (y_pred * (1 - y_true)).sum(0).float()
    fn_l = ((1 - y_pred) * y_true).sum(0).float()
    precision_label = tp_l.div(tp_l + fp_l + eps)
    recall_label = tp_l.div(tp_l + fn_l + eps)
    f1_label = (precision_label * recall_label).div(precision_label + recall_label + eps) * 2
    return f1_label

def accuracy_thresh(y_true: Tensor, y_pred: Tensor, thresh: float = CLASSIFICATION_THRESHOLD, sigmoid: bool = True):
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid:
        y_pred = y_pred.sigmoid()
    return ((y_pred > thresh) == y_true.bool()).float().mean().item()
#     return np.mean(((y_pred>thresh)==y_true.byte()).float().cpu().numpy(), axis=1).sum()


def fbeta(y_true: Tensor, y_pred: Tensor, thresh: float = 0.3, beta: float = 2, eps: float = 1e-9, sigmoid: bool = True):
    "Computes the f_beta between `preds` and `targets`"
    beta2 = beta ** 2
    if sigmoid:
        y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thresh).float()
    y_true = y_true.float()
    TP = (y_pred*y_true).sum(dim=1)
    prec = TP/(y_pred.sum(dim=1)+eps)
    rec = TP/(y_true.sum(dim=1)+eps)
    res = (prec*rec)/(prec*beta2+rec+eps)*(1+beta2)
    return res.mean().item()


def roc_auc(y_true: Tensor, y_pred: Tensor):
    # ROC-AUC calcualation
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    return roc_auc["micro"]


def Hamming_loss(y_true: Tensor, y_pred: Tensor, sigmoid: bool = True, thresh: float = CLASSIFICATION_THRESHOLD, sample_weight=None):
    if sigmoid:
        y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thresh).float()
    return hamming_loss(y_true, y_pred, sample_weight=sample_weight)


def Exact_Match_Ratio(y_true: Tensor, y_pred: Tensor, sigmoid: bool = True, thresh: float = CLASSIFICATION_THRESHOLD, normalize: bool = True, sample_weight=None):
    if sigmoid:
        y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thresh).float()
    return accuracy_score(y_true, y_pred, normalize=normalize, sample_weight=sample_weight)


def F1(y_true: Tensor, y_pred: Tensor, threshold: float = CLASSIFICATION_THRESHOLD):
    return fbeta(y_pred, y_true, thresh=threshold, beta=1)
