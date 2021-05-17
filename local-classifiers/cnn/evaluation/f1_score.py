from math import ceil
import time
import torch

def f1_score(y_true, y_pred, threshold, macro=False, eps=1e-9):
    """
    Computes the f1 score resulting from the comparison between y_true and y_pred after using the threshold set.

    Args:
      y_true: torch.tensor, 2-dimensional torch.tensor containing the true labels per record. The i-th row is the
      record i whereas the j-th column is the label j.
      y_pred: torch.tensor, 2-dimensional torch.tensor containing the probabilities assigned to each label. The i-th
      row is the record i whereas the j-th column is the label j.
      threshold: double, number between 0 and 1 that sets the threshold probability for a label to be truly assigned
      to a record.
      macro: bool, if false we will return the micro average but if true it will return the macro average.
      eps: double, it is just a very small value that avoids dividing by 0 when computing the precision and recall.

    Returns:
      f1: double, the resulting mean f1 score of all the labels (it will be a number between 0 and 1)
      precision: double, the resulting mean precision of all the labels (it will be a number between 0 and 1)
      recall: double, the resulting mean recall of all the labels (it will be a number between 0 and 1)
    """
    # Calculate False Positives, False Negatives, True positives
    y_pred = torch.ge(y_pred.float().to('cpu'), threshold).float().to('cpu')
    y_true = y_true.float().to('cpu')
    tp_l = (y_pred * y_true).sum(0).float()
    fp_l = (y_pred * (1 - y_true)).sum(0).float()
    fn_l = ((1 - y_pred) * y_true).sum(0).float()

    precision_label = tp_l.div(tp_l + fp_l + eps)
    recall_label = tp_l.div(tp_l + fn_l + eps)

    if macro: 
        f1_macro = torch.mean((precision_label * recall_label).div(precision_label + recall_label + eps) * 2)
        return f1_macro.item(), torch.mean(precision_label).item(), torch.mean(recall_label).item()
    else:
        tp = tp_l.sum()
        fp = fp_l.sum()
        fn = fn_l.sum()
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1_micro = (precision * recall).div(precision + recall + eps) * 2
        return f1_micro.item(), precision.item(), recall.item()

def f1_score_absence(y_true, y_pred, threshold, macro=False, eps=1e-9):
    """
    Computes the f1 score resulting from the comparison between y_true and y_pred after using the threshold set.

    Args:
      y_true: torch.tensor, 2-dimensional torch.tensor containing the true labels per record. The i-th row is the
      record i whereas the j-th column is the label j.
      y_pred: torch.tensor, 2-dimensional torch.tensor containing the probabilities assigned to each label. The i-th
      row is the record i whereas the j-th column is the label j.
      threshold: double, number between 0 and 1 that sets the threshold probability for a label to be truly assigned
      to a record.
      macro: bool, if false we will return the micro average but if true it will return the macro average.
      eps: double, it is just a very small value that avoids dividing by 0 when computing the precision and recall.

    Returns:
      f1: double, the resulting mean f1 score of all the labels (it will be a number between 0 and 1)
      precision: double, the resulting mean precision of all the labels (it will be a number between 0 and 1)
      recall: double, the resulting mean recall of all the labels (it will be a number between 0 and 1)
    """
    # Calculate False Positives, False Negatives, True positives
    y_pred = torch.lt(y_pred.float().to('cpu'), threshold).float().to('cpu')
    y_true_absence = 1 - y_true.float().to('cpu')
    tp_l = (y_pred * y_true_absence).sum(0).float()
    fp_l = (y_pred * (1 - y_true_absence)).sum(0).float()
    fn_l = ((1 - y_pred) * y_true_absence).sum(0).float()

    precision_label = tp_l.div(tp_l + fp_l + eps)
    recall_label = tp_l.div(tp_l + fn_l + eps)

    if macro: 
        f1_macro = torch.mean((precision_label * recall_label).div(precision_label + recall_label + eps) * 2)
        return f1_macro.item(), torch.mean(precision_label).item(), torch.mean(recall_label).item()
    else:
        tp = tp_l.sum()
        fp = fp_l.sum()
        fn = fn_l.sum()
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1_micro = (precision * recall).div(precision + recall + eps) * 2
        return f1_micro.item(), precision.item(), recall.item()


def f1_score_per_label(y_true, y_pred, threshold, eps=1e-9):
    """
    Computes the f1 score per label resulting from the comparison between y_true and y_pred after using the threshold
    set.

    Args:
        y_true: torch.tensor, 2-dimensional torch.tensor containing the true labels per record. The i-th row is the
        record i whereas the j-th column is the label j.
        y_pred: torch.tensor, 2-dimensional torch.tensor containing the probabilities assigned to each label. The i-th
        row is the record i whereas the j-th column is the label j.
        threshold: double, number between 0 and 1 that sets the threshold probability for a label to be truly assigned
        to a record.
        eps: double, it is just a very small value that avoids dividing by 0 when computing the precision and recall.

    Returns:
        f1: list, the resulting f1 score per label (it will be a number between 0 and 1)
        precision: list, the resulting precision per label (it will be a number between 0 and 1)
        recall: list, the resulting recall per label (it will be a number between 0 and 1)

    """

    y_pred = torch.ge(y_pred.float(), threshold).float().to('cpu')
    y_true = y_true.float().to('cpu')

    tp_l = (y_pred * y_true).sum(0).float()
    fp_l = (y_pred * (1 - y_true)).sum(0).float()
    fn_l = ((1 - y_pred) * y_true).sum(0).float()

    precision_label = tp_l.div(tp_l + fp_l + eps)
    recall_label = tp_l.div(tp_l + fn_l + eps)

    f1_label = (precision_label * recall_label).div(precision_label + recall_label + eps) * 2

    return f1_label, precision_label, recall_label


def f1_score_per_label_absence(y_true, y_pred, threshold, eps=1e-9):
    """
    Computes the f1 score per label resulting from the comparison between y_true and y_pred after using the threshold
    set.

    Args:
        y_true: torch.tensor, 2-dimensional torch.tensor containing the true labels per record. The i-th row is the
        record i whereas the j-th column is the label j.
        y_pred: torch.tensor, 2-dimensional torch.tensor containing the probabilities assigned to each label. The i-th
        row is the record i whereas the j-th column is the label j.
        threshold: double, number between 0 and 1 that sets the threshold probability for a label to be truly assigned
        to a record.
        eps: double, it is just a very small value that avoids dividing by 0 when computing the precision and recall.

    Returns:
        f1: list, the resulting f1 score per label (it will be a number between 0 and 1)
        precision: list, the resulting precision per label (it will be a number between 0 and 1)
        recall: list, the resulting recall per label (it will be a number between 0 and 1)

    """

    y_pred = torch.lt(y_pred.float().to('cpu'), threshold).float().to('cpu')
    y_true_absence = 1 - y_true.float().to('cpu')
    tp_l = (y_pred * y_true_absence).sum(0).float()
    fp_l = (y_pred * (1 - y_true_absence)).sum(0).float()
    fn_l = ((1 - y_pred) * y_true_absence).sum(0).float()
    
    precision_label = tp_l.div(tp_l + fp_l + eps)
    recall_label = tp_l.div(tp_l + fn_l + eps)

    f1_label = (precision_label * recall_label).div(precision_label + recall_label + eps) * 2

    return f1_label, precision_label, recall_label
