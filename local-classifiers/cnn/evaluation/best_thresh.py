import numpy as np
from .f1_score import f1_score_per_label

def get_best_thresholds(y_test, y_hat_test, labels):
    threshold_list = np.arange(0.0, 1, 0.01)
    best_f1_label = np.zeros((len(labels)))
    best_t_label = np.zeros((len(labels)))

    for label, index in labels.items():
        best_f1 = 0
        best_t = 0
        for t in threshold_list:
            current_f1 = f1_score_per_label(y_test, y_hat_test['child'], t)[0][labels[label]].item()
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_t = t

        best_f1_label[index] = best_f1
        best_t_label[index] = best_t
    return best_f1_label, best_t_label