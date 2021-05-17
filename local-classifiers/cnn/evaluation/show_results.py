# FUNCTIONS TO DISPLAY RESULTS

import torch
from .f1_score import f1_score, f1_score_absence, f1_score_per_label, f1_score_per_label_absence

import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from IPython.display import display
import pandas as pd

def print_results(model, data_collator, train_dataset, validation_dataset, threshold=0.5, batch_size=32):
    # Prepare model
    if torch.cuda.is_available(): device = 'cuda'
    else: device = 'cpu'
    
    print('Setting model to eval mode ...')
    model.eval()
    
    # Get data from datasets
    print('Extracting data ...')
    labels = train_dataset.labels

    x_train = data_collator(train_dataset)[0] #.to(device)
    y_train = train_dataset.labels_tensor #.to(device)
    
    x_validation = data_collator(validation_dataset)[0] #.to(device)
    y_validation = validation_dataset.labels_tensor #.to(device)
    
    # Get Correct Device
    if device == 'cuda':
        model.cuda(0)
        x_train = x_train.cuda(0)
        y_train = y_train.cuda(0)
        x_validation = x_validation.cuda(0)
        y_validation = y_validation.cuda(0)
    else:
        model.cpu()
        x_train = x_train.cpu()
        y_train = y_train.cpu()
        x_validation = x_validation.cpu()
        y_validation = y_validation.cpu()
    
    
    # DO evaluation on train set
    print('Evaluating on train set ...')
    n_batches = int(len(x_train)/batch_size)
    y_hat_train_ls = []

    for i,v in enumerate(range(n_batches+1)):
        idx = i*batch_size
        x = x_train[idx:idx+batch_size]
        if len(x)>0:
            y_hat_tmp = model(x)
            y_hat_train_ls.append(y_hat_tmp)
    
    if len(y_hat_train_ls) > 1:
        y_hat_train = torch.stack(y_hat_train_ls[:-1])
    else:
        y_hat_train = torch.stack(y_hat_train_ls)
    
    y_hat_train = y_hat_train.view(-1, y_hat_train.size()[-1])
    y_hat_train = torch.cat([y_hat_train, y_hat_train_ls[-1]],0)

    
    # DO evaluation on validation set
    print('Evaluating on validation set ...')
    n_batches = int(len(x_validation)/batch_size)
    y_hat_val_ls = []
    for i in range(n_batches+1):
        idx = i*batch_size
        x = x_validation[idx:idx+batch_size]           
        if len(x)>0:
            y_hat_tmp = model(x)
            y_hat_val_ls.append(y_hat_tmp)
    
    if len(y_hat_val_ls) > 1:
        y_hat_validation = torch.stack(y_hat_val_ls[:-1])
    else:
        y_hat_validation = torch.stack(y_hat_val_ls)
    # Flatten
    y_hat_validation = y_hat_validation.view(-1, y_hat_validation.size()[-1])
    
    if len(y_hat_val_ls) > 1:
        y_hat_validation = torch.cat([y_hat_validation, y_hat_val_ls[-1]],0)
    
    # This will be the x axis
    threshold_list = np.arange(0.0, 1, 0.01)

    # Validation results presence
    print('calculating scores ...')
    print()
    f1_scores_validation = [f1_score(
        y_validation, y_hat_validation,t)[0] for t in threshold_list]

    precisions_validation = [f1_score(
        y_validation, y_hat_validation, t)[1] for t in threshold_list]

    recalls_validation = [f1_score(
        y_validation, y_hat_validation, t)[2] for t in threshold_list]

    # Train results presence
    f1_scores_train = [f1_score(y_train, y_hat_train, t)[0] for t in threshold_list]
    precisions_train = [f1_score(y_train, y_hat_train, t)[1] for t in threshold_list]
    recalls_train = [f1_score(y_train, y_hat_train, t)[2] for t in threshold_list]
    
    """
    # validation results absence
    f1_scores_validation_absence = [f1_score_absence(
        y_validation, y_hat_validation,t)[0] for t in threshold_list]

    precisions_validation_absence = [f1_score_absence(
        y_validation, y_hat_validation, t)[1] for t in threshold_list]

    recalls_validation_absence = [f1_score_absence(
        y_validation, y_hat_validation, t)[2] for t in threshold_list]

    # Train results
    f1_scores_train_absence = [f1_score_absence(y_train, y_hat_train, t)[0] for t in threshold_list]
    precisions_train_absence = [f1_score_absence(y_train, y_hat_train, t)[1] for t in threshold_list]
    recalls_train_absence = [f1_score_absence(y_train, y_hat_train, t)[2] for t in threshold_list]

    
    """
    count_train = y_train.sum(0).div(len(y_train))
    count_valid = y_validation.sum(0).div(len(y_validation))
   
    print("{} Train Labels".format(y_train.sum()))
    print("{} Train Segments".format(len(y_train)))
    print("{} Validation Labels".format(y_validation.sum()))
    print("{} Validation Segments".format(len(y_validation)))
   
    
    
    """
    Here comes the pyplot code
    """
    """
    fig = plt.figure(figsize=(15, 4))
    # We start with the three pyplot axis we want. One for F1, another for precision and one last one for recall
    ax_f1 = fig.add_subplot(131)
    ax_precision = fig.add_subplot(132)
    ax_recall = fig.add_subplot(133)

    # We now plot all the data in te corresponding axis
    ax_f1.plot(threshold_list, f1_scores_validation, label='validation')
    ax_f1.plot(threshold_list, f1_scores_train, label='train')
    ax_f1.set_title('F1 Score vs Threshold')
    ax_f1.set_ylim(0, 1.05)
    ax_f1.legend()
    
    ax_precision.plot(threshold_list, precisions_validation, label='validation')
    ax_precision.plot(threshold_list, precisions_train, label='train')
    ax_precision.set_title('Precision vs Threshold')

    ax_precision.set_ylim(0, 1.05)
    ax_precision.legend()

    ax_recall.plot(threshold_list, recalls_validation, label='validation')
    ax_recall.plot(threshold_list, recalls_train, label='train')
    ax_recall.set_title('Recall vs Threshold')
    ax_recall.set_ylim(0, 1.05)
    ax_recall.legend()

    plt.show()
    """
    
    # We show the overall F1, precision and recall for a threshold of 0.5 given by the variable threshold
    
    f1_micro, precision_micro, recall_micro = f1_score(y_validation, y_hat_validation, 0.5)
    f1_macro, precision_macro, recall_macro = f1_score(y_validation,y_hat_validation, 0.5, macro=True)

    
    f1_micro_absence_score, precision_micro_absence_score, recall_micro_absence_score = f1_score_absence(y_validation, 
                                              y_hat_validation,0.5)
    f1_macro_absence_score, precision_macro_absence_score, recall_macro_absence_score  = f1_score_absence(y_validation,
                                                                                                          y_hat_validation,0.5, macro=True)
    
    
    scores_list = f1_score_per_label(y_validation,y_hat_validation,0.5)
    scores_list_absence = f1_score_per_label_absence(y_validation,y_hat_validation,0.5)


    print("-" * 35 * 3)
    print("\n" + "Score per label with " + str(threshold) + " threshold")
    print("-" * 35 * 3)

    row_format = "{:<48}" + "{:<10}" * 5
    print(row_format.format("Label", "F1", "Precision", "Recall", "C.Train", "C.Validation"))
    print("-" * 35 * 3)

    for  index, label in enumerate(labels):
        f1_label = ceil(((scores_list[0][index] + scores_list_absence[0][index])/2) * 100) / 100
        precision_label = ceil(((scores_list[1][index] + scores_list_absence[1][index])/2) * 100) / 100
        recall_label = ceil(((scores_list[2][index] + scores_list_absence[2][index])/2) * 100) / 100
        ct_label = ceil(ceil(count_train[index] * 100) / 100 * len(y_train))
        cv_label = ceil(ceil(count_valid[index] * 100) / 100 * len(y_validation))
        print(row_format.format(label, f1_label, precision_label, recall_label, ct_label, cv_label))
    print("-" * 35 * 3)
    print(row_format.format("micro avg ", ceil((f1_micro+f1_micro_absence_score)/2*100)/100, ceil((precision_micro+precision_micro_absence_score)/2*100)/100, ceil((recall_micro+recall_micro_absence_score)/2*100)/100, len(y_train), len(y_validation)))
    print(row_format.format("macro avg ", ceil((f1_macro+f1_macro_absence_score)/2*100)/100, ceil((precision_macro+precision_macro_absence_score)/2*100)/100, ceil((recall_macro+recall_macro_absence_score)/2*100)/100, len(y_train), len(y_validation)))


def print_results_best_t(y_validation, y_hat_validation, labels, best_threshold):
    best_t = torch.tensor(best_threshold).float()
    scores_list = f1_score_per_label(y_validation, y_hat_validation['child'], best_t)

    row_format = "{:<48}" + "{:<10}" * 3
    print(row_format.format("Label", "F1", "Precision", "Recall"))
    print("-" * 35 * 3)

    for label, index in labels.items():
        f1_label = ceil(scores_list[0][index] * 100) / 100
        precision_label = ceil(scores_list[1][index] * 100) / 100
        recall_label = ceil(scores_list[2][index] * 100) / 100
        print(row_format.format(label, f1_label, precision_label,
                                recall_label))

    f1_mean = torch.mean(scores_list[0]).item()
    precision_mean = torch.mean(scores_list[1]).item()
    recall_mean = torch.mean(scores_list[2]).item()
    
    print()
    print('Macro Averages')
    print('F1: {}'.format(f1_mean))
    print('Precision: {}'.format(precision_mean))
    print('Recall: {}'.format(recall_mean))
