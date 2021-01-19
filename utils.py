import numpy as np
import torch
from sklearn.metrics import mean_squared_error

def f1(pr, tr, class_num):
    """
    Calculates F1 score for a given class
    :param pr: list of predicted values
    :param tr: list of actual values
    :param class_num: indicates class
    :return: f1 score of class_num for predicted and true values in pr, tr
    """

    # Filter lists by class
    pred = [x == class_num for x in pr]
    truth = [x == class_num for x in tr]
    mix = list(zip(pred, truth))
    # Find true positives, false positives and false negatives
    tp = mix.count((True, True))
    fp = mix.count((False, True))
    fn = mix.count((True, False))
    # Return f1 score, if conditions are met
    if tp == 0 and fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    if tp == 0 and fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    if recall == 0 and precision == 0:
        return 0
    else:
        return 2 * recall * precision / (recall + precision)


def macro_f1(predictions, truth):
    """
    Calculates macro f1 score, where all classes have the same weight
    :param predictions: logits of model predictions
    :param truth: list of actual values
    :return: macro f1 between model predictions and actual values
    """
    different_f1s = [f1(predictions, truth, lab) for lab in set(truth)]
    return sum(different_f1s) / len(different_f1s)



def weighted_f1(predictions, truth):
    """
    Calculates weighted f1 score, where all classes have different weights based on appearance
    :param predictions: logits of model predictions
    :param truth: list of actual values
    :return: weighted f1 between model predictions and actual values
    """
    different_f1s = [f1(predictions, truth, lab) for lab in set(truth)]
    different_weights = [np.sum([x == lab for x in truth]) for lab in set(truth)]
    return sum(x*y for x, y in zip(different_f1s, different_weights)) / len(predictions)


def get_subset(to_subset, length):
    indices = torch.randperm(len(to_subset))[:length]
    return torch.utils.data.Subset(to_subset, indices)


def sep_val(pred_lab_short, idx):
    return list(zip([pred_lab_short[0][i] for i in idx], [pred_lab_short[1][i] for i in idx],
                    [pred_lab_short[2][i] for i in idx]))


def split(number, portion=0.9):
    return [round(portion*number), round((1-portion)*number)]


def validation_metrics(pred, labs):
    def isfloat(value):
        try:
            float(value)
            return True
        except ValueError:
            return False
    idx = np.where(np.array([isfloat(x) for x in pred]) == True)
    if idx[0].size > 0:
        preds = np.array([float(x) for x in pred[idx]])
        lab = np.array([float(x) for x in labs[idx]])
        new_idx = np.where(preds <= 1)
        print('\nInvalid validation examples: ', labs.size - new_idx[0].size)
        """
        val_acc = np.sum(acc_data[0] == acc_data[1]) / acc_data.shape[1]
        val_weighted = weighted_f1(acc_data[1], acc_data[0])
        val_macro = macro_f1(acc_data[1], acc_data[0])
        """
        return {"mse": mean_squared_error(lab[new_idx], preds[new_idx]),
                "acc": np.sum(lab[new_idx] == preds[new_idx]) / new_idx[0].size,
                "macro": macro_f1(preds[new_idx], lab[new_idx]),
                "weighted": weighted_f1(preds[new_idx], lab[new_idx])
                }
    else:
        print('\nInvalid validation')
        return {"mse": 1, "acc": 0, "macro": 0, "weighted": 0}

