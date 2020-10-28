import numpy as np


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

    f1_0 = f1(predictions, truth, "incorrect")
    f1_1 = f1(predictions, truth, "contradict")
    f1_2 = f1(predictions, truth, "correct")
    if np.sum([x == "contradict" for x in truth]) == 0:
        return (f1_0 + f1_2) / 2
    else:
        return (f1_0 + f1_1 + f1_2) / 3


def weighted_f1(predictions, truth):
    """
    Calculates weighted f1 score, where all classes have different weights based on appearance
    :param predictions: logits of model predictions
    :param truth: list of actual values
    :return: weighted f1 between model predictions and actual values
    """

    weight_0 = np.sum([x == "incorrect" for x in truth])
    weight_1 = np.sum([x == "contradict" for x in truth])
    weight_2 = np.sum([x == "correct" for x in truth])
    f1_0 = f1(predictions, truth, "incorrect")
    f1_1 = f1(predictions, truth, "contradict")
    f1_2 = f1(predictions, truth, "correct")
    return (weight_0 * f1_0 + weight_1 * f1_1 + weight_2 * f1_2) / len(truth)