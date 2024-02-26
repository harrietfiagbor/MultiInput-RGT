from tsai.metrics import *
from fastai.torch_core import *
import math


def threshold_lvef(inp, targ, healthy_lvef):
    "Threshold the values in `inp` and `targ` to 1 or 0 with respect to `healthy_lvef`"

    inp = tensor([int(lvef < healthy_lvef) for lvef in inp])
    targ = tensor([int(lvef < healthy_lvef) for lvef in targ])

    return inp, targ


def F1_regression(inp, targ, healthy_lvef=50, *args, **kwargs):
    "Compute F1 after thresholding `inp` and `targ` on `healthy_lvef` if they are of the same size."

    inp, targ = threshold_lvef(inp, targ, healthy_lvef)

    F1_score = F1_multi(inp, targ, *args, **kwargs)

    if math.isnan(F1_score):
        return 0
    return F1_score


def balanced_accuracy_regression(inp, targ, healthy_lvef=50, *args, **kwargs):
    "Compute balanced accuracy after thresholding `inp` and `targ` on `healthy_lvef` if they are of the same size."

    inp, targ = threshold_lvef(inp, targ, healthy_lvef)

    balanced_accuracy = balanced_accuracy_multi(inp, targ, *args, **kwargs)

    if math.isnan(balanced_accuracy):
        return 0
    return balanced_accuracy


def precision_regression(inp, targ, healthy_lvef=50, *args, **kwargs):
    "Compute precision after thresholding `inp` and `targ` on `healthy_lvef` if they are of the same size."

    inp, targ = threshold_lvef(inp, targ, healthy_lvef)

    precision = precision_multi(inp, targ, *args, **kwargs)

    if math.isnan(precision):
        return 0
    return precision


def recall_regression(inp, targ, healthy_lvef=50, *args, **kwargs):
    "Compute recall after thresholding `inp` and `targ` on `healthy_lvef` if they are of the same size."

    inp, targ = threshold_lvef(inp, targ, healthy_lvef)

    recall = recall_multi(inp, targ, *args, **kwargs)

    if math.isnan(recall):
        return 0
    return recall
