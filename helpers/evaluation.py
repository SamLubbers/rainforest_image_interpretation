import os
import sys
from collections import OrderedDict

import pandas as pd

from .data import extract_label_values

sys.path.append("../")
from config import DATASETS_PATH

from sklearn.metrics import fbeta_score


def evaluate_performance_validation(predictions, beta=2):
    labels = pd.read_csv(os.path.join(DATASETS_PATH, 'validation_labels.csv'))
    label_names = list(labels.columns[2:])
    labels = extract_label_values(labels)
    mean_f2 = fbeta_score(labels, predictions, beta, average='samples')
    per_class_f2 = fbeta_score(labels, predictions, beta, average=None)
    per_class_f2 = OrderedDict({l: v for l, v in zip(label_names, per_class_f2)})
    return mean_f2, per_class_f2
