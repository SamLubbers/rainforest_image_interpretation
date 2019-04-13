from sklearn.metrics import fbeta_score


def evaluate_performance(labels, predictions, beta=2):
    mean_f2 = fbeta_score(labels, predictions, beta, average='samples')
    per_class_f2 = fbeta_score(labels, predictions, beta, average=None)
    return mean_f2, per_class_f2