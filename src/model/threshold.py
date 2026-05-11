import numpy as np
from sklearn.metrics import roc_curve


def find_best_threshold_by_youden(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_score)) <= 1:
        raise ValueError('y_score가 모두 동일하여 threshold 계산이 불가능합니다.')
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    valid = ~np.isinf(thresholds)
    fpr, tpr, thresholds = fpr[valid], tpr[valid], thresholds[valid]
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx]), float(tpr[best_idx]), float(fpr[best_idx])
