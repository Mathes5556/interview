import numpy as np

def apk(actual, predicted, k=7):
    # https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0
    num_hits = 0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=7):
    """
    mean average precision
    source:
    https: // github.com / benhamner / Metrics / blob / master / Python / ml_metrics / average_precision.py
    :param actual:
    :param predicted: recommendation
    :param k: how many recommendation
    :return:
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])