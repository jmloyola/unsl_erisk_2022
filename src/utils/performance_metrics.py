"""
Performance metrics.
Copyright (C) 2022 Juan Martín Loyola

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License version 3 as
published by the Free Software Foundation.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import numpy as np
import sklearn.metrics as metrics


def latency_cost(k, break_point):
    """Latency cost function (lc_o in the original formula).

    Parameters
    ----------
    k : int
        Current delay.
    break_point : int
        Parameter to control the time where the cost grows more quickly. This
        parameter references the parameter `o` from the function `erde`.

    Returns
    -------
    latency_cost : float
        Latency cost.
    """
    return 1 - (1 / (1 + np.exp(k - break_point)))


def erde_user(label, true_label, delay, _c_tp, _c_fn, _c_fp, _o):
    """Calculate the ERDE measure for a user."""
    if label == 1 and true_label == 1:
        return latency_cost(k=delay, break_point=_o) * _c_tp
    elif label == 1 and true_label == 0:
        return _c_fp
    elif label == 0 and true_label == 1:
        return _c_fn
    elif label == 0 and true_label == 0:
        return 0


def erde(labels_list, true_labels_list, delay_list, c_fp, c_tp=1, c_fn=1, o=50):
    """Early risk detection error (ERDE).

    Metric proposed by Losada and Crestani in [1]_.

    Parameters
    ----------
    labels_list : list of int
        Predicted label for each user.
    true_labels_list : list of int
        True label for each user.
    delay_list : list of int
        Decision delay for each user.
    c_fp : float
        False positive cost.
    c_tp : float, default=1
        True positive cost.
    c_fn : float, default=1
        False negative cost.
    o : int, default=50
        Parameter to control the time where the cost grows more quickly.

    Returns
    -------
    erde_metric : float
        ERDE measure.

    References
    ----------
    .. [1] `Losada, D. E., & Crestani, F. (2016, September). A test collection
        for research on depression and language use. In International Conference
        of the Cross-Language Evaluation Forum for European Languages
        (pp. 28-39). Springer, Cham.`_
    """
    erde_list = [
        erde_user(
            label=l,
            true_label=true_labels_list[i],
            delay=delay_list[i],
            _c_tp=c_tp,
            _c_fn=c_fn,
            _c_fp=c_fp,
            _o=o,
        )
        for i, l in enumerate(labels_list)
    ]
    return np.mean(erde_list)


def value_p(k):
    """Get the penalty value for the F latency measure.

    Parameters
    ----------
    k : int
        Median number of posts from the positive users.

    Returns
    -------
    penalty : float
        Penalty to use.
    """
    return -(np.log(1 / 3) / (k - 1))


def f_penalty(k, _p):
    """Get the penalty of the current user delay.

    Parameters
    ----------
    k : int
        Current user delay.
    _p : float
        Penalty.

    Returns
    -------
    f_penalty : float
        Penalty latency.
    """
    return -1 + (2 / (1 + np.exp((-_p) * (k - 1))))


def speed(y_pred, y_true, d, p):
    """Get speed for every user correctly classified as positive."""
    penalty_list = [
        f_penalty(k=d[i], _p=p)
        for i in range(len(y_pred))
        if y_pred[i] == 1 and y_true[i] == 1
    ]

    if len(penalty_list) != 0:
        return 1 - np.median(penalty_list)
    else:
        return 0.0


def f_latency(labels, true_labels, delays, penalty):
    """F latency metric.

    Metric proposed by Sadeque and others in [1]_.

    Parameters
    ----------
    labels : list of int
        Predicted label for each user.
    true_labels : list of int
        True label for each user.
    delays : list of int
        Decision delay for each user.
    penalty : float
        Penalty. Defines how quickly the penalty should increase.

    Returns
    -------
    f_latency_metric : float
        F latency measure.

    References
    ----------
    .. [1] `Sadeque, F., Xu, D., & Bethard, S. (2018, February). Measuring the
        latency of depression detection in social media. In Proceedings of the
        Eleventh ACM International Conference on Web Search and Data Mining
        (pp. 495-503).`_
    """
    f1_score = metrics.f1_score(y_pred=labels, y_true=true_labels, average="binary")
    speed_value = speed(y_pred=labels, y_true=true_labels, d=delays, p=penalty)

    return f1_score * speed_value


def precision_at_k(scores, y_true, k=10):
    """Precision at k."""
    scores = np.asarray(scores)
    y_true = np.asarray(y_true)
    idx = np.argsort(-scores)
    scores_sorted = scores[idx]
    y_true_sorted = y_true[idx]
    if len(scores_sorted) > k:
        y_true_sorted = y_true_sorted[:k]

    return np.sum(y_true_sorted) / k


def dcg(relevance, rank):
    """Discounted cumulative gain."""
    relevance = np.asarray(relevance)[:rank]
    n_relevances = len(relevance)
    if n_relevances == 0:
        return 0.0

    discounts = np.log2(np.arange(n_relevances) + 2)
    return np.sum((2**relevance - 1) / discounts)


def ndcg(scores, y_true, p):
    """Normalized discounted cumulative gain."""
    best_dcg = dcg(relevance=sorted(y_true, reverse=True), rank=p)
    current_dcg = dcg(relevance=sorted(scores, reverse=True), rank=p)

    return current_dcg / best_dcg
