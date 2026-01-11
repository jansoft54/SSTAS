
import Levenshtein
from itertools import groupby
from typing import Iterable
from eval.metrics.mstcn_code import edit_score
import torch
import numpy as np


class Edit():
    def __init__(self, ignore_ids: Iterable[int] = (), window_size: int = 1):
        self.ignore_ids = ignore_ids
        self.values = []
        # self.deque.clear()

    """def get_deque_median(self):
        return np.median(self.deque)
    """

    def add(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> float:
        """

        :param targets: torch tensor with shape [batch_size, seq_len]
        :param predictions: torch tensor with shape [batch_size, seq_len]
        :return:
        """
        targets, predictions = np.array(targets), np.array(predictions)
        for target, pred in zip(targets, predictions):
            """ mask = np.logical_not(np.isin(target, self.ignore_ids))
            target = target[mask]
            pred = pred[mask]
            """
            current_score = edit_score(
                recognized=pred.tolist(),
                ground_truth=target.tolist(),
                ignored_classes=self.ignore_ids,
            )

            self.values.append(current_score)
            # self.deque.append(current_score)

        return current_score

    def summary(self) -> float:
        if len(self.values) > 0:
            return np.array(self.values).mean()
        else:
            return 0.0


def calc_subset_edit_score(pred, gt, mask, keep_ids):

    p_list = pred[mask.bool()].tolist()
    g_list = gt[mask.bool()].tolist()

    p_segs = [k for k, _ in groupby(p_list)]
    g_segs = [k for k, _ in groupby(g_list)]

    target_set = set(keep_ids)
    p_clean = [x for x in p_segs if x in target_set]
    g_clean = [x for x in g_segs if x in target_set]

    if not p_clean and not g_clean:
        return 100.0

    p_str = "".join(chr(int(x) + 5000) for x in p_clean)
    g_str = "".join(chr(int(x) + 5000) for x in g_clean)

    dist = Levenshtein.distance(p_str, g_str)
    maxlen = max(len(p_clean), len(g_clean))

    return (1 - dist / maxlen) * 100
