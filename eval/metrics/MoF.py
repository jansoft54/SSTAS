from typing import Iterable, Union
import torch
import numpy as np

class MacroMoFAccuracyMetric():
    def __init__(self, ignore_ids: Iterable[int] = (), window_size:  int = 1):
        self.ignore_ids = ignore_ids

        self.reset()

    # noinspection PyAttributeOutsideInit
    def reset(self):
        self.total = 0
        self.correct = 0
       # self.deque.clear()

    """def get_deque_median(self):
        return np.median(self.deque)
    """
    def careful_divide(correct: Union[int, float], total: int, zero_value: float = 0.0) -> float:
        if total == 0:
            return zero_value
        else:
            return correct / total
    
    def add(self, targets, predictions) -> float:
        """

        :param targets: torch tensor of shape [batch_size, seq_len]
        :param predictions: torch tensor of shape [batch_size, seq_len]
        :return:
        """
        targets, predictions = np.array(targets), np.array(predictions)
        masks = np.logical_not(np.isin(targets, self.ignore_ids))
        current_total = masks.sum()
        current_correct = (targets == predictions)[masks].sum()
        current_result = MacroMoFAccuracyMetric.careful_divide(current_correct, current_total)
        self.correct += current_correct
        self.total += current_total
        #self.deque.append(current_result)

        return current_result

    def summary(self) -> float:
        return MacroMoFAccuracyMetric.careful_divide(self.correct, self.total)

    def name(self) -> str:
        if self.ignore_ids:
            return "MoF-BG"
        else:
            return "MoF"