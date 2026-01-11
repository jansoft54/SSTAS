from typing import List
import numpy as np
import torch
from eval.metrics.mstcn_code import f_score_per_class


class F1Score():
    def __init__(
        self,
        overlaps: List[float] = (0.1, 0.25, 0.5),
        ignore_ids: List[int] = (),
        window_size: int = 1,
        num_classes: int = None,
    ):
        self.overlaps = overlaps
        self.ignore_ids = ignore_ids
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        if self.num_classes is None:
            # Fallback (sollte vermieden werden durch korrekte Init)
            shape = (len(self.overlaps), 1)
        else:
            shape = (len(self.overlaps), self.num_classes)

        self.tp = np.zeros(shape)
        self.fp = np.zeros(shape)
        self.fn = np.zeros(shape)

    def add(
            self,
            targets: torch.Tensor,
            predictions: torch.Tensor
    ) -> dict:
        """
        Fügt Batch-Daten hinzu und gibt den Score für diesen Batch zurück.
        """
        targets = np.array(targets)
        predictions = np.array(predictions)

        # 'current_result' speichert den Score des LETZTEN Videos im Batch
        # (oder man könnte den Durchschnitt des Batches zurückgeben)
        current_result = {}

        for target, pred in zip(targets, predictions):
            current_result = {}  # Reset pro Video für Rückgabewert

            for s in range(len(self.overlaps)):
                # 1. Hole Arrays pro Klasse
                tp_arr, fp_arr, fn_arr = f_score_per_class(
                    recognized=pred.tolist(),
                    ground_truth=target.tolist(),
                    overlap=self.overlaps[s],
                    num_classes=self.num_classes,
                    ignored_classes=self.ignore_ids,
                )

                # 2. Addiere zu globalen Stats (für summary am Ende)
                self.tp[s] += tp_arr
                self.fp[s] += fp_arr
                self.fn[s] += fn_arr

                # 3. Berechne lokalen Score für dieses Video (für Logs)
                # Wir nutzen hier auch die vektorisierte Funktion
                f1_vec_video = self.get_vectorized_f1(tp_arr, fp_arr, fn_arr)

                # Wir nehmen den Mittelwert über alle Klassen als "Video Score"
                current_f1 = np.mean(f1_vec_video)

                current_result[f"F1@{int(self.overlaps[s]*100)}"] = current_f1

        return current_result

    def summary(self, relevant_ids: List[int] = None) -> dict:
        result = {}
        for s in range(len(self.overlaps)):
            # 1. Berechne F1 für JEDE Klasse einzeln (basierend auf globalen Summen)
            f1_per_class = self.get_vectorized_f1(
                self.tp[s], self.fp[s], self.fn[s])

            # 2. Filtern nach IDs
            if relevant_ids is not None:
                # Nur gültige IDs nehmen
                valid_ids = [i for i in relevant_ids if i < len(f1_per_class)]

                # Debugging (optional)
                # print(f"Summary Filter: {len(valid_ids)} classes selected.")

                if len(valid_ids) > 0:
                    current_subset = f1_per_class[valid_ids]
                else:
                    current_subset = [0.0]
            else:
                # Global (alle Klassen)
                current_subset = f1_per_class

            # 3. Durchschnitt berechnen (Macro Average)
            result[f"F1@{int(self.overlaps[s]*100)}"] = np.mean(current_subset)

        return result

    @staticmethod
    def get_vectorized_f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
        """
        Berechnet F1 Score element-weise für Arrays.
        """
        # Kleines Epsilon gegen Division durch Null
        eps = 1e-7

        # Berechnung direkt auf den Arrays
        # (2 * TP) / (2 * TP + FP + FN)
        # Das ist mathematisch identisch zu 2*(Prec*Rec)/(Prec+Rec)

        numerator = 2 * tp
        denominator = 2 * tp + fp + fn + eps

        return numerator / denominator
