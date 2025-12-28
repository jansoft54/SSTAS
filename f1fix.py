from eval.metrics.F1Score import F1Score
import torch
import numpy as np

# --- Dummy f_score import (falls du das echte schon hast, weglassen)
# from eval.metrics.mstcn_code import f_score

# --- F1Score Klasse importieren
# from your_module import F1Score


def run_test_case(name, targets, predictions):
    print(f"\n=== {name} ===")
    metric = F1Score(overlaps=(0.1, 0.25, 0.5),num_classes=5)
    
    result = metric.add(targets, predictions)
    summary = metric.summary()

    print("Batch result:", result)
    print("Summary:", summary)


# -------------------------------
# Testfall 1: Perfekte Vorhersage
# -------------------------------
targets_1 = torch.tensor([
    [0, 0, 0, 1, 1, 1, 2, 2]
])

preds_1 = torch.tensor([
    [0, 0, 0, 1, 1, 1, 2, 2]
])

run_test_case("Perfect prediction", targets_1, preds_1)


# -----------------------------------
# Testfall 2: Kleine Segment-Verschiebung
# -----------------------------------
targets_2 = torch.tensor([
    [0, 0, 0, 1, 1, 1, 2, 2]
])

preds_2 = torch.tensor([
    [0, 0, 1, 1, 1, 2, 2, 2]
])

run_test_case("Slightly shifted segments", targets_2, preds_2)


# -----------------------------------
# Testfall 3: Schlechte Vorhersage
# -----------------------------------
targets_3 = torch.tensor([
    [0, 0, 0, 1, 1, 1, 2, 2]
])

preds_3 = torch.tensor([
    [2, 2, 2, 2, 2, 2, 2, 2]
])

run_test_case("Bad prediction", targets_3, preds_3)
