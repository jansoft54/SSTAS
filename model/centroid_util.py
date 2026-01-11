
import torch
from trainer_config import TrainerConfig
import wandb
import torch.nn.functional as F


def update_centroids(config: TrainerConfig, model, model_out, target_truth, known_mask, unknown_mask=None):
    """
    Universelle Funktion für das Zentren-Update.
    Nutzt den gemeinsamen Buffer self.model.class_centers.
    """

    with torch.no_grad():
        # Da du im forward result["embeddings"] = x (oder x_for_unk) setzt,
        # greifen wir hier immer auf den aktuellen Datenstrom zu.
        current_embs = model_out["embeddings"]
        logits = model_out["refine_logits"]
        # --- TEIL 1: Bekannte Klassen (immer 0 bis N-1) ---
        if known_mask.any():
            embs_k = current_embs[known_mask]
            lbls_k = target_truth[known_mask]

            for c in range(len(config.known_classes)):
                class_mask = (lbls_k == c)
                if class_mask.any():
                    batch_mean = embs_k[class_mask].mean(dim=0)
                    batch_mean = F.normalize(batch_mean, p=2, dim=0)

                    if not model.centers_initialized[c]:
                        model.class_centers[c] = batch_mean
                        model.centers_initialized[c] = True
                    else:
                        model.class_centers[c] = model.center_momentum * model.class_centers[c] + \
                            (1 - model.center_momentum) * batch_mean
                        model.class_centers[c] = F.normalize(
                            model.class_centers[c], p=2, dim=0)

        # --- TEIL 2: Globales Unknown-Zentrum (Index N) ---
        # Wir prüfen, ob der Buffer Platz für N+1 Klassen hat
        if model.class_centers.shape[0] > len(config.known_classes) and unknown_mask is not None and unknown_mask.any():

            unk_embs = current_embs[unknown_mask]     # [N_unk_frames, D]
            unk_logits = logits[unknown_mask]         # [N_unk_frames, 24]

            # 1. Bestimme für jeden Frame den "Winner-Slot" unter den 10 Pseudo-Klassen
            # Wir schauen nur auf die Indizes ab 14
            # [N_unk_frames, 10]
            pseudo_logits = unk_logits[:, len(config.known_classes):]
            winning_slots = torch.argmax(
                pseudo_logits, dim=-1)  # Werte von 0 bis 9

            # 2. Iteriere über die 10 Slots und update deren Zentren einzeln
         #   raise Exception(len(pseudo_logits[0]))
            num_pseudo = pseudo_logits.shape[1]
            for s in range(num_pseudo):
                slot_mask = (winning_slots == s)

                if slot_mask.any():
                    # Der echte Index im globalen class_centers Buffer
                    target_idx = len(config.known_classes) + s

                    # Berechne den Mittelwert der Frames, die diesem Slot zugeordnet wurden
                    batch_mean_s = unk_embs[slot_mask].mean(dim=0)
                    batch_mean_s = F.normalize(batch_mean_s, p=2, dim=0)

                    if not model.centers_initialized[target_idx]:
                        model.class_centers[target_idx] = batch_mean_s
                        model.centers_initialized[target_idx] = True
                    else:
                        # EMA Update
                        model.class_centers[target_idx] = model.center_momentum * model.class_centers[target_idx] + \
                            (1 - model.center_momentum) * batch_mean_s
                        model.class_centers[target_idx] = F.normalize(
                            model.class_centers[target_idx], p=2, dim=0)
