import torch
import torch.nn as nn
import torch.nn.functional as F
from trainer_config import TrainerConfig
import wandb


class VideoProgressLoss(nn.Module):
    def __init__(self, weight=5.0):
        """
        weight: Da MSE-Werte bei Werten zwischen 0-1 oft klein sind, 
                hilft ein höheres Gewicht (z.B. 5.0), um den Backbone zu trainieren.
        """
        super().__init__()
        self.weight = weight

    def forward(self, pred, padding_mask):
        """
        pred: [B, T, 1] - Der Sigmoid-Output deines Progress-Heads
        padding_mask: [B, T] - 1 für Daten, 0 für Padding
        """
        # 1. Dimension anpassen: [B, T, 1] -> [B, T]
        pred = pred.squeeze(-1)
        device = pred.device

        # 2. Ground Truth (0.0 bis 1.0) aus Maske generieren
        # Berechne tatsächliche Länge pro Video im Batch
        T_actual = padding_mask.sum(dim=-1, keepdim=True)  # [B, 1]

        # Erzeuge Frame-Indizes (0, 1, 2...)
        indices = torch.cumsum(padding_mask.long(), dim=-1) - 1
        indices = indices.clamp(min=0)  # Padding-Indizes auf 0 halten

        # Normieren auf Bereich [0, 1]
        # (T_actual - 1), damit der letzte Frame exakt 1.0 ist
        denom = (T_actual - 1).clamp(min=1)
        gt = indices.float() / denom.float()

        # 3. MSE berechnen
        # Wir berechnen den quadratischen Fehler für jeden Frame
        loss_map = (pred - gt).pow(2)

        # 4. Nur valide Frames (kein Padding) in den Durchschnitt nehmen
        # loss_map[padding_mask] gibt einen flachen Vektor aller validen Fehler zurück
        final_loss = loss_map[padding_mask].mean()

        return self.weight * final_loss


class ReconLoss(nn.Module):
    def __init__(self, trainer_config: TrainerConfig):
        super().__init__()
        self.criterion = nn.CosineEmbeddingLoss()
        self.l1_criterion = torch.nn.L1Loss()
        self.trainer_config = trainer_config

    def forward(self, recon_features, target_features, mask):
        target_ones = torch.ones(recon_features[mask].size(0)).to(
            recon_features.device)
        loss_recon = self.criterion(
            recon_features[mask], target_features[mask], target_ones)
        loss_l1 = self.l1_criterion(
            recon_features[mask], target_features[mask])
        wandb.log({
            "Reconstruction Cosine Loss": loss_recon,
            "Reconstruction L1 Loss": loss_l1,
        })
        return self.trainer_config.cosine_recon_weight * loss_recon + self.trainer_config.l1_recon_weight * loss_l1


class TemporalSmoothnessLoss(nn.Module):
    def __init__(self, trainer_config: TrainerConfig):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.trainer_config = trainer_config
        self.mse_clip_val = 16

    def forward(self, p, padding_mask):
        """
        p: Logits vom Modell [B, T, C]
        padding_mask: [B, T] (1 für Daten, 0 für Padding)
        """
        # 1. Log-Softmax über die KLASSEN (letzte Dimension)
        # LTContext nutzt dim=1, weil dort C an Stelle 1 steht.
        # Bei dir ist es dim=-1 oder dim=2!
        p_log = F.log_softmax(p, dim=-1)

        # 2. MSE zwischen t (1:) und t-1 (:-1)
        # Das .detach() am Vorgänger ist entscheidend für die Trägheit!
        loss = self.mse(p_log[:, 1:, :], p_log[:, :-1, :].detach())

        # 3. Truncation (Kappen der Gradienten an echten Grenzen)
        loss = torch.clamp(loss, min=0, max=self.mse_clip_val)

        # 4. Padding berücksichtigen
        # Die Maske muss um 1 Frame gekürzt werden, da wir Differenzen rechnen
        mask = padding_mask[:, 1:].unsqueeze(-1)

        return (loss * mask).mean()


class CentroidContrastiveLoss(nn.Module):
    def __init__(self, pull_weight=1.0, push_weight=0.75, margin=0.1):
        super().__init__()
        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.margin = margin  # Wie weit müssen andere Zentren mindestens weg sein?

    def forward(self, embeddings, targets, centers, prototypes, known_mask):
        """
        embeddings: [N, 256] (aus dem Transformer, normalisiert)
        targets: [N] (Ground Truth IDs)
        centers: [NumClasses, 256] (EMA Zentren aus dem Modell-Buffer)
        """
        if not known_mask.any():
            return torch.tensor(0.0, device=embeddings.device)

        emb = embeddings[known_mask]
        lbl = targets[known_mask]

        emb = F.normalize(emb, p=2, dim=-1)

        sim_matrix = torch.matmul(emb, centers.detach().t())

        temp = 0.07
        correct_sims = sim_matrix[torch.arange(emb.size(0)), lbl]
        pull_errors = 1.0 - correct_sims
        unique_labels = torch.unique(lbl)
        class_pull_losses = []
        for label in unique_labels:
            mask = (lbl == label)
            class_pull_losses.append(pull_errors[mask].mean())
        loss_pull = torch.stack(class_pull_losses).mean()

        scaled_sim_matrix = sim_matrix / temp

        K = centers.size(0)
        mask_other = torch.ones_like(scaled_sim_matrix, dtype=torch.bool)
        mask_other[torch.arange(emb.size(0)), lbl] = False

        class_push_losses = []
        for label in unique_labels:
            mask = (lbl == label)
            other_sims_for_class = scaled_sim_matrix[mask][mask_other[mask]]
            push_err = torch.clamp(
                other_sims_for_class - self.margin/temp, min=0)
            if push_err.numel() > 0:
                class_push_losses.append(push_err.mean())

        loss_push = torch.stack(class_push_losses).mean(
        ) if class_push_losses else torch.tensor(0.0).to(emb.device)

        w_norm = F.normalize(prototypes, p=2, dim=-1)
        class_sim = torch.matmul(w_norm, w_norm.t())

        K = w_norm.size(0)
        loss_ortho = (
            class_sim - torch.eye(K).to(prototypes.device)).pow(2).mean()

        wandb.log({
            "Contrastive LOSS loss_push": loss_push,
            "Contrastive LOSS loss_pull": loss_pull,
            "Contrastive LOSS loss_ortho": loss_ortho,
        })

        return (self.pull_weight * loss_pull) + (self.push_weight * loss_push) + 10.0 * loss_ortho


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0, alpha=None, reduction='mean'):

        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets, mask, is_focal=False):

      #  print(inputs.shape,targets.shape)
        inputs = inputs[mask]
        targets = targets[mask]
        ce_loss = F.cross_entropy(
            inputs, targets,  weight=self.alpha, label_smoothing=0)

        if not is_focal:
            return ce_loss
        pt = torch.exp(-ce_loss)

        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class MulticlassDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets, combined_mask):

        probs = torch.softmax(logits, dim=-1)

        C = probs.size(-1)
        targets_safe = torch.clamp(targets, 0, C - 1)

        targets_one_hot = F.one_hot(targets_safe, num_classes=C).float()

        m = combined_mask.unsqueeze(-1)
        probs = probs * m
        targets_one_hot = targets_one_hot * m

        intersection = (probs * targets_one_hot).sum(dim=1)
        union = probs.sum(dim=1) + targets_one_hot.sum(dim=1)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1. - dice.mean()


class TotalLoss(nn.Module):
    def __init__(self, trainer_config: TrainerConfig, class_weights):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=None)

        self.focal_loss = FocalLoss(alpha=None)
        self.unk_class_weights = torch.tensor([1.0, 5.0]).to(self.device)

        self.focal_loss_unknowns = FocalLoss(
            gamma=1.0, alpha=self.unk_class_weights)

        self.recon_loss = ReconLoss(trainer_config=trainer_config)
        self.smooth_loss = TemporalSmoothnessLoss(
            trainer_config=trainer_config)
        self.contrastive_loss = CentroidContrastiveLoss()
        self.dice_loss = MulticlassDiceLoss()
        self.video_progress_loss = VideoProgressLoss(weight=5.0)

        self.ce_weight = trainer_config.pim_loss_weight
        self.recon_weight = trainer_config.recon_loss_weight
        self.smooth_weight = trainer_config.smooth_loss_weight

        self.class_weights = class_weights

    def forward(self, loss_dict):

        zero = torch.tensor(0.0, device=self.device)
        """loss_pim = self.pim_loss(
            logits, target_labels, unknown_mask, known_mask) if self.pim_weight > 0 else zero"""
        logits = loss_dict["logits"]
        stage_output_logits = loss_dict["stages_output_logits"]

        known_mask = loss_dict["known_mask"]
        unknown_mask = loss_dict["unknown_mask"]

        target_labels = loss_dict["target_labels"]

        padding_mask = loss_dict["padding_mask"]
        embeddings = loss_dict["embeddings"]
        centers = loss_dict["centers"]
        prototypes = loss_dict["prototypes"]
        epoch = loss_dict["epoch"]

        loss_dice_stage_first = self.dice_loss(
            logits, target_labels, padding_mask & known_mask)
        loss_focal_stage_first = self.focal_loss(
            logits, target_labels, padding_mask & known_mask)
        """loss_recon_stage_first = self.recon_loss(
            recon_features, target_features, patch_mask) if self.recon_weight > 0 else zero"""

        loss_smooth_stage_first = self.smooth_loss(
            logits, padding_mask) if self.smooth_weight > 0 else zero

        loss_contrastive_stage_first = self.contrastive_loss(
            embeddings, target_labels, centers, prototypes, known_mask) if epoch >= 15 else zero

        loss_smooth_stage_refine = sum([self.smooth_loss(
            refine_logits_, padding_mask) for _, refine_logits_ in enumerate(stage_output_logits)]) / len(stage_output_logits)

        loss_dice_stage_refine = sum(
            [self.dice_loss(refine_logits_, target_labels, padding_mask & known_mask) for _, refine_logits_ in enumerate(stage_output_logits)]) / len(stage_output_logits)

        loss_focal_stage_refine = sum([self.focal_loss(
            refine_logits_, target_labels, padding_mask & known_mask) for _, refine_logits_ in enumerate(stage_output_logits)]) / len(stage_output_logits)

        total_loss = (1 * loss_dice_stage_first +
                      1 * loss_focal_stage_first +
                      1.5 * loss_smooth_stage_first +
                      1.5 * loss_contrastive_stage_first +
                      1 * loss_dice_stage_refine +
                      1.5 * loss_smooth_stage_refine +
                      1 * loss_focal_stage_refine


                      )

        wandb.log({
            "train/Dice Loss ": loss_dice_stage_first + loss_dice_stage_refine,
            "train/CE Loss ": loss_focal_stage_first + loss_focal_stage_refine,

            "train/Smoothness Loss ": loss_smooth_stage_refine,
            #  "train/Progress Loss ": ,




            "train/Prototype Contrastive Loss": loss_contrastive_stage_first,
            "train/Total Loss": total_loss,
        })
        print(total_loss.item(), f"DICE: {loss_dice_stage_refine}", f"CE: {loss_focal_stage_refine}",
              )

        return total_loss
