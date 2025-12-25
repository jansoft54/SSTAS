import torch
import torch.nn as nn
import torch.nn.functional as F
from trainer_config import TrainerConfig
import wandb




class PIMLoss(nn.Module):
    def __init__(self, l=0.2, momentum=0.9, eps=1e-8, known_classes=5, num_classes=10, device=None):
        super().__init__()
        self.l = l
        self.eps = eps
        self.momentum = momentum
        self.device = device
        self.known_classes = known_classes
        self.unknown_classes = num_classes - known_classes
        self.temperature = 5  # Speichern

        self.register_buffer('global_unk_pi', torch.ones(
            self.unknown_classes) / self.unknown_classes)

    def forward(self, logits, targets, unknown_mask, known_mask):
        probs = nn.functional.softmax(logits, dim=-1)
        batch_pi = probs.mean(dim=1)

        probs_unknown_clusters = probs[unknown_mask][:, self.known_classes:]
        probs_unknown_labels = probs[unknown_mask][:, :self.known_classes]

        probs_unknown_clusters = F.softmax(
            probs_unknown_clusters / self.temperature, dim=-1)

      #  print(probs_unknown_clusters[100])

        loss_conditional_unknown = torch.tensor(
            0.0, device=self.device, requires_grad=True)
        loss_marginal = torch.tensor(
            0.0, device=self.device, requires_grad=True)
        loss_separation = torch.tensor(
            0.0, device=self.device, requires_grad=True)
        loss_ce_known = torch.tensor(
            0.0, device=self.device, requires_grad=True)

        if known_mask.sum() > 0:
            loss_ce_known = nn.functional.cross_entropy(
                logits[known_mask], targets[known_mask])

         # conditional loss for unknowns H(y|x)
        if unknown_mask.sum() > 0:
            loss_conditional_unknown = torch.sum(
                probs_unknown_clusters * torch.log(probs_unknown_clusters + self.eps), dim=-1).mean()
            mass_of_unknowns_on_knowns = probs_unknown_labels.sum(dim=-1)
            loss_separation = torch.log(
                mass_of_unknowns_on_knowns + self.eps).mean()

        # marginal loss H(y)
        if unknown_mask.sum() > 0:
            batch_pi = probs_unknown_clusters.mean(dim=0)
        else:
            batch_pi = self.global_unk_pi.detach()

        if self.global_unk_pi.device != self.device:
            self.global_unk_pi = self.global_unk_pi.to(self.device)

        new_pi = self.momentum * self.global_unk_pi + \
            (1 - self.momentum) * batch_pi.detach()
        self.global_unk_pi.copy_(new_pi.squeeze())

        pi = self.momentum * self.global_unk_pi.detach() + (1 - self.momentum) * batch_pi

        log_pi = torch.log(pi + self.eps)
        loss_marginal = torch.sum(pi * log_pi)

        # + (  loss_separation) + (20.0 * loss_marginal)  - (self.l*loss_conditional_unknown)
        return loss_ce_known


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07, max_samples=1000, start=0):
        """
        temperature: Wie "streng" die Cluster sein sollen. 0.07 ist Standard.
        max_samples: Wie viele Frames maximal verglichen werden (Spart GPU Speicher).
        """
        super().__init__()
        self.temperature = temperature
        self.max_samples = max_samples
        self.start = start

    def forward(self, embeddings, labels, known_mask, epoch):
        """
        embeddings: [N, Dim] (Die Features aus ActionBERT)
        labels: [N] (Die Klassen-IDs, z.B. 0, 1, 2...)
        """
        device = embeddings.device

        if epoch < self.start:
            return torch.tensor(0, device=device)

        valid_indices = known_mask.bool()

        embeddings = embeddings[valid_indices]
        labels = labels[valid_indices]

        N = embeddings.shape[0]

        if N > self.max_samples:
            indices = torch.randperm(N, device=device)[:self.max_samples]

            embeddings = embeddings[indices]

            labels = labels[indices]
            N = self.max_samples

        embeddings = F.normalize(embeddings, dim=1)

        similarity_matrix = torch.matmul(embeddings, embeddings.T)

        labels = labels.view(-1, 1)

        mask = torch.eq(labels, labels.T).float()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(N).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        logits = similarity_matrix / self.temperature

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob_denom = torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        log_prob = logits - log_prob_denom

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        loss = -mean_log_prob_pos

        loss = loss[mask.sum(1) > 0].mean()

        if torch.isnan(loss):
            return torch.tensor(0.0, device=device)

        return loss


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

    def forward(self, logits, padding_mask):
        """
        logits: [Batch, Time, Classes]
        padding_mask: [Batch, Time] (True = Valid Data)
        """
        log_probs = F.log_softmax(logits, dim=-1)
        diff = self.mse(log_probs[:, :-1, :], log_probs[:, 1:, :])

        loss_per_frame = diff.mean(dim=-1)

        valid_mask = padding_mask[:, :-1] & padding_mask[:, 1:]

        loss = (loss_per_frame * valid_mask.float()
                ).sum() / (valid_mask.sum() + 1e-8)

        return loss


class SimpleCompactnessLoss(nn.Module):
    def __init__(self, start):
        super().__init__()
        self.start = start

    def forward(self, embeddings, labels, epoch):
        """
        Zieht Embeddings der gleichen Klasse im aktuellen Batch zusammen.
        """
        if epoch < self.start:
            return 0
        unique_labels = torch.unique(labels)
        loss = 0.0
        valid_classes = 0

        embeddings = F.normalize(embeddings, dim=-1)

        for label in unique_labels:

            mask = (labels == label)

            class_feats = embeddings[mask]

            if class_feats.shape[0] > 1:
                center = class_feats.mean(dim=0, keepdim=True)
                center = F.normalize(center, dim=1)
                sim = torch.mm(class_feats, center.t())

                class_loss = (1.0 - sim).mean()

                loss += class_loss
                valid_classes += 1

        if valid_classes > 0:
            return loss / valid_classes
        else:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):

        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        
      #  print(inputs.shape,targets.shape)
        ce_loss = F.cross_entropy(
            inputs, targets,  weight=self.alpha)

        return ce_loss
        pt = torch.exp(-ce_loss)

        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class TotalLoss(nn.Module):
    def __init__(self, trainer_config: TrainerConfig, class_weights):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.pim_loss = PIMLoss(l=1.0,
                                momentum=0.9,
                                eps=1e-8,
                                known_classes=trainer_config.knowns,
                                num_classes=trainer_config.knowns + trainer_config.K,
                                device=self.device)
        self.focal_loss = FocalLoss(alpha=None)
        self.recon_loss = ReconLoss(trainer_config=trainer_config)
        self.smooth_loss = TemporalSmoothnessLoss(
            trainer_config=trainer_config)
        self.info_contrastive_loss = SimpleCompactnessLoss(start=trainer_config.info_start)

        self.pim_weight = trainer_config.pim_loss_weight
        self.recon_weight = trainer_config.recon_loss_weight
        self.smooth_weight = trainer_config.smooth_loss_weight
        self.info_nce_weight = trainer_config.info_nce_weight

        self.class_weights= class_weights

    def forward(self, logits,
                embeddings,
                target_labels,
                recon_features,
                target_features,
                unknown_mask,
                foreground_mask,
                known_mask,
                patch_mask,
                padding_mask,
                epoch):

        zero = torch.tensor(0.0, device=self.device)
        """loss_pim = self.pim_loss(
            logits, target_labels, unknown_mask, known_mask) if self.pim_weight > 0 else zero"""

        
        loss_focal = self.focal_loss(
            logits[known_mask], target_labels[known_mask])
        loss_recon = self.recon_loss(
            recon_features, target_features, patch_mask ) if self.recon_weight > 0 else zero
        loss_smooth = self.smooth_loss(
            logits, padding_mask) if self.smooth_weight > 0 else zero
        loss_info_nce = self.info_contrastive_loss(
            embeddings[known_mask], target_labels[known_mask], epoch) if self.info_nce_weight > 0 else zero

        total_loss = (self.pim_weight * loss_focal +
                      self.recon_weight * loss_recon +
                      self.smooth_weight * loss_smooth +
                      self.info_nce_weight * loss_info_nce)

        """boundary_loss = self.l1(boundaries[:,:,0][~unknown_mask & padding_mask], padding_target_start[~unknown_mask & padding_mask].float()) + self.l1(boundaries[:,:,1][~unknown_mask & padding_mask], padding_target_end[~unknown_mask & padding_mask].float())
                """

        wandb.log({
            "train/CE Loss (Knowns)": loss_focal,
            "train/Recon Loss": loss_recon,
            "train/Smoothness Loss": loss_smooth,
            "train/Prototype Contrastive Loss": loss_info_nce,
            "train/Total Loss": total_loss,
        })
        print(total_loss.item(), f"PIM: {loss_focal}",
              f"Cos: {loss_recon}", f"Smooth: {loss_smooth}")

        return total_loss
