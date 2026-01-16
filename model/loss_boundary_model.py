import torch
import torch.nn as nn
import torch.nn.functional as F
from trainer_config import TrainerConfig
import wandb


class TemporalSmoothnessLoss(nn.Module):
    def __init__(self, trainer_config: TrainerConfig):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.trainer_config = trainer_config
        self.mse_clip_val = 16

    def forward(self, p, padding_mask):

        p_log = F.log_softmax(p, dim=-1)

        loss = self.mse(p_log[:, 1:, :], p_log[:, :-1, :].detach())

        loss = torch.clamp(loss, min=0, max=self.mse_clip_val)

        mask = padding_mask[:, 1:].unsqueeze(-1)

        return (loss * mask).mean()


class CentroidContrastiveLoss(nn.Module):
    def __init__(self, pull_weight=1.25, push_weight=1.25, margin=0):
        super().__init__()
        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.margin = margin

    def forward(self, embeddings, targets, centers, prototypes, known_mask):

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

 
import torch
import torch.nn as nn
import torch.nn.functional as F

class UnknownOrthogonalityLoss(nn.Module):
    def __init__(self, weight=1.0, margin=0.05):
        super().__init__()
        self.weight = weight
        self.margin = margin 

    def forward(self, embeddings, unknown_mask, global_centers):
        if not unknown_mask.any():
            return torch.tensor(0.0, device=embeddings.device)

        unk_emb = embeddings[unknown_mask]
        
      
        unk_norm = F.normalize(unk_emb, p=2, dim=-1)
        centers_norm = F.normalize(global_centers, p=2, dim=-1)
    
   
        sim_matrix = torch.abs(torch.matmul(unk_norm, centers_norm.detach().t()))
  
        closest_sim, _ = sim_matrix.max(dim=1) # [N_unk]
        
        loss = torch.clamp(closest_sim - self.margin, min=0).pow(2).mean()
        
        return self.weight * loss
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class UnknownSharpeningLoss(nn.Module):
    def __init__(self, weight=1.0, num_samples=512):
        super().__init__()
        self.weight = weight
        self.num_samples = num_samples 

    def forward(self, embeddings, unknown_mask):
        if not unknown_mask.any():
            return torch.tensor(0.0, device=embeddings.device)

        unk_emb = embeddings[unknown_mask]
        N = unk_emb.size(0)

        if N < 2: return torch.tensor(0.0, device=embeddings.device)
        
        limit = min(N, self.num_samples)
        
        idx_a = torch.randperm(N, device=embeddings.device)[:limit]
        idx_b = torch.randperm(N, device=embeddings.device)[:limit]

        feats_a = F.normalize(unk_emb[idx_a], p=2, dim=-1)
        feats_b = F.normalize(unk_emb[idx_b], p=2, dim=-1)

        sims = (feats_a * feats_b).sum(dim=-1)
        sims_clamped = torch.clamp(sims, min=0.0, max=1.0)
     
        loss_sharpen = 4 * sims_clamped * (1.0 - sims_clamped)

        return self.weight * loss_sharpen.mean()






import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedSmoothnessLoss(nn.Module):
    def __init__(self, steps=[1, 2, 4, 8, 16,32,64,128], decay_factor=0.5, weight=1.0):
        super().__init__()
        self.steps = steps
        self.decay_factor = decay_factor # Wie stark die Kraft mit der Distanz abnimmt
        self.weight = weight

    def forward(self, embeddings, padding_mask, unknown_mask=None):
   
        if unknown_mask is not None:
            valid_mask = padding_mask & unknown_mask
        else:
            valid_mask = padding_mask

        if not valid_mask.any():
            return torch.tensor(0.0, device=embeddings.device)

        emb_norm = F.normalize(embeddings, p=2, dim=-1)
        B, T, D = emb_norm.shape
        
        total_loss = torch.tensor(0.0, device=embeddings.device)
        total_weights = 0.0
        
        for i, step in enumerate(self.steps):
            if T <= step: continue
            
        
            sim = (emb_norm[:, :-step, :] * emb_norm[:, step:, :]).sum(dim=-1)
            
        
            mask_steps = valid_mask[:, :-step] & valid_mask[:, step:]
            
            if mask_steps.any():
             
                step_weight = self.decay_factor ** i 
                
                # Wir wollen Sim = 1.0
                loss_step = (1.0 - sim[mask_steps]).mean()
                
                total_loss += step_weight * loss_step
                total_weights += step_weight

        # Normalisieren, damit der Loss nicht wächst, wenn wir mehr Steps hinzufügen
        if total_weights > 0:
            return self.weight * (total_loss / total_weights)
        else:
            return torch.tensor(0.0, device=embeddings.device)



class TotalLoss(nn.Module):
    def __init__(self, trainer_config: TrainerConfig, class_weights=None):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.focal_loss = FocalLoss(alpha=None)

        self.smooth_loss = TemporalSmoothnessLoss(
            trainer_config=trainer_config)
        self.contrastive_loss = CentroidContrastiveLoss()
        self.dice_loss = MulticlassDiceLoss()
        
        self.unk_contrastive_loss = UnknownOrthogonalityLoss()
        self.unk_sharpening_loss = UnknownSharpeningLoss()
        self.unk_smoothing_loss = DilatedSmoothnessLoss()

        self.ce_weight = trainer_config.pim_loss_weight
        self.recon_weight = trainer_config.recon_loss_weight
        self.smooth_weight = trainer_config.smooth_loss_weight

    def forward(self, loss_dict):

        zero = torch.tensor(0.0, device=self.device)

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

        loss_smooth_stage_first = self.smooth_loss(
            logits, padding_mask & known_mask) if self.smooth_weight > 0 else zero

        loss_contrastive_stage_first = self.contrastive_loss(
            embeddings, target_labels, centers, prototypes, known_mask) if epoch >= 0 else zero

        loss_smooth_stage_refine = sum([self.smooth_loss(
            refine_logits_, padding_mask  & known_mask) for _, refine_logits_ in enumerate(stage_output_logits)]) / len(stage_output_logits)

        loss_dice_stage_refine = sum(
            [self.dice_loss(refine_logits_, target_labels, padding_mask & known_mask) for _, refine_logits_ in enumerate(stage_output_logits)]) / len(stage_output_logits)

        loss_focal_stage_refine = sum([self.focal_loss(
            refine_logits_, target_labels, padding_mask & known_mask) for _, refine_logits_ in enumerate(stage_output_logits)]) / len(stage_output_logits)
        

        loss_unk_contrastive = self.unk_contrastive_loss (embeddings,unknown_mask,centers)
        loss_unk_sharpening = self.unk_sharpening_loss(embeddings,unknown_mask)
        loss_unk_smooting = self.unk_smoothing_loss(embeddings,padding_mask,unknown_mask)
        
        total_loss = (1 * loss_dice_stage_first +
                      1 * loss_focal_stage_first +
                      1.5 * loss_smooth_stage_first +
                      1.75 * loss_contrastive_stage_first +
                      1 * loss_dice_stage_refine +
                      1.5 * loss_smooth_stage_refine +
                      1 * loss_focal_stage_refine +
                      
                      5 * loss_unk_contrastive + 
                      1.75* loss_unk_sharpening + 
                      1.5 * loss_unk_smooting



                      )
        wandb.log({
            "train/Dice Loss ": loss_dice_stage_first + loss_dice_stage_refine,
            "train/CE Loss ": loss_focal_stage_first + loss_focal_stage_refine,

            "train/Smoothness Loss ": loss_smooth_stage_refine,

            "train/UNK Contrastive Loss": loss_unk_contrastive,
            "train/UNK Contrastive Sharpening Loss": loss_unk_sharpening,
             "train/UNK SMOOTHING  Loss": loss_unk_smooting,
            
            "train/Prototype Contrastive Loss": loss_contrastive_stage_first,
            "train/Total Loss": total_loss,
        })
        print(total_loss.item(), f"DICE: {loss_dice_stage_refine}", f"CE: {loss_focal_stage_refine}",
              )

        return total_loss
