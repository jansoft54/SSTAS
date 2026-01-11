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
    def __init__(self, pull_weight=1.0, push_weight=0.75, margin=0.1):
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



class KnownVotingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_dists, gt_start,gt_end):
        
        valid = (gt_start != -1)
        
        if not valid.any():
            return torch.tensor(0.0, device=pred_dists.device)

        p_start = pred_dists[:, 0, :]
        p_end   = pred_dists[:, 1, :]

  
        loss_s = F.mse_loss((p_start[valid]), torch.log1p(gt_start[valid]))
        loss_e = F.mse_loss((p_end[valid]),   torch.log1p(gt_end[valid]))

        return loss_s + loss_e

import torch
import torch.nn as nn

class MMDLoss(nn.Module):
    def __init__(self, max_samples=2000, kernel_num=5, kernel_mul=2.0):
        super().__init__()
        self.max_samples = max_samples 
        self.kernel_num = kernel_num   
        self.kernel_mul = kernel_mul   

    def forward(self, pred_log_dists, known_mask, unknown_mask):
        d_start = torch.expm1(pred_log_dists[:, 0, :])
        d_end   = torch.expm1(pred_log_dists[:, 1, :])
        
   
        total_len = d_start + d_end + 1e-6
        relative_progress = d_start / total_len
        
     
        source_samples = relative_progress[known_mask].detach().unsqueeze(1)
        
        target_samples = relative_progress[unknown_mask].unsqueeze(1)
        
        if source_samples.numel() < 10 or target_samples.numel() < 10:
            return torch.tensor(0.0, device=pred_log_dists.device)

   
        
        if source_samples.shape[0] > self.max_samples:
            idx = torch.randperm(source_samples.shape[0])[:self.max_samples]
            source_samples = source_samples[idx]
            
        if target_samples.shape[0] > self.max_samples:
            idx = torch.randperm(target_samples.shape[0])[:self.max_samples]
            target_samples = target_samples[idx]
            
        min_len = min(source_samples.shape[0], target_samples.shape[0])
        source_samples = source_samples[:min_len]
        target_samples = target_samples[:min_len]

        loss = self._compute_mmd(source_samples, target_samples)
        
        return loss


    def _gaussian_kernel(self, source, target):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        
        # Effiziente Distanzmatrix [N+M, N+M] durch Broadcasting
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1)**2).sum(2) 
        
        # Heuristik für Sigma (Bandwidth): Median der Distanzen
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
        
        # Multi-Scale Kernel (Verschiedene Sigmas für feine und grobe Details)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul**i) for i in range(self.kernel_num)]
        
        # Summe der Kernel-Antworten
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def _compute_mmd(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self._gaussian_kernel(source, target)
        
        # Die 4 Quadranten der Kernel-Matrix
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        
        return torch.mean(XX + YY - XY - YX)




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
        self.action_boundary_loss  = KnownVotingLoss()
        self.mmd_loss = MMDLoss()
        
        
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
        
        gt_dist_start = loss_dict["gt_dist_start"]
        gt_dist_end  = loss_dict["gt_dist_end"]
    
        action_progress = loss_dict["action_progress"]

        
        loss_dice_stage_first = self.dice_loss(
            logits, target_labels, padding_mask & known_mask)
        loss_focal_stage_first = self.focal_loss(
            logits, target_labels, padding_mask & known_mask)
      
        loss_smooth_stage_first = self.smooth_loss(
            logits, padding_mask& known_mask) if self.smooth_weight > 0 else zero



        loss_contrastive_stage_first = self.contrastive_loss(
            embeddings, target_labels, centers, prototypes, known_mask) if epoch >= 15 else zero



        loss_smooth_stage_refine = sum([self.smooth_loss(
            refine_logits_, padding_mask& known_mask) for _, refine_logits_ in enumerate(stage_output_logits)]) / len(stage_output_logits)

        loss_dice_stage_refine = sum(
            [self.dice_loss(refine_logits_, target_labels, padding_mask & known_mask) for _, refine_logits_ in enumerate(stage_output_logits)]) / len(stage_output_logits)

        loss_focal_stage_refine = sum([self.focal_loss(
            refine_logits_, target_labels, padding_mask & known_mask) for _, refine_logits_ in enumerate(stage_output_logits)]) / len(stage_output_logits)


        loss_action_boundary =  self.action_boundary_loss(
            action_progress, gt_dist_start, gt_dist_end)
        
        
        loss_mmd = self.mmd_loss(action_progress, known_mask, unknown_mask)
        
        
        total_loss = (1 * loss_dice_stage_first +
                      1 * loss_focal_stage_first +
                      1.5 * loss_smooth_stage_first +
                      1 * loss_contrastive_stage_first +
                      1 * loss_dice_stage_refine +
                      1.5* loss_smooth_stage_refine +
                      1 * loss_focal_stage_refine +
                      1 * loss_action_boundary + 
                      5 * loss_mmd

                      )
        wandb.log({
            "train/Dice Loss ": loss_dice_stage_first + loss_dice_stage_refine,
            "train/CE Loss ": loss_focal_stage_first + loss_focal_stage_refine,

            "train/Smoothness Loss ": loss_smooth_stage_refine,
            "train/Action Boundary Loss ":loss_action_boundary ,
            "train/MMD Loss":loss_mmd ,

            "train/Prototype Contrastive Loss": loss_contrastive_stage_first,
            "train/Total Loss": total_loss,
        })
        print(total_loss.item(), f"DICE: {loss_dice_stage_refine}", f"CE: {loss_focal_stage_refine}",
              )

        return total_loss
