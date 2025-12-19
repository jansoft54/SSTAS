import torch
import torch.nn as nn
import torch.nn.functional as F


class PIMLoss(nn.Module ):
    def __init__(self,l=0.2,momentum=0.9,eps=1e-8,known_classes = 5,num_classes=10,device = None):
        super().__init__()
        self.l = l
        self.eps = eps 
        self.momentum = momentum 
        self.device = device
        self.known_classes = known_classes
        self.unknown_classes = num_classes - known_classes
        self.temperature = 5 # Speichern

        self.register_buffer('global_unk_pi', torch.ones(self.unknown_classes) / self.unknown_classes)
        
    def forward(self, logits,targets, unknown_mask,known_mask):
        probs = nn.functional.softmax(logits, dim=-1)
        batch_pi = probs.mean(dim=1)
        
        
        probs_unknown_clusters = probs[unknown_mask][:,self.known_classes:]
        probs_unknown_labels= probs[unknown_mask][:,:self.known_classes]
      
        probs_unknown_clusters = F.softmax(probs_unknown_clusters / self.temperature, dim=-1)

      #  print(probs_unknown_clusters[100])

        loss_conditional_unknown = torch.tensor(0.0, device=self.device, requires_grad=True)
        loss_marginal = torch.tensor(0.0, device=self.device, requires_grad=True)
        loss_separation = torch.tensor(0.0, device=self.device, requires_grad=True)
        loss_ce_known = torch.tensor(0.0, device=self.device, requires_grad=True)


        if known_mask.sum() >0:
            loss_ce_known =  nn.functional.cross_entropy(logits[known_mask], targets[known_mask])
       
         #conditional loss for unknowns H(y|x)
        if unknown_mask.sum() >0:
            loss_conditional_unknown = torch.sum(probs_unknown_clusters * torch.log(probs_unknown_clusters + self.eps), dim=-1).mean()
            mass_of_unknowns_on_knowns = probs_unknown_labels.sum(dim=-1)
            loss_separation = torch.log(mass_of_unknowns_on_knowns + self.eps).mean()
       
        #marginal loss H(y)
        if unknown_mask.sum() >0:
            batch_pi = probs_unknown_clusters.mean(dim=0)
        else:
            batch_pi = self.global_unk_pi.detach()


        if self.global_unk_pi.device != self.device:
            self.global_unk_pi = self.global_unk_pi.to(self.device)  
            
        new_pi = self.momentum * self.global_unk_pi + (1 - self.momentum) * batch_pi.detach()
        self.global_unk_pi.copy_(new_pi.squeeze())
       
        pi= self.momentum * self.global_unk_pi.detach() + (1 - self.momentum) * batch_pi
       
        log_pi = torch.log(pi + self.eps)
        loss_marginal = torch.sum(pi * log_pi)

        return     loss_ce_known#+ (  loss_separation) + (20.0 * loss_marginal)  - (self.l*loss_conditional_unknown)
    
    
class ReconLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CosineEmbeddingLoss()
        self.l1_criterion = torch.nn.L1Loss()

    def forward(self, recon_features, target_features, mask):
        target_ones = torch.ones(recon_features[mask].size(0)).to(recon_features.device)        
        loss_recon = self.criterion(recon_features[mask], target_features[mask], target_ones)
        loss_l1 = self.l1_criterion(recon_features[mask], target_features[mask])
        return loss_recon + loss_l1
    
    
class TemporalSmoothnessLoss(nn.Module):
    def __init__(self, threshold=0.0):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, logits, padding_mask):
        """
        logits: [Batch, Time, Classes]
        padding_mask: [Batch, Time] (True = Valid Data)
        """
        log_probs = F.log_softmax(logits, dim=-1)
        diff = self.mse(log_probs[:, :-1, :], log_probs[:, 1:, :]) 
        
        loss_per_frame = diff.mean(dim=-1) 
   
        valid_mask = padding_mask[:, :-1] & padding_mask[:, 1:]
        
        loss = (loss_per_frame * valid_mask.float()).sum() / (valid_mask.sum() + 1e-8)
        
        return loss
    
    
class TotalLoss(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pim_loss = PIMLoss(l=1.0,
                           momentum=0.9,
                           eps=1e-8,
                           known_classes=config.knowns,
                           num_classes=config.knowns + config.K,
                           device=self.device)
        self.recon_loss = ReconLoss()
        self.smooth_loss = TemporalSmoothnessLoss()
        self.pim_weight = 1.0
        self.recon_weight = 1.0
        self.smooth_weight = 10

    def forward(self, logits, target_labels,recon_features, target_features,
                unknown_mask,
                known_mask,
                patch_mask,
                padding_mask):
        loss_pim = self.pim_loss(logits, target_labels, unknown_mask, known_mask)
        
        loss_recon = self.recon_loss(recon_features, target_features, patch_mask)
        loss_smooth = self.smooth_loss(logits, padding_mask)

        total_loss = (self.pim_weight * loss_pim +
                      self.recon_weight * loss_recon +
                      self.smooth_weight * loss_smooth)
        """boundary_loss = self.l1(boundaries[:,:,0][~unknown_mask & padding_mask], padding_target_start[~unknown_mask & padding_mask].float()) + self.l1(boundaries[:,:,1][~unknown_mask & padding_mask], padding_target_end[~unknown_mask & padding_mask].float())
                """
        print(total_loss.item(),f"PIM: {loss_pim}",f"Cos: {loss_recon}",f"Smooth: {loss_smooth}")

        return total_loss