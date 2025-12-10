import torch
import torch.nn as nn
class PIMLoss(nn.Module ):
    def __init__(self,l=1,momentum=0.9,eps=1e-8,known_classes = 5,num_classes=10,device = None):
        super().__init__()
        self.l = l
        self.eps = eps 
        self.momentum = momentum 
        self.device = device
        self.known_classes = known_classes
        self.unknown_classes = num_classes - known_classes
        
        self.register_buffer('global_unk_pi', torch.ones(self.unknown_classes) / self.unknown_classes)
        
    def forward(self, logits,targets, unknown_mask,known_mask):
        probs = nn.functional.softmax(logits, dim=-1)
        batch_pi = probs.mean(dim=1)
        
        
        probs_unknown_clusters = probs[unknown_mask][:,self.known_classes:]
        probs_unknown_labels= probs[unknown_mask][:,:self.known_classes]

        probs_unknown_clusters = probs_unknown_clusters / (probs_unknown_clusters.sum(dim=-1,keepdim=True)+self.eps)


        loss_conditional_unknown = torch.tensor(0.0, device=self.device, requires_grad=True)
        loss_marginal = torch.tensor(0.0, device=self.device, requires_grad=True)
        loss_separation = torch.tensor(0.0, device=self.device, requires_grad=True)
        loss_ce_known = torch.tensor(0.0, device=self.device, requires_grad=True)


        if known_mask.sum() >0:
            loss_ce_known =  nn.functional.cross_entropy(logits[known_mask], targets[known_mask])
       
        #conditional loss for unknowns H(y|x)
        if unknown_mask.sum() >0:
            loss_conditional_unknown = torch.sum(probs_unknown_clusters * torch.log(probs_unknown_clusters + self.eps), dim=-1).mean()
            mass_on_unknowns = probs_unknown_labels.sum(dim=-1)
            loss_separation = torch.log(mass_on_unknowns + self.eps).mean()
       
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
        
        return loss_separation +   loss_ce_known + loss_marginal  - (self.l*loss_conditional_unknown)