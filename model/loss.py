import torch
import torch.nn as nn
class PIMLoss(nn.Module ):
    def __init__(self,l=1,momentum=0.9,eps=1e-8,num_classes=10,device = None):
        super().__init__()
        self.l = l
        self.eps = eps 
        self.momentum = momentum 
        self.device = device
    
        self.register_buffer('global_pi', torch.ones(num_classes) / num_classes)
        
    def forward(self, logits,targets, unknown_mask,known_mask):
        probs = nn.functional.softmax(logits, dim=-1)
        batch_pi = probs.mean(dim=1)
        if self.global_pi.device != self.device:
            self.global_pi = self.global_pi.to(self.device)  
        new_pi = self.momentum * self.global_pi + (1 - self.momentum) * batch_pi.detach()
        self.global_pi.copy_(new_pi.squeeze())
       
        pi= self.momentum * self.global_pi.detach() + (1 - self.momentum) * batch_pi
       
        log_pi = torch.log(pi + self.eps)
        loss_marginal = torch.sum(pi * log_pi)
        loss_ce = nn.functional.cross_entropy(logits[known_mask], targets[known_mask])        
        loss_conditional = torch.sum(probs[unknown_mask] * torch.log(probs[unknown_mask] + self.eps), dim=-1).mean()
        
        return loss_marginal + loss_ce - (self.l*loss_conditional)