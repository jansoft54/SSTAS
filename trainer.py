


from loader.dataloader import VideoDataSet,VideoDataLoader
from model.loss import TotalLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataclasses import dataclass

import torch




@dataclass
class TrainerConfig:
    dataset: str = "50salads"
    split: str = "train.split1.bundle"
    default_path: str = "./data/data/"
    output_name: str = "output"

    batch_size: int = 1
    knowns: int = 0
    unknowns: int = 0
    K: int = 0

    learning_rate: float = 2e-4
    epsilon: float = 1e-8
    num_epochs: int = 50
    mask_ratio: float = 0.5
    min_span: int = 5
    max_span: int = 20
    
    
class Trainer():
    def __init__(self,config: TrainerConfig):
        self.config = config
        self.video_dataset = VideoDataSet(dataset=config.dataset,
                               split=config.split,
                               default_path=config.default_path,
                               knowns=config.knowns,
                               unknowns=config.unknowns,
                               total_classes=config.knowns + config.K)
        self.data_loader = VideoDataLoader(self.video_dataset, batch_size=config.batch_size, shuffle=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.total_loss = TotalLoss(config=config)
        
    def add_model(self,model):
        self.model = model
        self.model.to(self.device)
        self.optim = torch.optim.AdamW(model.parameters(),
                                       lr=self.config.learning_rate,
                                       eps=self.config.epsilon )

        self.scheduler = CosineAnnealingLR(
            self.optim, 
            T_max=self.config.num_epochs, 
            eta_min=5e-5
        )
    def _generate_span_mask(self,features,mask_ratio=0,min_span=5,max_span=20):
        """
        Generate span mask for input features.
        features: [B, S, D]
        """
        B, S, D = features.size()
        mask = torch.zeros((B, S), dtype=torch.bool)
        num_mask_frames = int(S * mask_ratio)
        for b in range(B):
            masked_count = 0
            while masked_count < num_mask_frames:
                span_len = torch.randint(min_span, max_span + 1, (1,)).item()
                if S - span_len <= 0:
                    start_idx = 0
                else:
                    start_idx = torch.randint(0, S - span_len, (1,)).item()
                mask[b, start_idx : start_idx + span_len] = True
                masked_count += span_len 
                
        return mask.to(self.device)
    def train(self):
        for epoch in range(self.config.num_epochs):
            break
            for batch in self.data_loader:
                features = batch["features"].to(self.device)

                unknown_mask = batch["unknown_mask"].to(self.device)
                target_truth = batch["target_truth"].to(self.device)
                padding_mask = batch["padding_mask"].to(self.device)
                padding_target_start = batch["target_start"].to(self.device)
                padding_target_end = batch["target_end"].to(self.device)
                
                patch_mask = self._generate_span_mask(features,mask_ratio=0.5,min_span=30,max_span=150)
                patch_mask = patch_mask & (padding_mask.bool()) 
                
                self.model.train()
                recon_feat, class_logits, boundaries = self.model(features,patch_mask=patch_mask,padding_mask=padding_mask)
                
                
                loss = self.total_loss(class_logits,
                                target_truth,
                                recon_feat,
                                features,
                                unknown_mask,
                                ~unknown_mask & padding_mask,
                                patch_mask,
                                padding_mask) 
                
                self.optim.zero_grad()
                loss.backward() 
                    
                self.optim.step()
            print(f"-------------------- Epoch {epoch+1}/{self.config.num_epochs} -------------------- ")
            self.scheduler.step()

        print(f"Training completed. Saving model {self.config.output_name} ...")
        torch.save(self.model.state_dict(), f"./output/{self.config.output_name}.pth")
                 
        