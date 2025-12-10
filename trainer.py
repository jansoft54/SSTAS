


from loader.dataloader import VideoDataSet,VideoDataLoader
from model.loss import PIMLoss
import torch


class TrainerConfig:
    def __init__(self,
                 dataset:str ="50salads",
                 split: str = "train.split1.bundle",
                 default_path="./data/data/",
                 output_name:str ="output",
                 batch_size:int=1,
                 knowns:int=0,
                 unknowns:int=0,
                 K:int=0):
        self.dataset = dataset
        self.split = split
        self.default_path = default_path
        self.knowns = knowns
        self.unknowns = unknowns
        self.K = K
        self.learning_rate = 3e-4
        self.epsilon = 1e-8
        self.num_epochs = 100
        self.batch_size = 1
        self.mask_ratio = 0.5
        self.min_span = 5
        self.max_span = 20
        self.batch_size = batch_size
        self.output_name = output_name
        
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
        self.mse = torch.nn.MSELoss()
        self.pim = PIMLoss(l=1.0,momentum=0.9,eps=1e-8,num_classes=config.knowns + config.K,device=self.device)
        self.l1 = torch.nn.L1Loss()
    def add_model(self,model):
        self.model = model
        self.model.to(self.device)
        self.optim = torch.optim.AdamW(model.parameters(),
                                       lr=self.config.learning_rate,
                                       eps=self.config.epsilon )

        
    def _generate_span_mask(self,features,mask_ratio=0.5,min_span=5,max_span=20):
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
    def train(self,epochs:int):
        for epoch in range(epochs):
            for batch in self.data_loader:
                features = batch["features"].to(self.device)

                unknown_mask = batch["unknown_mask"].to(self.device)
                target_truth = batch["target_truth"].to(self.device)
                padding_mask = batch["padding_mask"].to(self.device)
                padding_target_start = batch["target_start"].to(self.device)
                padding_target_end = batch["target_end"].to(self.device)
                
                patch_mask = self._generate_span_mask(features,mask_ratio=0.5,min_span=5,max_span=20)
                patch_mask = patch_mask & (padding_mask.bool()) 
                
                self.model.train()
                recon_feat, class_logits, boundaries = self.model(features,patch_mask=patch_mask,padding_mask=padding_mask)
                
                pim_loss = self.pim(class_logits,
                                    target_truth,
                                    unknown_mask=unknown_mask,
                                    known_mask=~unknown_mask & padding_mask)
                mse_loss = self.mse(recon_feat[patch_mask & padding_mask ],features[patch_mask & padding_mask])
                

                boundary_loss = self.l1(boundaries[:,:,0][~unknown_mask & padding_mask], padding_target_start[~unknown_mask & padding_mask].float()) + self.l1(boundaries[:,:,1][~unknown_mask & padding_mask], padding_target_end[~unknown_mask & padding_mask].float())
            
        
                loss = mse_loss + boundary_loss #+ pim_loss# + boundary_loss
                self.optim.zero_grad()
                loss.backward()
                print(loss.item())
                self.optim.step()
            print(f"-------------------- Epoch {epoch+1}/{epochs} -------------------- ")


        print(f"Training completed. Saving model {self.config.output_name} ...")
        torch.save(self.model.state_dict(), f"./output/{self.config.output_name}.pth")
                 
        
        