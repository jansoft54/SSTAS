from pathlib import Path
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader

import numpy as np
import torch

def collate_fn(batch):
    features = [item["features"] for item in batch]
    targets = [item["target"] for item in batch]
    lengths = [f.shape[0] for f in features]
    max_len = max(lengths)
    feat_dim = features[0].shape[1]
    num_classes = targets[0].shape[1]
    
    padded_feats = torch.zeros(len(batch), max_len, feat_dim)
    padded_tgts = torch.zeros(len(batch), max_len, num_classes)
    mask = torch.zeros(len(batch), max_len)

    for i, (f, t) in enumerate(zip(features, targets)):
        padded_feats[i, :f.shape[0]] = f
        padded_tgts[i, :t.shape[0]] = t
        mask[i, :f.shape[0]] = 1.

    return padded_feats, padded_tgts, mask
    
class VideoDataLoader(Dataset):
    def __init__(self,
                dataset:str ="50salads",
                split: str = "train.split1.bundle",
                default_path="../data/data/"):
        self.default_path = default_path
        self.split = split
        self.dataset = dataset
        self.data = []
 
        path = Path(f"{default_path}/{dataset}/splits/{split}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                feature_name = line.split(".txt\n")[0]
              
                features = self._load_features_(feature_name)
                target = self._load_target_(feature_name)
                data_obj = {}
                data_obj["dataset"] = dataset
                data_obj["split"] = split
                data_obj["features"] = features
                data_obj["target"] = target
                
                self.data.append(data_obj)
                

    def _load_features_(self,feature_name):
        path = Path(f"{self.default_path}/{self.dataset}/features/{feature_name}.npy")
        features_np = np.load(path)         
        return torch.from_numpy(features_np).float().T  
        

    def _load_target_(self,target_name):
        mapping_path = Path(f"{self.default_path}/{self.dataset}/mapping.txt")
        label_to_index = {}
        with open(mapping_path, "r", encoding="utf-8") as f:
            for line in f:
                idx, label = line.strip().split(" ")
                label_to_index[label] = int(idx)
                
        target_path = Path(f"{self.default_path}/{self.dataset}/groundTruth/{target_name}.txt")
        with open(target_path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f]
            
        target_indices = torch.tensor([label_to_index[lbl] for lbl in labels], dtype=torch.long)
        num_classes = len(label_to_index)
        return torch.nn.functional.one_hot(target_indices, num_classes=num_classes).float()



        
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]
    
    
    
    
dataset = VideoDataLoader(split="train.split3.bundle")
loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn, shuffle=True)

for features, targets, mask in loader:
    print(features.shape,targets.shape,mask.shape)