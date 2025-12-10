from pathlib import Path
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader

import numpy as np
import torch


class VideoDataSet(Dataset):
    def __init__(self,
                dataset:str ="50salads",
                split: str = "train.split1.bundle",
                default_path="../data/data/",
                knowns:int=0,
                unknowns:int=0,
                total_classes:int=10):
        self.default_path = default_path
        self.split = split
        self.dataset = dataset
        self.data = []
 
        path = Path(f"{default_path}/{dataset}/splits/{split}")
        self.knowns = knowns
        self.unknowns = unknowns
        self.total_classes = total_classes
        
        
        mapping_path = Path(f"{self.default_path}/{self.dataset}/mapping.txt")
        label_to_index = {}
        with open(mapping_path, "r", encoding="utf-8") as f:
            for line in f:
                idx, label = line.strip().split(" ")
                label_to_index[label] = int(idx)    
        
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                feature_name = line.split(".txt\n")[0]
              
                features = self._load_features_(feature_name)
                unknown_mask, target_truth = self._load_target_(feature_name,label_to_index)
                target_start, target_end = self.load_start_end_indices(target_truth,unknown_mask)

                data_obj = {}
                data_obj["dataset"] = dataset
                data_obj["split"] = split
                data_obj["features"] = features
                data_obj["unknown_mask"] = unknown_mask
                data_obj["target_truth"] = target_truth
                data_obj["labels_dict"] = label_to_index
                data_obj["target_start"] = target_start
                data_obj["target_end"] = target_end
                
                self.data.append(data_obj)
                
    def load_start_end_indices(self,targets,unknown_mask):
        _, counts = torch.unique_consecutive(targets, return_counts=True)
        start_ramps = [torch.arange(c, dtype=torch.float) for c in counts]
        d_start = torch.cat(start_ramps)
        d_end = torch.cat([torch.flip(r, dims=[0]) for r in start_ramps])
        
        d_start = torch.log(d_start + 1) 
        d_end = torch.log(d_end + 1)
       # d_start[unknown_mask] = 0.0
        #d_end[unknown_mask] = 0.0
        return d_start, d_end

    def _load_features_(self,feature_name):
        path = Path(f"{self.default_path}/{self.dataset}/features/{feature_name}.npy")
        features_np = np.load(path)       
        return torch.from_numpy(features_np).float().T  
        

    def _load_target_(self,target_name,label_to_index):
        target_path = Path(f"{self.default_path}/{self.dataset}/groundTruth/{target_name}.txt")
        with open(target_path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f]
            
        target_indices = torch.tensor([label_to_index[lbl] for lbl in labels], dtype=torch.long)
        unknown_mask = (target_indices < self.unknowns).bool()
        return unknown_mask,target_indices

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]
   
    
class VideoDataLoader(DataLoader):
    def __init__(self,
                 dataset:Dataset,
                 batch_size:int=8,
                 shuffle:bool=True,
                 collate_fn=None):
        collate_fn = self.collate_fn if collate_fn is None else collate_fn
        
        super().__init__(dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         collate_fn=collate_fn)
    
    def collate_fn(self,batch):
        features = [item["features"] for item in batch]
        unknown_mask = [item["unknown_mask"] for item in batch]
        targets_truth = [item["target_truth"] for item in batch]
        target_start = [item["target_start"] for item in batch]
        target_end = [item["target_end"] for item in batch]
        
        lengths = [f.shape[0] for f in features]
        max_len = max(lengths)
        feat_dim = features[0].shape[1]
        
        padded_feats = torch.zeros(len(batch), max_len, feat_dim)
        padded_tgts_truth= torch.zeros(len(batch), max_len).long()
        padded_unknown_mask= torch.zeros(len(batch), max_len).bool()
        padding_mask = torch.zeros(len(batch), max_len).bool()
        padding_target_start = torch.zeros(len(batch), max_len)
        padding_target_end= torch.zeros(len(batch), max_len)

        for i, (f, u, t_t,t_s,t_e) in enumerate(zip(features,unknown_mask ,targets_truth,target_start,target_end)):
            padded_feats[i, :f.shape[0]] = f
            padded_tgts_truth[i, :t_t.shape[0]] = t_t
            padding_mask[i, :f.shape[0]] = True
            padded_unknown_mask[i, :u.shape[0]] = u
            padding_target_start[i, :t_s.shape[0]] = t_s
            padding_target_end[i, :t_e.shape[0]] = t_e
            

        return {"features":padded_feats,
                "unknown_mask":padded_unknown_mask,
                "padding_mask":padding_mask,
                "target_truth":padded_tgts_truth,
                "target_start":padding_target_start,
                "target_end":padding_target_end,
                "labels_dict": batch[0]["labels_dict"]
                }
        
