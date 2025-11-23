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
                unknowns:int=0):
        self.default_path = default_path
        self.split = split
        self.dataset = dataset
        self.data = []
 
        path = Path(f"{default_path}/{dataset}/splits/{split}")
        self.knowns = knowns
        self.unknowns = unknowns
        
        
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
                target_masked, target_truth = self._load_target_(feature_name,label_to_index)
                data_obj = {}
                data_obj["dataset"] = dataset
                data_obj["split"] = split
                data_obj["features"] = features
                data_obj["target_mask"] = target_masked
                data_obj["target_truth"] = target_truth
               
                self.data.append(data_obj)
                

    def _load_features_(self,feature_name):
        path = Path(f"{self.default_path}/{self.dataset}/features/{feature_name}.npy")
        features_np = np.load(path)       
      
        return torch.from_numpy(features_np).float().T  
        

    def _load_target_(self,target_name,label_to_index):
      
        target_path = Path(f"{self.default_path}/{self.dataset}/groundTruth/{target_name}.txt")
        with open(target_path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f]
            
        target_indices = torch.tensor([label_to_index[lbl] for lbl in labels], dtype=torch.long)

        num_classes = len(label_to_index)
        target_indices_masked = torch.where((target_indices ) < self.unknowns,label_to_index["UNK"],target_indices)
        
        onehot_target_masked = torch.nn.functional.one_hot(target_indices_masked, num_classes=num_classes).float()
        onehot_target = torch.nn.functional.one_hot(target_indices, num_classes=num_classes ).float()

        return onehot_target_masked,onehot_target

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
        targets_masked = [item["target_mask"] for item in batch]
        targets_truth = [item["target_truth"] for item in batch]

        lengths = [f.shape[0] for f in features]
        max_len = max(lengths)
        feat_dim = features[0].shape[1]
        num_classes = targets_masked[0].shape[1]
        
        padded_feats = torch.zeros(len(batch), max_len, feat_dim)
        padded_tgts_masked = torch.zeros(len(batch), max_len, num_classes)
        padded_tgts_truth= torch.zeros(len(batch), max_len, num_classes)

        mask = torch.zeros(len(batch), max_len)

        for i, (f, t_m,t_t) in enumerate(zip(features, targets_masked,targets_truth)):
            padded_feats[i, :f.shape[0]] = f
            padded_tgts_masked[i, :t_m.shape[0]] = t_m
            padded_tgts_truth[i, :t_t.shape[0]] = t_t
            mask[i, :f.shape[0]] = 1.


        return {"features":padded_feats,
                "target_masked":padded_tgts_masked,
                "mask":mask,
                "target_truth":padded_tgts_truth}
        
