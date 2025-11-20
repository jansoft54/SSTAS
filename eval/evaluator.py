
from pathlib import Path
from metrics.Edit import Edit
from metrics.MoF import MoFAccuracyMetric
from metrics.F1Score import F1Score
import torch
from scipy.optimize import linear_sum_assignment

class Evaluator():
    def __init__(self,
                 evaluation_name = "default_evaluation",
                 dataset:str ="50salads",
                 default_path="../data/data/",
                 known_classes:int =0,
                 unkown_classes:int =0):
        self.dataset = dataset
        self.default_path = default_path
        self.known_classes = known_classes
        self.unkown_classes = unkown_classes
        self.evaluation_name = evaluation_name
        mapping_path = Path(f"{self.default_path}/{self.dataset}/mapping.txt")
        self.label_to_index = {}
        with open(mapping_path, "r", encoding="utf-8") as f:
            for line in f:
                idx, label = line.strip().split(" ")
                self.label_to_index[label] = int(idx)  
        
    def compute_hungarian_cost(self,pred,targets):
        #Simulate prdecitions with 7 random assign clusters
        clusters = 5
        unknown_mask = pred[:, :, -1] == 1  
        pred[unknown_mask] = torch.zeros_like(pred[unknown_mask])
        padding = torch.zeros(pred.shape[0], pred.shape[1],clusters, device=pred.device)
        random_clusters = torch.randint(0, clusters, (pred.shape[0], pred.shape[1], 1), device=pred.device).long()
        random_clusters = torch.nn.functional.one_hot(random_clusters.squeeze(), num_classes=clusters).unsqueeze(dim=0).float()

        padding[unknown_mask] = random_clusters[unknown_mask]
        pred_resized = torch.cat([pred, padding], dim=2)
    

        ####
       # print(pred_resized.shape,targets.shape)
        
        orig_classes = targets.shape[-1]  
        pred_indices = pred_resized.argmax(dim=2)   
             
        targets_indices = targets.argmax(dim=2)  
    
        cost_matrix = torch.zeros(clusters,self.unkown_classes, device=pred.device)
        
        for cluster in range(clusters):
            for label in range(self.unkown_classes):
               
                pred_copy = pred_indices.clone()
                pred_copy[pred_copy == orig_classes + cluster] = label
               
                cost_matrix[cluster, label] = ((pred_copy == targets_indices) & unknown_mask).sum(dim=-1).item()
                print( cost_matrix[cluster, label])
        
        
        
        cm = cost_matrix.detach().cpu().numpy()
        cm_inv = cm.max() - cm

        row_ind, col_ind = linear_sum_assignment(cm_inv)

        
        mapping = list(zip(row_ind, col_ind))

        print("Cluster → Label Zuordnung:")
        for c, l in mapping:
            print(f"Cluster {c} → Label {l}")
            pred_indices[pred_indices == orig_classes + c] = l
        
        return pred_indices, targets_indices
    def evaluate(self,model_pred, ground_truth,mask):
        model_pred,ground_truth = self.compute_hungarian_cost(model_pred, ground_truth)
        mask = mask == 1
        
        model_pred = model_pred[mask]
        ground_truth = ground_truth[mask]

        mof_metric = MoFAccuracyMetric()
        edit_metric = Edit()
        f1_metric = F1Score(num_classes=self.unkown_classes+self.known_classes)
        mof_metric.add(ground_truth, model_pred)
        edit_metric.add(ground_truth, model_pred)
        f1_metric.add(ground_truth.unsqueeze(dim=0), model_pred.unsqueeze(dim=0))

       
       
        results = [
            {"name":  self.evaluation_name,
             "mof": mof_metric.summary(),
             "edit": edit_metric.summary(),
             "f1_10": f1_metric.summary()["F1@10"],
             "f1_25": f1_metric.summary()["F1@25"],
             "f1_50": f1_metric.summary()["F1@50"],},
        ]
        
        self._print_results_table(results)

    def _print_results_table(self,results):
        headers = ["Model", "MOF", "Edit", "F1@0.10", "F1@0.25", "F1@0.50"]
        line = "+-----------+--------+--------+---------+---------+---------+"
        print(line)
        print("| {:9} | {:6} | {:6} | {:7} | {:7} | {:7} |".format(*headers))
        print(line)
        for r in results:
            print("| {:9} | {:6.1f} | {:6.1f} | {:7.1f} | {:7.1f} | {:7.1f} |".format(
                r["name"], r["mof"], r["edit"], r["f1_10"], r["f1_25"], r["f1_50"]
            ))
        print(line)

        
