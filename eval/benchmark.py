import torch.nn.functional as F
import torch
from loader.dataloader import VideoDataLoader, VideoDataSet
from eval.evaluator import Evaluator
import sys
sys.path.append('../')

knowns = 14
unknowns = 5


class DataEvaluation:
    def __init__(self, dataloader, train=False):
        self.loader = dataloader
        self.train = train
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def eval(self, model, epoch, console_log=False):

        model.eval()
        model = model.to(self.device)
        with torch.no_grad():
            for batch in self.loader:

                features = batch["features"].to(self.device)
                target_truth = batch["target_truth"].to(self.device)
                padding_mask = batch["padding_mask"].to(self.device)
                unknown_mask = batch["unknown_mask"].to(self.device)

                result = model(features, None, padding_mask)

                softmax_logits = F.softmax(result["refine_logits"], dim=-1)
                class_labels = torch.argmax(softmax_logits, dim=-1).cpu()
                target_truth = target_truth.cpu()
                padding_mask = padding_mask.cpu()
                unknown_mask = unknown_mask.cpu()

                evaluator = Evaluator(evaluation_name="Evaluation",
                                      dataset="50salads",
                                      default_path="./data/data/",
                                      train=self.train,
                                      known_classes=knowns,
                                      unkown_classes=unknowns)
                known_pref, unkown_perf = evaluator.evaluate(model_pred=class_labels,
                                                             ground_truth=target_truth,
                                                             padding_mask=padding_mask,
                                                             unknown_mask=unknown_mask)

                known_pref["epoch"] = (epoch)
                unkown_perf["epoch"] = (epoch)
                if console_log:
                    print(known_pref)
                else:

                    import wandb
                    wandb.log(known_pref)
                break
