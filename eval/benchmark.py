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

    def get_label_ids(self, labels_list, labels_dict):
        return [labels_dict[name] for name in labels_list]

    def get_unk_mask(self, unk, target_truth, labels_dict):

        unk_ids = torch.tensor(
            [labels_dict[name] for name in unk],
            device=target_truth.device
        )
        return torch.isin(target_truth, unk_ids)

    def eval(self, model, epoch, known_ids, unknown_ids):

        model.eval()
        model = model.to(self.device)
        gather_known = {}
        gather_unknown = {}
        with torch.no_grad():
            for batch in self.loader:
               # print("Evaluating batch...")
                features = batch["features"].to(self.device)
                target_truth = batch["target_truth"].to(self.device)
                padding_mask = batch["padding_mask"].to(self.device)
                labels_dict = batch["labels_dict"]
                unknown_mask = self.get_unk_mask(
                    unknown_ids, target_truth, labels_dict)

                result = model(features, padding_mask)

                softmax_logits = F.softmax(result["refine_logits"], dim=-1)
                class_labels = torch.argmax(softmax_logits, dim=-1).cpu()
                target_truth = target_truth.cpu()
                padding_mask = padding_mask.cpu()
                unknown_mask = unknown_mask.cpu()

                known_ids_ = self.get_label_ids(
                    known_ids, labels_dict)
                unknown_ids_ = self.get_label_ids(unknown_ids, labels_dict)

                evaluator = Evaluator(evaluation_name="Evaluation",
                                      dataset="50salads",
                                      default_path="./data/data/",
                                      train=self.train,
                                      known_ids=known_ids_,
                                      unknown_ids=unknown_ids_)

                known_perf, unkown_perf = evaluator.evaluate(model_pred_knowns=class_labels,
                                                             ground_truth_knowns=target_truth,
                                                             padding_mask=padding_mask,
                                                             unknown_mask=unknown_mask)

              #  print("hallo")

                for metric, value in known_perf.items():
                    if metric not in gather_known:
                        gather_known[metric] = []
                    gather_known[metric].append(value)

                for metric, value in unkown_perf.items():
                    if metric not in gather_unknown:
                        gather_unknown[metric] = []
                    gather_unknown[metric].append(value)

                known_perf["epoch"] = (epoch)
              #  unkown_perf["epoch"] = (epoch)
                """if console_log:
                    pass
                else:

                    import wandb
                    wandb.log(known_pref)
                    # wandb.log(unkown_perf)"""

                # IMPORTANT !!!!!!!!!!!!!!!! break

        final_metrics_known = {}
        for metric, values in gather_known.items():
            final_metrics_known[metric] = sum(values) / len(values)
        final_metrics_unknown = {}
        for metric, values in gather_unknown.items():
            final_metrics_unknown[metric] = sum(values) / len(values)

        return final_metrics_known, final_metrics_unknown
