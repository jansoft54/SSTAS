

from eval.benchmark import DataEvaluation
from loader.dataloader import VideoDataSet, VideoDataLoader
from model.bert import ActionBERTConfig
from model.centroid_util import update_centroids
from model.loss_boundary_model import TotalLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
from trainer_config import TrainerConfig
import wandb
import torch.nn.functional as F


class Trainer():
    def __init__(self, trainer_config: TrainerConfig, model_config: ActionBERTConfig):
        self.config = trainer_config
        self.model_config = model_config
        self.train_video_dataset = VideoDataSet(dataset=trainer_config.dataset,
                                                split=trainer_config.train_split,
                                                default_path=trainer_config.default_path,
                                                knowns=trainer_config.known_classes,
                                                unknowns=trainer_config.unknowns,
                                                holdout_set=trainer_config.hold_outs
                                                )
        self.train_data_loader = VideoDataLoader(
            self.train_video_dataset, batch_size=trainer_config.batch_size, shuffle=True)

        self.test_video_dataset = VideoDataSet(dataset=trainer_config.dataset,
                                               split=trainer_config.test_split,
                                               default_path=trainer_config.default_path,
                                               knowns=trainer_config.known_classes,
                                               unknowns=trainer_config.unknowns,
                                                holdout_set=trainer_config.hold_outs
                                               )
        self.test_data_loader = VideoDataLoader(
            self.test_video_dataset, len(self.test_video_dataset), shuffle=True)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.total_loss = TotalLoss(
            trainer_config=trainer_config, class_weights=None)

        self.test_evaluator = DataEvaluation(
            self.test_data_loader, train=False)
        self.train_evaluator = DataEvaluation(
            self.train_data_loader, train=True)

    def add_model(self, model):
        self.model = model
        self.model.to(self.device)
        self.optim = torch.optim.AdamW(model.parameters(),
                                       lr=self.config.learning_rate,
                                       weight_decay=self.config.weight_decay,
                                       eps=self.config.epsilon)

        self.scheduler = CosineAnnealingLR(
            self.optim,
            T_max=self.config.num_epochs,
            eta_min=5e-5
        )
        config = vars(self.config) | vars(self.model_config)

        self.run = wandb.init(
            project="SSAD",
            name="NewTest",
            config=config
        )
        wandb.define_metric("train-eval/*", step_metric="epoch")
        wandb.define_metric("test-eval/*", step_metric="epoch")

    def target_gen(self, targets, known_mask):
        with torch.no_grad():
            B, T = targets.shape
            gt_dist_start = torch.full((B, T), -1.0, device=targets.device)
            gt_dist_end = torch.full((B, T), -1.0, device=targets.device)

            for b in range(B):

                vals, counts = torch.unique_consecutive(
                    targets[b], return_counts=True)

                current_idx = 0
                for length in counts:
                    length = length.item()

                    if known_mask[b, current_idx]:

                        gt_dist_start[b, current_idx: current_idx+length] = \
                            torch.arange(length, dtype=torch.float,
                                         device=targets.device)

                        gt_dist_end[b, current_idx: current_idx+length] = \
                            torch.arange(length, dtype=torch.float,
                                         device=targets.device).flip(0)

                    current_idx += length

            return gt_dist_start, gt_dist_end

    def _get_loss_dict(self, model_out, target_truth, unknown_mask, padding_mask, epoch):

        gt_dist_start, gt_dist_end = self.target_gen(
            target_truth, (~unknown_mask) & padding_mask)

        loss_inputs = {
            "logits": model_out["prototype_logits"],
            "embeddings": model_out["embeddings"],
            "target_labels": target_truth,
            #   "multi_stage_embeddings": model_out["multi_stage_embeddings"],
            "gt_dist_start": gt_dist_start,
            "gt_dist_end": gt_dist_end,

            "centers": self.model.class_centers,
            "prototypes": self.model.prototypes.weight,

            "refine_logits": model_out["refine_logits"],
            "stages_output_logits": model_out["stages_output_logits"],

            "known_mask": (~unknown_mask) & padding_mask,
            "unknown_mask": (unknown_mask) & padding_mask,
            "padding_mask": padding_mask,
            "epoch": epoch,
        }
        return loss_inputs

    def train(self):
        for epoch in range(self.config.num_epochs):
            self.model.train()
            for batch in self.train_data_loader:

                features = batch["features"].to(self.device)

                unknown_mask = batch["unknown_mask"].to(self.device)
                target_truth = batch["target_truth"].to(self.device)
                padding_mask = batch["padding_mask"].to(self.device)
                holdout_mask = batch["holdout_mask"].to(self.device)

             #   print(features.shape)
                #only on holdouts and knowns
                features = features[(~unknown_mask | holdout_mask)].unsqueeze(0)
                target_truth = target_truth[(~unknown_mask | holdout_mask)].unsqueeze(0)
                padding_mask = padding_mask[(~unknown_mask | holdout_mask)].unsqueeze(0)
                unknown_mask = holdout_mask[(~unknown_mask | holdout_mask)].unsqueeze(0)
                
              #  print(features.shape)
                
              #  raise Exception()

                self.model.train()

                result = self.model(
                    features,  padding_mask=padding_mask)

                update_centroids(
                    self.config, self.model, result, target_truth, ~unknown_mask & padding_mask, unknown_mask & padding_mask)

                loss_inputs = self._get_loss_dict(
                    result, target_truth, unknown_mask, padding_mask, epoch)

                loss = self.total_loss(loss_inputs)

                wandb.log({
                    "epoch": epoch,
                })
                self.optim.zero_grad()
                loss.backward()

                self.optim.step()
            self.test_evaluator.eval(
                self.model, epoch, self.config.known_classes, self.config.unknowns)
            self.train_evaluator.eval(
                self.model, epoch, self.config.known_classes, self.config.unknowns)

            print(
                f"-------------------- Epoch {epoch+1}/{self.config.num_epochs} -------------------- ")
            self.scheduler.step()

        print(
            f"Training completed. Saving model {self.config.output_name} ...")
        torch.save(self.model.state_dict(),
                   f"./output/{self.config.output_name}.pth")
        self.run.finish()
