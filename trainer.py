

from eval.benchmark import DataEvaluation
from loader.dataloader import VideoDataSet, VideoDataLoader
from model.bert import ActionBERTConfig
from model.centroid_util import update_centroids
from model.loss import TotalLoss
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
                                                )
        self.train_data_loader = VideoDataLoader(
            self.train_video_dataset, batch_size=trainer_config.batch_size, shuffle=True)

        self.test_video_dataset = VideoDataSet(dataset=trainer_config.dataset,
                                               split=trainer_config.test_split,
                                               default_path=trainer_config.default_path,
                                               knowns=trainer_config.known_classes,
                                               unknowns=trainer_config.unknowns)
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
        print(self.config.weight_decay)
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
            project="ActionBERT",
            name="NewTest",
            config=config
        )
        wandb.define_metric("train-eval/*", step_metric="epoch")
        wandb.define_metric("test-eval/*", step_metric="epoch")

    def _generate_structured_mask(self, features, mask_ratio=0.75, block_size=64):

        B, S, D = features.size()
        mask = torch.zeros((B, S), dtype=torch.bool)

        for t in range(0, S, block_size):
            end_t = min(t + block_size, S)
            actual_block_len = end_t - t

            curr_mask_len = int(actual_block_len * mask_ratio)

            if curr_mask_len == 0:
                continue

            max_start = actual_block_len - curr_mask_len

            if max_start > 0:

                rel_starts = torch.randint(0, max_start, (B,))

                for b in range(B):
                    abs_start = t + rel_starts[b].item()
                    mask[b, abs_start: abs_start + curr_mask_len] = True
            else:
                mask[:, t:end_t] = True

        return mask.to(self.device)

    def _get_loss_dict(self, model_out, target_truth, unknown_mask, patch_mask, padding_mask, epoch):

        if self.config.train_for_knowns:
            loss_inputs = {
                "logits": model_out["prototype_logits"],
                "embeddings": model_out["embeddings"],
                "target_labels": target_truth,
                "recon_features": model_out["recon_features"],
                "target_features": model_out["recon_target"],

                "centers": self.model.class_centers,
                "prototypes": self.model.prototypes.weight,

                "refine_logits": model_out["refine_logits"],
                "stages_output_logits": model_out["stages_output_logits"],
                "progress_pred": model_out["progress_pred"],

                "known_mask": (~unknown_mask) & padding_mask,
                "unknown_mask": (unknown_mask) & padding_mask,
                "patch_mask": patch_mask,
                "padding_mask": padding_mask,
                "epoch": epoch,
            }
        else:

            new_targets = target_truth.clone()

            if unknown_mask.any():
                unk_logits = model_out["refine_logits"][unknown_mask][:,
                                                                      len(self.config.known_classes):]
                winning_slots = torch.argmax(unk_logits.detach(), dim=-1)
                pseudo_labels = winning_slots + len(self.config.known_classes)
                new_targets[unknown_mask] = pseudo_labels

            loss_inputs = {
                "logits": model_out["prototype_logits"],
                "embeddings": model_out["embeddings"],
                "target_labels": new_targets,
                "recon_features": model_out["recon_features"],
                "target_features": model_out["recon_target"],

                "centers": self.model.class_centers,
                "prototypes": self.model.prototypes_unk.weight,

                "refine_logits": model_out["refine_logits"],
                "stages_output_logits": model_out["stages_output_logits"],
                "progress_pred": model_out["progress_pred"],

                "known_mask": torch.ones_like(unknown_mask).bool() & padding_mask,
                "unknown_mask": (unknown_mask) & padding_mask,
                "patch_mask": patch_mask,
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

                patch_mask = self._generate_structured_mask(features,
                                                            mask_ratio=self.config.mask_ratio,
                                                            block_size=self.config.block_size,)

                patch_mask = patch_mask & (padding_mask.bool())

                self.model.train()

                result = self.model(
                    features, padding_mask=padding_mask)

                update_centroids(
                    self.config, self.model, result, target_truth, ~unknown_mask & padding_mask, unknown_mask & padding_mask)

                loss_inputs = self._get_loss_dict(
                    result, target_truth, unknown_mask, patch_mask, padding_mask, epoch)

                loss = self.total_loss(loss_inputs)

                wandb.log({
                    "epoch": epoch,
                })
                self.optim.zero_grad()
                loss.backward()

                self.optim.step()
            """self.test_evaluator.eval(
                self.model, epoch, self.config.known_classes, self.config.unknowns)
            self.train_evaluator.eval(
                self.model, epoch, self.config.known_classes, self.config.unknowns)"""

            print(
                f"-------------------- Epoch {epoch+1}/{self.config.num_epochs} -------------------- ")
            self.scheduler.step()

        print(
            f"Training completed. Saving model {self.config.output_name} ...")
        torch.save(self.model.state_dict(),
                   f"./output/{self.config.output_name}.pth")
        self.run.finish()
