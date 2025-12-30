

from eval.benchmark import DataEvaluation
from loader.dataloader import VideoDataSet, VideoDataLoader
from model.bert import ActionBERTConfig
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
                                                knowns=trainer_config.knowns,
                                                unknowns=trainer_config.unknowns,
                                                total_classes=trainer_config.knowns + trainer_config.K)
        self.train_data_loader = VideoDataLoader(
            self.train_video_dataset, batch_size=trainer_config.batch_size, shuffle=True)

        self.test_video_dataset = VideoDataSet(dataset=trainer_config.dataset,
                                               split=trainer_config.test_split,
                                               default_path=trainer_config.default_path,
                                               knowns=trainer_config.knowns,
                                               unknowns=trainer_config.unknowns,
                                               total_classes=trainer_config.knowns + trainer_config.K)
        self.test_data_loader = VideoDataLoader(
            self.test_video_dataset, len(self.test_video_dataset), shuffle=True)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        """self.class_weights = self.calculate_class_weights(self.train_data_loader,
                                                          trainer_config.knowns + trainer_config.K,
                                                          self.device)"""

        self.total_loss = TotalLoss(
            trainer_config=trainer_config, class_weights=None)

        self.test_evaluator = DataEvaluation(
            self.test_data_loader, train=False)
        self.train_evaluator = DataEvaluation(
            self.train_data_loader, train=True)

    def calculate_class_weights(self, dataloader, num_classes, device):
        print("Berechne Class Weights über das ganze Dataset...")

        counts = torch.zeros(num_classes, dtype=torch.long)

        for batch in dataloader:

            targets = batch["target_truth"]

            targets_flat = targets.flatten()

            mask = (targets_flat >= 0) & (targets_flat < num_classes)
            valid_targets = targets_flat[mask]

            if valid_targets.numel() > 0:
                batch_counts = torch.bincount(
                    valid_targets, minlength=num_classes)
                counts += batch_counts

        counts = counts.float() + 1

        total_samples = counts.sum()

        weights = total_samples / counts

        weights[self.config.knowns:] = 1.0
        weights = weights / weights.mean()

        print(f"Absolute weights: {weights} {total_samples}")
        return weights.to(device)

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

    def update_centroids(self, model_out, target_truth, known_mask):
        with torch.no_grad():
            # Normalisierte Embeddings nutzen!
            current_embs = model_out["embeddings"][known_mask]
            current_lbls = target_truth[known_mask]

            for c in range(self.config.knowns):
                class_mask = (current_lbls == c)
                if class_mask.any():
                    batch_mean = current_embs[class_mask].mean(dim=0)
                    batch_mean = F.normalize(batch_mean, p=2, dim=0)

                    if not self.model.centers_initialized[c]:
                        self.model.class_centers[c] = batch_mean
                        self.model.centers_initialized[c] = True
                    else:
                        self.model.class_centers[c] = self.model.center_momentum * self.model.class_centers[c] + \
                            (1 - self.model.center_momentum) * batch_mean
                        self.model.class_centers[c] = F.normalize(
                            self.model.class_centers[c], p=2, dim=0)

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

    def train(self):
        for epoch in range(self.config.num_epochs):
            self.model.train()
            for batch in self.train_data_loader:

                features = batch["features"].to(self.device)

                unknown_mask = batch["unknown_mask"].to(self.device)
                foreground_mask = batch["foreground_mask"].to(self.device)
                target_truth = batch["target_truth"].to(self.device)
                padding_mask = batch["padding_mask"].to(self.device)

                patch_mask = self._generate_structured_mask(features,
                                                            mask_ratio=self.config.mask_ratio,
                                                            block_size=self.config.block_size,)

                patch_mask = patch_mask & (padding_mask.bool())

                self.model.train()

                result = self.model(
                    features, patch_mask=patch_mask, padding_mask=padding_mask)

                self.update_centroids(
                    result, target_truth, ~unknown_mask & padding_mask)

                loss_inputs = {
                    "logits": result["prototype_logits"],
                    "embeddings": result["embeddings"],
                    "target_labels": target_truth,
                    "recon_features": result["recon_features"],
                    "target_features": result["recon_target"],

                    "centers": self.model.class_centers,
                    "prototypes": self.model.prototypes.weight,

                    "refine_logits": result["refine_logits"],
                    "stages_output_logits": result["stages_output_logits"],

                    "unkown_logits": result["unkown_logits"],
                    "stages_output_unknown_logits": result["stages_output_unknown_logits"],

                    "known_mask": (~unknown_mask) & padding_mask,
                    "unknown_mask": (unknown_mask) & padding_mask,
                    "patch_mask": patch_mask,
                    "padding_mask": padding_mask,
                    "epoch": epoch,
                }
                loss = self.total_loss(loss_inputs)

                wandb.log({
                    "epoch": epoch,
                })
                self.optim.zero_grad()
                loss.backward()

                self.optim.step()
            self.test_evaluator.eval(self.model, epoch)
            self.train_evaluator.eval(self.model, epoch)

            print(
                f"-------------------- Epoch {epoch+1}/{self.config.num_epochs} -------------------- ")
            self.scheduler.step()

        print(
            f"Training completed. Saving model {self.config.output_name} ...")
        torch.save(self.model.state_dict(),
                   f"./output/{self.config.output_name}.pth")
        self.run.finish()
