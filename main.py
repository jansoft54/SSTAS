from model.bert import ActionBERT, ActionBERTConfig
from trainer import Trainer
from trainer_config import TrainerConfig


import random
import os
import numpy as np
import torch


def set_deterministic(seed=42):
    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print(f"Random Seed set to: {seed} (Deterministic Mode)")


set_deterministic(42)


knowns = 14
unknowns = 5
prototypes = 0

trainer_config = TrainerConfig(
    train_split="train.split2.bundle",
    test_split="test.split2.bundle",
    knowns=knowns,
    unknowns=unknowns,
    K=prototypes,
    batch_size=1,
    num_epochs=35,
    output_name="actionbert_second_try"
)
bert_conf = ActionBERTConfig(
    total_classes=knowns + prototypes,
    input_dim=2048,
    d_model=256,
    num_heads=8,
    num_layers=4,
    local_window_size=128,
    window_dilation=32,
    dropout=0)
model = ActionBERT(config=bert_conf)

trainer = Trainer(trainer_config=trainer_config, model_config=bert_conf)


print("Active parameters: ", sum(p.numel()
      for p in model.parameters() if p.requires_grad))
trainer.add_model(model=model)
trainer.train()
