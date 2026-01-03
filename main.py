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
train_for_knowns = False

for i in range(2, 3):
    
    trainer_config = TrainerConfig(
        train_split=f"train.split{i}.bundle",
        test_split=f"test.split{i}.bundle",
        train_for_knowns=train_for_knowns,
        known_classes=knowns,
        unknowns=unknowns,
        K=prototypes,
        batch_size=1,
        num_epochs=45,
        output_name=f"actionbert_unk_split{i}"
    )
    bert_conf = ActionBERTConfig(
        known_classes=knowns,
        input_dim=2048,
        d_model=256,
        num_heads=8,
        num_layers=3,
        local_window_size=128,
        window_dilation=32,
        dropout=0)


    model = ActionBERT(config=bert_conf, train_for_knowns=train_for_knowns)

    trainer = Trainer(trainer_config=trainer_config, model_config=bert_conf)


    print("Active parameters: ", sum(p.numel()
        for p in model.parameters() if p.requires_grad))
    trainer.add_model(model=model)
    trainer.train()
