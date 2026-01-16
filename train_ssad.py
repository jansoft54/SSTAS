from model.bert import ActionBERT, ActionBERTConfig
from model.boundary_model import ActionBoundary
from trainer_ssad import Trainer
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

knowns = ["cut_tomato",
         
          "cut_lettuce",

        

          "peel_cucumber",
          "cut_cucumber",
          "place_cucumber_into_bowl",
          "add_dressing",

          "serve_salad_onto_plate",
          "action_start",
          "action_end",
          ]
unknowns = [
    "add_vinegar",
    "add_oil",
    "cut_cheese",
    "mix_dressing",
    "place_lettuce_into_bowl",
    "mix_ingredients",
    
     "place_tomato_into_bowl",

          "place_cheese_into_bowl",
          
     "add_salt",

     "add_pepper",
          
]

hold_outs = [
    "place_tomato_into_bowl",
    "place_cheese_into_bowl",
    "add_salt",
    "add_pepper",
]
train_for_knowns = True


trainer_config = TrainerConfig(
    train_split=f"train.split2.bundle",
    test_split=f"test.split2.bundle",
    train_for_knowns=train_for_knowns,
    known_classes=knowns,
    unknowns=unknowns,
    hold_outs=hold_outs,
    batch_size=1,
    num_epochs=40,
    output_name=f"ssad_holdouts_try"
)
bert_conf = ActionBERTConfig(
    known_classes=knowns,
    input_dim=2048,
    d_model=128,
    num_heads=4,
    num_layers=8,
    local_window_size=128,
    window_dilation=32,
    dropout=0)


path = "./output/ssad_first_try.pth"
model = ActionBoundary(config=bert_conf, train_for_knowns=train_for_knowns)

# state_dict = torch.load(path, map_location=torch.device('cuda'))
# model.load_state_dict(state_dict, strict=False)
# model = model.to('cuda')

trainer = Trainer(trainer_config=trainer_config, model_config=bert_conf)


print("Active parameters: ", sum(p.numel()
                                 for p in model.parameters() if p.requires_grad))
trainer.add_model(model=model)

trainer.train()
