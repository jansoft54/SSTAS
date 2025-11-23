from model.bert import ActionBERT, ActionBERTConfig
from trainer import Trainer,TrainerConfig


knows = 14
unknowns = 5
prototypes = 15
trainer_config = TrainerConfig()
trainer = Trainer(config=trainer_config)

model = ActionBERT(ActionBERTConfig(total_classes=knows + prototypes))

print("Active parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
trainer.add_model(model=model)  # Hier sollte das eigentliche Modell übergeben werden
trainer.train_step()