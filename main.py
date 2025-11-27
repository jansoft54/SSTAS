from model.bert import ActionBERT, ActionBERTConfig
from trainer import Trainer,TrainerConfig


knows = 14
unknowns = 5
prototypes = 30
trainer_config = TrainerConfig(knowns=knows,
                               unknowns=unknowns,
                               K=prototypes)
trainer = Trainer(config=trainer_config)

model = ActionBERT(ActionBERTConfig(total_classes=knows + prototypes))

print("Active parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
trainer.add_model(model=model)  # Hier sollte das eigentliche Modell übergeben werden
trainer.train(epochs=10)