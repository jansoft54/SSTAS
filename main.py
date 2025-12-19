from model.bert import ActionBERT, ActionBERTConfig
from trainer import Trainer,TrainerConfig




knowns = 14
unknowns = 5
prototypes = 30

trainer_config = TrainerConfig(
    split = "train.split1.bundle",
    knowns=knowns,
    unknowns=unknowns,
    K=prototypes,
    batch_size=1,
    num_epochs=1,
    output_name="actionbert_dummy"
)
trainer = Trainer(config=trainer_config)
bert_conf = ActionBERTConfig(
    total_classes=knowns + prototypes,
    input_dim=2048,
    d_model=128,
    num_heads=8,
    num_layers=6,
    dropout=0)
model = ActionBERT(config=bert_conf)


print("Active parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
trainer.add_model(model=model)  
trainer.train()