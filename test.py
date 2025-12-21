from eval.benchmark import TestDataEvaluation
from loader.dataloader import VideoDataLoader, VideoDataSet
from model.bert import ActionBERT, ActionBERTConfig
import torch.nn.functional as F
import torch 
knowns = 14
prototypes = 30
bert_conf = ActionBERTConfig(
    total_classes=knowns + prototypes,
    input_dim=2048,
    d_model=128,
    num_heads=8,
    num_layers=4,
    dropout=0)
model = ActionBERT(config=bert_conf)
path = "./output/actionbert_first_try.pth"


state_dict = torch.load(path, map_location=torch.device('cuda'))
model.load_state_dict(state_dict, strict=False)
model = model.to('cuda')
model.eval()
print("Modell erfolgreich geladen.")

data_set = VideoDataSet(dataset="50salads",split="test.split1.bundle",knowns=14,unknowns=5)
loader = VideoDataLoader(data_set, batch_size=len(data_set), shuffle=True)
test_eval = TestDataEvaluation(loader)
test_eval.eval(model=model)