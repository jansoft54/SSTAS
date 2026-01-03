

from eval.benchmark import DataEvaluation
from loader.dataloader import VideoDataSet, VideoDataLoader
from model.bert import ActionBERT, ActionBERTConfig
from model.centroid_util import update_centroids
from model.loss import TotalLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
from trainer_config import TrainerConfig
import wandb
import torch.nn.functional as F

knowns = 14
unknowns = 5
prototypes = 0
train_for_knowns = True


for i in range(1,6):
    trainer_config = TrainerConfig(
        dataset="50salads", # Ergänzt, da im Pfad oben genutzt
        default_path="./data/data/", # Ergänzt laut Evaluator Pfad
        train_split=f"train.split{i}.bundle",
        test_split=f"test.split{i}.bundle",
        train_for_knowns=train_for_knowns,
        known_classes=knowns,   # 14
        unknowns=unknowns,      # Deine Liste der Unknown IDs
        K=prototypes,           # 10
        batch_size=1,
        num_epochs=45,
        output_name="actionbert_second_try_unk"
    )

    # 2. Dataset und DataLoader initialisieren (Dein erster Schnipsel)
    test_video_dataset = VideoDataSet(
        dataset=trainer_config.dataset,
        split=trainer_config.test_split,
        default_path=trainer_config.default_path,
        knowns=trainer_config.known_classes,
        unknowns=trainer_config.unknowns,
        total_classes=trainer_config.known_classes + trainer_config.K
    )

    test_data_loader = VideoDataLoader(
        test_video_dataset, 
        1, 
        shuffle=False
    )


    # 3. DataEvaluation Instanz erstellen
    # (Wir gehen davon aus, dass die Klasse 'DataEvaluation' im Scope definiert ist)
    evaluator_service = DataEvaluation(test_data_loader, train=False)




    bert_conf_known = ActionBERTConfig(
        known_classes=knowns,
        input_dim=2048,
        d_model=256,
        num_heads=8,
        num_layers=3,
        local_window_size=128,
        window_dilation=32,
        dropout=0)
    bert_conf_unknown = ActionBERTConfig(
        known_classes=knowns,
        input_dim=2048,
        d_model=256,
        num_heads=4,
        num_layers=3,
        local_window_size=128,
        window_dilation=32,
        dropout=0)
    model_known = ActionBERT(config=bert_conf_known, train_for_knowns=train_for_knowns)
    model_unknown = ActionBERT(config=bert_conf_unknown, train_for_knowns=not train_for_knowns)
    
    path_known = f"./output/actionbert_known_split{i}.pth"
    path_unknown = f"./output/actionbert_unk_split{i}.pth"


    state_dict = torch.load(path_known, map_location=torch.device('cuda'))
    model_known.load_state_dict(state_dict, strict=False)
    model_known = model_known.to('cuda')
    model_known.eval()
    
    state_dict = torch.load(path_unknown, map_location=torch.device('cuda'))
    model_unknown.load_state_dict(state_dict, strict=False)
    model_unknown = model_unknown.to('cuda')
    model_unknown.eval()
    #print("Modell erfolgreich geladen.")


    eval_results = evaluator_service.eval(model_known,model_unknown, epoch=None, console_log=True)
    gather = {}
    for metric, value in eval_results.items():
        if  metric not in gather:
            gather[metric] = []
            gather[metric].append(value)

final_metrics = {}
for metric, values in gather.items():
    final_metrics[metric] = sum(values) / len(values)
            
print(final_metrics)