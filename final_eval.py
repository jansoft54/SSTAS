

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


knowns_ids = ["cut_tomato",
              "place_tomato_into_bowl",
              "cut_cheese",
              "place_cheese_into_bowl",
              "cut_lettuce",
              "place_lettuce_into_bowl",
              "add_salt",
              "add_vinegar",
              "add_oil",
              "add_pepper",
              "mix_dressing",
              "peel_cucumber",
              "cut_cucumber",
              "place_cucumber_into_bowl",
              "add_dressing",
              "mix_ingredients",
              "serve_salad_onto_plate",
              "action_start",
              "action_end",
              ]
unknowns_ids = [
]


known_for_eval = ["cut_tomato",
                  "place_tomato_into_bowl",

                  "place_cheese_into_bowl",
                  "cut_lettuce",

                  "add_salt",


                  "add_pepper",

                  "peel_cucumber",
                  "cut_cucumber",
                  "place_cucumber_into_bowl",
                  "add_dressing",

                  "serve_salad_onto_plate",
                  "action_start",
                  "action_end",
                  ]
unk_for_eval = ["add_oil",
                "add_vinegar",
                "cut_cheese",
                "mix_dressing",
                "mix_ingredients",
                "place_lettuce_into_bowl",]

prototypes = 0
train_for_knowns = True

gather_known = {}
gather_unknown = {}

for i in range(1, 6):
    trainer_config = TrainerConfig(
        dataset="50salads",  # Ergänzt, da im Pfad oben genutzt
        default_path="./data/data/",  # Ergänzt laut Evaluator Pfad
        train_split=f"train.split{i}.bundle",
        test_split=f"test.split{i}.bundle",
        train_for_knowns=train_for_knowns,
        known_classes=knowns_ids,   # 14
        unknowns=unknowns_ids,      # Deine Liste der Unknown IDs
        K=prototypes,           # 10
        batch_size=1,
        num_epochs=45,
        output_name="actionbert_second_try_unk"
    )

    test_video_dataset = VideoDataSet(
        dataset=trainer_config.dataset,
        split=trainer_config.test_split,
        default_path=trainer_config.default_path,
        knowns=trainer_config.known_classes,
        unknowns=trainer_config.unknowns,

    )

    test_data_loader = VideoDataLoader(
        test_video_dataset,
        1,
        shuffle=False
    )

    evaluator_service = DataEvaluation(test_data_loader, train=False)

    bert_conf = ActionBERTConfig(
        known_classes=knowns_ids,
        input_dim=2048,
        d_model=256,
        num_heads=8,
        num_layers=3,
        local_window_size=128,
        window_dilation=32,
        dropout=0)

    model = ActionBERT(
        config=bert_conf, train_for_knowns=train_for_knowns)

    path_unknown = f"./output/actionbert_full_known_split{i}.pth"

    state_dict = torch.load(path_unknown, map_location=torch.device('cuda'))
    model.load_state_dict(state_dict, strict=False)
    model = model.to('cuda')
    model.eval()

    known_result, unk_result = evaluator_service.eval(
        model, None, known_for_eval, unk_for_eval)

    for metric, value in known_result.items():
        if metric not in gather_known:
            gather_known[metric] = []
        gather_known[metric].append(value)

    for metric, value in unk_result.items():
        if metric not in gather_unknown:
            gather_unknown[metric] = []
        gather_unknown[metric].append(value)

final_metrics_known = {}
for metric, values in gather_known.items():
    final_metrics_known[metric] = sum(values) / len(values)
final_metrics_unknown = {}
for metric, values in gather_unknown.items():
    final_metrics_unknown[metric] = sum(values) / len(values)


print("-------------------- KNOWN --------------------")
print(final_metrics_known)
print("-------------------- UNKNOWN --------------------")
print(final_metrics_unknown)
"""
{'test-eval/mof Known': 86.40855484491786,
'test-eval/edit Known': 69.15581739942354,
'test-eval/f1_10 Known': 79.15880414963728,
'test-eval/f1_25 Known': 77.84697620125482,
'test-eval/f1_50 Known': 71.1583029869843}
"""


"""
FULLY SUPERVISED ON ALL CLASSES:

-------------------- KNOWN --------------------
{'test-eval/mof Known': np.float64(88.14097443478549), 'test-eval/edit Known': 81.75495984064095, 'test-eval/f1_10 Known': np.float64(85.95970287712649), 'test-eval/f1_25 Known': np.float64(84.9853439408017), 'test-eval/f1_50 Known': np.float64(79.818311188183)}
-------------------- UNKNOWN --------------------
{'test-eval/mof Unknown': np.float64(82.64508849237689), 'test-eval/edit Unknown': 77.82063492063493, 'test-eval/f1_10 Unknown': np.float64(84.03332946720391), 'test-eval/f1_25 Unknown': np.float64(82.67777397172242), 'test-eval/f1_50 Unknown': np.float64(77.32221865216684)}


"""
