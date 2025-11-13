from datasets import load_dataset

SPLIT = 1
dataset = load_dataset("dinggd/breakfast", name=f"split{SPLIT}")

# Trainings‐Daten
print(len(dataset["train"]))
"""for x in dataset["train"]:
    video_id, video_feature, video_label = x
    # z. B. print(video_id, video_feature.shape, video_label)

# Test‐Daten
for x in dataset["test"]:
    video_id, video_feature, video_label = x
    # z. B. print(video_id, video_feature.shape, video_label)
"""