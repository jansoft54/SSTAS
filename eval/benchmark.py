import sys
sys.path.append('../')
from loader.dataloader import VideoDataLoader, VideoDataSet
from evaluator import Evaluator


knows = 14
unknowns = 5
        
dataset = VideoDataSet(dataset="50salads",split="train.split3.bundle",knowns=knows,unknowns=unknowns)
loader = VideoDataLoader(dataset, batch_size=8, shuffle=True)

for batch in loader:
    features = batch["features"]
    target_truth = batch["target_truth"]
    padding_mask = batch["padding_mask"]
    unknown_mask = batch["unknown_mask"]
    
    print(target_truth.shape,padding_mask.shape)
    
    evaluator = Evaluator(evaluation_name="First",
                          dataset="50salads",
                          default_path="../data/data/",
                          known_classes=knows,
                          unkown_classes=unknowns)
    evaluator.evaluate(model_pred=target_truth[0:1,:],
                       ground_truth=target_truth[0:1,:],
                       padding_mask=padding_mask[0:1,:],
                       unknown_mask=unknown_mask[0:1,:])
    break