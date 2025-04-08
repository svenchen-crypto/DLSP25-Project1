import torch.optim as optim
import json
import os

cifar_10_mean_std = {
    "mean": [0.491400808095932, 0.48215898871421814, 0.44653093814849854], 
    "std": [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]
}


optimizer_map = {
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
    "SGD": optim.SGD,
    "RMSprop": optim.RMSprop
}

scheduler_map = {
    "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
    "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
    "OneCycleLR": optim.lr_scheduler.OneCycleLR,
    "CyclicLR": optim.lr_scheduler.CyclicLR
}

def update_study_details(checkpoint_dir, trial_num, trial_details):
    file_path = os.path.join(checkpoint_dir, "study_details.json")
    
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            json.dump({}, f, indent=4)
    
    with open(file_path, "r") as f:
        study_details = json.load(f)

    study_details[str(trial_num)] = trial_details
    
    with open(file_path, "w") as f:
        json.dump(study_details, f, indent=4)

# See the total number of trainable parameters
def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)