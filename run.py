import torch
import torch.nn as nn
import os

from data_loader import get_cifar10_dataloaders
from trainer import train_model
from helper import optimizer_map, scheduler_map

def single_run(
    model,
    transform,
    num_epochs=20,
    batch_size=64,
    optimizer_type="SGD",
    optimizer_params={"lr": 0.01},
    scheduler_type=None,
    scheduler_params=None,
    criterion_params={}
):
    train_loader, valid_loader = get_cifar10_dataloaders(
        transform,
        subset_percent=1, 
        valid_size=0.1,
        batch_size=batch_size,
        num_workers=8,
        use_kaggle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optimizer_map[optimizer_type](model.parameters(), **optimizer_params)
    scheduler = scheduler_map[scheduler_type](optimizer, **scheduler_params)
    criterion = nn.CrossEntropyLoss(**criterion_params)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # Training
    best_val_accuracy = train_model(
        model, train_loader, criterion, optimizer, valid_loader=valid_loader, 
        num_epochs=num_epochs, device=device,scheduler=scheduler
    )
    return best_val_accuracy
    