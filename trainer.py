import torch
import optuna
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from datetime import datetime
import os

from helper import num_params
from cifar10_1_dataloader import get_dataloader_10_1

def train_model(
    model, train_loader, criterion, optimizer, valid_loader=None, num_epochs=10, 
    device='cpu', log_interval=10, scheduler=None, trial=None, chkpt_dir="checkpoints", plot_dir="plots"
):
    """
    Train model with flexible parameters.
    """
    if num_params(model) > 5_000_000:
        raise Exception("Model size greater than 5M parameters")
        
    # Track metrics
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_accuracy = 0
    
    for epoch in range(num_epochs):
        model.train()
        correct_train = 0
        total_train = 0
        train_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
            train_loss += loss.item()
            train_acc = 100. * float(correct_train) / float(total_train)
            
            if (i + 1) % log_interval == 0:  # Print every log_interval batches
                print(f"\r  Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Train Acc: {train_acc:.4f} Loss: {loss.item():.4f}", end="")

            # Step the scheduler after every epoch if it's not ReduceLROnPlateau
            if scheduler and scheduler.__class__ != lr_scheduler.ReduceLROnPlateau:
                scheduler.step()
                
        print()
        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss / len(train_loader))
        
        # Validation phase
        val_acc, val_loss = evaluate_model(model, valid_loader, criterion=criterion, device=device)
        history["val_loss"].append(val_loss / len(valid_loader))
        history["val_acc"].append(val_acc)
        
        best_val_accuracy = max(best_val_accuracy, val_acc)  # Track best accuracy
        print(f"  Validation Accuracy after Epoch {epoch + 1}: {val_acc:.4f}")
        
        # Test on cifar10.1, good indicator of kaggle performance
        dataloader_10_1 = get_dataloader_10_1(num_samples=2000)
        cifar10_1_acc, _ = evaluate_model(model, dataloader_10_1, device)
        print(f"  Cidar10.1 Accuracy: {cifar10_1_acc}")

        if trial:
            # Report intermediate result to Optuna
            trial.report(val_acc, epoch)

            # Handle pruning (optional)
            if trial.should_prune():
                print("  Trial pruned due to no improvement.")
                raise optuna.exceptions.TrialPruned()
            
        # Step scheduler for ReduceLROnPlateau based on validation loss
        if scheduler and scheduler.__class__ == lr_scheduler.ReduceLROnPlateau:
            scheduler.step(val_loss)



    if trial:
        print(f"Trial {trial.number} complete. ", end="")
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}\n")

    # define file names
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    training_id = f"{model.__class__.__name__}_{best_val_accuracy:.4f}_{ts}"

    chkpt_file = f"val_acc_{training_id}.pth"
    plot_loss_file = f"loss_{training_id}.png"
    plot_acc_file = f"acc_{training_id}.png"

    if trial:
        chkpt_file = f"trial_{trial.number}_{chkpt_file}"
        plot_loss_file = f"trail_{trial.number}_{plot_loss_file}"
        plot_acc_file = f"trail_{trial.number}_{plot_acc_file}"
        
    plot_loss_file = os.path.join(plot_dir, plot_loss_file)
    plot_acc_file = os.path.join(plot_dir, plot_acc_file)
    chkpt_file = os.path.join(chkpt_dir, chkpt_file)

    # save model
    torch.save(model.state_dict(), chkpt_file)        
    
    # Plot loss
    plt.figure(figsize=(10, 4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig(plot_loss_file)  # Save the plot
    plt.close()
    
    # Plot accuracy
    plt.figure(figsize=(10, 4))
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.savefig(plot_acc_file)  # Save the plot
    plt.close()
    
    # Return the best validation accuracy across all epochs
    return best_val_accuracy


def evaluate_model(model, data_loader, device='cpu', criterion=None):
    """Evaluate model performance"""
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    loss = 0.0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if criterion:
                loss += criterion(outputs, labels).item()

    # Calculate accuracy
    accuracy = 100. * float(correct) / float(total)
    return accuracy, loss

