import torch
import optuna
from torch.optim import lr_scheduler

def train_model(trial, model, train_loader, criterion, optimizer,
                valid_loader=None, num_epochs=10, device='cpu', log_interval=10,
                scheduler=None
               ):
    """
    Train model with flexible parameters.
    """
    best_val_accuracy = 0  # Track the best validation accuracy
    
    # Training and validation loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        correct_train = 0
        total_train = 0
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)  # Get class index with highest probability
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
            train_acc = float(correct_train) / float(total_train)
            running_loss += loss.item()
            
            if (i + 1) % log_interval == 0:  # Print every log_interval batches
                print(f"\r  Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Train Acc: {train_acc:.4f} Loss: {loss.item():.4f}", end="")

            # Step the scheduler after every epoch if it's not ReduceLROnPlateau
            if scheduler and scheduler.__class__ != lr_scheduler.ReduceLROnPlateau:
                scheduler.step()
                
        print()
        # Validation phase
        val_accuracy = evaluate_model(model, valid_loader, device)
        best_val_accuracy = max(best_val_accuracy, val_accuracy)  # Track best accuracy
        print(f"  Validation Accuracy after Epoch {epoch + 1}: {val_accuracy:.4f}")

        # Report intermediate result to Optuna
        trial.report(val_accuracy, epoch)

        # Handle pruning (optional)
        if trial.should_prune():
            print("  Trial pruned due to no improvement.")
            raise optuna.exceptions.TrialPruned()
            
        # Step scheduler for ReduceLROnPlateau based on validation loss
        if scheduler and scheduler.__class__ == lr_scheduler.ReduceLROnPlateau:
            scheduler.step(running_loss)

    # Print summary for the trial
    print(f"Trial {trial.number} complete. Best Validation Accuracy: {best_val_accuracy:.4f}\n")

    # Return the best validation accuracy across all epochs
    return best_val_accuracy


def evaluate_model(model, data_loader, device='cpu'):
    """Evaluate model performance"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate accuracy
    accuracy = float(correct) / float(total)
    return accuracy
