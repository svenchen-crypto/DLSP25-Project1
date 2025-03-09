import torch

def train_model(model, train_loader, criterion, optimizer, valid_loader=None,
                num_epochs=10, device='cuda', scheduler=None, log_interval=100):
    """
    Train a PyTorch model with flexible parameters.
    """
    
    # Move model to the appropriate device
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Log progress
            if (batch_idx + 1) % log_interval == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], "
                      f"Step [{batch_idx + 1}/{len(train_loader)}], "
                      f"Loss: {running_loss / log_interval:.4f}")
                running_loss = 0.0

        # Validation step (if validation loader is provided)
        if valid_loader is not None:
            valid_loss, valid_acc = evaluate_model(model, valid_loader, criterion, device)
            print(f"Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_acc:.2f}%")
        
        # Step the scheduler if provided
        if scheduler is not None:
            scheduler.step()

    print("Training complete!")


def evaluate_model(model, data_loader, criterion, device='cuda'):
    """Evaluate model performance on a dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy
