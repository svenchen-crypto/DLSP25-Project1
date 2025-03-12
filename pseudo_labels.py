import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_pil_image

# Generate Pseudo Labels for Unlabeled Kaggle Test Data
def generate_pseudo_labels(model, data_loader, threshold=0.9, device="cpu"):
    model.eval()
    pseudo_labels = []
    confident_samples = []

    with torch.no_grad():
        for images, in data_loader:  # Unlabeled data
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            max_probs, predicted_labels = torch.max(probs, dim=1)

            # Select samples with high confidence
            mask = max_probs > threshold
            confident_samples.append(images[mask])
            pseudo_labels.append(predicted_labels[mask])

    if len(confident_samples) > 0:
        confident_samples = torch.cat(confident_samples)
        pseudo_labels = torch.cat(pseudo_labels)
        return confident_samples, pseudo_labels
    else:
        return None, None

# Custom Dataset for Pseudo Labels
class PseudoDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images.cpu()
        self.labels = labels.cpu()
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert Tensor to PIL Image
        image = to_pil_image(image)
        
        if self.transform:
            image = self.transform(image)

        # Convert label to tensor
        return image, label

class CustomCIFAR10Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images.cpu()  # Already in tensor format
        self.labels = labels.cpu()  # Labels in tensor format
        self.transform = transform  # Data augmentation

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert Tensor to PIL Image
        image = to_pil_image(image)

        # Apply transformations (if provided)
        if self.transform:
            image = self.transform(image)

        return image, label