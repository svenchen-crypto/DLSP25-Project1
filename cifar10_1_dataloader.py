import torch
import numpy as np
import random
import time
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class CIFAR101Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images  # Already extracted subset
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert image to correct format (HWC -> CHW)
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

def get_dataloader_10_1(num_samples=1000, batch_size=64, seed=None):
    if seed is None:
        seed = int(time.time()) # Different seed every time

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Define transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # File paths
    image_path = "datasets/cifar10_1/cifar10.1_v6_data.npy"
    label_path = "datasets/cifar10_1/cifar10.1_v6_labels.npy"

    images = np.load(image_path, allow_pickle=True).copy()  # Force fresh load
    labels = np.load(label_path, allow_pickle=True).copy()
    
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    
    subset_size = min(num_samples, len(images))
    selected_indices = indices[:subset_size]
    selected_images = images[selected_indices]
    selected_labels = labels[selected_indices]

    subset_dataset = CIFAR101Dataset(selected_images, selected_labels, transform=transform)
    data_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    return data_loader
