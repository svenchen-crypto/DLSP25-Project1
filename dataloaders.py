import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import kaggle
import zipfile
import pickle
import os


# Specify the directory containing CIFAR-10 batches
cifar10_dir = 'cifar-10-python/cifar-10-batches-py'
competition_name = "deep-learning-spring-2025-project-1"

# Function to load CIFAR-10 dataset
def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def download_kaggle_dataset(
    competition_name=competition_name,
    fn="deep-learning-spring-2025-project-1.zip"
):
    # Download dataset from kaggle
    kaggle.api.competition_download_cli(competition_name)
    with zipfile.ZipFile(fn, 'r') as zip_ref:
        # Extract all contents to a directory
        zip_ref.extractall('.')

def get_train_dataloaders(transform, batch_size=256, subset_percent=1, train_percent=0.7):
    # Load metadata (labels)
    meta_data_dict = load_cifar_batch(os.path.join(cifar10_dir, 'batches.meta'))
    label_names = [label.decode('utf-8') for label in meta_data_dict[b'label_names']]

    # Load training data
    train_data = []
    train_labels = []
    for i in range(1, 6):
        batch = load_cifar_batch(os.path.join(cifar10_dir, f'data_batch_{i}'))
        train_data.append(batch[b'data'])
        train_labels += batch[b'labels']

    train_data = np.vstack(train_data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # Convert to HWC format
    train_labels = np.array(train_labels)

    # Convert to TensorDataset and apply transformations
    train_dataset = [(transform(img), label) for img, label in zip(train_data, train_labels)]

    subset_size = int(subset_percent * len(train_dataset))
    subset, _ = random_split(train_dataset, [subset_size, len(train_dataset) - subset_size])
    train_size = int(subset_size * train_percent)
    val_size = int(subset_size - train_size)
    
    # Split the dataset
    train_subset, valid_subset = random_split(subset, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False, num_workers=4)

    image, label = subset[0]
    print(f"Image shape: {image.shape}")
    print(f"Label: {label}")
    print(f"Number of training data: {len(train_subset)}")
    print(f"Number of validation data: {len(valid_subset)}")
    
    return train_loader, valid_loader

def get_test_dataloader(
    transform,
    cifar_test_path = 'cifar_test_nolabel.pkl',
    batch_size=128
):
    test_batch = load_cifar_batch(cifar_test_path)
    test_images = test_batch[b'data'].astype(np.float32) / 255.0
    test_images = test_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # Convert to HWC format
    
    # Convert test dataset to Tensor
    test_dataset = [(transform(img)) for img in test_images]
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    return test_loader
