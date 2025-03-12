import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
import kaggle
import zipfile
import pickle
import os

from helper import cifar_10_mean_std


# Specify the directory containing CIFAR-10 batches
cifar10_dir = "datasets/cifar-10-python"
lib_data_dir = "datasets/data"

def download_kaggle_dataset(
    competition_name="deep-learning-spring-2025-project-1",
    fn="deep-learning-spring-2025-project-1.zip"
):
    if os.path.exists(cifar10_dir):
        return 
        
    print(f"Dataset directory '{cifar10_dir}' not found. Downloading from Kaggle")
        
    # Download dataset from kaggle
    kaggle.api.competition_download_cli(competition_name)
    with zipfile.ZipFile(fn, 'r') as zip_ref:
        # Extract all contents to a directory
        zip_ref.extractall('datasets/')
    os.remove(fn)
    print("Successfully downloaded dataset from Kaggle")

def check_kaggle():
    if use_kaggle:
        # Check if the data directory exists
        if not os.path.exists(cifar10_dir):
            print(f"Dataset directory '{cifar10_dir}' not found. Downloading from Kaggle")
            download_kaggle_dataset()

        data_dir = cifar10_dir
    else:
        data_dir = "lib_data/"


def get_cifar10_dataloaders(
    transform_train, batch_size=64, num_workers=4, valid_size=0.1, 
    subset_percent=1.0, random_seed=42,  use_kaggle=True, verbos=False
):
    # Set the random seed for reproducibility
    torch.manual_seed(random_seed)
    
    if use_kaggle:
        download_kaggle_dataset()
        data_dir = cifar10_dir
    else:
        data_dir = lib_data_dir
    
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=not use_kaggle, transform=transform_train)
    
    subset_size = int(subset_percent * len(train_dataset))
    
    # Get the indices for the subset if specified
    indices = torch.randperm(len(train_dataset))[:subset_size]
    train_dataset = Subset(train_dataset, indices)
    
    # Calculate the number of samples for validation
    num_train = len(train_dataset)
    split = int(valid_size * num_train)
    
    # Split the dataset into training and validation sets
    train_indices, valid_indices = torch.utils.data.random_split(range(num_train), (num_train - split, split))
    
    train_subset = Subset(train_dataset, train_indices)
    valid_subset = Subset(train_dataset, valid_indices)
    
    # Create data loaders for the training and validation sets
    train_loader = DataLoader(train_subset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = DataLoader(valid_subset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    if verbos:
        image, label = train_subset[0]
        print(f"Image shape: {image.shape}")
        print(f"Label: {label}")
        print(f"Number of training data: {len(train_subset)}")
        print(f"Number of validation data: {len(valid_subset)}")

    return train_loader, valid_loader


def get_test_dataloader(
    batch_size=64, num_workers=4, grayscale=False, use_kaggle=True, verbos=False
):
    if use_kaggle:
        download_kaggle_dataset()
        data_dir = cifar10_dir
    else:
        data_dir = lib_data_dir  

    if grayscale:
        test_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Convert to grayscale but keep 3 channels
            transforms.ToTensor(),
            transforms.Normalize(**cifar_10_mean_std)  # Normalize with mean and std of CIFAR-10
        ])
    else:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**cifar_10_mean_std)  # Normalize with mean and std of CIFAR-10
        ])

    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=not use_kaggle, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    if verbos:
        print(f"Number of test data: {len(test_dataset)}")
    return test_loader


def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    return batch
    

def get_kaggle_test_dataloader(
    cifar_test_path='datasets/cifar_test_nolabel.pkl', batch_size=64
):
    download_kaggle_dataset()
    cifar10_batch = load_cifar_batch(cifar_test_path) # Load the batch
    test_images = cifar10_batch[b'data']
    test_images = torch.tensor(test_images, dtype=torch.float32)  # Convert to float32 for normalization

    # Transform the images from [N, W, H, C] to [N, C, W, H] and normalize the images 
    test_images = test_images.permute(0, 3, 1, 2) / 255.0
    normalize = transforms.Normalize(**cifar_10_mean_std)
    test_images = normalize(test_images)

    # Create a TensorDataset and DataLoader (without labels)
    test_dataset = TensorDataset(test_images)  # Only images, no labels
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader
