# # Aggressive Augmentation
# train_transform = transforms.Compose([
#     transforms.RandomCrop(32, padding=4, fill=0),  
#     transforms.RandomHorizontalFlip(p=0.5),  
#     transforms.RandomRotation(degrees=15),  
#     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  
#     transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),  
#     transforms.ToTensor(),
#     transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)), # Random Erasing (Mimics `Cutout`)
#     transforms.Normalize(**cifar_10_mean_std)
# ])

# Min Augmentation
# train_transform = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(**cifar_10_mean_std),
# ])


# # Medium Augmentation
# train_transform = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(0.5),
#     transforms.RandomRotation(15),
#     # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
#     transforms.ToTensor(),
#     transforms.Normalize(**cifar_10_mean_std)  # Normalize with mean and std of CIFAR-10
# ])

# # Realistic tranformation for better generalization
# train_transform = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),  # Mild color variations
#     transforms.ToTensor(),
#     transforms.Normalize(**cifar_10_mean_std),
# ])


