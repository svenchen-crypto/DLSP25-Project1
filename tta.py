import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

# Define mild TTA augmentations
tta_transforms = [
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomRotation(degrees=5),
]

def tta_predict_batched(model, images, device="cuda", conf_threshold=0.8):
    """ Apply TTA only for low-confidence predictions """
    model.eval()
    images = images.to(device)  # Move batch to GPU

    # Get original prediction
    with torch.no_grad():
        orig_logits = model(images)  # Get logits
        orig_pred = F.softmax(orig_logits, dim=-1)  # Convert to probabilities
        orig_conf, orig_class = torch.max(orig_pred, dim=1)  # Confidence & predicted class

    # Identify low-confidence samples
    low_conf_mask = orig_conf < conf_threshold  # Boolean mask for uncertain predictions

    # ✅ FIX: Ensure TTA is skipped when threshold is too low
    if low_conf_mask.sum().item() == 0:
        return orig_pred  # Skip TTA and return original predictions

    # Select only uncertain images for TTA
    uncertain_images = images[low_conf_mask]

    # Generate augmentations
    batch_augmented = [uncertain_images]  # Start with original uncertain images
    for t in tta_transforms:
        batch_augmented.append(t(uncertain_images))  # Apply TTA

    # Stack augmented images: [TTA * Uncertain_B, C, H, W]
    batch_augmented = torch.cat(batch_augmented, dim=0)

    # Run inference for TTA batch
    with torch.no_grad():
        preds = model(batch_augmented)  # Get logits (not softmax)

    # Reshape: [TTA, Uncertain_B, num_classes]
    num_tta = len(tta_transforms) + 1  # +1 for original image
    preds = preds.view(num_tta, uncertain_images.size(0), -1)

    # Average logits over TTA versions
    final_tta_pred = torch.mean(preds, dim=0)  # [Uncertain_B, num_classes]
    final_tta_pred = F.softmax(final_tta_pred, dim=-1)  # Apply softmax at the end

    # Merge TTA predictions with original confident predictions
    final_pred = orig_pred.clone()  # Copy original predictions
    final_pred[low_conf_mask] = final_tta_pred  # Replace low-confidence predictions

    return final_pred