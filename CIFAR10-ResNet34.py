import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 32  # Reduced from 64 to 32

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)  # Reduced from 64 to 32
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)  # Reduced from 64
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)  # Reduced from 128
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)  # Reduced from 256
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)  # Reduced from 512
        self.linear = nn.Linear(256 * block.expansion, num_classes)  # Reduced from 512

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

##########################################################
# A "SmallResNet" Architecture
##########################################################
class SmallResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2,2,2,2],
                 num_classes=10, base_channels=32):
        """
        Args:
            block:          Residual block type (BasicBlock).
            num_blocks:     List of 4 integers, # of blocks in each layer.
            num_classes:    Number of output classes (10 for CIFAR-10).
            base_channels:  # of channels in first layer. Adjust to control total params.
        """
        super(SmallResNet, self).__init__()
        self.in_planes = base_channels

        # Initial conv layer
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)

        # Create 4 stages (layers)
        self.layer1 = self._make_layer(block, base_channels,   num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base_channels*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, base_channels*8, num_blocks[3], stride=2)

        # Add dropout before linear layer
        self.dropout = nn.Dropout(0.2)

        # Global average pooling and linear layer
        self.linear = nn.Linear(base_channels*8*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # Global Average Pooling
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out
    
Xtr_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
Xts_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
Xtr = datasets.CIFAR10(root='./data', train=True, download=True, transform=Xtr_transform)
Xts = datasets.CIFAR10(root='./data', download=False, transform=Xts_transform)

Xtr_loader = DataLoader(Xtr, batch_size=256, shuffle=True, num_workers=2)
Xts_loader = DataLoader(Xts, batch_size=256, shuffle=False, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
um_classes = 10
num_epochs = 50
learning_rate = 0.01

model = SmallResNet(base_channels=24, num_blocks=[3, 4, 6, 3]).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 5e-4, momentum = 0.9)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Trainable Parameters: {total_params}")

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

def train_model(model, trainloader, testloader, device='cuda'):
    #model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() # zero outs gradients
            outputs = model(inputs) # forward pass : model outsputs predictions
            loss = criterion(outputs, labels) # computes the loss
            loss.backward() #backward pass to compute the gradients
            optimizer.step() #updates the parameters.

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(Xtr_loader)
        train_acc = 100.*correct/total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        scheduler.step()

        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs_test, labels_test in testloader:
                inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)
                outputs_test = model(inputs_test)
                loss_test = criterion(outputs_test, labels_test)
                test_loss += loss_test.item()
                _, predicted_test = outputs_test.max(1)
                total_test += labels_test.size(0)
                correct_test += predicted_test.eq(labels_test).sum().item()

        test_acc = 100.*correct_test/total_test
        test_loss = test_loss / len(Xts_loader)
        test_acc = 100. * correct_test / total_test

        # Save test stats
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Loss: {running_loss/len(trainloader):.3f}, "
              f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

    return model

trained_model = train_model(model, Xtr_loader, Xts_loader)