import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# ============================================================
# Device
# ============================================================

device = torch.device("cpu")

print("="*60)
print("DEVICE :", device)
print("="*60)

# ============================================================
# Image Transform
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ============================================================
# Dataset
# ============================================================

train_dataset = datasets.ImageFolder(
    root="data/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root="data/val",
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)

print("Training Images :", len(train_dataset))
print("Validation Images :", len(val_dataset))
print("Classes :", train_dataset.classes)

# ============================================================
# Load MobileNetV2
# ============================================================

model = mobilenet_v2(
    weights=MobileNet_V2_Weights.DEFAULT
)

print("\nPretrained MobileNetV2 Loaded Successfully")

# ============================================================
# Freeze Feature Extractor
# ============================================================

for param in model.features.parameters():
    param.requires_grad = False

# ============================================================
# Replace Final Classifier
# ============================================================

model.classifier[1] = nn.Linear(
    in_features=1280,
    out_features=2
)

model = model.to(device)

print("\nClassifier Replaced Successfully")
print(model.classifier)

# ============================================================
# Loss Function
# ============================================================

criterion = nn.CrossEntropyLoss()

# ============================================================
# Optimizer
# ============================================================

optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)

# ============================================================
# Epochs
# ============================================================

EPOCHS = 5

print("\nTraining for",EPOCHS,"epochs")

# ============================================================
# Training Loop
# ============================================================

print("\n" + "="*60)
print("START TRAINING")
print("="*60)

best_accuracy = 0.0

for epoch in range(EPOCHS):

    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

    # -----------------------------
    # Training Phase
    # -----------------------------
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear old gradients
        optimizer.zero_grad()

        # Forward Pass
        outputs = model(images)

        # Calculate Loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update Weights
        optimizer.step()

        # Statistics
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total
    train_loss = running_loss / len(train_loader)

    print(f"Training Loss     : {train_loss:.4f}")
    print(f"Training Accuracy : {train_accuracy:.2f}%")

    # -----------------------------
    # Validation Phase
    # -----------------------------
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total

    print(f"Validation Accuracy : {val_accuracy:.2f}%")

    # Save Best Model
    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        os.makedirs("models", exist_ok=True)

        torch.save(
            model.state_dict(),
            "models/mobilenet_fp32_trained.pth"
        )

        print("Best Model Saved!")

print("\n" + "="*60)
print("TRAINING COMPLETED")
print("="*60)

print(f"Best Validation Accuracy : {best_accuracy:.2f}%")
# ============================================================
# FP32 Model Size
# ============================================================

fp32_model_path = "models/mobilenet_fp32_trained.pth"

fp32_size = os.path.getsize(fp32_model_path) / (1024 * 1024)

print("\n" + "="*60)
print("FP32 MODEL SIZE")
print("="*60)
print(f"Model Size : {fp32_size:.2f} MB")

# ============================================================
# FP32 Inference Time
# ============================================================

images, labels = next(iter(val_loader))

images = images.to(device)

model.eval()

start = time.time()

with torch.no_grad():
    outputs = model(images)

end = time.time()

fp32_time = (end - start) * 1000

print("\n" + "="*60)
print("FP32 INFERENCE")
print("="*60)
print(f"Batch Size      : {images.size(0)}")
print(f"Inference Time  : {fp32_time:.2f} ms")