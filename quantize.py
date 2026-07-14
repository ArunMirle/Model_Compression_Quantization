import os
import time

import torch
import torch.nn as nn
import torch.ao.quantization as quant

from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms
from torch.ao.quantization.quantize_fx import prepare_fx
from torch.ao.quantization.quantize_fx import convert_fx
from torch.ao.quantization import get_default_qconfig_mapping
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# ============================================================
# Device
# ============================================================

device = torch.device("cpu")

print("="*60)
print("DEVICE :", device)
print("="*60)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(
    root="data/train",
    transform=transform
)

val_dataset = datasets.ImageFolder(
    root="data/val",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)

print("Training Images :", len(train_dataset))
print("Validation Images :", len(val_dataset))
print("Classes :", train_dataset.classes)

model_fp32 = mobilenet_v2(
    weights=MobileNet_V2_Weights.DEFAULT,
    
)

model_fp32.classifier[1] = nn.Linear(
    in_features=1280,
    out_features=2
)

load_result = model_fp32.load_state_dict(
    torch.load(
        "models/mobilenet_fp32_trained.pth",
        map_location=device
    )
)

print(load_result)


model_fp32.eval()

print("\nFP32 Model Loaded Successfully")

def evaluate(model, dataloader):   
    '''This is evalute the accuracy'''

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in dataloader:

            outputs = model(images)

            _, predicted = torch.max(outputs,1)

            total += labels.size(0)

            correct += (predicted==labels).sum().item()

    return 100*correct/total


fp32_accuracy = evaluate(model_fp32, val_loader)

print("\nFP32 Accuracy")

print(f"{fp32_accuracy:.2f}%")


# ============================================================
# FP32 Model Size
# ============================================================

fp32_size = os.path.getsize(
    "models/mobilenet_fp32_trained.pth"
) / (1024 * 1024)

print("FP32 Model Size")
print(f"{fp32_size:.2f} MB")

# ============================================================
# FP32 Inference Time
# ============================================================

images, labels = next(iter(val_loader))

start = time.time()

with torch.no_grad():
    _ = model_fp32(images)

end = time.time()

fp32_time = (end - start) * 1000

print("FP32 Inference Time")

print(f"{fp32_time:.2f} ms")
# ============================================================
# Step 3 : Set Quantization Backend
# ============================================================

torch.backends.quantized.engine = "fbgemm"

print("\nQuantization Backend")
print("--------------------------------")
print(torch.backends.quantized.engine)



qconfig_mapping = get_default_qconfig_mapping("fbgemm")

print("\nQConfig Mapping Created")

example_inputs = (torch.randn(1, 3, 224, 224),)
print("\nQConfig Mapping Created")

# ============================================================
# Step 5 : Prepare Model for PTQ
# ============================================================

print("\n" + "="*60)
print("PREPARING MODEL FOR PTQ")
print("="*60)

model_fp32.eval()

prepared_model = prepare_fx(
    model_fp32,
    qconfig_mapping,
    example_inputs
)

print("✓ Model Prepared Successfully")

# ============================================================
# Step 6 : Calibration
# ============================================================

print("\n" + "="*60)
print("CALIBRATION")
print("="*60)

prepared_model.eval()

with torch.no_grad():

    for images, labels in train_loader:

        prepared_model(images)

print("✓ Calibration Completed")

# ============================================================
# Step 7 : Convert to INT8
# ============================================================

print("\n" + "="*60)
print("CONVERTING TO INT8")
print("="*60)

int8_model = convert_fx(prepared_model)

print("✓ INT8 Model Created")

# ============================================================
# Step 8 : INT8 Accuracy
# ============================================================

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


int8_accuracy = evaluate(int8_model, val_loader)

print("\nINT8 Accuracy")
print("--------------------------------")
print(f"{int8_accuracy:.2f}%")

# ============================================================
# Step 9 : Model Size
# ============================================================

os.makedirs("models", exist_ok=True)

torch.save(int8_model.state_dict(), "models/mobilenet_int8.pth")

int8_size = os.path.getsize("models/mobilenet_int8.pth") / (1024*1024)

print("\nINT8 Model Size :", round(int8_size, 2), "MB")

# ============================================================
# Step 10 : Inference Time
# ============================================================

images, labels = next(iter(val_loader))

start = time.time()

with torch.no_grad():
    _ = int8_model(images)

end = time.time()
int8_time=((end-start)*1000)
print("\nINT8 Inference Time :", (end-start)*1000, "ms")


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)

print(f"FP32 Accuracy      : {fp32_accuracy:.2f}%")
print(f"INT8 Accuracy      : {int8_accuracy:.2f}%")

print(f"FP32 Model Size    : {fp32_size:.2f} MB")
print(f"INT8 Model Size    : {int8_size:.2f} MB")



print(f"FP32 Inference     : {fp32_time:.2f} ms")
print(f"INT8 Inference     : {int8_time:.2f} ms")



print(f"Compression Ratio  : {fp32_size / int8_size:.2f}x")