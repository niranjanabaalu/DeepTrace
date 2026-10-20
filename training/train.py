import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# ==========================================
# SETTINGS
# ==========================================

IMAGE_SIZE = 224
BATCH_SIZE = 16

# ==========================================
# TRANSFORMS
# ==========================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

validation_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# DATASETS
# ==========================================

train_dataset = datasets.ImageFolder(
    "dataset/train",
    transform=train_transform
)

validation_dataset = datasets.ImageFolder(
    "dataset/validation",
    transform=validation_transform
)

# ==========================================
# DATA LOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ==========================================
# LOAD PRETRAINED EFFICIENTNET-B0
# ==========================================

print("\nLoading EfficientNet-B0...")

weights = models.EfficientNet_B0_Weights.DEFAULT

model = models.efficientnet_b0(
    weights=weights
)

# ==========================================
# FREEZE FEATURE EXTRACTOR
# ==========================================

for parameter in model.features.parameters():
    parameter.requires_grad = False

# ==========================================
# REPLACE CLASSIFIER
# ==========================================

input_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    input_features,
    2
)

model = model.to(device)

# ==========================================
# LOSS FUNCTION
# ==========================================

criterion = nn.CrossEntropyLoss()

# ==========================================
# OPTIMIZER
# ==========================================

optimizer = torch.optim.Adam(
    model.classifier[1].parameters(),
    lr=0.001
)

# ==========================================
# MODEL INFORMATION
# ==========================================

print("\n========================================")
print("MODEL INFORMATION")
print("========================================")

print("Model: EfficientNet-B0")
print("Classes:", train_dataset.classes)
print("Output classes: 2")
print("Trainable layer: Final classifier")
print("Frozen layers: Feature extractor")

# ==========================================
# TEST ONE FORWARD PASS
# ==========================================

images, labels = next(iter(train_loader))

images = images.to(device)
labels = labels.to(device)

outputs = model(images)

print("\n========================================")
print("FORWARD PASS TEST")
print("========================================")

print("Input shape:", images.shape)
print("Output shape:", outputs.shape)

print("Expected output shape:")
print("(batch_size, 2)")

print("\nForward pass successful!")

# ==========================================
# LOSS TEST
# ==========================================

loss = criterion(outputs, labels)

print("\nInitial loss:", loss.item())

# ==========================================
# BACKWARD PASS TEST
# ==========================================

optimizer.zero_grad()

loss.backward()

optimizer.step()

print("\nBackward pass successful!")
print("Optimizer step successful!")

print("\n========================================")
print("EFFICIENTNET SETUP SUCCESSFUL")
print("========================================")