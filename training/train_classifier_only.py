
"""
DeepTrace
EfficientNet-B0: Classifier-only training
Uses the corrected source-group dataset split.
"""

from pathlib import Path
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights


# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "deeptrace_efficientnet_b0_v2.pth"

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# --------------------------------------------------
# 2. Image transformations
# --------------------------------------------------

weights = EfficientNet_B0_Weights.DEFAULT
mean = weights.transforms().mean
std = weights.transforms().std

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


# --------------------------------------------------
# 3. Load datasets
# --------------------------------------------------

train_data = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform
)

val_data = datasets.ImageFolder(
    DATA_DIR / "validation",
    transform=eval_transform
)

test_data = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=eval_transform
)

expected_classes = {"fake": 0, "real": 1}

for name, dataset in [
    ("train", train_data),
    ("validation", val_data),
    ("test", test_data),
]:
    print(
        f"{name}: {len(dataset)} images, "
        f"class mapping = {dataset.class_to_idx}"
    )
    assert dataset.class_to_idx == expected_classes, (
        f"Unexpected class mapping in {name}"
    )

train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0
)

val_loader = DataLoader(
    val_data, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0
)

print("Test set loaded for later evaluation.")
print("Test images will NOT be used during training.")


# --------------------------------------------------
# 4. Load pretrained EfficientNet-B0
# --------------------------------------------------

model = models.efficientnet_b0(weights=weights)

# Freeze the pretrained feature extractor
for param in model.features.parameters():
    param.requires_grad = False

# Replace the original classification layer
input_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(input_features, 2)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE
)

print("\nModel: EfficientNet-B0")
print("Training strategy: classifier-only")
print("Classes: 0 = fake, 1 = real")


# --------------------------------------------------
# 5. Training and validation
# --------------------------------------------------

best_val_loss = float("inf")
best_val_acc = 0.0
best_state = None

for epoch in range(EPOCHS):

    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)

        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    train_loss /= train_total
    train_acc = 100 * train_correct / train_total

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)

            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_acc = 100 * val_correct / val_total

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.2f}%"
    )

    # Save the model with the lowest validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc = val_acc
        best_state = copy.deepcopy(model.state_dict())

        torch.save(best_state, MODEL_PATH)
        print("  Best model saved.")

print("\nTRAINING COMPLETED")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Validation accuracy at best loss: {best_val_acc:.2f}%")
print("Saved model:", MODEL_PATH)
