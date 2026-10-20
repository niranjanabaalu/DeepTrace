import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os


# ==========================================
# 1. DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================
# 2. SETTINGS
# ==========================================

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10

CLASSIFIER_LR = 0.001
FINE_TUNE_LR = 0.0001


# ==========================================
# 3. TRAIN TRANSFORM
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


# ==========================================
# 4. VALIDATION TRANSFORM
# ==========================================

validation_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 5. LOAD DATASETS
# ==========================================

train_dataset = datasets.ImageFolder(
    "dataset/train",
    transform=train_transform
)

validation_dataset = datasets.ImageFolder(
    "dataset/validation",
    transform=validation_transform
)


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


print("\n========================================")
print("DATASET")
print("========================================")

print(
    "Training images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(validation_dataset)
)

print(
    "Classes:",
    train_dataset.classes
)


# ==========================================
# 6. LOAD PRETRAINED EFFICIENTNET
# ==========================================

print("\nLoading EfficientNet-B0...")

weights = models.EfficientNet_B0_Weights.DEFAULT

model = models.efficientnet_b0(
    weights=weights
)


# ==========================================
# 7. FREEZE ALL FEATURES
# ==========================================

for parameter in model.features.parameters():

    parameter.requires_grad = False


# ==========================================
# 8. UNFREEZE LAST FEATURE BLOCK
# ==========================================

for parameter in model.features[-1].parameters():

    parameter.requires_grad = True


# ==========================================
# 9. REPLACE CLASSIFIER
# ==========================================

input_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    input_features,
    2
)


# ==========================================
# 10. MOVE MODEL TO DEVICE
# ==========================================

model = model.to(device)


# ==========================================
# 11. LOSS FUNCTION
# ==========================================

criterion = nn.CrossEntropyLoss()


# ==========================================
# 12. OPTIMIZER
# ==========================================

optimizer = torch.optim.Adam(
    [
        {
            "params": model.features[-1].parameters(),
            "lr": FINE_TUNE_LR
        },
        {
            "params": model.classifier.parameters(),
            "lr": CLASSIFIER_LR
        }
    ]
)


# ==========================================
# 13. MODEL DIRECTORY
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)


MODEL_PATH = (
    "models/"
    "deeptrace_efficientnet_b0_finetuned.pth"
)


best_validation_accuracy = 0.0


# ==========================================
# 14. TRAINING
# ==========================================

print("\n========================================")
print("FINE-TUNING STARTED")
print("========================================")


for epoch in range(EPOCHS):


    # ======================================
    # TRAINING
    # ======================================

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0


    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()

        optimizer.step()


        running_loss += loss.item()


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_loss = (
        running_loss /
        len(train_loader)
    )


    train_accuracy = (
        100 *
        correct /
        total
    )


    # ======================================
    # VALIDATION
    # ======================================

    model.eval()

    validation_loss = 0.0

    validation_correct = 0
    validation_total = 0


    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(device)
            labels = labels.to(device)


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            validation_loss += loss.item()


            _, predicted = torch.max(
                outputs,
                1
            )


            validation_total += (
                labels.size(0)
            )


            validation_correct += (
                predicted == labels
            ).sum().item()


    validation_loss = (
        validation_loss /
        len(validation_loader)
    )


    validation_accuracy = (
        100 *
        validation_correct /
        validation_total
    )


    # ======================================
    # PRINT RESULTS
    # ======================================

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Loss: {validation_loss:.4f} "
        f"Val Acc: {validation_accuracy:.2f}%"
    )


    # ======================================
    # SAVE BEST MODEL
    # ======================================

    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = (
            validation_accuracy
        )


        torch.save(
            model.state_dict(),
            MODEL_PATH
        )


        print(
            "  ✓ Best model saved!"
        )


# ==========================================
# 15. FINAL RESULT
# ==========================================

print("\n========================================")
print("FINE-TUNING COMPLETED")
print("========================================")


print(
    f"Best validation accuracy: "
    f"{best_validation_accuracy:.2f}%"
)


print("Model saved at:")

print(MODEL_PATH)

print("========================================")