
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "deeptrace_efficientnet_b0_v2.pth"
TEST_DIR = ROOT / "dataset" / "test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Use the same normalization as the pretrained model
weights = EfficientNet_B0_Weights.DEFAULT
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std
    )
])

test_data = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(
    test_data, batch_size=32, shuffle=False, num_workers=0
)

print("Test images:", len(test_data))
print("Classes:", test_data.class_to_idx)

# Recreate the model architecture
model = models.efficientnet_b0(weights=None)
input_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(input_features, 2)

checkpoint = torch.load(
    MODEL_PATH, map_location=device, weights_only=True
)
model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())

# Print evaluation results
accuracy = accuracy_score(all_labels, all_predictions)

print(f"\nTEST ACCURACY: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(
    all_labels,
    all_predictions,
    labels=[0, 1],
    target_names=["fake", "real"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_predictions, labels=[0, 1]))
print("\nEvaluation completed.")


