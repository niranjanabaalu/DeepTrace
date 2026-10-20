
from pathlib import Path
from collections import defaultdict
import re

import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "dataset" / "test"
MODEL_PATH = ROOT / "models" / "deeptrace_efficientnet_b0_v2.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# --------------------------------------------------
# 2. Image preprocessing
# --------------------------------------------------

weights = EfficientNet_B0_Weights.DEFAULT

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std
    )
])

# --------------------------------------------------
# 3. Load the saved model
# --------------------------------------------------

model = models.efficientnet_b0(weights=None)
input_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(input_features, 2)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# ImageFolder mapping: fake=0, real=1
# Softmax column 0 = fake probability

# --------------------------------------------------
# 4. Group test crops by video
# --------------------------------------------------

video_images = defaultdict(list)
unmatched = []

for label in ("fake", "real"):
    folder = TEST_DIR / label

    for path in folder.iterdir():
        if not path.is_file():
            continue

        # Examples:
        # real_183_frame_000390.jpg -> video 183
        # fake_183_253_frame_000390.jpg -> video 183_253
        match = re.match(
            r"^(?:real|fake)_(\d+(?:_\d+)?)_frame_",
            path.name
        )

        if not match:
            unmatched.append(path.name)
            continue

        video_id = match.group(1)

        # The folder gives the ground-truth class.
        true_label = 0 if label == "fake" else 1

        video_images[(label, video_id)].append(
            (path, true_label)
        )

if unmatched:
    print("Unrecognized filenames:", len(unmatched))
    print("Examples:", unmatched[:5])

print("Videos found:", len(video_images))

# --------------------------------------------------
# 5. Predict each face crop and average by video
# --------------------------------------------------

video_results = []

with torch.no_grad():
    for (folder_label, video_id), items in sorted(video_images.items()):

        probabilities = []

        for path, _ in items:
            with Image.open(path) as img:
                image = img.convert("RGB")
                tensor = transform(image).unsqueeze(0).to(device)

            output = model(tensor)
            probs = torch.softmax(output, dim=1)

            fake_probability = probs[0, 0].item()
            probabilities.append(fake_probability)

        # Average the fake probability across this video's face crops
        mean_fake_probability = float(np.mean(probabilities))

        predicted_label = (
            0 if mean_fake_probability >= 0.5 else 1
        )

        true_label = 0 if folder_label == "fake" else 1

        video_results.append({
            "video": video_id,
            "actual": true_label,
            "predicted": predicted_label,
            "fake_probability": mean_fake_probability,
            "crops": len(items)
        })

# --------------------------------------------------
# 6. Display per-video results
# --------------------------------------------------

print("\nVIDEO-LEVEL PREDICTIONS")
print("-" * 72)

for result in video_results:
    actual = "FAKE" if result["actual"] == 0 else "REAL"
    predicted = "FAKE" if result["predicted"] == 0 else "REAL"

    print(
        f"{result['video']:12s} "
        f"Actual={actual:4s} "
        f"Predicted={predicted:4s} "
        f"Fake probability={result['fake_probability']:.3f} "
        f"Crops={result['crops']}"
    )

# --------------------------------------------------
# 7. Evaluate video-level performance
# --------------------------------------------------

y_true = [r["actual"] for r in video_results]
y_pred = [r["predicted"] for r in video_results]

if not y_true:
    raise RuntimeError("No videos were evaluated. Check filenames.")

print("\nVIDEO-LEVEL EVALUATION")
print("Videos evaluated:", len(video_results))
print(f"Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["fake", "real"],
    zero_division=0
))

print("Confusion Matrix (actual rows, predicted columns):")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print("\nVideo-level evaluation completed.")
