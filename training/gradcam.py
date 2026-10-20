
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights


# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "deeptrace_efficientnet_b0_v2.pth"
TEST_DIR = ROOT / "dataset" / "test"
OUTPUT_DIR = ROOT / "outputs" / "gradcam"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# --------------------------------------------------
# 2. Load the saved EfficientNet-B0 model
# --------------------------------------------------

weights = EfficientNet_B0_Weights.DEFAULT

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

# EfficientNet-B0's final convolutional feature block
target_layer = model.features[-1]

activations = []
gradients = []


# --------------------------------------------------
# 3. Register hooks to capture features and gradients
# --------------------------------------------------

def forward_hook(module, inputs, output):
    activations.clear()
    activations.append(output)


def backward_hook(module, grad_input, grad_output):
    gradients.clear()
    gradients.append(grad_output[0])


forward_handle = target_layer.register_forward_hook(forward_hook)
backward_handle = target_layer.register_full_backward_hook(backward_hook)


# --------------------------------------------------
# 4. Preprocess the face image
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std
    )
])


# --------------------------------------------------
# 5. Select a test image
# --------------------------------------------------

# First try a fake face crop.
image_paths = sorted((TEST_DIR / "fake").glob("*.jpg"))

# If no fake image exists, use a real face crop.
if not image_paths:
    image_paths = sorted((TEST_DIR / "real").glob("*.jpg"))

if not image_paths:
    raise FileNotFoundError(
        "No JPG face crops found in dataset/test/fake or dataset/test/real."
    )

image_path = image_paths[0]
print("Input image:", image_path.name)

original_image = Image.open(image_path).convert("RGB")
original_image = original_image.resize((224, 224))

input_tensor = transform(original_image).unsqueeze(0).to(device)


# --------------------------------------------------
# 6. Predict and calculate Grad-CAM
# --------------------------------------------------

model.zero_grad(set_to_none=True)

logits = model(input_tensor)
probabilities = torch.softmax(logits, dim=1)

predicted_class = int(logits.argmax(dim=1).item())
predicted_label = "FAKE" if predicted_class == 0 else "REAL"
confidence = float(probabilities[0, predicted_class].item())

# Backpropagate the score of the predicted class
logits[0, predicted_class].backward()

feature_maps = activations[0]
feature_gradients = gradients[0]

# Average gradients across spatial dimensions
weights_cam = feature_gradients.mean(dim=(2, 3), keepdim=True)

# Weighted sum of the feature maps
cam = (weights_cam * feature_maps).sum(dim=1, keepdim=True)

# Keep positive contributions
cam = F.relu(cam)

# Resize heatmap to the image dimensions
cam = F.interpolate(
    cam,
    size=(224, 224),
    mode="bilinear",
    align_corners=False
)

cam = cam[0, 0].detach().cpu().numpy()

# Normalize to 0–255
cam_min = cam.min()
cam_max = cam.max()

if cam_max > cam_min:
    cam = (cam - cam_min) / (cam_max - cam_min)

heatmap = np.uint8(255 * cam)


# --------------------------------------------------
# 7. Overlay the heatmap on the face
# --------------------------------------------------

image_array = np.array(original_image)
image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

colored_heatmap = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)

overlay = cv2.addWeighted(
    image_bgr,
    0.6,
    colored_heatmap,
    0.4,
    0
)


# --------------------------------------------------
# 8. Save outputs
# --------------------------------------------------

heatmap_path = OUTPUT_DIR / f"{image_path.stem}_heatmap.jpg"
overlay_path = OUTPUT_DIR / f"{image_path.stem}_gradcam.jpg"

cv2.imwrite(str(heatmap_path), colored_heatmap)
cv2.imwrite(str(overlay_path), overlay)

forward_handle.remove()
backward_handle.remove()

print("\nGRAD-CAM COMPLETED")
print("Predicted class:", predicted_label)
print(f"Prediction confidence: {confidence * 100:.2f}%")
print("Heatmap saved:", heatmap_path)
print("Overlay saved:", overlay_path)
print(
    "Note: Grad-CAM shows influential regions; "
    "it does not prove a manipulation."
)
