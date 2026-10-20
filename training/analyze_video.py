
"""
DeepTrace - Video Deepfake Analysis
Uses YuNet for face detection and EfficientNet-B0
for real/fake face classification.
"""

from pathlib import Path
import argparse
import csv

import cv2
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights


# --------------------------------------------------
# 1. Project paths and settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "deeptrace_efficientnet_b0_v2.pth"
YUNET_PATH = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
OUTPUT_DIR = ROOT / "outputs" / "video_analysis"

FRAME_INTERVAL = 10
FACE_SIZE = 224
FACE_PADDING = 20
FAKE_THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# 2. Load EfficientNet-B0
# --------------------------------------------------

def load_model():
    weights = EfficientNet_B0_Weights.DEFAULT

    model = models.efficientnet_b0(weights=None)
    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, 2)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((FACE_SIZE, FACE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=weights.transforms().mean,
            std=weights.transforms().std
        )
    ])

    return model, transform


# --------------------------------------------------
# 3. Load YuNet face detector
# --------------------------------------------------

def load_face_detector():
    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError(
            "This OpenCV installation does not expose FaceDetectorYN."
        )

    detector = cv2.FaceDetectorYN.create(
        str(YUNET_PATH),
        "",
        (320, 320),
        0.8,
        0.3,
        5000
    )

    return detector


# --------------------------------------------------
# 4. Detect and crop faces
# --------------------------------------------------

def detect_faces(frame, detector):
    height, width = frame.shape[:2]

    detector.setInputSize((width, height))
    _, detections = detector.detect(frame)

    faces = []

    if detections is None:
        return faces

    for detection in detections:
        x, y, w, h = detection[:4].astype(int)

        x1 = max(0, x - FACE_PADDING)
        y1 = max(0, y - FACE_PADDING)
        x2 = min(width, x + w + FACE_PADDING)
        y2 = min(height, y + h + FACE_PADDING)

        if x2 <= x1 or y2 <= y1:
            continue

        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        faces.append(face_rgb)

    return faces


# --------------------------------------------------
# 5. Predict fake probability for face crops
# --------------------------------------------------

def predict_faces(faces, model, transform):
    if not faces:
        return []

    batch = torch.stack([
        transform(Image.fromarray(face))
        for face in faces
    ]).to(DEVICE)

    with torch.no_grad():
        logits = model(batch)
        probabilities = torch.softmax(logits, dim=1)

    # ImageFolder mapping: fake = 0, real = 1
    return probabilities[:, 0].cpu().numpy().tolist()


# --------------------------------------------------
# 6. Analyze the video
# --------------------------------------------------

def analyze_video(video_path, model, transform, detector):
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    sampled_frames = 0
    frames_with_faces = 0
    total_faces = 0

    frame_results = []
    all_fake_probabilities = []

    print("\nAnalyzing:", video_path.name)
    print("Video FPS:", round(fps, 2))
    print("Total frames:", total_frames)
    print("Sampling every", FRAME_INTERVAL, "frames")

    frame_index = 0

    while True:
        success, frame = capture.read()

        if not success:
            break

        if frame_index % FRAME_INTERVAL == 0:
            sampled_frames += 1

            faces = detect_faces(frame, detector)
            probabilities = predict_faces(faces, model, transform)

            if probabilities:
                frames_with_faces += 1
                total_faces += len(probabilities)

                frame_fake_probability = float(np.mean(probabilities))
                all_fake_probabilities.extend(probabilities)

                frame_results.append({
                    "frame": frame_index,
                    "time_seconds": (
                        frame_index / fps if fps and fps > 0 else 0.0
                    ),
                    "faces": len(probabilities),
                    "fake_probability": frame_fake_probability,
                    "prediction": (
                        "FAKE"
                        if frame_fake_probability >= FAKE_THRESHOLD
                        else "REAL"
                    )
                })

        frame_index += 1

    capture.release()

    if not all_fake_probabilities:
        raise RuntimeError(
            "No faces were detected in the sampled frames. "
            "Try a clearer video with visible faces."
        )

    # Average all face-crop probabilities across sampled frames.
    video_fake_probability = float(np.mean(all_fake_probabilities))

    video_prediction = (
        "FAKE"
        if video_fake_probability >= FAKE_THRESHOLD
        else "REAL"
    )

    return {
        "video": video_path.name,
        "prediction": video_prediction,
        "fake_probability": video_fake_probability,
        "real_probability": 1.0 - video_fake_probability,
        "sampled_frames": sampled_frames,
        "frames_with_faces": frames_with_faces,
        "faces_analyzed": total_faces,
        "frame_results": frame_results
    }


# --------------------------------------------------
# 7. Save frame-level timeline
# --------------------------------------------------

def save_timeline(result, output_dir):
    timeline_path = output_dir / "frame_timeline.csv"

    with timeline_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "frame",
                "time_seconds",
                "faces",
                "fake_probability",
                "prediction"
            ]
        )

        writer.writeheader()
        writer.writerows(result["frame_results"])

    return timeline_path


# --------------------------------------------------
# 8. Main program
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze a video with DeepTrace."
    )

    parser.add_argument(
        "video",
        help="Path to an input video, such as raw_dataset/example.mp4"
    )

    args = parser.parse_args()

    video_path = Path(args.video)

    if not video_path.is_absolute():
        video_path = ROOT / video_path

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not YUNET_PATH.is_file():
        raise FileNotFoundError(f"YuNet model not found: {YUNET_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Device:", DEVICE)
    print("Loading EfficientNet-B0...")
    model, transform = load_model()

    print("Loading YuNet...")
    detector = load_face_detector()

    result = analyze_video(
        video_path,
        model,
        transform,
        detector
    )

    timeline_path = save_timeline(result, OUTPUT_DIR)

    print("\n" + "=" * 45)
    print("DEEPTRACE VIDEO ANALYSIS REPORT")
    print("=" * 45)
    print("Video:", result["video"])
    print("Prediction:", result["prediction"])
    print(
        f"Fake probability: "
        f"{result['fake_probability'] * 100:.2f}%"
    )
    print(
        f"Real probability: "
        f"{result['real_probability'] * 100:.2f}%"
    )
    print("Sampled frames:", result["sampled_frames"])
    print("Frames with faces:", result["frames_with_faces"])
    print("Faces analyzed:", result["faces_analyzed"])
    print("Timeline CSV:", timeline_path)
    print("\nAnalysis complete.")
    print(
        "Experimental result only: this model is not yet validated "
        "for reliable real-world deepfake detection."
    )


if __name__ == "__main__":
    main()
