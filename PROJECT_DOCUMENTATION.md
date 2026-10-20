# DeepTrace — AI-Powered Deepfake Detection System

**Author:** Niranj  
**Current Progress:** 30% Complete  
**Date:** September 2026

---

## 1. Project Overview

### 1.1 What is DeepTrace?

**DeepTrace** is an Artificial Intelligence (AI) system designed to automatically detect whether a video or image of a person's face is **real** (authentic) or **fake** (AI-generated / deepfake). With the rapid rise of deepfake technology — where AI can swap faces in videos with alarming realism — this tool serves as a safeguard against misinformation, identity fraud, and digital impersonation.

### 1.2 Why This Matters

Deepfakes pose a serious threat today:

- 🎭 **Fake news & propaganda**: Political figures can be "shown" saying things they never said.
- 💰 **Financial fraud**: Voice + face deepfakes are used to scam people and companies.
- 👤 **Reputation damage**: Innocent people can be framed with fabricated videos.
- 🛡️ **Legal & forensic needs**: Courts and investigators need tools to verify media authenticity.

DeepTrace aims to solve this problem by using **Computer Vision** and **Deep Learning** to analyze facial features at a pixel level and spot the subtle artifacts that human eyes miss.

---

## 2. How Deepfake Detection Works (High-Level)

A deepfake video is created by swapping one person's face onto another person's video. While these fakes look convincing to the naked eye, the AI generation process leaves tiny **imperfections** or **artifacts**:

- Blending seams around the face boundary
- Inconsistent eye blinking patterns
- Slight lighting or color mismatches
- Pixel-level distortions during encoding/decoding

DeepTrace works like this:

```
Input Video
    ↓
Extract face frames every 10 frames
    ↓
Run each face through EfficientNet-B0 (trained CNN)
    ↓
Output: Probability → REAL (0) or FAKE (1)
    ↓
Aggregate results per video → Final verdict
```

---

## 3. Project Architecture (End-to-End Pipeline)

The project is structured in **4 major stages**. Currently Stages 1–3 are complete (~30% overall progress, since Stage 4 — the user-facing app — is upcoming).

```
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1 — DATA ACQUISITION  (Complete ✅)                  │
│  → Download FaceForensics++ dataset (Real + Deepfake videos)│
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2 — PREPROCESSING  (Complete ✅)                     │
│  a) Extract faces from each video using YuNet detector      │
│  b) Split dataset → Train 70% / Val 15% / Test 15%          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3 — MODEL TRAINING & EVALUATION  (Complete ✅)       │
│  a) Load pre-trained EfficientNet-B0                        │
│  b) Fine-tune on extracted face dataset                     │
│  c) Evaluate with Accuracy, Precision, Recall, F1, Confusion│
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4 — WEB APPLICATION  (Upcoming 🔜)                   │
│  a) Flask/FastAPI backend for inference                     │
│  b) React frontend with drag & drop upload                  │
│  c) Display results with confidence scores                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Dataset: FaceForensics++

### 4.1 Source

DeepTrace uses the **FaceForensics++** benchmark dataset — the industry standard for deepfake detection research. It contains **1,000 original YouTube videos** + **manipulated versions** using 5 different AI techniques:

| Manipulation Type | Description                                                |
| ----------------- | ---------------------------------------------------------- |
| **Deepfakes**     | Classic GAN-based face swap (what DeepTrace targets first) |
| Face2Face         | Re-enacting facial expressions from source to target       |
| FaceSwap          | Open-source face swapping software                         |
| NeuralTextures    | Neural rendering-based manipulation                        |
| FaceShifter       | High-fidelity face swapping                                |

Currently, DeepTrace is trained on **Original (Real)** vs **Deepfakes** binary classification.

### 4.2 Download Script

**File:** [faceforensics_download_v4.py](file:///d:/Niranj/DeepTraceProject/DeepTrace/faceforensics_download_v4.py)

This official script downloads raw videos from TUM servers with configurable options:

```bash
python faceforensics_download_v4.py raw_dataset \
    -d original \        # Download real videos
    -d Deepfakes \       # Download fake videos
    -c c23 \             # Medium compression (balanced)
    -t videos            # Download video files
```

### 4.3 Video Quality / Compression Levels

FaceForensics provides 3 quality tiers — DeepTrace uses **c23** (medium) as a balance between quality and file size:

- `raw` — Lossless (~500 GB+)
- `c23` — H.264 medium compression (recommended)
- `c40` — Highly compressed (~low quality)

---

## 5. Stage 2 — Data Preprocessing (Detailed)

### 5.1 Step A: Face Extraction

**File:** [extract_faces.py](file:///d:/Niranj/DeepTraceProject/DeepTrace/preprocessing/extract_faces.py)

Raw videos contain full scenes — the model only needs to see the **face region**. This script does the heavy lifting:

#### **How it works:**

1. **Input folders:**
   - Real videos → `raw_dataset/original_sequences/youtube/c23/videos/`
   - Fake videos → `raw_dataset/manipulated_sequences/Deepfakes/c23/videos/`

2. **Face Detector — YuNet:**
   - Uses OpenCV's built-in **YuNet** (a state-of-the-art, ultra-fast face detector)
   - Model file: `models/face_detection_yunet_2023mar.onnx`
   - Confidence threshold: 0.8 → only high-certainty faces are extracted

3. **Frame Sampling:**
   - Instead of processing every single frame (redundant), samples every **10th frame**
   - This reduces dataset size drastically while keeping enough unique face angles

4. **Extraction per face:**

   ```
   Bounding box found → Add 20px padding (includes hair/jaw context)
       → Crop face from frame → Resize to 224×224 pixels
       → Save as JPEG with naming convention:
         {label}_{video_name}_frame_{NNNNNN}_face_{N}.jpg
   ```

5. **Idempotency:** Videos already processed are **skipped** (prevents duplicate work if script is re-run).

#### **Example output:**

```
real_183_frame_000010_face_0.jpg
fake_183_253_frame_000010_face_0.jpg
```

### 5.2 Step B: Dataset Splitting

**File:** [split_dataset.py](file:///d:/Niranj/DeepTraceProject/DeepTrace/preprocessing/split_dataset.py)

After face extraction, the images are split into **Train / Validation / Test** sets.

#### **Key Design Decision: Split by Video ID (NOT by individual images!)**

This is **critical for valid evaluation**. If frames from the same video appear in both train and test sets, the model will overfit and give inflated fake accuracy — this is called **data leakage**.

**How splitting works:**

1. Extract **video ID** from filename (`real_183_...` → video ID = `183`)
2. Group all face images belonging to the **same video** together
3. Shuffle video IDs (random seed = 42 for reproducibility)
4. Split the **videos themselves**, not the frames:
   - **70%** → Training (used to learn weights)
   - **15%** → Validation (used to tune hyperparameters & pick best model)
   - **15%** → Test (held out, used only once for final score)

#### **Final Dataset Structure:**

```
dataset/
├── train/
│   ├── real/        (70% of real video faces)
│   └── fake/        (70% of fake video faces)
├── validation/
│   ├── real/        (15%)
│   └── fake/        (15%)
└── test/
    ├── real/        (15%)
    └── fake/        (15%)
```

---

## 6. Stage 3 — Model Architecture & Training

### 6.1 Why EfficientNet-B0?

DeepTrace uses **EfficientNet-B0** from Google Research as its backbone CNN model. This choice is intentional:

| Property                    | Why it matters for DeepTrace                                                                      |
| --------------------------- | ------------------------------------------------------------------------------------------------- |
| **Pre-trained on ImageNet** | Already knows millions of visual patterns → faster learning, less data needed (Transfer Learning) |
| **Balanced scaling**        | Optimized ratio of depth × width × resolution → great accuracy without being huge                 |
| **Small size (~20 MB)**     | Fast inference, runs on CPU without GPU for demo purposes                                         |
| **Proven in forensics**     | Widely used in academic papers on deepfake detection                                              |

### 6.2 Transfer Learning + Fine-Tuning Strategy

**Files:**

- Setup/Sanity check: [train.py](file:///d:/Niranj/DeepTraceProject/DeepTrace/training/train.py)
- Full training loop: [train_model.py](file:///d:/Niranj/DeepTraceProject/DeepTrace/training/train_model.py)

Instead of training a neural network from scratch (which requires millions of images), we use **Transfer Learning**:

```
Phase 1 — Feature Extractor (FROZEN ❄️)
├─  MBConv blocks 0–6 of EfficientNet-B0
└─  Pre-trained ImageNet weights are "locked"
    → These extract edges, textures, facial patterns

Phase 2 — Last Block (UNFROZEN 🔥, Fine-tune LR = 0.0001)
├─  MBConv block 7 (the deepest features)
└─  Learning slowly adapts high-level features
    to detect deepfake artifacts

Phase 3 — Classifier Head (NEW & UNFROZEN 🔥, LR = 0.001)
├─  Dropout + Linear(1280 → 2) layer
└─  Outputs logits for [REAL, FAKE] classes
```

#### **Two-Layer Optimizer Trick:**

The Adam optimizer uses **different learning rates** per layer group:

- Deep features: `lr = 0.0001` (slow, don't break good features!)
- Classifier: `lr = 0.001` (fast, this is brand new)

### 6.3 Data Augmentation (Training Only)

To make the model **robust** and prevent overfitting, training images are randomly transformed:

- `RandomHorizontalFlip(p=0.5)` — Flips image left-right
- `RandomRotation(±10°)` — Slight tilt to simulate camera angles
- `Normalize(imagenet_mean, imagenet_std)` — Standardize pixel values

Validation & test images are **only resized + normalized** (no augmentation — we need a fair score).

### 6.4 Full Training Loop

**File:** [train_model.py](file:///d:/Niranj/DeepTraceProject/DeepTrace/training/train_model.py)

| Hyperparameter | Value                           |
| -------------- | ------------------------------- |
| Image size     | 224 × 224                       |
| Batch size     | 16                              |
| Epochs         | 10                              |
| Optimizer      | Adam (per-layer LR)             |
| Loss function  | CrossEntropyLoss                |
| Device         | CUDA GPU if available, else CPU |

**Each Epoch does:**

1. **Training Phase** — Forward pass → compute loss → backpropagation → update weights
2. **Validation Phase** — No gradients; compute accuracy on unseen validation set
3. **Best Model Save** — If validation accuracy beats the previous best, save checkpoint to `models/deeptrace_efficientnet_b0_finetuned.pth`

### 6.5 Model Evaluation

**File:** [evaluate_model.py](file:///d:/Niranj/DeepTraceProject/DeepTrace/training/evaluate_model.py)

The **trained model checkpoint** is evaluated on the **held-out test set** (never seen during training):

#### **Metrics Computed:**

1. **Accuracy** = `(TP + TN) / Total` → Overall % correct
2. **Precision** → When model says "fake", how often is it actually fake?
3. **Recall** (Sensitivity) → Of all real fakes, what % did we catch?
4. **F1-Score** → Harmonic mean of Precision & Recall (balanced metric)
5. **Classification Report** → Per-class breakdown of the above
6. **Confusion Matrix** → 2×2 table showing:
   ```
                  Predicted Real    Predicted Fake
   Actual Real    [TN (True Neg)]   [FP (False Pos)]
   Actual Fake    [FN (False Neg)]  [TP (True Pos)]
   ```

The model file `models/deeptrace_efficientnet_b0.pth` is the saved checkpoint ready for production inference.

---

## 7. Tech Stack Summary

| Component               | Technology/Library             | Purpose                                       |
| ----------------------- | ------------------------------ | --------------------------------------------- |
| **Language**            | Python 3.12                    | Core development                              |
| **Deep Learning**       | PyTorch + torchvision          | Build & train EfficientNet                    |
| **Face Detection**      | OpenCV (cv2) + YuNet ONNX      | Extract faces from video frames               |
| **Numerical Computing** | NumPy, Pandas                  | Data handling & math                          |
| **Evaluation**          | scikit-learn                   | Accuracy, precision, recall, confusion matrix |
| **Progress Bars**       | tqdm                           | Download & training progress                  |
| **Dataset**             | FaceForensics++ (FF++)         | Benchmark real/fake videos                    |
| **Saved Models**        | .pth (PyTorch) + .onnx (YuNet) | Inference-ready checkpoints                   |
| **Upcoming**            | FastAPI + React                | User-friendly web interface                   |

---

## 8. Project Folder Structure

```
DeepTrace/
├── faceforensics_download_v4.py     # Step 1: Download FF++ dataset
│
├── preprocessing/
│   ├── extract_faces.py             # Step 2a: YuNet face extraction from videos
│   └── split_dataset.py             # Step 2b: Train/Val/Test split by video ID
│
├── training/
│   ├── train.py                     # Step 3a: Model setup & forward/backward sanity check
│   ├── train_model.py               # Step 3b: Full 10-epoch fine-tuning loop
│   └── evaluate_model.py            # Step 3c: Test-set evaluation + metrics
│
├── models/
│   ├── face_detection_yunet_2023mar.onnx   # Face detector (pre-trained)
│   └── deeptrace_efficientnet_b0.pth       # Our trained deepfake classifier
│
├── raw_dataset/                     # Raw videos from FaceForensics++ (generated)
│   ├── original_sequences/
│   └── manipulated_sequences/
│
├── processed_dataset/               # Extracted 224×224 face JPEGs (generated)
│   ├── real/
│   └── fake/
│
└── dataset/                         # Final Train/Val/Test split (generated)
    ├── train/{real,fake}/
    ├── validation/{real,fake}/
    └── test/{real,fake}/
```

---

## 9. Current Progress — 30% Complete

### ✅ Completed Work (Core AI Pipeline)

| Task                                                     | Status  | Code Files                                                                                                |
| -------------------------------------------------------- | ------- | --------------------------------------------------------------------------------------------------------- |
| Dataset download script                                  | ✅ Done | [faceforensics_download_v4.py](file:///d:/Niranj/DeepTraceProject/DeepTrace/faceforensics_download_v4.py) |
| Face extraction pipeline (YuNet)                         | ✅ Done | [extract_faces.py](file:///d:/Niranj/DeepTraceProject/DeepTrace/preprocessing/extract_faces.py)           |
| Train/Val/Test split (video-level, no leakage)           | ✅ Done | [split_dataset.py](file:///d:/Niranj/DeepTraceProject/DeepTrace/preprocessing/split_dataset.py)           |
| EfficientNet-B0 architecture setup                       | ✅ Done | [train.py](file:///d:/Niranj/DeepTraceProject/DeepTrace/training/train.py)                                |
| Fine-tuning training loop (10 epochs, per-layer LR)      | ✅ Done | [train_model.py](file:///d:/Niranj/DeepTraceProject/DeepTrace/training/train_model.py)                    |
| Model evaluation script (Accuracy, F1, Confusion Matrix) | ✅ Done | [evaluate_model.py](file:///d:/Niranj/DeepTraceProject/DeepTrace/training/evaluate_model.py)              |
| Trained model checkpoint saved                           | ✅ Done | `models/deeptrace_efficientnet_b0.pth`                                                                    |

### 🔜 Upcoming Work (Next 70%)

| Task                                                                                           | Priority | Estimated Effort |
| ---------------------------------------------------------------------------------------------- | -------- | ---------------- |
| **Inference API** — Write `predict.py` to run model on a single video/image                    | High     | Medium           |
| **Backend Server** — Build FastAPI endpoint to accept uploads & return result                  | High     | Medium           |
| **Frontend UI** — React page with drag & drop, confidence bar, loading spinner                 | High     | High             |
| **Aggregation Logic** — Combine per-frame predictions to get video-level verdict               | Medium   | Low              |
| **Confusion Matrix Visualization** — Plot heatmaps using matplotlib/seaborn                    | Medium   | Low              |
| **Extend to 4 more manipulations** — Face2Face, FaceSwap, NeuralTextures, FaceShifter          | Low      | High             |
| **Explainability (Grad-CAM)** — Show which regions of the face triggered the "fake" prediction | Low      | Medium           |
| **ROC-AUC curve** — Plot tradeoff between True Positive & False Positive rate                  | Low      | Low              |
| **Deployment** — Containerize with Docker + deploy to Vercel/Render                            | Low      | Medium           |

---

## 10. Key Design Decisions & Learning Points

### Why these choices were made:

1. **EfficientNet-B0 over ResNet / Xception?**  
   EfficientNet family is optimized for speed-accuracy. B0 is the smallest and can run inference on a normal laptop CPU in ~100ms per face — perfect for a web demo.

2. **Why split by VIDEO ID, not random shuffle?**  
   Randomly splitting individual images would let the model "memorize" a person's identity rather than learn fake detection. Video-level split ensures a true measure of generalization.

3. **Fine-tune only last block + classifier?**  
   Fine-tuning the entire network would cause **catastrophic forgetting** — the model would lose its ImageNet knowledge. A graduated unfreezing strategy is safer.

4. **YuNet face detector over MTCNN or Haar Cascades?**  
   YuNet is 5–10× faster than MTCNN with comparable accuracy. For processing thousands of video frames, speed matters enormously.

---

## 11. How to Run the Project (Reproducibility)

### Step 1: Activate Environment

```bash
cd DeepTrace
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # Mac/Linux
```

### Step 2: Download Dataset (requires ~10-50 GB free space)

```bash
python faceforensics_download_v4.py raw_dataset -d original -c c23 -t videos
python faceforensics_download_v4.py raw_dataset -d Deepfakes -c c23 -t videos
```

### Step 3: Extract Faces from Videos

```bash
python preprocessing/extract_faces.py
```

_Output → `processed_dataset/real/_.jpg`and`processed_dataset/fake/_.jpg`_

### Step 4: Split Dataset

```bash
python preprocessing/split_dataset.py
```

_Output → `dataset/train`, `dataset/validation`, `dataset/test`_

### Step 5: Verify Model Setup (Sanity Check)

```bash
python training/train.py
```

_Should print "Forward pass successful!" + "Backward pass successful!"_

### Step 6: Train the Model (takes ~30 min to a few hours depending on GPU)

```bash
python training/train_model.py
```

_Best checkpoint → `models/deeptrace_efficientnet_b0_finetuned.pth`_

### Step 7: Evaluate on Test Set

```bash
python training/evaluate_model.py
```

_Prints Test Accuracy %, Classification Report, and Confusion Matrix_

---

## 12. Conclusion

DeepTrace represents a complete, production-intent deepfake detection **AI pipeline** from raw data to trained model. The next phase — building a web interface — will make this technology accessible to non-technical users, which is where the real-world impact lies.

The project demonstrates understanding of:

- ✅ Modern deep learning (Transfer Learning, fine-tuning schedules)
- ✅ Computer vision (face detection, image preprocessing, augmentation)
- ✅ ML best practices (train/val/test split, data leakage prevention, reproducibility)
- ✅ Real-world dataset handling (FaceForensics++ standard)
- ✅ Evaluation beyond simple accuracy (precision, recall, confusion matrix)

**Thank you for reading! 😊**
