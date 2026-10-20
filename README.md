# DeepTrace — AI-Powered Deepfake Video Detection

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Django](https://img.shields.io/badge/Framework-Django-green.svg)
![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-orange.svg)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-OpenCV-blueviolet.svg)
![Project Status](https://img.shields.io/badge/Status-Experimental-yellow.svg)

> ⚠️ **Disclaimer:** DeepTrace is an experimental academic project developed for learning and research purposes. Its predictions are not definitive proof that a video is real or fake, and the system is not a certified forensic tool.

---

DeepTrace is an AI-powered web application developed to analyze videos for potential deepfake content using deep learning and computer vision techniques. The project combines face detection, facial image classification, video-level prediction, and visual explainability in a Django-based web interface.

![DeepTrace Project](docs/images/DeepTraceHomePage.JPG)

---

## Overview

**DeepTrace** explores how deep learning can be applied to deepfake video detection. It processes video frames, detects faces, analyzes facial crops using a trained EfficientNet-B0 model, and combines predictions to produce an overall video-level result.

The application also incorporates Grad-CAM to help visualize image regions that influence the model's predictions.

---

## Tech Stack

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Python, Django
- **Deep Learning Framework:** PyTorch
- **Classification Model:** EfficientNet-B0
- **Face Detection:** YuNet, ONNX
- **Video Processing:** OpenCV
- **Explainable AI:** Grad-CAM
- **Database:** SQLite
- **Dataset:** FaceForensics++ (c23 subset)

---

## Screenshots

### 1. Home Page

The main interface for accessing the DeepTrace application.

![DeepTrace Home Page](docs/images/DeepTraceHomePage.JPG)

### 2. User Authentication

Registration and login interfaces for users.

![DeepTrace Sign Up Page](docs/images/DeepTraceSignupPage.JPG)

![DeepTrace Sign In Page](docs/images/DeepTraceSigninPage.JPG)

### 3. Video Analysis Steps

An overview of the video analysis workflow.

![DeepTrace Analysis Steps](docs/images/DeepTraceStepsPage.JPG)

### 4. Analysis Page

The interface for initiating and viewing video analysis.

![DeepTrace Analysis Page](docs/images/DeepTraceAnalyzePage.JPG)

### 5. Analysis Output

The interface displaying the analysis results.

![DeepTrace Output Page](docs/images/DeepTraceOutputPage.JPG)

### 6. Command-Line Output

A view of the analysis output from the command-line workflow.

![DeepTrace Command Output](docs/images/DeepTraceCmdOutputPage.JPG)

---

## Key Features

- **Video Analysis:** Processes uploaded videos for potential deepfake indicators.
- **Face Detection:** Uses YuNet to detect faces in selected video frames.
- **Deepfake Classification:** Uses a trained EfficientNet-B0 model to classify detected facial crops.
- **Video-Level Prediction:** Aggregates predictions from analyzed face crops into an overall video result.
- **Explainable AI:** Uses Grad-CAM visualizations to help interpret model predictions.
- **User Authentication:** Provides signup and signin interfaces.
- **Result Visualization:** Presents analysis output through the web application.

---

## How It Works

1. The user uploads a video through the application.
2. The video is processed and frames are extracted.
3. YuNet detects faces in selected frames.
4. Detected facial regions are prepared for model inference.
5. EfficientNet-B0 predicts the class of each processed face crop.
6. The individual predictions are aggregated into a video-level result.
7. Grad-CAM can be used to visualize regions influencing the model's predictions.
8. The application displays the analysis output.

---

## Dataset and Experimental Results

DeepTrace was experimentally evaluated using a subset of the **FaceForensics++ dataset with c23 compression**.

### Dataset Summary

| Dataset Detail        | Value |
| --------------------- | ----: |
| Original videos       |    50 |
| Fake videos           |    50 |
| Extracted face crops  | 5,878 |
| Training face crops   | 3,784 |
| Validation face crops |   638 |
| Testing face crops    | 1,456 |

### Experimental Evaluation

| Evaluation Metric         |             Result |
| ------------------------- | -----------------: |
| Face-crop test accuracy   |             66.48% |
| Video-level test accuracy | 80% (16/20 videos) |

These results represent the project's experimental evaluation setup. The video-level result is based on a test set of 20 videos and should not be interpreted as evidence of reliable performance on arbitrary real-world videos.

---

## Installation and Setup

### Prerequisites

- Python 3.12 or a compatible Python version
- Git
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/niranjanabaalu/DeepTrace.git
cd DeepTrace
```

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Apply Database Migrations

```bash
python manage.py migrate
```

### 5. Run the Development Server

```bash
python manage.py runserver
```

Open the following address in your browser:

`http://127.0.0.1:8000/`

**Note:** Running the application may require additional model files, including the trained EfficientNet-B0 checkpoint and YuNet ONNX model. Ensure that these files and any required configuration are available. Do not upload restricted datasets, private videos, credentials, or secrets to the repository.

---

## Project Purpose

DeepTrace was developed as an academic project to explore the practical application of artificial intelligence, deep learning, and computer vision in synthetic media analysis.

The project demonstrates the integration of a deep learning model with a web application, video processing, face detection, and explainable AI techniques.

---

## Limitations

- The model's performance may vary across datasets, video compression levels, lighting conditions, and unseen deepfake generation techniques.
- The experimental dataset and evaluation set are limited in size.
- Face-crop classification accuracy does not directly represent real-world video detection accuracy.
- Grad-CAM visualizations indicate regions influencing a model prediction; they do not independently prove that a video is manipulated.
- DeepTrace should not be used as the sole basis for forensic, legal, or other high-stakes decisions.

---

## Future Enhancements

- Train and evaluate models using larger and more diverse datasets.
- Improve generalization to unseen deepfake generation methods.
- Optimize processing speed for longer videos.
- Enhance confidence-score presentation and explainability visualizations.
- Expand testing across different video formats and compression levels.
- Improve deployment and model-management workflows.

---

## Developer

**Niranjana B**

- GitHub: [github.com/niranjanabaalu](https://github.com/niranjanabaalu)

---

_DeepTrace demonstrates an academic exploration of deepfake video detection using deep learning, computer vision, and explainable AI._
