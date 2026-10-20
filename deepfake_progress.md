DeepTrace: An Explainable AI-Based Deepfake Detection System
Project progress and technical documentation
Project workspace: D:\Niranj\DeepTraceProject\DeepTrace
Primary task: analyze an uploaded video and estimate whether it is real or manipulated using face detection and an EfficientNet-B0 image classifier.

1. Project Overview
   DeepTrace is a web-based prototype for deepfake video analysis. A user uploads a video through the website. The backend samples frames from the video, detects faces, crops the detected face regions, and passes those crops to a trained deep-learning classifier. The classifier produces scores for the fake and real classes. DeepTrace aggregates the scores from the analyzed crops and returns a video-level prediction.

The project also includes Grad-CAM experiments to help visualize which regions of a face crop influenced a classifier prediction.

Important limitation: this is an experimental prototype, not a forensic guarantee. Its output is a model prediction, not proof that a video is authentic or manipulated. The current evaluation dataset is small, and the displayed scores should not be interpreted as calibrated probabilities.

2. Main Technologies

- Python: video processing, training, evaluation, and inference.
- PyTorch / torchvision: EfficientNet-B0 model and inference pipeline.
- OpenCV: video reading, frame sampling, and face detection.
- YuNet: face detector loaded from an ONNX model.
- Pillow: image handling where needed.
- scikit-learn: train/validation/test evaluation metrics.
- Django: web application and upload workflow.
- HTML / CSS / JavaScript: user interface.
- SQLite: available in the project's Python/Django environment for local application data, depending on the configured Django database.

3. Dataset Used
   3.1 Primary dataset source
   The project used a subset of FaceForensics++, specifically:

- original_sequences/youtube/c23/videos for real/original videos.
- manipulated_sequences/Deepfakes/c23/videos for manipulated videos.
- c23 is the compressed-video version used for this experiment.
- The manipulated subset is the Deepfakes manipulation category.
  Official project and access instructions:
- FaceForensics++ project: https://github.com/ondyari/faceforensics
- Dataset instructions: https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md
  FaceForensics++ is distributed under its own access terms. Follow those terms when obtaining, storing, using, or redistributing the videos. Do not redistribute dataset videos unless the license/access terms permit it.
  3.2 Downloaded subset
  The download script was used to obtain 50 original videos and 50 Deepfakes videos for the project experiment. The script used was faceforensics_download_v4.py.

Commands recorded during the project:
python faceforensics_download_v4.py raw_dataset -d original -c c23 -t videos -n 50 --server EU2
python faceforensics_download_v4.py raw_dataset -d Deepfakes -c c23 -t videos -n 50 --server EU2
Expected raw data directories:
raw_dataset/
├── original_sequences/
│ └── youtube/
│ └── c23/
│ └── videos/
└── manipulated_sequences/
└── Deepfakes/
└── c23/
└── videos/
The 50 original videos and 50 manipulated videos are the source video subset. They should not be confused with the number of face crops used for training: a single video can provide many sampled frames and face crops.
3.3 Source-aware dataset splitting
The project used preprocessing/split_by_source.py to separate source-related videos into training, validation, and test partitions. This was done to reduce the risk of closely related source videos appearing across different splits and making the evaluation overly optimistic.

Recorded source split:

- Training source IDs: 34
- Validation source IDs: 6
- Test source IDs: 10
  Recorded face-crop counts:
  Partition Real crops Fake crops Total
  Train 1,891 1,893 3,784
  Validation 319 319 638
  Test 726 730 1,456
  Total 2,936 2,942 5,878

These are the recorded crop counts from the current experiment, not the number of unique videos. Because several crops can come from one video, crop-level metrics are not the same as video-level metrics.

The split script reported 25 related source groups and no unmatched files. The source-aware split is intended to reduce leakage, but it does not by itself guarantee that the model generalizes to every camera, subject, compression method, or deepfake generator.
3.4 Face-crop dataset
After video frames and faces were processed, the classifier dataset was organized in a class-folder structure compatible with torchvision.datasets.ImageFolder:
dataset/
├── train/
│ ├── fake/
│ └── real/
├── validation/
│ ├── fake/
│ └── real/
└── test/
├── fake/
└── real/
The recorded class indices were:

- fake = 0
- real = 1
  The exact preprocessing script outputs and counts should be checked against the local folders if the dataset is rebuilt.

4. Project Development Progress
   The project evolved through these stages.
   Stage 1 — Environment and project structure

- Created the project workspace and Python virtual environment.
- Set up Python and installed the main deep-learning and computer-vision packages.
- Created folders for raw data, processed data, preprocessing, models, training, and outputs.
  Recorded environment:
- Python 3.12.4
- PyTorch 2.14.0+cpu
- OpenCV 5.0.0
- NumPy 2.5.0-series
- Pillow 12.3.0
- CPU-only execution; CUDA was not available.
  Package versions can change. Run python --version and python -m pip freeze to record the exact versions in the active environment.
  Stage 2 — Dataset acquisition
- Obtained access to FaceForensics++.
- Downloaded 50 original and 50 Deepfakes c23 videos.
- Stored the files under raw_dataset/.
  Stage 3 — Preprocessing and face extraction
- Wrote/used scripts for organizing the source videos and splitting the dataset by related source IDs.
- Extracted face crops from video frames for image-based classifier training.
- Stored processed images in real/fake class folders.
- Used a source-aware split to reduce overlap between related videos across train, validation, and test partitions.
  Stage 4 — Initial model experiments
  Several EfficientNet-B0 checkpoints were produced:
- models/deeptrace_efficientnet_b0.pth
- models/deeptrace_efficientnet_b0_finetuned.pth
- models/deeptrace_efficientnet_b0_v2.pth
  Recorded earlier experiments:
- The baseline checkpoint achieved 51.25% accuracy on an earlier random-split test.
- The fine-tuned experiment showed overfitting.
- These results motivated a simpler classifier-only training experiment.
  These are separate experiments and should not be combined as if they were one evaluation.
  Stage 5 — Classifier-only EfficientNet-B0 training
  The current main checkpoint is:

models/deeptrace_efficientnet_b0_v2.pth

Recorded training approach:

1. Started with a pretrained EfficientNet-B0.
2. Froze the feature-extraction backbone.
3. Trained the classifier head for the two classes (fake, real).
4. Used horizontal-flip augmentation during training.
5. Used batch size 32, 10 epochs, and Adam with learning rate 0.001.
6. Evaluated on validation data during training and then evaluated on the held-out test data.
   Recorded first-epoch metrics:

- Training loss: 0.4937
- Training accuracy: 77.14%
- Validation loss: 0.4380
- Validation accuracy: 80.72%
  The first-epoch values are not the final training metrics. Do not describe them as final model performance.
  Stage 6 — Evaluation
  On the source-aware held-out face-crop test set, the v2 model achieved:
- Face-crop accuracy: 66.48%
- Fake: precision 0.68, recall 0.63, F1-score 0.65, support 730
- Real: precision 0.65, recall 0.70, F1-score 0.68, support 726
  Recorded confusion matrix (rows = actual class; columns = predicted class; order = fake, real):
  Predicted fake Predicted real
  Actual fake 461 269
  Actual real 219 507
  This result shows that crop-level classification is imperfect. Some manipulated crops are classified as real, and some real crops are classified as fake.
  Stage 7 — Video-level evaluation
  The project also evaluated video-level aggregation over 20 test videos.

Recorded result:

- Video-level accuracy: 80% (16/20 videos)
- Fake class precision/recall/F1: 0.80 / 0.80 / 0.80
- Real class precision/recall/F1: 0.80 / 0.80 / 0.80
  Recorded confusion matrix, with class order fake then real:
  Predicted fake Predicted real
  Actual fake 8 2
  Actual real 2 8
  This is a preliminary result from only 20 videos. It should not be presented as a reliable estimate of performance on arbitrary real-world videos. Video-level evaluation is useful because the final website returns one result for a video, but the sample size needs to be increased for stronger conclusions.
  Stage 8 — Grad-CAM explainability
- Created training/gradcam.py.
- Generated Grad-CAM heatmaps for selected face crops.
- Saved outputs under outputs/gradcam/.
- One inspected fake crop was predicted as real with 86.14% model confidence.
  Grad-CAM highlights image regions that influenced the classifier's output. It does not prove that those regions are fake or explain the model's decision perfectly.
  Stage 9 — Video inference script
  The script training/analyze_video.py connects video decoding, face detection, crop classification, and video-level score aggregation.

Recorded behavior:

- Loads models/deeptrace_efficientnet_b0_v2.pth.
- Loads models/face_detection_yunet_2023mar.onnx.
- Samples every 10th frame.
- Detects faces in sampled frames and classifies face crops.
- Averages the fake-class scores to obtain a video-level fake score.
- Uses a 0.5 decision threshold for the final fake/real label.
- Writes/overwrites outputs/video_analysis/frame_timeline.csv.
  The score is an average of model outputs and is not calibrated. A displayed “fake probability” is best described as a model score unless calibration has been tested.
  Stage 10 — Django website and user interface
- Began integrating the analysis pipeline into a Django website.
- Built a homepage with a video-upload form and result display.
- Used a dark wine/red floral theme with Cormorant Garamond and DM Sans fonts.
- Added a Sign In and Sign Up page design and Django authentication views/routes in the development plan.
- Planned static assets under detector/static/detector/images/ and CSS under detector/static/detector/css/.
  Integration status note: the pages and code snippets have been discussed, but the documentation does not independently verify that every Django view, URL, static asset, and authentication flow has been installed and tested in the local project. Verify these in the working tree before describing them as fully completed.

5. How the Model Was Trained
   5.1 What the model learns
   EfficientNet-B0 is a convolutional neural network (CNN) architecture for image classification. In this project, it receives a cropped face image and outputs scores for two labels:

- Fake
- Real
  The pretrained backbone contains visual feature-extraction layers. In the classifier-only experiment, those layers were frozen and the final classification head was trained for the project's two classes.

The model learns patterns that help separate examples in the training dataset. It is not explicitly checking a list of rules such as “blinking incorrectly” or “mouth movement mismatch.” It learns statistical visual patterns from the training crops.
5.2 Training workflow

1. Read face-crop images from the training and validation folders.
2. Resize and normalize images according to the EfficientNet-B0 preprocessing expected by the training code.
3. Apply horizontal flipping to training images as augmentation.
4. Pass batches through EfficientNet-B0.
5. Calculate the classification loss against the known training labels.
6. Update the classifier-head weights using Adam (learning rate 0.001).
7. Repeat for 10 epochs with batch size 32.
8. Track validation loss/accuracy.
9. Save the trained model checkpoint.
10. Evaluate the saved checkpoint on the separate test partition.
    The exact resize and normalization values should be confirmed in training/train_classifier_only.py before putting them into a final academic report.
    
5.3 Why split the dataset?

- Training set: updates model parameters.
- Validation set: helps monitor performance and guide training choices.
- Test set: provides an independent evaluation after training.
  A source-aware split is important for video datasets because different manipulated videos can be derived from the same original source. If related frames or videos occur in both training and test data, the evaluation may look better than real generalization.

6. How an Uploaded Video Is Analyzed
   The end-to-end conceptual workflow is:
   User uploads a video
   |
   v
   Django receives the uploaded file
   |
   v
   Video is opened by OpenCV
   |
   v
   Sample frames at the configured interval
   (currently every 10th frame in analyze_video.py)
   |
   v
   YuNet detects faces in each sampled frame
   |
   v
   Crop and preprocess each detected face
   |
   v
   EfficientNet-B0 predicts fake/real scores for each crop
   |
   v
   Aggregate crop-level fake scores across the video
   |
   v
   Compare the aggregate score with threshold 0.50
   |
   v
   Display video-level FAKE or REAL prediction and scores
   Step 1 — Upload
   The user selects a video on the website and submits the form. The Django view receives the uploaded file. The exact upload handling depends on the current analyze_video view implementation.

A production deployment should validate file type and size, generate a safe temporary filename, avoid trusting the original filename, and delete temporary uploads when they are no longer needed.
Step 2 — Open and sample the video
OpenCV opens the uploaded video and reads its frames. Instead of analyzing every frame, the current script samples every 10th frame. This reduces processing time.

For example, if a video has 900 frames, sampling every 10th frame would inspect about 90 frames, subject to how the loop handles frame numbering and end-of-video conditions.

Consequence: frames between sampled frames are not directly analyzed. A brief manipulation appearing only in skipped frames could be missed.
Step 3 — Detect faces with YuNet
The script loads the YuNet face-detection model from:

models/face_detection_yunet_2023mar.onnx

For each sampled frame, the detector finds face bounding boxes. A bounding box identifies the approximate position of a face in the frame.

Frames with no detected faces cannot contribute a face-crop classification. Detection quality can be affected by small faces, profile views, blur, occlusion, lighting, and image resolution.
Step 4 — Crop and preprocess faces
Each detected face region is cropped from its frame. The crop is resized and normalized into the format expected by EfficientNet-B0.

The classifier does not receive the full video as one temporal sequence in this implementation. It receives face images extracted from sampled frames.
Step 5 — Classify each face crop
The EfficientNet-B0 classifier produces class scores for each crop. The recorded class index mapping is:

- 0 → fake
- 1 → real
  A crop could produce a result such as:
- Fake score: 0.72
- Real score: 0.28
  That means the model favors the fake class for that crop. It does not mean there is a verified 72% chance the video is a deepfake.
  Step 6 — Aggregate predictions across the video
  The inference script averages the fake scores of the analyzed crops to form a video-level fake score. The corresponding real score is calculated from the model's output aggregation in the script.

A simplified explanation is:
\[
S*{\text{fake, video}} =
\frac{1}{N}\sum*{i=1}^{N} S\_{\text{fake},i}
\]
where:

- \(N\) is the number of analyzed face crops.
- \(S\_{\text{fake},i}\) is the fake-class score for crop \(i\).
  This formula describes the recorded average-score approach. If the implementation aggregates scores in a different order or handles multiple faces differently, use the exact code in analyze_video.py as the final authority.
  Step 7 — Apply the decision threshold
  The recorded threshold is 0.50:
- If the aggregate fake score is at least 0.50, the script labels the video FAKE.
- If it is below 0.50, the script labels the video REAL.
  This threshold is a simple decision rule. It has not been shown to be optimal or calibrated for all video types.
  Step 8 — Show the result
  The Django page can display:
- Final label (FAKE or REAL)
- Aggregate fake and real scores
- Analysis output or timeline information, if the view passes these values to the template
  The project script writes outputs/video_analysis/frame_timeline.csv, but whether this CSV and every per-frame detail are shown on the web page depends on the Django integration.

7. Recorded Example Results
   These are results from previous local experiments, not guarantees for future uploads.
   Example Recorded outcome
   033.mp4 (real test video) REAL; fake score 34.87%, real score 65.13%; 81 sampled frames, 81 faces, 214 crops recorded
   183_253.mp4 (fake test video) FAKE; fake score 81.43%, real score 18.57%; 39 sampled frames and 39 faces recorded
   5695c2c725f24e0783ebb8b9187a1621.mov (online-shopping clip) REAL prediction; fake score 33.75%, real score 66.25%; 88 sampled frames, 88 frames with faces, 176 faces analyzed

For the first example, the crop count is larger than the number of faces because multiple crops may be produced from a frame or detected face-processing path. Confirm the local CSV/logs if exact counting behavior needs to be explained in a presentation.

The online-shopping clip's REAL prediction does not establish that the source is authentic. It only records how the current model classified that file. 8. Explainability with Grad-CAM
Grad-CAM generates a heatmap showing which spatial regions in a selected image had stronger influence on the model's class score.

In this project:

1. A face crop is passed through the trained classifier.
2. Gradients associated with the selected class are used to weight feature maps from a convolutional layer.
3. The weighted maps are combined into a heatmap.
4. The heatmap is overlaid on the original crop and saved under outputs/gradcam/.
   Use this to discuss model attention, not as definitive evidence of forgery. Heatmaps can be misleading and do not establish causal explanations by themselves.
5. Current Evaluation and Limitations
6. Small video sample: only 20 videos were used for the recorded video-level evaluation.
7. Crop-level performance is moderate: the recorded held-out face-crop accuracy is 66.48%.
8. Dataset scope: the current experiment uses original YouTube videos and the Deepfakes manipulation category from FaceForensics++ at c23 compression. It does not cover all deepfake methods.
9. Frame sampling: every 10th frame is inspected, so short-lived evidence may be skipped.
10. Face detection: videos with no detectable faces or very poor-quality faces may not be analyzable.
11. Domain shift: social-media compression, filters, low resolution, lighting, and different generators may change results.
12. Score calibration: model scores are not proven calibrated probabilities.
13. Temporal evidence: this version classifies face crops and aggregates scores; it is not a temporal model trained to understand motion across the entire video.
14. False positives and false negatives: a real video can be flagged fake, and a fake video can be labeled real.
15. Security and privacy: uploaded videos can contain sensitive personal data. A deployed service should use secure upload handling, size/type limits, restricted storage, and a clear retention/deletion policy.
    For a stronger evaluation, test more source-independent videos, report precision/recall/F1 and confusion matrices at video level, evaluate threshold choices on validation data, and consider testing additional manipulation methods.
16. Project File Structure
    The main recorded folders and files are:
    DeepTrace/
    ├── raw_dataset/
    ├── dataset/
    │ ├── train/
    │ ├── validation/
    │ └── test/
    ├── preprocessing/
    │ ├── extract_faces.py
    │ ├── split_by_source.py
    │ └── split_dataset.py
    ├── processed_dataset/
    │ ├── real/
    │ └── fake/
    ├── models/
    │ ├── deeptrace_efficientnet_b0.pth
    │ ├── deeptrace_efficientnet_b0_finetuned.pth
    │ ├── deeptrace_efficientnet_b0_v2.pth
    │ └── face_detection_yunet_2023mar.onnx
    ├── training/
    │ ├── train_model.py
    │ ├── train_classifier_only.py
    │ ├── evaluate_model.py
    │ ├── evaluate_video_level.py
    │ ├── gradcam.py
    │ └── analyze_video.py
    ├── outputs/
    │ ├── gradcam/
    │ └── video_analysis/
    ├── faceforensics_download_v4.py
    └── PROJECT_DOCUMENTATION.md
    The Django app and template/static files may be present in the current local tree but were not all independently verified while this document was prepared. Add or correct the Django files in this diagram to match the actual local project.
17. How to Run the Main Video Analyzer
    From the project root, activate the virtual environment and run the existing analyzer script with its required arguments. Check the script's argument parser first, because its exact command-line interface was not verified in this document.

For example, to inspect supported options:
python training/analyze_video.py --help
If the script does not support --help, open training/analyze_video.py and inspect the section that reads command-line arguments or defines the input-video path.

The Django site should call the analyzer through its backend view, or import a reusable analysis function from the script. Avoid duplicating the model-loading and frame-processing logic in the template. 12. Suggested Demonstration Flow

1. Start the Django development server.
2. Open the DeepTrace homepage.
3. Upload a known real sample from the held-out test set.
4. Show the sampled-frame count, detected faces, crop count, final score, and predicted label if the interface exposes them.
5. Upload a known manipulated sample.
6. Compare the outputs.
7. Show a Grad-CAM heatmap to explain which image regions influenced a crop prediction.
8. Explain the limitations honestly: the model can make mistakes, and a prediction is not proof of authenticity.
   Use only videos you are allowed to demonstrate, and clearly identify known dataset examples as evaluation samples.
9. Summary
   DeepTrace combines a Django upload interface with an OpenCV/YuNet video-processing pipeline and an EfficientNet-B0 face-crop classifier. The model was trained using face crops extracted from a small FaceForensics++ subset. During inference, the system samples frames, detects faces, classifies the resulting crops, averages fake-class scores, and applies a 0.50 threshold to produce one video-level label. Grad-CAM provides a visual aid for interpreting selected crop predictions.

The project demonstrates a complete experimental deepfake-detection workflow, while the current dataset size, moderate crop-level accuracy, frame sampling, and lack of score calibration mean that its results must be presented as preliminary.
References

1. FaceForensics++ project: https://github.com/ondyari/faceforensics
2. FaceForensics++ dataset access instructions: https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md
3. PyTorch: https://pytorch.org/
4. OpenCV documentation: https://docs.opencv.org/
5. Django documentation: https://docs.djangoproject.com/
