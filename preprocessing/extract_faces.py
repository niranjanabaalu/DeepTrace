import cv2
import os


# ==========================================
# 1. FOLDERS
# ==========================================

REAL_FOLDER = "raw_dataset/original_sequences/youtube/c23/videos"
FAKE_FOLDER = "raw_dataset/manipulated_sequences/Deepfakes/c23/videos"

OUTPUT_REAL = "processed_dataset/real"
OUTPUT_FAKE = "processed_dataset/fake"

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"


# ==========================================
# 2. SETTINGS
# ==========================================

FRAME_INTERVAL = 10
FACE_SIZE = (224, 224)
PADDING = 20


# ==========================================
# 3. CREATE OUTPUT FOLDERS
# ==========================================

os.makedirs(OUTPUT_REAL, exist_ok=True)
os.makedirs(OUTPUT_FAKE, exist_ok=True)


# ==========================================
# 4. LOAD YUNET
# ==========================================

face_detector = cv2.FaceDetectorYN.create(
    MODEL_PATH,
    "",
    (320, 320),
    0.8,
    0.3,
    5000
)


# ==========================================
# 5. CHECK WHETHER VIDEO IS ALREADY DONE
# ==========================================

def video_already_processed(video_name, output_folder, label):

    prefix = f"{label}_{video_name}_"

    for filename in os.listdir(output_folder):

        if filename.startswith(prefix):
            return True

    return False


# ==========================================
# 6. EXTRACT FACES
# ==========================================

def extract_faces(video_path, output_folder, label):

    video_name = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    # Skip already processed videos
    if video_already_processed(
        video_name,
        output_folder,
        label
    ):

        print(
            f"\nSKIPPING: {video_name} "
            f"(already processed)"
        )

        return 0


    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print(
            "ERROR: Could not open:",
            video_path
        )

        return 0


    frame_number = 0
    saved_faces = 0

    print("\n----------------------------------------")
    print("Processing:", video_name)
    print("Label:", label)
    print("----------------------------------------")


    while True:

        ret, frame = cap.read()

        if not ret:
            break


        if frame_number % FRAME_INTERVAL == 0:

            height, width = frame.shape[:2]

            face_detector.setInputSize(
                (width, height)
            )

            _, faces = face_detector.detect(
                frame
            )


            if faces is not None:

                for face_index, detection in enumerate(faces):

                    x, y, w, h = detection[:4]

                    x = int(x)
                    y = int(y)
                    w = int(w)
                    h = int(h)


                    # Add padding
                    x1 = max(
                        0,
                        x - PADDING
                    )

                    y1 = max(
                        0,
                        y - PADDING
                    )

                    x2 = min(
                        width,
                        x + w + PADDING
                    )

                    y2 = min(
                        height,
                        y + h + PADDING
                    )


                    face = frame[
                        y1:y2,
                        x1:x2
                    ]


                    if face.size == 0:
                        continue


                    # Resize
                    face = cv2.resize(
                        face,
                        FACE_SIZE
                    )


                    filename = (
                        f"{label}_{video_name}_"
                        f"frame_{frame_number:06d}_"
                        f"face_{face_index}.jpg"
                    )


                    output_path = os.path.join(
                        output_folder,
                        filename
                    )


                    cv2.imwrite(
                        output_path,
                        face
                    )

                    saved_faces += 1


        frame_number += 1


    cap.release()


    print(
        "Frames processed:",
        frame_number
    )

    print(
        "Faces saved:",
        saved_faces
    )


    return saved_faces


# ==========================================
# 7. PROCESS REAL VIDEOS
# ==========================================

real_total = 0

real_videos = sorted([
    f
    for f in os.listdir(REAL_FOLDER)
    if f.lower().endswith(".mp4")
])


print("\n========================================")
print("REAL VIDEOS FOUND:", len(real_videos))
print("========================================")


for video in real_videos:

    video_path = os.path.join(
        REAL_FOLDER,
        video
    )

    real_total += extract_faces(
        video_path,
        OUTPUT_REAL,
        "real"
    )


# ==========================================
# 8. PROCESS FAKE VIDEOS
# ==========================================

fake_total = 0

fake_videos = sorted([
    f
    for f in os.listdir(FAKE_FOLDER)
    if f.lower().endswith(".mp4")
])


print("\n========================================")
print("FAKE VIDEOS FOUND:", len(fake_videos))
print("========================================")


for video in fake_videos:

    video_path = os.path.join(
        FAKE_FOLDER,
        video
    )

    fake_total += extract_faces(
        video_path,
        OUTPUT_FAKE,
        "fake"
    )


# ==========================================
# 9. FINAL RESULT
# ==========================================

print("\n========================================")
print("FACE EXTRACTION COMPLETED")
print("========================================")

print(
    "Real videos:",
    len(real_videos)
)

print(
    "Fake videos:",
    len(fake_videos)
)

print(
    "New real faces:",
    real_total
)

print(
    "New fake faces:",
    fake_total
)

print("========================================")