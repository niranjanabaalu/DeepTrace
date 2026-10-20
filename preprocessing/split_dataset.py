import os
import shutil
import random


# ==========================================
# 1. FOLDERS
# ==========================================

PROCESSED_REAL = "processed_dataset/real"
PROCESSED_FAKE = "processed_dataset/fake"

DATASET_ROOT = "dataset"


# ==========================================
# 2. SETTINGS
# ==========================================

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(RANDOM_SEED)


# ==========================================
# 3. CREATE DATASET FOLDERS
# ==========================================

for split in ["train", "validation", "test"]:

    for label in ["real", "fake"]:

        folder = os.path.join(
            DATASET_ROOT,
            split,
            label
        )

        os.makedirs(
            folder,
            exist_ok=True
        )


# ==========================================
# 4. CLEAR OLD SPLIT
# ==========================================

print("Clearing old dataset split...")

for split in ["train", "validation", "test"]:

    for label in ["real", "fake"]:

        folder = os.path.join(
            DATASET_ROOT,
            split,
            label
        )

        for filename in os.listdir(folder):

            filepath = os.path.join(
                folder,
                filename
            )

            if os.path.isfile(filepath):

                os.remove(filepath)


# ==========================================
# 5. GET VIDEO ID FROM FILENAME
# ==========================================

def get_video_id(filename, label):

    name = os.path.splitext(filename)[0]

    parts = name.split("_")

    if label == "real":

        # Example:
        # real_183_frame_000010_face_0

        return parts[1]


    else:

        # Example:
        # fake_183_253_frame_000010_face_0

        return parts[1]


# ==========================================
# 6. GROUP FACES BY VIDEO
# ==========================================

def group_by_video(folder, label):

    video_groups = {}

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(".jpg")
    ]

    for filename in files:

        video_id = get_video_id(
            filename,
            label
        )

        if video_id not in video_groups:

            video_groups[video_id] = []

        video_groups[video_id].append(
            filename
        )

    return video_groups


# ==========================================
# 7. SPLIT VIDEOS
# ==========================================

def split_video_ids(video_ids):

    video_ids = list(video_ids)

    random.shuffle(video_ids)

    total = len(video_ids)

    train_count = int(
        total * TRAIN_RATIO
    )

    validation_count = int(
        total * VALIDATION_RATIO
    )

    train_ids = video_ids[
        :train_count
    ]

    validation_ids = video_ids[
        train_count:
        train_count + validation_count
    ]

    test_ids = video_ids[
        train_count + validation_count:
    ]

    return (
        train_ids,
        validation_ids,
        test_ids
    )


# ==========================================
# 8. COPY FILES
# ==========================================

def copy_files(
    groups,
    video_ids,
    destination
):

    count = 0

    for video_id in video_ids:

        for filename in groups[video_id]:

            source = os.path.join(
                groups["__folder__"],
                filename
            )

            target = os.path.join(
                destination,
                filename
            )

            shutil.copy2(
                source,
                target
            )

            count += 1

    return count


# ==========================================
# 9. PROCESS EACH CLASS
# ==========================================

def process_class(
    source_folder,
    label
):

    print("\n========================================")
    print("PROCESSING:", label.upper())
    print("========================================")


    groups = group_by_video(
        source_folder,
        label
    )

    groups["__folder__"] = source_folder


    video_ids = [
        video_id
        for video_id in groups
        if video_id != "__folder__"
    ]


    print(
        "Videos found:",
        len(video_ids)
    )


    train_ids, validation_ids, test_ids = (
        split_video_ids(video_ids)
    )


    print(
        "Training videos:",
        len(train_ids)
    )

    print(
        "Validation videos:",
        len(validation_ids)
    )

    print(
        "Test videos:",
        len(test_ids)
    )


    train_count = copy_files(
        groups,
        train_ids,
        os.path.join(
            DATASET_ROOT,
            "train",
            label
        )
    )


    validation_count = copy_files(
        groups,
        validation_ids,
        os.path.join(
            DATASET_ROOT,
            "validation",
            label
        )
    )


    test_count = copy_files(
        groups,
        test_ids,
        os.path.join(
            DATASET_ROOT,
            "test",
            label
        )
    )


    print(
        "Training faces:",
        train_count
    )

    print(
        "Validation faces:",
        validation_count
    )

    print(
        "Test faces:",
        test_count
    )


# ==========================================
# 10. RUN SPLIT
# ==========================================

process_class(
    PROCESSED_REAL,
    "real"
)

process_class(
    PROCESSED_FAKE,
    "fake"
)


# ==========================================
# 11. FINAL RESULT
# ==========================================

print("\n========================================")
print("DATASET SPLIT COMPLETED")
print("========================================")

print("Dataset location:")
print("dataset/")

print("\nSplit:")
print("70% Training")
print("15% Validation")
print("15% Test")

print("\nRandom seed:", RANDOM_SEED)

print("========================================")