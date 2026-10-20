

from pathlib import Path
from collections import defaultdict
import random
import re
import shutil

# Project paths
ROOT = Path(__file__).resolve().parent.parent
REAL_VIDEOS = ROOT / "raw_dataset/original_sequences/youtube/c23/videos"
FAKE_VIDEOS = ROOT / "raw_dataset/manipulated_sequences/Deepfakes/c23/videos"
PROCESSED = ROOT / "processed_dataset"
DATASET = ROOT / "dataset"

SEED = 42
random.seed(SEED)

# 1. Build connected groups of related source videos
parent = {}

def find(x):
    parent.setdefault(x, x)
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

real_ids = {
    p.stem for p in REAL_VIDEOS.glob("*.mp4")
}

fake_pairs = []
for p in FAKE_VIDEOS.glob("*.mp4"):
    match = re.fullmatch(r"(\d+)_(\d+)", p.stem)
    if match:
        a, b = match.groups()
        fake_pairs.append((a, b))
        union(a, b)

# Include all real videos in the grouping
for video_id in real_ids:
    find(video_id)

groups = defaultdict(set)
for video_id in parent:
    groups[find(video_id)].add(video_id)

components = list(groups.values())
random.shuffle(components)

# 2. Split whole source groups, never individual frames
n = len(components)
n_train = int(n * 0.70)
n_val = int(n * 0.15)

group_split = {}

for i, group in enumerate(components):
    if i < n_train:
        split = "train"
    elif i < n_train + n_val:
        split = "validation"
    else:
        split = "test"

    for video_id in group:
        group_split[video_id] = split

print("Related source groups:", n)
print("Source videos by split:")
for split in ("train", "validation", "test"):
    ids = sorted(
        vid for vid, assigned in group_split.items()
        if assigned == split
    )
    print(f"{split}: {len(ids)} videos -> {', '.join(ids)}")

# 3. Clear old split folders and recreate them
for split in ("train", "validation", "test"):
    for label in ("real", "fake"):
        target = DATASET / split / label
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

# 4. Copy face crops according to their source group
copied = defaultdict(int)
unmatched = []

for label in ("real", "fake"):
    source_dir = PROCESSED / label

    for image_path in source_dir.glob("*"):
        if not image_path.is_file():
            continue

        # Expected examples:
        # real_183_frame_000390.jpg
        # fake_183_253_frame_000390.jpg
        match = re.match(
            r"^(?:real|fake)_(\d+)(?:_(\d+))?_",
            image_path.name
        )

        if not match:
            unmatched.append(image_path.name)
            continue

        first_id, second_id = match.groups()

        if first_id not in group_split:
            unmatched.append(image_path.name)
            continue

        split = group_split[first_id]

        if label == "fake" and second_id:
            if second_id not in group_split:
                unmatched.append(image_path.name)
                continue

            if group_split[first_id] != group_split[second_id]:
                raise RuntimeError(
                    f"Related sources split apart: {image_path.name}"
                )

        destination = DATASET / split / label / image_path.name
        shutil.copy2(image_path, destination)
        copied[(split, label)] += 1

print("\nFACE CROP COUNTS")
for split in ("train", "validation", "test"):
    print(
        f"{split}: "
        f"real={copied[(split, 'real')]}, "
        f"fake={copied[(split, 'fake')]}"
    )

print("\nUnmatched files:", len(unmatched))
if unmatched:
    print("First unmatched filenames:", unmatched[:10])

print("\nSOURCE-GROUP SPLIT COMPLETED")


