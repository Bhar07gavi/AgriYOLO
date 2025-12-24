import os

paths = {
    "train_healthy": "dataset/train/healthy",
    "train_dead": "dataset/train/dead",
    "valid_healthy": "dataset/valid/healthy",
    "valid_dead": "dataset/valid/dead"
}

for name, path in paths.items():
    if os.path.exists(path):
        count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"{name}: {count} images")
    else:
        print(f"{name}: folder not found")
