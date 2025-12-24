import glob
import os
from ultralytics import YOLO

# 1) Try to find crop_health_classifier_v2 best.pt automatically
candidates = glob.glob("runs/**/crop_health_classifier_v2/weights/best.pt", recursive=True)

if not candidates:
    print("❌ crop_health_classifier_v2 best.pt not found.")
    print("Searching for ANY best.pt inside runs/ ...")
    candidates = glob.glob("runs/**/best.pt", recursive=True)

if not candidates:
    print("❌ No best.pt found anywhere under runs/.")
else:
    model_path = candidates[0]
    print(f"✅ Using model: {model_path}")

    model = YOLO(model_path)
    print("model.names =", model.names)
