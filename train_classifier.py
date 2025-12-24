from ultralytics import YOLO

# Load YOLOv8 classification model
model = YOLO('yolov8n-cls.pt')  # pre-trained model

# Train on your dataset
model.train(
    data='dataset',             # path to your dataset folder
    epochs=30,                  # number of training epochs
    imgsz=224,                  # image size
    batch=8,                    # adjust based on RAM
    project='runs',             # where results are stored
    name='crop_health_classifier'
)
