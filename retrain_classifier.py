from ultralytics import YOLO

# Load base YOLO classification model
model = YOLO('yolov8n-cls.pt')

# Train again on balanced dataset
model.train(
    data='dataset',
    epochs=50,          # more epochs = deeper learning
    imgsz=224,          # good for plant-size images
    batch=8,
    lr0=0.001,          # stable learning rate
    project='runs',
    name='crop_health_classifier_v2'
)
