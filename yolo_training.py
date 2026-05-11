# Import required libraries
from ultralytics import YOLO

# Define paths
data_yaml = r"D:\Master Folder\Documents List\projects\weedsdata\data.yaml"  # Path to your data.yaml file
model_type = "yolov8s.pt"  # Pretrained YOLOv8 model, options: yolov8n.pt, yolov8s.pt, etc.

# Load the YOLOv8 model
model = YOLO(model_type)

# Train the model
model.train(
    data=data_yaml,  # Path to your data.yaml file
    epochs=10,  
    lrf=0.01,     # Number of epochs
    batch=16,        # Batch size
    imgsz=640,       # Image size
    workers=4,       # Number of workers for data loading
    name=r"D:\Master Folder\Documents List\projects\weedsdata\experimentResults"  # Name of the training run
)