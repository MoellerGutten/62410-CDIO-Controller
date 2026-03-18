from ultralytics import YOLO

# Load the smallest, fastest model (perfect for real-time EV3)
model = YOLO('yolo11n.pt') 

# Train for 50-100 rounds (epochs)
model.train(data='YOLO_data/data.yaml', epochs=100, imgsz=640)    