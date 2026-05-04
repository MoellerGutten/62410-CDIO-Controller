from ultralytics import YOLO

model = YOLO('image_recon/yolo11n-pose.pt')

model.train(data='image_recon/YOLO_data_5.0/data.yaml', epochs=100, imgsz=640)  