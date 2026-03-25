from ultralytics import YOLO

model = YOLO('image_recon/yolo11n-seg.pt')

model.train(data='image_recon/YOLO_data_2.0/data.yaml', epochs=100, imgsz=640)