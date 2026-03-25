from ultralytics import YOLO

model = YOLO('yolo11n-seg.pt')

model.train(data='YOLO_data_2.0/data.yaml', epochs=100, imgsz=640)