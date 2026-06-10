from ultralytics import YOLO

model = YOLO("image_recon/yolo11n-pose.pt")

if __name__ == '__main__':
    model.train(data='image_recon/YOLO_data_6.0/data.yaml', epochs=100, imgsz=640)