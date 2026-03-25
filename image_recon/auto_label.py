from ultralytics import YOLO

MODEL_PATH = "../runs/segment/train3/weights/best.pt"
model = YOLO(MODEL_PATH)

NEW_IMAGES_FOLDER = "../data/data_2.0/rest_of_photos_2.0"

def main():
    print(f"Starting auto-labeling for images in: {NEW_IMAGES_FOLDER}")
    
    results = model.predict(
        source=NEW_IMAGES_FOLDER,
        conf=0.25,        
        save_txt=True,     
        save_conf=False,
    )
    
    print("\nSuccess! Auto-labeling complete.")
    print("Check your 'runs/segment/predict/labels' folder for the new .txt files!")

if __name__ == "__main__":
    main()