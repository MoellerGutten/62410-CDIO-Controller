import cv2
from ultralytics import YOLO

MODEL_PATH = "runs/segment/train5/weights/best.pt"
IMAGE_PATH  = "image_recon/data/data_2.0/all_photos_2.0/WIN_20260318_12_25_46_Pro.jpg"

def main():
    model = YOLO(MODEL_PATH)

    image = cv2.imread(IMAGE_PATH)
    
    if image is None:
        print(f"Error: Could not read image at {IMAGE_PATH}. Check the path!")
        return

    print("Running YOLO inference on the image...")

    results = model(image, verbose=True, conf=0.25)[0] 
    
    if len(results.boxes) > 0:
        print(f"Success! {len(results.boxes)} objects detected.")
    else:
        print("No objects detected above the confidence threshold.")

    vis_image = results.plot()
    
    window_name = "YOLO Image Test (Press any key to close)"
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    cv2.resizeWindow(window_name, 800, 600) 
    
    cv2.imshow(window_name, vis_image)

    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    print("Test complete!")

if __name__ == "__main__":
    main()