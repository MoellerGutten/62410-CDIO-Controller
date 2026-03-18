import cv2
from ultralytics import YOLO

MODEL_PATH = "runs/detect/train7/weights/best.pt"  # Pretrained COCO model (80 classes)

def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam (try index 1 or 2)")
        return

    print("Starting YOLO webcam test... Press 'q' to quit")
    print("Detected classes:", list(model.names.values())[:10], "...")  # Show first 10 classes

    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to read frame")
            break

        # Run inference (fast on CPU)
        results = model(frame, verbose=False, conf=0.25)[0]  # conf=0.25 filters weak detections
        
        # Show detections info
        if len(results.boxes) > 0:
            print(f"Frame {frame_count}: {len(results.boxes)} objects detected")

        # Visualize
        vis_frame = results.plot()
        cv2.imshow("YOLO Webcam Test (q=quit)", vis_frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam test complete!")

if __name__ == "__main__":
    main()
