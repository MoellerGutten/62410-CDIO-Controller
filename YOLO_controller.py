import socket
import cv2
from ultralytics import YOLO

# Configuration
EV3_IP = "10.42.0.182"
EV3_PORT = 9999
MODEL_PATH = "best.pt"  # Your trained YOLOv11 model

def main():
    # 1. Load your trained model
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0) # Use 0 for default webcam

    print("Starting AI Controller...")

    # Keep a single persistent connection if possible, or connect per loop
    # For real-time, one persistent connection is better for latency
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((EV3_IP, EV3_PORT))
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            # 2. Run Inference
            results = model(frame, verbose=False)[0]
            
            ball_pos = None
            center_x_pos = None

            # 3. Extract Coordinates
            for box in results.boxes:
                cls = int(box.cls[0])
                name = model.names[cls]
                # Get center (x, y) of the bounding box
                x_c, y_c, w, h = box.xywh[0] 

                if name == "ball":
                    ball_pos = (float(x_c), float(y_c))
                elif name == "center_x":
                    center_x_pos = (float(x_c), float(y_c))

            # 4. Calculate Relative Error
            if ball_pos and center_x_pos:
                # Math: Ball Position - X Position = Error
                err_x = ball_pos[0] - center_x_pos[0]
                err_y = ball_pos[1] - center_x_pos[1]
                
                # Format command: "err_x,err_y"
                cmd = f"{err_x:.2f},{err_y:.2f}"
                sock.sendall((cmd + "\n").encode("utf-8"))
                
                # Optional: Read EV3 Ack
                # ack = sock.recv(1024) 
            
            # Visual Feedback
            cv2.imshow("YOLO EV3 Monitor", results.plot())
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()