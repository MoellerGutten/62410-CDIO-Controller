import cv2
import YOLO_controller # This imports your script

# 1. Initialize everything (Loads YOLO and the saved corners)
model, M, M_inv = YOLO_controller.initialize_vision()

if model is None:
    print("Please run arena_tracker.py manually first to calibrate the corners!")
    exit()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # 2. Call the scan function directly! 
    # It automatically updates the JSON file and returns the data to this script.
    robot_data, vis_frame = YOLO_controller.scan(frame, model, M, M_inv)
    
    # Do whatever you want with robot_data here...
    print(robot_data["balls"])