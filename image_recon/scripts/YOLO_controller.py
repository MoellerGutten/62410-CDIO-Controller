import math

import cv2
import numpy as np
import json
import os
from ultralytics import YOLO

MODEL_PATH = "runs/pose/train2/weights/best.pt"
CAMERA_INDEX = 0

ARENA_W_CM = 167.0
ARENA_H_CM = 121.5
CONFIG_FILE = "image_recon/arena_config.json"

# Global variables
corners = []
goal_a_pts = []
goal_b_pts = []
setup_step = "CORNERS"  # States: CORNERS, GOAL_A, GOAL_B, DONE

# --- HELPER FUNCTIONS ---
def draw_text_with_outline(img, text, pos, font_scale, color, thickness):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 3)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def handle_mouse(event, x, y, flags, param):
    global corners, goal_a_pts, goal_b_pts, setup_step
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if setup_step == "CORNERS":
            if len(corners) < 4:
                corners.append((x, y))
                print(f"✓ Corner {len(corners)}/4 set: ({x}, {y})")
                if len(corners) == 4:
                    setup_step = "GOAL_A"
                    print("\n-> Next: Click Top-Left then Bottom-Right of GOAL A (Right, Small)")
                    
        elif setup_step == "GOAL_A":
            if len(goal_a_pts) < 2:
                goal_a_pts.append((x, y))
                print(f"✓ Goal A Pt {len(goal_a_pts)}/2 set: ({x}, {y})")
                if len(goal_a_pts) == 2:
                    setup_step = "GOAL_B"
                    print("\n-> Next: Click Top-Left then Bottom-Right of GOAL B (Left, Large)")
                    
        elif setup_step == "GOAL_B":
            if len(goal_b_pts) < 2:
                goal_b_pts.append((x, y))
                print(f"✓ Goal B Pt {len(goal_b_pts)}/2 set: ({x}, {y})")
                if len(goal_b_pts) == 2:
                    setup_step = "DONE"
                    print("\n-> Setup Complete! Press ENTER to confirm and save.")

def load_calibration():
    global corners, goal_a_pts, goal_b_pts
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                corners = [tuple(pt) for pt in data.get("corners", [])]
                goal_a_pts = [tuple(pt) for pt in data.get("goal_a_pts", [])]
                goal_b_pts = [tuple(pt) for pt in data.get("goal_b_pts", [])]
                if len(corners) == 4:
                    print("\n[INFO] Loaded previous arena calibration.")
                    return True
        except Exception as e:
            print(f"[WARNING] Could not load config: {e}")
    return False

def save_calibration():
    data = {
        "corners": corners,
        "goal_a_pts": goal_a_pts,
        "goal_b_pts": goal_b_pts
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=4)
    print("\n[INFO] Arena calibration saved.")

def setup_arena(cap):
    global corners, goal_a_pts, goal_b_pts, setup_step
    corners.clear()
    goal_a_pts.clear()
    goal_b_pts.clear()
    setup_step = "CORNERS"

    print("\n--- MANUAL ARENA SETUP ---")
    print("Warming up camera...")
    for _ in range(5):
        ret, frame = cap.read()

    window = "Arena Setup"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)
    cv2.setMouseCallback(window, handle_mouse)

    while True:
        ret, frame = cap.read()
        if not ret: continue
        display = frame.copy()

        if setup_step == "CORNERS":
            draw_text_with_outline(display, f"1. Click 4 INSIDE Corners (BL, BR, TR, TL): {len(corners)}/4", (20, 40), 0.7, (0, 255, 255), 2)
        elif setup_step == "GOAL_A":
            draw_text_with_outline(display, f"2. GOAL A (Right, Small) - Click Top-Left then Bottom-Right: {len(goal_a_pts)}/2", (20, 40), 0.7, (255, 150, 0), 2)
        elif setup_step == "GOAL_B":
            draw_text_with_outline(display, f"3. GOAL B (Left, Large) - Click Top-Left then Bottom-Right: {len(goal_b_pts)}/2", (20, 40), 0.7, (0, 0, 255), 2)
        elif setup_step == "DONE":
            draw_text_with_outline(display, "Setup Complete! Press ENTER to lock and save | 'R' to reset", (20, 40), 0.7, (0, 255, 0), 2)

        for i, (px, py) in enumerate(corners):
            cv2.circle(display, (px, py), 7, (0, 255, 0), -1)
        if len(corners) > 1:
            pts = np.array(corners, dtype=np.int32)
            cv2.polylines(display, [pts], isClosed=(len(corners) == 4), color=(0, 255, 0), thickness=2)

        if len(goal_a_pts) >= 1:
            cv2.circle(display, goal_a_pts[0], 5, (255, 150, 0), -1)
        if len(goal_a_pts) == 2:
            cv2.rectangle(display, goal_a_pts[0], goal_a_pts[1], (255, 150, 0), 2)
            draw_text_with_outline(display, "Goal A", (goal_a_pts[0][0], goal_a_pts[0][1] - 10), 0.5, (255, 150, 0), 2)

        if len(goal_b_pts) >= 1:
            cv2.circle(display, goal_b_pts[0], 5, (0, 0, 255), -1)
        if len(goal_b_pts) == 2:
            cv2.rectangle(display, goal_b_pts[0], goal_b_pts[1], (0, 0, 255), 2)
            draw_text_with_outline(display, "Goal B", (goal_b_pts[0][0], goal_b_pts[0][1] - 10), 0.5, (0, 0, 255), 2)

        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):
            corners.clear()
            goal_a_pts.clear()
            goal_b_pts.clear()
            setup_step = "CORNERS"
            print("\n--- Setup Reset ---")
        elif key == 13 and setup_step == "DONE":
            save_calibration()
            break
        elif key == ord('q'):
            cv2.destroyWindow(window)
            return False

    cv2.destroyWindow(window)
    return True

def get_perspective_transform():
    # Since we manually click the exact bounds, no offsets are needed.
    src = np.array(corners, dtype=np.float32)
    dst = np.array([
        [0,          0         ],
        [ARENA_W_CM, 0         ],
        [ARENA_W_CM, ARENA_H_CM],
        [0,          ARENA_H_CM],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    return M, M_inv

def to_arena_coords(x, y, M):
    pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, M)
    ax, ay = transformed[0][0]
    return round(float(ax), 1), round(float(ay), 1)

def get_yolo_detections(results, model, M):
    detections = []
    if results.boxes is not None:
        for i in range(len(results.boxes)):
            cls = int(results.boxes[i].cls[0].item())
            label = model.names[cls]
            
            # Get Box Center
            x1, y1, x2, y2 = results.boxes[i].xyxy[0].tolist()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # Convert center to arena coords
            ax, ay = to_arena_coords(cx, cy, M)
            
            # Keep only items inside the arena bounds
            if ax < 0 or ax > ARENA_W_CM or ay < 0 or ay > ARENA_H_CM:
                continue
                
            tl = to_arena_coords(x1, y1, M)
            tr = to_arena_coords(x2, y1, M)
            br = to_arena_coords(x2, y2, M)
            bl = to_arena_coords(x1, y2, M)

            # --- POSE EXTRACTION ---
            keypoints_arena = []
            heading = None
            
            if results.keypoints is not None and len(results.keypoints.xy) > i:
                kpts = results.keypoints.xy[i].cpu().numpy()
                for kx, ky in kpts:
                    # Check if keypoint is visible/valid (not 0,0)
                    if kx > 0 and ky > 0:
                        k_ax, k_ay = to_arena_coords(kx, ky, M)
                        keypoints_arena.append({"x": k_ax, "y": k_ay, "px": int(kx), "py": int(ky)})

                # Calculate Heading (Adjust indices to match your dataset's front/back keypoints)
                if "robot" in label.lower() and len(keypoints_arena) >= 2:
                    FRONT_KP_INDEX = 0  # <--- Change to your front keypoint index
                    BACK_KP_INDEX = 1   # <--- Change to your back keypoint index
                    
                    try:
                        front_kp = keypoints_arena[FRONT_KP_INDEX]
                        back_kp = keypoints_arena[BACK_KP_INDEX]
                        
                        # Calculate angle using atan2
                        dy = front_kp["y"] - back_kp["y"]
                        dx = front_kp["x"] - back_kp["x"]
                        
                        # Get angle in degrees (0 is straight right, 90 is straight up, etc.)
                        angle_deg = math.degrees(math.atan2(dy, dx))
                        heading = round(angle_deg, 1)
                    except IndexError:
                        pass # Not enough keypoints detected for this frame

            detections.append({
                "label": label, "cx": cx, "cy": cy,
                "ax": ax, "ay": ay, "corners": [tl, tr, br, bl],
                "keypoints": keypoints_arena,
                "heading": heading
            })
    return detections

def draw_arena_overlay(vis_frame, M_inv):
    if len(corners) == 4:
        pts = np.array(corners, dtype=np.int32)
        cv2.polylines(vis_frame, [pts], isClosed=True, color=(0, 200, 255), thickness=3)

    if len(goal_a_pts) == 2:
        cv2.rectangle(vis_frame, goal_a_pts[0], goal_a_pts[1], (255, 150, 0), 2)
        draw_text_with_outline(vis_frame, "Goal A", (goal_a_pts[0][0], goal_a_pts[0][1] - 10), 0.5, (255, 150, 0), 2)
    if len(goal_b_pts) == 2:
        cv2.rectangle(vis_frame, goal_b_pts[0], goal_b_pts[1], (0, 0, 255), 2)
        draw_text_with_outline(vis_frame, "Goal B", (goal_b_pts[0][0], goal_b_pts[0][1] - 10), 0.5, (0, 0, 255), 2)
    return vis_frame

def draw_positions(vis_frame, detections):
    for d in detections:
        label_lower = d['label'].lower()
        if "ball" in label_lower:
            cv2.drawMarker(vis_frame, (d["cx"], d["cy"]), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            draw_text_with_outline(vis_frame, f"{d['label']} ({d['ax']}cm, {d['ay']}cm)", (d["cx"] + 10, d["cy"] - 10), 0.5, (0, 255, 0), 2)
        elif "robot" in label_lower:
            cv2.drawMarker(vis_frame, (d["cx"], d["cy"]), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
            heading_text = f"HDG: {d['heading']}*" if d['heading'] is not None else ""
            draw_text_with_outline(vis_frame, f"{d['label']} {heading_text}", (d["cx"] + 10, d["cy"] - 10), 0.5, (0, 255, 255), 2)
            
            # Draw Heading Arrow based on Keypoints
            if d['heading'] is not None and len(d['keypoints']) >= 2:
                # Using the pixel coords for drawing
                front_px = (d['keypoints'][0]['px'], d['keypoints'][0]['py']) # Adjust index if needed
                back_px = (d['keypoints'][1]['px'], d['keypoints'][1]['py'])  # Adjust index if needed
                cv2.arrowedLine(vis_frame, back_px, front_px, (0, 255, 255), 3, tipLength=0.3)
        else:
            cv2.drawMarker(vis_frame, (d["cx"], d["cy"]), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
            draw_text_with_outline(vis_frame, f"{d['label']} (Obstacle)", (d["cx"] + 10, d["cy"] - 10), 0.5, (255, 0, 255), 2)
    return vis_frame

# --- EXPOSED API FUNCTIONS ---

def initialize_vision():
    """Call this from external scripts to setup the tracker without UI."""
    if not load_calibration():
        print("[ERROR] Cannot initialize vision. No calibration file found.")
        print("Please run arena_tracker.py manually to calibrate the corners!")
        return None, None, None
        
    model = YOLO(MODEL_PATH)
    M, M_inv = get_perspective_transform()
    os.makedirs("image_recon", exist_ok=True)
    return model, M, M_inv

def scan(frame, model, M, M_inv):
    """Call this from external scripts to process a frame and safely update the JSON."""
    global corners, goal_a_pts, goal_b_pts
    
    # Apply Blackout Mask
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    if len(corners) == 4:
        pts = np.array(corners, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Run YOLO (Make sure model returns keypoints)
    results = model(masked_frame, verbose=False, conf=0.25)[0]
    detections = get_yolo_detections(results, model, M)

    # Construct JSON Output
    goal_a_cm = [{"x": to_arena_coords(pt[0], pt[1], M)[0], "y": to_arena_coords(pt[0], pt[1], M)[1]} for pt in goal_a_pts] if len(goal_a_pts) == 2 else []
    goal_b_cm = [{"x": to_arena_coords(pt[0], pt[1], M)[0], "y": to_arena_coords(pt[0], pt[1], M)[1]} for pt in goal_b_pts] if len(goal_b_pts) == 2 else []

    robot_data = {
        "arena": {
            "width_cm": ARENA_W_CM,
            "height_cm": ARENA_H_CM,
            "corners_pixel": [
                {"position": "bottom-left", "x": corners[0][0], "y": corners[0][1]},
                {"position": "bottom-right", "x": corners[1][0], "y": corners[1][1]},
                {"position": "top-right", "x": corners[2][0], "y": corners[2][1]},
                {"position": "top-left", "x": corners[3][0], "y": corners[3][1]}
            ] if len(corners) == 4 else []
        },
        "goals": {"A": goal_a_cm, "B": goal_b_cm},
        "robot": {}, 
        "cross": {}, 
        "balls": []
    }
    
    for d in detections:
        label_lower = d["label"].lower()
        corners_list = [
            {"position": "top-left", "x": d["corners"][0][0], "y": d["corners"][0][1]},
            {"position": "top-right", "x": d["corners"][1][0], "y": d["corners"][1][1]},
            {"position": "bottom-right", "x": d["corners"][2][0], "y": d["corners"][2][1]},
            {"position": "bottom-left", "x": d["corners"][3][0], "y": d["corners"][3][1]}
        ]

        if "ball" in label_lower:
            robot_data["balls"].append({"label": d["label"], "x": d["ax"], "y": d["ay"]})
        elif "robot" in label_lower:
            robot_data["robot"] = {
                "label": d["label"],
                "x": d["ax"],
                "y": d["ay"],
                "heading": d["heading"],
                "keypoints": [{"x": kp["x"], "y": kp["y"]} for kp in d["keypoints"]],
                "corners": corners_list
            }
        else: # Obstacles / Cross
            robot_data["cross"] = {
                "label": d["label"],
                "corners": corners_list
            }
    
    json_output = json.dumps(robot_data, indent=2)

    # Atomic Save
    temp_file = "image_recon/robot_coords_temp.json"
    final_file = "image_recon/robot_coords.json"
    with open(temp_file, "w") as json_file:
        json_file.write(json_output)
    os.replace(temp_file, final_file) 
    
    # Draw visual frame for returning
    vis_frame = results.plot()
    vis_frame = draw_arena_overlay(vis_frame, M_inv)
    vis_frame = draw_positions(vis_frame, detections)
    
    return robot_data, vis_frame

def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {CAMERA_INDEX}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    os.makedirs("image_recon", exist_ok=True)

    if not load_calibration():
        if not setup_arena(cap):
            print("Setup cancelled.")
            cap.release()
            return

    M, M_inv = get_perspective_transform()
    print(f"\nArena locked: {ARENA_W_CM} x {ARENA_H_CM} cm (Bottom-Left is 0x0)")
    
    window_name = "Live Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    show_visuals = False 
    continuous_mode = False

    while True:
        ret, frame = cap.read()
        if not ret: break

        preview_frame = frame.copy()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('v'):
            show_visuals = not show_visuals
        elif key == ord('c'):
            continuous_mode = not continuous_mode
            print(f"\n-> Continuous Mode: {'ON' if continuous_mode else 'OFF'}")
        elif key == ord('r'):
            cv2.destroyWindow(window_name) 
            if setup_arena(cap):
                M, M_inv = get_perspective_transform()
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 1280, 720)
            else:
                break
        
        do_scan = continuous_mode or (key == ord('s'))

        if do_scan:
            robot_data, vis_frame = scan(frame, model, M, M_inv)

            if key == ord('s'):
                print("\n--- SINGLE SCAN COMPLETE ---")
                print(json.dumps(robot_data, indent=2))
                
                if show_visuals:
                    cv2.imwrite("image_recon/latest_scan.jpg", vis_frame)
                    result_window = "Screenshot Result (Press ANY KEY to resume)"
                    cv2.namedWindow(result_window, cv2.WINDOW_NORMAL)
                    cv2.imshow(result_window, vis_frame)
                    cv2.waitKey(0) 
                    cv2.destroyWindow(result_window)
                else:
                    cv2.imwrite("image_recon/latest_scan_silent.jpg", vis_frame)
            
            if continuous_mode:
                preview_frame = vis_frame
        else:
            preview_frame = draw_arena_overlay(preview_frame, M_inv)

        vis_status = "ON" if show_visuals else "OFF"
        cont_status = "ON" if continuous_mode else "OFF"
        draw_text_with_outline(preview_frame, f"'s': SCAN | 'c': CONT [{cont_status}] | 'v': VIS [{vis_status}] | 'r': RESET | 'q': QUIT", 
                               (20, 40), 0.7, (0, 255, 0), 2)
        
        cv2.imshow(window_name, preview_frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()