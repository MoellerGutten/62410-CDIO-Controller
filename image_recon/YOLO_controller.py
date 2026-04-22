import cv2
import numpy as np
import json
import os
from ultralytics import YOLO

MODEL_PATH = "runs/segment/train5/weights/best.pt"
CAMERA_INDEX = 0

ARENA_W_CM = 167.0
ARENA_H_CM = 121.5

# --- THE MAGIC OFFSETS ---
# Since color detection might lock onto the outside edge or center of the red wood,
# use these to push the true (0,0) coordinate grid inward to the playing surface.
OFFSET_X_CM = 1.5  # cm inward 
OFFSET_Y_CM = 1.5  # cm inward 

# Global variables
corners = []
goal_a_pts = []
goal_b_pts = []
setup_step = "FIND_ARENA"  # States: FIND_ARENA, GOAL_A, GOAL_B, DONE

# --- HELPER FUNCTIONS ---
def draw_text_with_outline(img, text, pos, font_scale, color, thickness):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 3)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def order_points(pts):
    """Sorts corners into: Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-Left
    rect[2] = pts[np.argmax(s)] # Bottom-Right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-Right
    rect[3] = pts[np.argmax(diff)] # Bottom-Left
    return rect

def find_red_arena(frame):
    """Finds the largest 4-point red polygon in the frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Red wraps around the HSV hue cylinder, so we need two masks
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Clean up the mask (remove small noise)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Only proceed if it's reasonably large
        if cv2.contourArea(largest_contour) > 10000:
            # Approximate the contour to a polygon
            peri = cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
            
            # If the polygon has 4 corners, we found the rectangular arena!
            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                # Ensure they are in the correct order for the perspective transform
                ordered_pts = order_points(pts)
                
                # YOLO uses BL, BR, TR, TL mapping internally in our get_perspective_transform
                tl, tr, br, bl = ordered_pts
                return [tuple(bl), tuple(br), tuple(tr), tuple(tl)]
                
    return None

def handle_mouse(event, x, y, flags, param):
    global goal_a_pts, goal_b_pts, setup_step
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if setup_step == "GOAL_A":
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
                    print("\n-> Setup Complete! Press ENTER to confirm or 'r' to restart.")

def setup_arena(cap):
    global corners, goal_a_pts, goal_b_pts, setup_step
    corners.clear()
    goal_a_pts.clear()
    goal_b_pts.clear()
    setup_step = "FIND_ARENA"

    print("\n--- ARENA SETUP ---")
    print("Warming up camera...")
    for _ in range(5):
        ret, frame = cap.read()

    window = "Arena Setup"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)
    cv2.setMouseCallback(window, handle_mouse)

    print("Step 1: Auto-detecting red borders...")

    while True:
        ret, frame = cap.read()
        if not ret: continue
        display = frame.copy()

        if setup_step == "FIND_ARENA":
            draw_text_with_outline(display, "1. Auto-detecting Red Arena borders...", 
                                   (20, 40), 0.7, (0, 255, 255), 2)
            
            detected_corners = find_red_arena(frame)
            
            if detected_corners:
                corners = detected_corners
                pts = np.array(corners, dtype=np.int32)
                cv2.polylines(display, [pts], isClosed=True, color=(0, 255, 255), thickness=3)
                for i, (px, py) in enumerate(corners):
                    cv2.circle(display, (int(px), int(py)), 7, (0, 255, 0), -1)
                    
                draw_text_with_outline(display, "Arena Found! Press ENTER to lock, or adjust lighting.", 
                                       (20, 80), 0.7, (0, 255, 0), 2)
            else:
                corners.clear()
                draw_text_with_outline(display, "No 4-corner red shape found. Ensure borders are visible.", 
                                       (20, 80), 0.7, (0, 0, 255), 2)

        elif setup_step == "GOAL_A":
            draw_text_with_outline(display, f"2. GOAL A (Right, Small) - Click Top-Left then Bottom-Right: {len(goal_a_pts)}/2", 
                                   (20, 40), 0.7, (255, 150, 0), 2)
            if len(corners) == 4:
                pts = np.array(corners, dtype=np.int32)
                cv2.polylines(display, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        elif setup_step == "GOAL_B":
            draw_text_with_outline(display, f"3. GOAL B (Left, Large) - Click Top-Left then Bottom-Right: {len(goal_b_pts)}/2", 
                                   (20, 40), 0.7, (0, 0, 255), 2)
            if len(corners) == 4:
                pts = np.array(corners, dtype=np.int32)
                cv2.polylines(display, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        elif setup_step == "DONE":
            draw_text_with_outline(display, "Setup Complete! Press ENTER to start | 'R' to reset", 
                                   (20, 40), 0.7, (0, 255, 0), 2)
            if len(corners) == 4:
                pts = np.array(corners, dtype=np.int32)
                cv2.polylines(display, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw Goals during setup
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
            setup_step = "FIND_ARENA"
            print("\n--- Setup Reset: Scanning for red borders ---")
        elif key == 13: # ENTER
            if setup_step == "FIND_ARENA" and len(corners) == 4:
                setup_step = "GOAL_A"
                print("\n✓ Arena Locked!")
            elif setup_step == "DONE":
                break
        elif key == ord('q'):
            cv2.destroyWindow(window)
            return False

    cv2.destroyWindow(window)
    return True

def get_perspective_transform():
    src = np.array(corners, dtype=np.float32)
    
    # We apply the physical offset to pull the 0,0 coordinate inward
    dst = np.array([
        [OFFSET_X_CM,              OFFSET_Y_CM             ],   # Bottom-Left
        [ARENA_W_CM - OFFSET_X_CM, OFFSET_Y_CM             ],   # Bottom-Right
        [ARENA_W_CM - OFFSET_X_CM, ARENA_H_CM - OFFSET_Y_CM],   # Top-Right
        [OFFSET_X_CM,              ARENA_H_CM - OFFSET_Y_CM],   # Top-Left
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    return M, M_inv

def point_in_arena(x, y):
    # YOLO scans the whole screen, coordinates are filtered later
    return True

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
            
            if results.masks is not None and len(results.masks.xy) > i:
                mask = results.masks.xy[i]
                cx = int(np.mean(mask[:, 0]))
                cy = int(np.mean(mask[:, 1]))
            else:
                x1, y1, x2, y2 = results.boxes[i].xyxy[0].tolist()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

            ax, ay = to_arena_coords(cx, cy, M)
            
            # Filter out things physically outside the true 0 to Width boundaries
            if ax < 0 or ax > ARENA_W_CM or ay < 0 or ay > ARENA_H_CM:
                continue
            
            x1, y1, x2, y2 = results.boxes[i].xyxy[0].tolist()
            tl = to_arena_coords(x1, y1, M)
            tr = to_arena_coords(x2, y1, M)
            br = to_arena_coords(x2, y2, M)
            bl = to_arena_coords(x1, y2, M)
            
            detections.append({
                "label": label,
                "cx": cx, "cy": cy,
                "ax": ax, "ay": ay,
                "corners": [tl, tr, br, bl] 
            })
    return detections

def draw_arena_overlay(vis_frame, M_inv):
    if len(corners) == 4:
        # Define the 4 corners of the TRUE physical arena in CM
        true_corners_cm = np.array([
            [[0.0, 0.0]],                           # Bottom-Left
            [[ARENA_W_CM, 0.0]],                    # Bottom-Right
            [[ARENA_W_CM, ARENA_H_CM]],             # Top-Right
            [[0.0, ARENA_H_CM]]                     # Top-Left
        ], dtype=np.float32)
        
        # Convert the true CM corners back to pixel locations on the screen
        true_corners_px = cv2.perspectiveTransform(true_corners_cm, M_inv)
        pts = np.int32(true_corners_px)
        
        # Draw the inner boundary line
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
        if "ball" in d['label'].lower():
            cx, cy = d["cx"], d["cy"]
            cv2.drawMarker(vis_frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            text = f"{d['label']} ({d['ax']}cm, {d['ay']}cm)"
            draw_text_with_outline(vis_frame, text, (cx + 10, cy - 10), 0.5, (0, 255, 0), 2)
        else:
            cx, cy = d["cx"], d["cy"]
            cv2.drawMarker(vis_frame, (cx, cy), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
            text = f"{d['label']} (Obstacle)"
            draw_text_with_outline(vis_frame, text, (cx + 10, cy - 10), 0.5, (255, 0, 255), 2)
    return vis_frame

def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {CAMERA_INDEX}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    os.makedirs("image_recon", exist_ok=True)

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

    while True:
        ret, frame = cap.read()
        if not ret: break

        preview_frame = frame.copy()
        preview_frame = draw_arena_overlay(preview_frame, M_inv)
        
        visual_status = "ON" if show_visuals else "OFF"
        draw_text_with_outline(preview_frame, f"'s': SCAN | 'v': VISUALS [{visual_status}] | 'r': RE-DRAW | 'q': QUIT", 
                               (20, 40), 0.8, (0, 255, 0), 2)
        
        cv2.imshow(window_name, preview_frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
            
        elif key == ord('v'):
            show_visuals = not show_visuals
            print(f"\n-> Visual output toggled: {'ON' if show_visuals else 'OFF'}")
            
        elif key == ord('r'):
            print("\n--- Re-drawing Arena ---")
            cv2.destroyWindow(window_name) 
            if setup_arena(cap):
                M, M_inv = get_perspective_transform()
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 1280, 720)
            else:
                break
                
        elif key == ord('s'):
            print("\n--- Processing Scan ---")
            
            results = model(frame, verbose=False, conf=0.25)[0]
            detections = get_yolo_detections(results, model, M)

            goal_a_cm = []
            if len(goal_a_pts) == 2:
                for pt in goal_a_pts:
                    ax, ay = to_arena_coords(pt[0], pt[1], M)
                    goal_a_cm.append({"x": ax, "y": ay})
                    
            goal_b_cm = []
            if len(goal_b_pts) == 2:
                for pt in goal_b_pts:
                    ax, ay = to_arena_coords(pt[0], pt[1], M)
                    goal_b_cm.append({"x": ax, "y": ay})

            true_corners_cm = np.array([[[0.0, 0.0]], [[ARENA_W_CM, 0.0]], [[ARENA_W_CM, ARENA_H_CM]], [[0.0, ARENA_H_CM]]], dtype=np.float32)
            true_corners_px = cv2.perspectiveTransform(true_corners_cm, M_inv)

            robot_data = {
                "arena": {
                    "width_cm": ARENA_W_CM,
                    "height_cm": ARENA_H_CM,
                    "corners_pixel": [
                        {"position": "bottom-left", "x": round(float(true_corners_px[0][0][0]), 1), "y": round(float(true_corners_px[0][0][1]), 1)},
                        {"position": "bottom-right", "x": round(float(true_corners_px[1][0][0]), 1), "y": round(float(true_corners_px[1][0][1]), 1)},
                        {"position": "top-right", "x": round(float(true_corners_px[2][0][0]), 1), "y": round(float(true_corners_px[2][0][1]), 1)},
                        {"position": "top-left", "x": round(float(true_corners_px[3][0][0]), 1), "y": round(float(true_corners_px[3][0][1]), 1)}
                    ] if len(corners) == 4 else []
                },
                "goals": {
                    "A": goal_a_cm, 
                    "B": goal_b_cm
                },
                "cross": {}, 
                "balls": []
            }
            
            for d in detections:
                if "ball" not in d["label"].lower():
                    robot_data["cross"] = {
                        "label": d["label"],
                        "corners": [
                            {"position": "top-left", "x": d["corners"][0][0], "y": d["corners"][0][1]},
                            {"position": "top-right", "x": d["corners"][1][0], "y": d["corners"][1][1]},
                            {"position": "bottom-right", "x": d["corners"][2][0], "y": d["corners"][2][1]},
                            {"position": "bottom-left", "x": d["corners"][3][0], "y": d["corners"][3][1]}
                        ]
                    }
                else:
                    robot_data["balls"].append({
                        "label": d["label"],
                        "x": d["ax"],
                        "y": d["ay"]
                    })
            
            json_output = json.dumps(robot_data, indent=2)
            print("\n--- DATA FOR ROBOT ---")
            print(json_output)
            
            with open("image_recon/robot_coords.json", "w") as json_file:
                json_file.write(json_output)

            if show_visuals:
                vis_frame = results.plot()
                vis_frame = draw_arena_overlay(vis_frame, M_inv)
                vis_frame = draw_positions(vis_frame, detections)
                cv2.imwrite("image_recon/latest_scan.jpg", vis_frame)
                
                result_window = "Screenshot Result (Press ANY KEY to resume)"
                cv2.namedWindow(result_window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(result_window, 1280, 720)
                cv2.imshow(result_window, vis_frame)
                cv2.waitKey(0) 
                cv2.destroyWindow(result_window)
                print("\nResuming preview...")
            else:
                vis_frame = results.plot()
                vis_frame = draw_arena_overlay(vis_frame, M_inv)
                vis_frame = draw_positions(vis_frame, detections)
                cv2.imwrite("image_recon/latest_scan_silent.jpg", vis_frame)
                print("-> Scan complete. (Files saved to 'image_recon')")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()