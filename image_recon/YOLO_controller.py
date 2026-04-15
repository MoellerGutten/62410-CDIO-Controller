import cv2
import numpy as np
import json
from ultralytics import YOLO

MODEL_PATH = "runs/segment/train5/weights/best.pt"
CAMERA_INDEX = 0

ARENA_W_CM = 167.0
ARENA_H_CM = 121.5

# Global variables for setup
corners = []
goal_a_pts = []
goal_b_pts = []
setup_step = "CORNERS"  # States: CORNERS, GOAL_A, GOAL_B, DONE

# Helper function for highly visible text without blocking the screen
def draw_text_with_outline(img, text, pos, font_scale, color, thickness):
    # Draw thick black outline
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 3)
    # Draw colored text over it
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
                    print("\n-> Setup Complete! Press ENTER to confirm or 'r' to restart.")

def setup_arena(cap):
    global corners, goal_a_pts, goal_b_pts, setup_step
    corners.clear()
    goal_a_pts.clear()
    goal_b_pts.clear()
    setup_step = "CORNERS"

    print("\n--- ARENA SETUP ---")
    print("Warming up camera...")
    for _ in range(5):
        ret, frame = cap.read()

    if not ret:
        print("Error: Could not grab frame.")
        return False

    window = "Arena Setup"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)
    cv2.setMouseCallback(window, handle_mouse)

    print("Step 1: Click 4 corners (BL, BR, TR, TL)")

    while True:
        display = frame.copy()

        # Dynamic Instructions using outlined text (No black bars)
        if setup_step == "CORNERS":
            draw_text_with_outline(display, f"1. Click 4 Corners (BL, BR, TR, TL): {len(corners)}/4", 
                                   (20, 40), 0.7, (0, 255, 255), 2)
        elif setup_step == "GOAL_A":
            draw_text_with_outline(display, f"2. GOAL A (Right, Small) - Click Top-Left then Bottom-Right: {len(goal_a_pts)}/2", 
                                   (20, 40), 0.7, (255, 150, 0), 2)
        elif setup_step == "GOAL_B":
            draw_text_with_outline(display, f"3. GOAL B (Left, Large) - Click Top-Left then Bottom-Right: {len(goal_b_pts)}/2", 
                                   (20, 40), 0.7, (0, 0, 255), 2)
        elif setup_step == "DONE":
            draw_text_with_outline(display, "Setup Complete! Press ENTER to start | 'R' to reset", 
                                   (20, 40), 0.7, (0, 255, 0), 2)

        # Draw Corners
        for i, (px, py) in enumerate(corners):
            cv2.circle(display, (px, py), 7, (0, 255, 0), -1)
        if len(corners) > 1:
            pts = np.array(corners, dtype=np.int32)
            cv2.polylines(display, [pts], isClosed=(len(corners) == 4), color=(0, 255, 0), thickness=2)

        # Draw Goal A (Blueish)
        if len(goal_a_pts) >= 1:
            cv2.circle(display, goal_a_pts[0], 5, (255, 150, 0), -1)
        if len(goal_a_pts) == 2:
            cv2.rectangle(display, goal_a_pts[0], goal_a_pts[1], (255, 150, 0), 2)
            draw_text_with_outline(display, "Goal A", (goal_a_pts[0][0], goal_a_pts[0][1] - 10), 0.5, (255, 150, 0), 2)

        # Draw Goal B (Red)
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
            break
        elif key == ord('q'):
            cv2.destroyWindow(window)
            return False

    cv2.destroyWindow(window)
    return True

def get_perspective_transform():
    src = np.array(corners, dtype=np.float32)
    dst = np.array([
        [0,          0         ],   # Bottom-left  -> (0, 0)
        [ARENA_W_CM, 0         ],   # Bottom-right -> (167, 0)
        [ARENA_W_CM, ARENA_H_CM],   # Top-right    -> (167, 121.5)
        [0,          ARENA_H_CM],   # Top-left     -> (0, 121.5)
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return M

def point_in_arena(x, y):
    if len(corners) < 4: return True
    poly = np.array(corners, dtype=np.int32)
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

def to_arena_coords(x, y, M):
    pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, M)
    ax, ay = transformed[0][0]
    return round(float(np.clip(ax, 0, ARENA_W_CM)), 1), round(float(np.clip(ay, 0, ARENA_H_CM)), 1)

def get_yolo_detections(results, model, M):
    detections = []
    if results.boxes is not None:
        for i in range(len(results.boxes)):
            cls = int(results.boxes[i].cls[0].item())
            label = model.names[cls]
            
            # Determine center (mask if available, otherwise bounding box)
            if results.masks is not None and len(results.masks.xy) > i:
                mask = results.masks.xy[i]
                cx = int(np.mean(mask[:, 0]))
                cy = int(np.mean(mask[:, 1]))
            else:
                x1, y1, x2, y2 = results.boxes[i].xyxy[0].tolist()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

            if not point_in_arena(cx, cy):
                continue
                
            ax, ay = to_arena_coords(cx, cy, M)
            
            # Extract the 4 corners of the YOLO bounding box
            x1, y1, x2, y2 = results.boxes[i].xyxy[0].tolist()
            tl = to_arena_coords(x1, y1, M)
            tr = to_arena_coords(x2, y1, M)
            br = to_arena_coords(x2, y2, M)
            bl = to_arena_coords(x1, y2, M)
            
            detections.append({
                "label": label,
                "cx": cx, "cy": cy,
                "ax": ax, "ay": ay,
                "corners": [tl, tr, br, bl] # Store all 4 corners
            })
    return detections

def draw_arena_overlay(vis_frame):
    # Draw Arena Polygon
    if len(corners) == 4:
        pts = np.array(corners, dtype=np.int32)
        cv2.polylines(vis_frame, [pts], isClosed=True, color=(0, 200, 255), thickness=2)
    
    # Draw Goal A (Right, Small)
    if len(goal_a_pts) == 2:
        cv2.rectangle(vis_frame, goal_a_pts[0], goal_a_pts[1], (255, 150, 0), 2)
        draw_text_with_outline(vis_frame, "Goal A", (goal_a_pts[0][0], goal_a_pts[0][1] - 10), 0.5, (255, 150, 0), 2)
        
    # Draw Goal B (Left, Large)
    if len(goal_b_pts) == 2:
        cv2.rectangle(vis_frame, goal_b_pts[0], goal_b_pts[1], (0, 0, 255), 2)
        draw_text_with_outline(vis_frame, "Goal B", (goal_b_pts[0][0], goal_b_pts[0][1] - 10), 0.5, (0, 0, 255), 2)
                    
    return vis_frame

def draw_positions(vis_frame, detections):
    for d in detections:
        # If it's a ball, just put the crosshair and center coordinate
        if "ball" in d['label'].lower():
            cx, cy = d["cx"], d["cy"]
            cv2.drawMarker(vis_frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            text = f"{d['label']} ({d['ax']}cm, {d['ay']}cm)"
            draw_text_with_outline(vis_frame, text, (cx + 10, cy - 10), 0.5, (0, 255, 0), 2)
        else:
            # If it's the cross/obstacle, draw a bounding box around it
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

    if not setup_arena(cap):
        print("Setup cancelled.")
        cap.release()
        return

    M = get_perspective_transform()
    print(f"\nArena locked: {ARENA_W_CM} x {ARENA_H_CM} cm (Bottom-Left is 0x0)")
    
    window_name = "Live Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    show_visuals = False  # Set to False by default

    while True:
        ret, frame = cap.read()
        if not ret: break

        preview_frame = frame.copy()
        preview_frame = draw_arena_overlay(preview_frame)
        
        # Developer UI: Outlined text on the Live Preview only
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
                M = get_perspective_transform()
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 1280, 720)
            else:
                break
                
        elif key == ord('s'):
            print("\n--- Processing Scan ---")
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            
            if len(corners) == 4:
                pts = np.array(corners, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
            
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
            results = model(masked_frame, verbose=False, conf=0.25)[0]
            detections = get_yolo_detections(results, model, M)

            # ----- CONVERT GOAL PIXELS TO ARENA (CM) COORDINATES -----
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

            # ----- PACK DATA INTO JSON DICTIONARY -----
            robot_data = {
                "goals": {
                    "A": goal_a_cm,  # Contains [{"x": .., "y": ..}, {"x": .., "y": ..}]
                    "B": goal_b_cm
                },
                "cross": {}, # Will hold the 4 corners of the cross
                "balls": []
            }
            
            for d in detections:
                # If the YOLO label DOES NOT contain the word "ball", treat it as the cross
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
                    # Treat it as a standard ball
                    robot_data["balls"].append({
                        "label": d["label"],
                        "x": d["ax"],
                        "y": d["ay"]
                    })
            
            json_output = json.dumps(robot_data, indent=2)
            print("\n--- DATA FOR ROBOT ---")
            print(json_output)
            
            # Save JSON to file
            with open("image_recon/robot_coords.json", "w") as json_file:
                json_file.write(json_output)

            # ----- HANDLE VISUALS -----
            if show_visuals:
                vis_frame = results.plot()
                vis_frame = draw_arena_overlay(vis_frame)
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
                vis_frame = draw_arena_overlay(vis_frame)
                vis_frame = draw_positions(vis_frame, detections)
                cv2.imwrite("image_recon/latest_scan_silent.jpg", vis_frame)
                print("-> Scan complete. (Image saved to latest_scan_silent.jpg, JSON saved to robot_coords.json)")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()