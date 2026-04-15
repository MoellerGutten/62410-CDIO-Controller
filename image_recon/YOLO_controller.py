import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "runs/segment/train5/weights/best.pt"
CAMERA_INDEX = 0

ARENA_W_CM = 167.0
ARENA_H_CM = 121.5

corners = []

def click_corner(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(corners) < 4:
            corners.append((x, y))
            print(f"✓ Corner {len(corners)} set: ({x}, {y})")
        else:
            print("Already have 4 corners. Press 'r' to reset.")

def select_corners(cap):
    global corners
    corners = []

    print("Warming up camera...")
    for _ in range(30):
        ret, frame = cap.read()

    if not ret:
        print("Error: Could not grab frame.")
        return False

    # Order: BL, BR, TR, TL
    LABELS = ["1: Bottom-Left", "2: Bottom-Right", "3: Top-Right", "4: Top-Left"]

    window = "Corner Selection"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)
    cv2.setMouseCallback(window, click_corner)

    print("\nClick the 4 corners in order:")
    print("  1) Bottom-left  2) Bottom-right  3) Top-right  4) Top-left")
    print("'r' = reset  |  Enter = confirm  |  'q' = quit\n")

    while True:
        display = frame.copy()

        # Small guide text in top-left
        if len(corners) < 4:
            cv2.putText(display, f"Click {LABELS[len(corners)]}",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        else:
            cv2.putText(display, "Enter to confirm  |  R to reset",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        cv2.putText(display, f"Corners: {len(corners)}/4   Arena: {ARENA_W_CM}x{ARENA_H_CM} cm",
                    (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Draw placed corners
        for i, (px, py) in enumerate(corners):
            cv2.circle(display, (px, py), 7, (0, 255, 0), -1)
            cv2.circle(display, (px, py), 8, (0, 0, 0), 1)
            cv2.putText(display, LABELS[i], (px + 10, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if len(corners) > 1:
            pts = np.array(corners, dtype=np.int32)
            cv2.polylines(display, [pts], isClosed=(len(corners) == 4),
                          color=(0, 255, 0), thickness=1)

        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):
            corners.clear()
            print("Corners reset.")
        elif key == 13 and len(corners) == 4:
            print(f"\nCorners confirmed: {corners}")
            break
        elif key == ord('q'):
            cv2.destroyWindow(window)
            return False

    cv2.destroyWindow(window)
    return True

def get_perspective_transform():
    # Corners clicked as: BL, BR, TR, TL
    src = np.array(corners, dtype=np.float32)
    # Adjusted destination mapping so Bottom-Left is exactly (0, 0)
    dst = np.array([
        [0,          0         ],   # Bottom-left  -> (0, 0)
        [ARENA_W_CM, 0         ],   # Bottom-right -> (167, 0)
        [ARENA_W_CM, ARENA_H_CM],   # Top-right    -> (167, 121.5)
        [0,          ARENA_H_CM],   # Top-left     -> (0, 121.5)
    ], dtype=np.float32)
    
    M     = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    return M, M_inv

def point_in_arena(x, y):
    if len(corners) < 4:
        return True
    poly = np.array(corners, dtype=np.int32)
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

def to_arena_coords(x, y, M):
    pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, M)
    ax, ay = transformed[0][0]
    ax = float(np.clip(ax, 0, ARENA_W_CM))
    ay = float(np.clip(ay, 0, ARENA_H_CM))
    return round(ax, 1), round(ay, 1)

def get_ball_positions(results, model, M):
    positions = []
    if results.masks is not None:
        for i, mask in enumerate(results.masks.xy):
            cx = int(np.mean(mask[:, 0]))
            cy = int(np.mean(mask[:, 1]))
            if not point_in_arena(cx, cy):
                continue
            cls   = int(results.boxes[i].cls[0].item())
            conf  = results.boxes[i].conf[0].item()
            label = model.names[cls]
            ax, ay = to_arena_coords(cx, cy, M)
            positions.append({
                "id": i, "label": label,
                "cx": cx, "cy": cy,
                "ax": ax, "ay": ay,
                "conf": conf
            })
    return positions

def draw_positions(vis_frame, positions):
    for ball in positions:
        cx, cy = ball["cx"], ball["cy"]
        cv2.drawMarker(vis_frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(vis_frame,
                    f"{ball['label']} ({ball['ax']}cm, {ball['ay']}cm)",
                    (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return vis_frame

def draw_arena_overlay(vis_frame):
    if len(corners) == 4:
        pts = np.array(corners, dtype=np.int32)
        cv2.polylines(vis_frame, [pts], isClosed=True, color=(0, 200, 255), thickness=2)
    return vis_frame

def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {CAMERA_INDEX}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not select_corners(cap):
        print("Corner selection cancelled.")
        cap.release()
        return

    M, M_inv = get_perspective_transform()
    print(f"Arena locked: {ARENA_W_CM} x {ARENA_H_CM} cm (Bottom-Left is 0x0)")
    
    window_name = "Live Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    print("\nReady. Press 's' to take a screenshot and get coordinates.")

    while True:
        # Keep reading frames so the camera buffer doesn't get stale
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Show a lightweight preview (No YOLO processing here to save resources)
        preview_frame = frame.copy()
        preview_frame = draw_arena_overlay(preview_frame)
        
        # Updated Instructions
        cv2.putText(preview_frame, "Press 's' to SCAN | 'r' to RE-DRAW | 'q' to QUIT", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow(window_name, preview_frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
            
        elif key == ord('r'):
            print("\n--- Re-drawing Corners ---")
            cv2.destroyWindow(window_name) # Hide preview temporarily
            
            if select_corners(cap):
                M, M_inv = get_perspective_transform()
                print(f"New arena locked: {ARENA_W_CM} x {ARENA_H_CM} cm")
                
                # Bring the preview window back
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 1280, 720)
            else:
                print("Corner selection cancelled. Quitting...")
                break
            
        elif key == ord('s'):
            print("\n--- Processing Screenshot ---")
            
            # 1. Create a black mask the same size as the frame
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            
            # 2. Draw a white polygon inside your 4 corners
            if len(corners) == 4:
                pts = np.array(corners, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
            
            # 3. Apply the mask to the frame (everything outside becomes pure black)
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

            # 4. Give the masked frame to YOLO instead of the raw frame
            results = model(masked_frame, verbose=False, conf=0.25)[0]
            positions = get_ball_positions(results, model, M)

            if positions:
                pos_str = "  |  ".join(
                    [f"{b['label']}: ({b['ax']}cm, {b['ay']}cm)" for b in positions]
                )
                print(f"Detected: {pos_str}")
            else:
                print("No detections found in this frame.")

            # Create the annotated screenshot (YOLO only plots what was inside the mask)
            vis_frame = results.plot()
            vis_frame = draw_arena_overlay(vis_frame)
            vis_frame = draw_positions(vis_frame, positions)
            
            # Save the image locally (optional, but handy)
            cv2.imwrite("latest_scan.jpg", vis_frame)

            # Display the result and pause until the user acknowledges
            result_window = "Screenshot Result (Press ANY KEY to resume preview)"
            cv2.namedWindow(result_window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(result_window, 1280, 720)
            cv2.imshow(result_window, vis_frame)
            
            # Wait indefinitely until any key is pressed
            cv2.waitKey(0) 
            cv2.destroyWindow(result_window)
            print("\nResuming preview. Ready for next scan.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()