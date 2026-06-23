import cv2
import numpy as np
from src.state.arena_tracker import ArenaTracker

tracker = ArenaTracker()
cap = tracker._open_camera()
tracker._warm_up_camera(cap)
cam_mtx, cam_dist = tracker._load_camera_calibration()

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, params)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    if cam_mtx is not None:
        frame = cv2.undistort(frame, cam_mtx, cam_dist, None, cam_mtx)

    corners, ids, rejected = detector.detectMarkers(frame)

    vis = frame.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(vis, corners, ids, (0, 255, 0))
    for r in rejected:
        pts = r.reshape(-1, 2).astype(int)
        cv2.polylines(vis, [pts], True, (0, 0, 255), 1)

    perim = "n/a" if ids is None else f"{cv2.arcLength(corners[0][0].astype(np.float32), True):.0f}px"
    cv2.putText(vis, f"detected={ids is not None}  rejected_count={len(rejected)}  marker_perimeter={perim}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("ArUco Debug (q to quit)", vis)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()