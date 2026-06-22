from __future__ import annotations

import json
import math
import os
import platform
import threading as _threading
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from src.lib.detect_camera import find_camera_index
from src.model.ball import Ball
from src.model.cross import Cross
from src.model.robot import Robot
from src.model.arena_state import ArenaState
from src.state.arena_config import ArenaConfig
from src.debug.log import get_logger, setup_logger

class CameraReader:
    def __init__(self, cap: cv2.VideoCapture) -> None:
        self._cap     = cap
        self._frame:  Optional[np.ndarray] = None
        self._lock    = _threading.Lock()
        self._running = True
        self._ready = _threading.Event()
        self._thread  = _threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)
        if self._frame is None:
            raise RuntimeError("camera thread failed to deliver a frame within 5 seconds")

    def _capture_loop(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
                self._ready.set()

    def get_latest(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        self._cap.release()

class ArenaTracker:
    """
    Singleton camera + YOLO + ArUco tracker.
    scan() returns an ArenaState with all positions in cm.
    """

    _instance:  Optional["ArenaTracker"] = None
    _init_lock: _threading.Lock          = _threading.Lock()

    # ------------------------------------------------------------------ #
    #  Singleton boilerplate                                               #
    # ------------------------------------------------------------------ #

    def __new__(cls, **kwargs):
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialised = False
        return cls._instance

    def __init__(self, config: ArenaConfig = ArenaConfig()) -> None:
        if self._initialised:
            return
        self._initialised = True

        self._cfg = config

        self._corners:    list[tuple[int, int]] = []
        self._goal_a_pts: list[tuple[int, int]] = []
        self._goal_b_pts: list[tuple[int, int]] = []
        self._setup_step: str = "CORNERS"

        self._model:          Optional[YOLO]             = None
        self._cap:            Optional[cv2.VideoCapture] = None
        self._M:              Optional[np.ndarray]       = None
        self._M_inv:          Optional[np.ndarray]       = None
        self._cam_mtx:        Optional[np.ndarray]       = None
        self._cam_dist:       Optional[np.ndarray]       = None
        self._running:        bool                       = False
        self._resolved_index: int                        = 0
        self._scan_lock:      _threading.Lock            = _threading.Lock()

        self.logger = get_logger("ArenaTracker")

        try:
            self._aruco_dict     = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            self._aruco_params   = cv2.aruco.DetectorParameters()
            self._aruco_detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._aruco_params)
        except AttributeError:
            self._aruco_dict     = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            self._aruco_params   = cv2.aruco.DetectorParameters()
            self._aruco_detector = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if self._running:
            return

        os.makedirs(os.path.dirname(self._cfg.arena_config_file) or ".", exist_ok=True)

        cap = self._open_camera()
        self._frame_shape = self._warm_up_camera(cap)

        if not self._load_calibration():
            raise RuntimeError(
                "No arena calibration found. "
                "Run `python3 -m src.state.arena_tracker` to calibrate, "
                "then try again."
            )

        self._M, self._M_inv          = self._compute_perspective()
        self._model                   = YOLO(self._cfg.model_path)
        self._cam_mtx, self._cam_dist = self._load_camera_calibration()
        self._camera_reader = CameraReader(cap)

        self._running = True
        self.logger.debug(
            f"Ready — camera index {self._resolved_index}, "
            f"arena {self._cfg.width_cm} × {self._cfg.height_cm} cm"
        )

    def scan(self) -> ArenaState:
        if not self._running:
            raise RuntimeError("Call start() before scan().")
        with self._scan_lock:
            frame = self._grab_frame()
            return self._process_frame(frame)

    def stop(self) -> None:
        if hasattr(self, "_camera_reader"):
            self._camera_reader.stop()
            self._camera_reader = None
        self._cap = None   # cap ejes nu af CameraReader
        self._running = False
        self.logger.info("Stopped")

    def __enter__(self) -> "ArenaTracker":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------ #
    #  Interactive calibration mode  (run once from terminal)             #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        model = YOLO(self._cfg.model_path)
        cap   = self._open_camera()
        if not cap.isOpened():
            self.logger.error(f"Cannot open camera (index {self._resolved_index}).")
            return
        
        self._frame_shape = self._warm_up_camera(cap)

        os.makedirs(os.path.dirname(self._cfg.arena_config_file) or ".", exist_ok=True)

        if not self._load_calibration():
            if not self._setup_arena(cap):
                cap.release()
                return

        self._compute_perspective()
        cam_mtx, cam_dist = self._load_camera_calibration()
        self.logger.info(f"Arena locked: {self._cfg.width_cm} × {self._cfg.height_cm} cm")

        win = "Live Preview"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        h, w = self._frame_shape
        cv2.resizeWindow(win, w, h)
        #cv2.resizeWindow(win, self._cfg.frame_width, self._cfg.frame_height)

        show_visuals = False
        continuous   = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if cam_mtx is not None:
                frame = cv2.undistort(frame, cam_mtx, cam_dist, None, cam_mtx)

            key = cv2.waitKey(1) & 0xFF
            if   key == ord("q"):
                break
            elif key == ord("v"):
                show_visuals = not show_visuals
            elif key == ord("c"):
                continuous = not continuous
                self.logger.info(f"-> Continuous: {'ON' if continuous else 'OFF'}")
            elif key == ord("r"):
                cv2.destroyWindow(win)
                if self._setup_arena(cap):
                    self._compute_perspective()
                    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                    h, w = self._frame_shape
                    cv2.resizeWindow(win, w, h)
                    # cv2.resizeWindow(win, self._cfg.frame_width, self._cfg.frame_height)
                else:
                    break

            preview = frame.copy()

            if continuous or key == ord("s"):
                result = self._process_frame(frame, model=model)
                vis    = self._render_debug_frame(frame, result, model)

                if key == ord("s"):
                    self.logger.info(repr(result))
                    cv2.imwrite("image_recon/latest_scan.jpg", vis)
                    if show_visuals:
                        w2 = "Result — any key to resume"
                        cv2.namedWindow(w2, cv2.WINDOW_NORMAL)
                        cv2.imshow(w2, vis)
                        cv2.waitKey(0)
                        cv2.destroyWindow(w2)

                if continuous:
                    preview = vis
            else:
                preview = self._draw_arena_overlay(preview, self._M_inv)

            self._draw_text_with_outline(
                preview,
                f"s:SCAN  c:CONT[{'ON' if continuous else 'OFF'}]  "
                f"v:VIS[{'ON' if show_visuals else 'OFF'}]  r:RESET  q:QUIT",
                (20, 40), 0.65, (0, 255, 0), 2,
            )
            cv2.imshow(win, preview)

        cap.release()
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------ #
    #  Core processing                                                   #
    # ------------------------------------------------------------------ #
    
    def _grab_frame(self) -> np.ndarray:
        frame = self._camera_reader.get_latest()
        if frame is None:
            raise RuntimeError("No frame available yet from camera thread.")
        if self._cam_mtx is not None:
            frame = cv2.undistort(frame, self._cam_mtx, self._cam_dist, None, self._cam_mtx)
        return frame

    def _process_frame(
        self,
        frame: np.ndarray,
        *,
        model: Optional[YOLO] = None,
    ) -> ArenaState:
        model = model or self._model

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if len(self._corners) == 4:
            cv2.fillPoly(mask, [np.array(self._corners, dtype=np.int32)], 255)
        masked = cv2.bitwise_and(frame, frame, mask=mask)

        results    = model(masked, verbose=False, conf=self._cfg.detection_conf)[0]
        detections = self._parse_detections(results, model)
        return self._build_state(detections, frame)
    

    def _parse_detections(self, results, model: YOLO) -> list[dict]:
        out = []
        if results.boxes is None:
            return out

        has_keypoints = results.keypoints is not None

        for i in range(len(results.boxes)):
            cls   = int(results.boxes[i].cls[0].item())
            label = model.names[cls].lower()

            if "robot" in label:
                continue

            x1, y1, x2, y2 = results.boxes[i].xyxy[0].tolist()
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            ax, ay = self._to_cm(cx, cy)

            if not (0 <= ax <= self._cfg.width_cm and 0 <= ay <= self._cfg.height_cm):
                continue

            d = {
                "label": label,
                "ax":    ax,
                "ay":    ay,
                "corners_cm": [
                    self._to_cm(x1, y1),
                    self._to_cm(x2, y1),
                    self._to_cm(x2, y2),
                    self._to_cm(x1, y2),
                ],
            }

            # Extract keypoints for cross orientation
            if has_keypoints:
                kpts = results.keypoints[i].xy[0]  # shape [4, 2]
                if len(kpts) == 4:
                    d["keypoints_cm"] = [
                        self._to_cm(float(kp[0]), float(kp[1]))
                        for kp in kpts
                    ]

            out.append(d)

        return out


    def _build_state(self, detections: list[dict], frame: np.ndarray) -> ArenaState:
        state = ArenaState()

        # ---- Robot (ArUco only) ----------------------------------------
        state.robot = self._get_aruco_robot(frame)

        # ---- Balls + Cross (YOLO only) ---------------------------------
        for d in detections:
            lbl = d["label"]
            if "ball" in lbl:
                state.balls.append(Ball(
                    position=(d["ax"], d["ay"]),
                    is_vip=("orange" in lbl or "vip" in lbl or lbl == "oball"),
                ))
            elif "x" in lbl:
                corners_cm = d["corners_cm"]
                cx = sum(x for x, y in corners_cm) / 4
                cy = sum(y for x, y in corners_cm) / 4

                if "keypoints_cm" in d:
                    orientation = self._cross_orientation(d["keypoints_cm"])
                else:
                    orientation = 0.0 

                state.cross = Cross(position=(cx, cy), orientation=orientation, bounding_box=corners_cm)

        return state

    # ------------------------------------------------------------------ #
    #  ArUco robot detection                                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _correct_for_height(
            measured_pos_cm: tuple[float, float],
            camera_pos_cm: tuple[int, int],
            camera_height: int,
            aruco_height: int,
    ) -> tuple[float, float]:
        """
        Korrigerer for parallax-fejl når ArUco-markøren sidder hævet over banen.

        measured_pos_cm : (x, y) – rå position fra homografien
        camera_pos_cm   : (cx, cy) – kameraets nadir-punkt i bane-koordinater (cm)
        camera_pos_cm   : (cx, cy) – kameraets nadir-punkt i bane-koordinater (cm)
        camera_height   : kameraets højde over banen (cm)
        aruco_height    : ArUco-markørens højde over banen (cm)
        """
        cx, cy = camera_pos_cm
        mx, my = measured_pos_cm
        scale = (camera_height - aruco_height) / camera_height

        return (
            cx + (mx - cx) * scale,
            cy + (my - cy) * scale,
        )

    def _get_aruco_robot(self, frame: np.ndarray) -> Optional[Robot]:
        if self._aruco_detector:
            corners, ids, _ = self._aruco_detector.detectMarkers(frame)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                frame, self._aruco_dict, parameters=self._aruco_params
            )

        if ids is None:
            return None

        ids_flat = ids.flatten()
        if self._cfg.aruco_target_id not in ids_flat:
            return None

        idx = np.where(ids_flat == self._cfg.aruco_target_id)[0][0]
        marker_px = corners[idx][0]

        tl, tr, br, bl = [self._to_cm(px, py) for px, py in marker_px]

        center_x = (tl[0] + tr[0] + br[0] + bl[0]) / 4.0
        center_y = (tl[1] + tr[1] + br[1] + bl[1]) / 4.0
        front_mid_x = (tl[0] + tr[0]) / 2.0
        front_mid_y = (tl[1] + tr[1]) / 2.0

        heading = round(math.degrees(math.atan2(front_mid_y - center_y, front_mid_x - center_x)), 1)

        corrected_x, corrected_y = self._correct_for_height(
            measured_pos_cm=(center_x, center_y),
            camera_pos_cm=(self._cfg.camera_x, self._cfg.camera_y),
            camera_height=self._cfg.camera_height,
            aruco_height=self._cfg.aruco_height,
        )

        # Offset the robot center of the robot compared to the center of the ArUco marker.
        # Offset values can be found in arena_config.py
        rad = math.radians(heading)
        robot_center_x = corrected_x + self._cfg.aruco_offset_x * math.cos(rad) - self._cfg.aruco_offset_y * math.sin(rad)
        robot_center_y = corrected_y + self._cfg.aruco_offset_x * math.sin(rad) + self._cfg.aruco_offset_y * math.cos(rad)


        return Robot(position=(round(robot_center_x, 1), round(robot_center_y, 1)), orientation=heading)

    # ------------------------------------------------------------------ #
    #  Geometry helpers                                                    #
    # ------------------------------------------------------------------ #

    def _to_cm(self, x: float, y: float) -> tuple[float, float]:
        pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
        tx, ty = cv2.perspectiveTransform(pt, self._M)[0][0]
        return round(float(tx), 1), round(float(ty), 1)

    @staticmethod
    def _cross_orientation(keypoints_cm: list[tuple[float, float]]) -> float:
        """
        Estimate cross orientation from its 4 bounding-box corner keypoints.
        Uses the diagonal vectors to find the box tilt.
        Returns angle in degrees relative to upright.
        """
        tl, tr, bl, br = keypoints_cm  # order from your output

        # Two diagonals of the bounding box
        dx1 = br[0] - tl[0]
        dy1 = br[1] - tl[1]
        dx2 = bl[0] - tr[0]
        dy2 = bl[1] - tr[1]

        angle1 = math.degrees(math.atan2(dy1, dx1)) + 45 % 90
        angle2 = math.degrees(math.atan2(dy2, dx2)) + 45 % 90

        return (angle1 + angle2) / 2

    def _compute_perspective(self) -> tuple[np.ndarray, np.ndarray]:
        src = np.array(self._corners, dtype=np.float32)
        dst = np.array([
            [0,                    0                   ],
            [self._cfg.width_cm,   0                   ],
            [self._cfg.width_cm,   self._cfg.height_cm ],
            [0,                    self._cfg.height_cm ],
        ], dtype=np.float32)
        self._M     = cv2.getPerspectiveTransform(src, dst)
        self._M_inv = cv2.getPerspectiveTransform(dst, src)
        return self._M, self._M_inv

    # ------------------------------------------------------------------ #
    #  Calibration — load / save / interactive setup                      #
    # ------------------------------------------------------------------ #

    def _load_calibration(self) -> bool:
        if not os.path.exists(self._cfg.arena_config_file):
            return False
        try:
            with open(self._cfg.arena_config_file) as f:
                data = json.load(f)
            corners     = [tuple(p) for p in data.get("corners",    [])]
            goal_a_pts  = [tuple(p) for p in data.get("goal_a_pts", [])]
            goal_b_pts  = [tuple(p) for p in data.get("goal_b_pts", [])]
            saved_shape = tuple(data.get("frame_shape", []))  # (h, w)

            if len(corners) != 4:
                return False

            current_shape = getattr(self, "_frame_shape", None)
            if saved_shape and current_shape and tuple(saved_shape) != tuple(current_shape):
                sh, sw = saved_shape
                ch, cw = current_shape
                scale_x = cw / sw
                scale_y = ch / sh
                self.logger.warning(
                    f"Calibration was saved at {sw}x{sh} but camera now delivers "
                    f"{cw}x{ch}. Rescaling stored points (scale {scale_x:.3f}x / {scale_y:.3f}x)."
                )
                def _rescale(pts):
                    return [(round(px * scale_x), round(py * scale_y)) for px, py in pts]
                corners    = _rescale(corners)
                goal_a_pts = _rescale(goal_a_pts)
                goal_b_pts = _rescale(goal_b_pts)

            self._corners    = corners
            self._goal_a_pts = goal_a_pts
            self._goal_b_pts = goal_b_pts
            self.logger.debug("Arena calibration loaded.")
            return True
        except Exception as exc:
            self.logger.warning(f"Could not load calibration: {exc}")
        return False

    def _save_calibration(self) -> None:
        with open(self._cfg.arena_config_file, "w") as f:
            json.dump(
                {
                    "corners":    self._corners,
                    "goal_a_pts": self._goal_a_pts,
                    "goal_b_pts": self._goal_b_pts,
                    "frame_shape": list(getattr(self, "_frame_shape", [])),
                    
                },
                f, indent=4,
            )
        self.logger.info("Calibration saved.")

    def _load_camera_calibration(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if os.path.exists(self._cfg.camera_calib_file):
            with np.load(self._cfg.camera_calib_file) as data:
                self.logger.debug("Lens calibration loaded.")
                return data["mtx"], data["dist"]
        return None, None

    def _validate_corners(self) -> bool:
        """Warn if the clicked corners have a suspiciously wrong aspect ratio."""
        if len(self._corners) != 4:
            return False
        tl, tr, br, bl = self._corners
        width  = (math.dist(tl, tr) + math.dist(bl, br)) / 2
        height = (math.dist(tl, bl) + math.dist(tr, br)) / 2
        measured = width / height
        expected = self._cfg.width_cm / self._cfg.height_cm
        if abs(measured - expected) > 0.15:
            self.logger.warning(
                f"The corners look crooked!"
                f"Measured ratio: {measured:.2f}, expected: {expected:.2f}"
            )
            return False
        return True

    # ------------------------------------------------------------------ #
    #  Camera                                                              #
    # ------------------------------------------------------------------ #

    def _open_camera(self) -> cv2.VideoCapture:
        system = platform.system()
        backend = {
            "Linux":   cv2.CAP_V4L2,
            "Darwin":  cv2.CAP_AVFOUNDATION,
            "Windows": cv2.CAP_DSHOW,
        }.get(system, cv2.CAP_ANY)

        idx = find_camera_index() if self._cfg.camera_index == -1 else self._cfg.camera_index
        self._resolved_index = idx

        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {idx} on {system}.")

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._cfg.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.frame_height)
        cap.set(cv2.CAP_PROP_FPS,          self._cfg.frame_fps)
        cap.set(cv2.CAP_PROP_FOURCC,       cv2.VideoWriter_fourcc(*"MJPG"))
        return cap
    
    @staticmethod
    def _warm_up_camera(cap: cv2.VideoCapture, max_frames: int = 30) -> tuple[int, int]:
        """Reads frames until the camera's actual resolution stabilizes."""
        last_shape = None
        shape = None
        for _ in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                continue
            shape = frame.shape[:2]
            if shape == last_shape:
                break
            last_shape = shape
        if shape is None:
            raise RuntimeError("Camera never produced a frame during warm-up.")
        return shape

    # ------------------------------------------------------------------ #
    #  Interactive arena setup                                             #
    # ------------------------------------------------------------------ #

    def _setup_arena(self, cap: cv2.VideoCapture) -> bool:
        self._corners.clear()
        self._goal_a_pts.clear()
        self._goal_b_pts.clear()
        self._setup_step = "CORNERS"

        self._frame_shape = self._warm_up_camera(cap)
        cam_mtx, cam_dist = self._load_camera_calibration()

        win = "Arena Setup"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        h, w = self._frame_shape
        cv2.resizeWindow(win, w, h)
        # cv2.resizeWindow(win, self._cfg.frame_width, self._cfg.frame_height)
        cv2.setMouseCallback(win, self._handle_mouse)

        _STEPS = {
            "CORNERS": lambda: (f"1. Click 4 corners (BL BR TR TL): {len(self._corners)}/4", (0, 255, 255)),
            "GOAL_A":  lambda: (f"2. Goal A — TL then BR: {len(self._goal_a_pts)}/2",           (255, 150, 0)),
            "GOAL_B":  lambda: (f"3. Goal B — TL then BR: {len(self._goal_b_pts)}/2",           (0, 0, 255)),
            "DONE":    lambda: ("Done! ENTER to save | R to restart",             (0, 255, 0)),
        }

        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            if cam_mtx is not None:                                          # <-- add this
                frame = cv2.undistort(frame, cam_mtx, cam_dist, None, cam_mtx)
            disp = frame.copy()
            msg, color = _STEPS[self._setup_step]()
            self._draw_text_with_outline(disp, msg, (20, 40), 0.7, color, 2)

            for px, py in self._corners:
                cv2.circle(disp, (px, py), 7, (0, 255, 0), -1)
            if len(self._corners) > 1:
                cv2.polylines(
                    disp, [np.array(self._corners, dtype=np.int32)],
                    isClosed=(len(self._corners) == 4), color=(0, 255, 0), thickness=2,
                )
            self._draw_goal_on_frame(disp, self._goal_a_pts, (255, 150, 0), "Goal A")
            self._draw_goal_on_frame(disp, self._goal_b_pts, (0, 0, 255),   "Goal B")

            cv2.imshow(win, disp)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("r"):
                self._corners.clear()
                self._goal_a_pts.clear()
                self._goal_b_pts.clear()
                self._setup_step = "CORNERS"
            elif key == 13 and self._setup_step == "DONE":
                self._validate_corners()
                self._save_calibration()
                break
            elif key == ord("q"):
                cv2.destroyWindow(win)
                return False

        cv2.destroyWindow(win)
        return True

    def _handle_mouse(self, event, x: int, y: int, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self._setup_step == "CORNERS" and len(self._corners) < 4:
            self._corners.append((x, y))
            if len(self._corners) == 4:
                self._setup_step = "GOAL_A"
        elif self._setup_step == "GOAL_A" and len(self._goal_a_pts) < 2:
            self._goal_a_pts.append((x, y))
            if len(self._goal_a_pts) == 2:
                self._setup_step = "GOAL_B"
        elif self._setup_step == "GOAL_B" and len(self._goal_b_pts) < 2:
            self._goal_b_pts.append((x, y))
            if len(self._goal_b_pts) == 2:
                self._setup_step = "DONE"

    # ------------------------------------------------------------------ #
    #  Debug rendering                                                     #
    # ------------------------------------------------------------------ #

    def _render_debug_frame(
        self,
        frame: np.ndarray,
        state: ArenaState,
        model: YOLO,
    ) -> np.ndarray:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if len(self._corners) == 4:
            cv2.fillPoly(mask, [np.array(self._corners, dtype=np.int32)], 255)
        yolo_res = model(
            cv2.bitwise_and(frame, frame, mask=mask),
            verbose=False,
            conf=self._cfg.detection_conf,
        )[0]
        vis = yolo_res.plot()
        vis = self._draw_arena_overlay(vis, self._M_inv)

        # Draw robot heading arrow if available
        if state.robot is not None:
            def cm_to_px(pos: tuple[float, float]) -> tuple[int, int]:
                pt = np.array([[[pos[0], pos[1]]]], dtype=np.float32)
                ox, oy = cv2.perspectiveTransform(pt, self._M_inv)[0][0]
                return int(ox), int(oy)

            hv   = state.robot.heading_vector()
            base = state.robot.position
            tip  = (base[0] + hv[0] * 15, base[1] + hv[1] * 15)
            cv2.arrowedLine(vis, cm_to_px(base), cm_to_px(tip), (0, 255, 255), 3, tipLength=0.3)

        return vis

    def _draw_arena_overlay(self, frame: np.ndarray, _M_inv: np.ndarray) -> np.ndarray:
        if len(self._corners) == 4:
            cv2.polylines(
                frame, [np.array(self._corners, dtype=np.int32)],
                isClosed=True, color=(0, 200, 255), thickness=3,
            )
        self._draw_goal_on_frame(frame, self._goal_a_pts, (255, 150, 0), "Goal A")
        self._draw_goal_on_frame(frame, self._goal_b_pts, (0, 0, 255),   "Goal B")
        return frame

    def _draw_goal_on_frame(
        self,
        frame: np.ndarray,
        pts:   list[tuple[int, int]],
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        if len(pts) >= 1:
            cv2.circle(frame, pts[0], 5, color, -1)
        if len(pts) == 2:
            cv2.rectangle(frame, pts[0], pts[1], color, 2)
            self._draw_text_with_outline(frame, label, (pts[0][0], pts[0][1] - 10), 0.5, color, 2)

    @staticmethod
    def _draw_text_with_outline(
        img:        np.ndarray,
        text:       str,
        pos:        tuple[int, int],
        font_scale: float,
        color:      tuple[int, int, int],
        thickness:  int,
    ) -> None:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 3)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,     thickness)


if __name__ == "__main__":
    setup_logger()
    ArenaTracker().run()
