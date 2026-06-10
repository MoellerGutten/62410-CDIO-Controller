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
from src.debug.log import get_logger

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

        if not self._load_calibration():
            raise RuntimeError(
                "No arena calibration found. "
                "Run `python3 -m src.state.arena_tracker` to calibrate, "
                "then try again."
            )

        self._M, self._M_inv          = self._compute_perspective()
        self._model                   = YOLO(self._cfg.model_path)
        self._cam_mtx, self._cam_dist = self._load_camera_calibration()
        self._cap                     = self._open_camera()

        for _ in range(5):          # flush stale frames from buffer
            self._cap.read()

        self._running = True
        get_logger().debug(
            f"[ArenaTracker] Ready — camera index {self._resolved_index}, "
            f"arena {self._cfg.width_cm} × {self._cfg.height_cm} cm"
        )

    def scan(self) -> ArenaState:
        if not self._running:
            raise RuntimeError("Call start() before scan().")
        with self._scan_lock:
            frame = self._grab_frame()
            return self._process_frame(frame)

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._running = False
        get_logger().info("[ArenaTracker] Stopped")

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
            get_logger().error(f"[ERROR] Cannot open camera (index {self._resolved_index}).")
            return

        os.makedirs(os.path.dirname(self._cfg.arena_config_file) or ".", exist_ok=True)

        if not self._load_calibration():
            if not self._setup_arena(cap):
                cap.release()
                return

        self._compute_perspective()
        cam_mtx, cam_dist = self._load_camera_calibration()
        get_logger().info(f"[ArenaTracker] Arena locked: {self._cfg.width_cm} × {self._cfg.height_cm} cm")

        win = "Live Preview"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, self._cfg.frame_width, self._cfg.frame_height)

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
                get_logger().info(f"-> Continuous: {'ON' if continuous else 'OFF'}")
            elif key == ord("r"):
                cv2.destroyWindow(win)
                if self._setup_arena(cap):
                    self._compute_perspective()
                    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(win, self._cfg.frame_width, self._cfg.frame_height)
                else:
                    break

            preview = frame.copy()

            if continuous or key == ord("s"):
                result = self._process_frame(frame, model=model)
                vis    = self._render_debug_frame(frame, result, model)

                if key == ord("s"):
                    get_logger().info(repr(result))
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
    #  Core processing                                                     #
    # ------------------------------------------------------------------ #

    def _grab_frame(self) -> np.ndarray:
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera.")
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

        for i in range(len(results.boxes)):
            cls   = int(results.boxes[i].cls[0].item())
            label = model.names[cls].lower()

            if "robot" in label:        # robot comes strictly from ArUco
                continue

            x1, y1, x2, y2 = results.boxes[i].xyxy[0].tolist()
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            ax, ay = self._to_cm(cx, cy)

            if not (0 <= ax <= self._cfg.width_cm and 0 <= ay <= self._cfg.height_cm):
                continue

            out.append({
                "label": label,
                "ax":    ax,
                "ay":    ay,
                "corners_cm": [
                    self._to_cm(x1, y1),
                    self._to_cm(x2, y1),
                    self._to_cm(x2, y2),
                    self._to_cm(x1, y2),
                ],
            })

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
            elif "cross" in lbl:
                corners_cm = d["corners_cm"]
                cx = sum(x for x, y in corners_cm) / 4
                cy = sum(y for x, y in corners_cm) / 4
                orientation = self._cross_orientation(corners_cm)
                state.cross = Cross(position=(cx, cy), orientation=orientation)

        return state

    # ------------------------------------------------------------------ #
    #  ArUco robot detection                                               #
    # ------------------------------------------------------------------ #
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

        return Robot(position=(round(center_x, 1), round(center_y, 1)), orientation=heading)

    # ------------------------------------------------------------------ #
    #  Geometry helpers                                                    #
    # ------------------------------------------------------------------ #

    def _to_cm(self, x: float, y: float) -> tuple[float, float]:
        pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
        tx, ty = cv2.perspectiveTransform(pt, self._M)[0][0]
        return round(float(tx), 1), round(float(ty), 1)

    @staticmethod
    def _cross_orientation(corners_cm: list[tuple[float, float]]) -> float:
        """
        Estimate cross orientation from its bounding-box corners.
        Returns angle in degrees, clamped to [0, 90) due to 4-fold symmetry.
        """
        (x1, y1), (x2, _), _, (_, y2) = corners_cm
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        return angle % 90

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
            self._corners    = [tuple(p) for p in data.get("corners",    [])]
            self._goal_a_pts = [tuple(p) for p in data.get("goal_a_pts", [])]
            self._goal_b_pts = [tuple(p) for p in data.get("goal_b_pts", [])]
            if len(self._corners) == 4:
                get_logger().debug("[ArenaTracker] Arena calibration loaded.")
                return True
        except Exception as exc:
            get_logger().warning(f"[WARNING] Could not load calibration: {exc}")
        return False

    def _save_calibration(self) -> None:
        with open(self._cfg.arena_config_file, "w") as f:
            json.dump(
                {
                    "corners":    self._corners,
                    "goal_a_pts": self._goal_a_pts,
                    "goal_b_pts": self._goal_b_pts,
                },
                f, indent=4,
            )
        get_logger().info("[ArenaTracker] Calibration saved.")

    def _load_camera_calibration(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if os.path.exists(self._cfg.camera_calib_file):
            with np.load(self._cfg.camera_calib_file) as data:
                get_logger().debug("[ArenaTracker] Lens calibration loaded.")
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
            get_logger().warning(
                f"[WARNING] Hjørner ser skæve ud! "
                f"Målt ratio: {measured:.2f}, forventet: {expected:.2f}"
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

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._cfg.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.frame_height)
        cap.set(cv2.CAP_PROP_FPS,          self._cfg.frame_fps)
        cap.set(cv2.CAP_PROP_FOURCC,       cv2.VideoWriter_fourcc(*"MJPG"))
        return cap

    # ------------------------------------------------------------------ #
    #  Interactive arena setup                                             #
    # ------------------------------------------------------------------ #

    def _setup_arena(self, cap: cv2.VideoCapture) -> bool:
        self._corners.clear()
        self._goal_a_pts.clear()
        self._goal_b_pts.clear()
        self._setup_step = "CORNERS"

        for _ in range(5):
            cap.read()

        win = "Arena Setup"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, self._cfg.frame_width, self._cfg.frame_height)
        cv2.setMouseCallback(win, self._handle_mouse)

        _STEPS = {
            "CORNERS": lambda: (f"1. Klik 4 hjørner (BL BR TR TL): {len(self._corners)}/4", (0, 255, 255)),
            "GOAL_A":  lambda: (f"2. Mål A — TL så BR: {len(self._goal_a_pts)}/2",           (255, 150, 0)),
            "GOAL_B":  lambda: (f"3. Mål B — TL så BR: {len(self._goal_b_pts)}/2",           (0, 0, 255)),
            "DONE":    lambda: ("Færdig! ENTER for at gemme | R for at nulstille",             (0, 255, 0)),
        }

        while True:
            ret, frame = cap.read()
            if not ret:
                continue
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
            self._draw_goal_on_frame(disp, self._goal_a_pts, (255, 150, 0), "Mål A")
            self._draw_goal_on_frame(disp, self._goal_b_pts, (0, 0, 255),   "Mål B")

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
        self._draw_goal_on_frame(frame, self._goal_a_pts, (255, 150, 0), "Mål A")
        self._draw_goal_on_frame(frame, self._goal_b_pts, (0, 0, 255),   "Mål B")
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
    ArenaTracker().run()