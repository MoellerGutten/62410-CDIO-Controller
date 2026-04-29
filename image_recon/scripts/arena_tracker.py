"""
arena_tracker.py
================
Headless-first arena tracker.  The tracker owns its camera and model;
callers never touch frames or transform matrices.

ArenaTracker is a **singleton**: every call to ``ArenaTracker()`` returns
the same object.  This prevents multiple threads / modules from trying to
open the same camera simultaneously (which causes errno=16 on Linux V4L2).

Typical third-party usage
--------------------------
    from arena_tracker import ArenaTracker

    tracker = ArenaTracker()
    tracker.start()                 # opens camera, loads calibration, warms up

    data = tracker.scan()           # returns a ScanResult – no GUI, no args
    print(data.robot.heading)
    print(data.to_json())           # serialise to JSON string

    tracker.stop()                  # releases camera

Context-manager usage
---------------------
    with ArenaTracker() as tracker:
        result = tracker.scan()

Interactive / debug usage
--------------------------
    ArenaTracker().run()            # full OpenCV window with keyboard controls
                                    # also used to create the calibration file
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


# --------------------------------------------------------------------------- #
#  Camera auto-detection                                                       #
# --------------------------------------------------------------------------- #

# USB vendor:product IDs for the Logitech C930e
_LOGITECH_C930E_VID = "046d"
_LOGITECH_C930E_PID = "0843"

# Fallback: any Logitech webcam (vendor ID only)
_LOGITECH_VID = "046d"

# Keywords used as a last resort when USB IDs cannot be read
_LOGITECH_KEYWORDS = ("c930", "logitech", "logi")


def find_logitech_c930e(max_index: int = 10) -> int:
    """
    Return the OpenCV camera index for the Logitech C930e (or the best
    available Logitech webcam) on Linux, macOS, and Windows.

    Detection strategy (in priority order):
      1. Platform-native USB enumeration to match vid:pid exactly.
      2. Scan indices 0..max_index-1 and check the backend's reported name.
      3. Return 0 as a last resort (original behaviour).

    Raises RuntimeError if a platform-specific tool fails unexpectedly
    (missing executable, etc.); that error is caught and we fall through
    to the next strategy.
    """
    system = platform.system()

    # --- 1. Platform-native USB enumeration ---
    try:
        index = _usb_enumerate(system)
        if index is not None:
            print(f"[CameraDetect] Found Logitech C930e via USB enumeration → index {index}")
            return index
    except Exception as exc:
        print(f"[CameraDetect] USB enumeration skipped ({exc})")

    # --- 2. Scan OpenCV backends for a matching name ---
    index = _scan_by_name(max_index)
    if index is not None:
        print(f"[CameraDetect] Found Logitech camera by name scan → index {index}")
        return index

    # --- 3. Fallback ---
    print("[CameraDetect] Could not identify C930e; falling back to index 0.")
    return 0


# ── Linux ──────────────────────────────────────────────────────────────────

def _linux_find_camera() -> Optional[int]:
    """
    Walk /sys/class/video4linux and match by idVendor/idProduct.
    Returns the lowest-numbered /dev/videoN index whose USB device matches
    the C930e vid:pid (or any Logitech vid if the exact pid is absent).
    """
    v4l_root = "/sys/class/video4linux"
    if not os.path.isdir(v4l_root):
        return None

    exact_match: list[int]    = []
    vendor_match: list[int]   = []

    for node in sorted(os.listdir(v4l_root)):
        # node is e.g. "video0", "video1", …
        match = re.match(r"video(\d+)$", node)
        if not match:
            continue
        cam_idx = int(match.group(1))

        # Walk up the sysfs tree to find the USB device attributes
        device_path = os.path.realpath(os.path.join(v4l_root, node, "device"))
        vid, pid = _sysfs_vid_pid(device_path)

        if vid == _LOGITECH_C930E_VID and pid == _LOGITECH_C930E_PID:
            exact_match.append(cam_idx)
        elif vid == _LOGITECH_VID:
            vendor_match.append(cam_idx)

    if exact_match:
        return exact_match[0]
    if vendor_match:
        return vendor_match[0]
    return None


def _sysfs_vid_pid(device_path: str) -> tuple[str, str]:
    """Climb sysfs parents until we find idVendor / idProduct files."""
    path = device_path
    for _ in range(6):                    # max depth to climb
        vid_file = os.path.join(path, "idVendor")
        pid_file = os.path.join(path, "idProduct")
        if os.path.isfile(vid_file) and os.path.isfile(pid_file):
            vid = open(vid_file).read().strip().lower()
            pid = open(pid_file).read().strip().lower()
            return vid, pid
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return "", ""


# ── macOS ──────────────────────────────────────────────────────────────────

def _macos_find_camera() -> Optional[int]:
    """
    Use `system_profiler SPCameraDataType` to list cameras and their
    USB IDs, then match to an OpenCV index by probing in order.

    system_profiler output (macOS 13+) looks like:
        FaceTime HD Camera:
          Vendor ID: 0x046d
          Product ID: 0x0843
          ...
    """
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPCameraDataType"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode(errors="replace")
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    # Collect camera names that match C930e
    matching_names: list[str] = []
    current_name  = ""
    current_vid   = ""
    current_pid   = ""

    for line in out.splitlines():
        line = line.strip()
        # A camera entry starts with a name followed by a colon at column 0
        if line.endswith(":") and not line.startswith(" ") and ":" not in line[:-1]:
            # save previous
            if current_name and _is_logitech(current_vid, current_pid, current_name):
                matching_names.append(current_name)
            current_name = line[:-1].strip()
            current_vid  = current_pid = ""
        elif "Vendor ID" in line or "vendorID" in line.lower():
            m = re.search(r"0x([0-9a-fA-F]+)", line)
            if m:
                current_vid = m.group(1).lower()
        elif "Product ID" in line or "productID" in line.lower():
            m = re.search(r"0x([0-9a-fA-F]+)", line)
            if m:
                current_pid = m.group(1).lower()

    if current_name and _is_logitech(current_vid, current_pid, current_name):
        matching_names.append(current_name)

    if not matching_names:
        return None

    # Now map names → OpenCV indices by opening cameras and reading the name
    # (AVFoundation backend exposes the name via CAP_PROP_BACKEND_NAME on
    # recent OpenCV builds; fall back to positional matching if not available)
    target = matching_names[0].lower()
    for idx in range(10):
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            continue
        # Try to get the backend name (OpenCV 4.5.3+)
        backend_name = ""
        try:
            backend_name = cap.getBackendName().lower()
        except AttributeError:
            pass
        cap.release()

        # If AVFoundation is available we can try cv2.videoio_registry
        # but the safest cross-version approach is just positional order:
        # system_profiler lists cameras in the same order as AVFoundation.
        # Return the index equal to the position of our match in the list.
        return matching_names.index(matching_names[0])   # index within OS list

    return None


# ── Windows ────────────────────────────────────────────────────────────────

def _windows_find_camera() -> Optional[int]:
    """
    Query WMI (via PowerShell) to find the PnP device index for the C930e.

    We list Win32_PnPEntity items whose DeviceID contains the vid:pid,
    then map to a DirectShow/MSMF camera index by scanning OpenCV.
    """
    ps_script = (
        "Get-WmiObject Win32_PnPEntity | "
        "Where-Object { $_.DeviceID -match 'VID_046D' } | "
        "Select-Object -ExpandProperty Name"
    )
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps_script],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode(errors="replace")
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    names = [l.strip().lower() for l in out.splitlines() if l.strip()]
    if not names:
        return None

    # Match camera names → OpenCV index by scanning
    for idx in range(10):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        cap.release()
        # If any WMI name looks like our camera, assign this index.
        # We scan in order, so the first open index that matches is it.
        for name in names:
            if any(k in name for k in _LOGITECH_KEYWORDS):
                return idx

    return None


# ── Shared helpers ─────────────────────────────────────────────────────────

def _usb_enumerate(system: str) -> Optional[int]:
    if system == "Linux":
        return _linux_find_camera()
    elif system == "Darwin":
        return _macos_find_camera()
    elif system == "Windows":
        return _windows_find_camera()
    return None


def _scan_by_name(max_index: int) -> Optional[int]:
    """
    Open each camera index and check its name (where supported by the
    OpenCV backend) for Logitech keywords.  Falls back to returning the
    first camera that successfully opens if a keyword match isn't possible.
    """
    system = platform.system()
    backend = {
        "Linux":   cv2.CAP_V4L2,
        "Darwin":  cv2.CAP_AVFOUNDATION,
        "Windows": cv2.CAP_DSHOW,
    }.get(system, cv2.CAP_ANY)

    first_open: Optional[int] = None

    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            continue

        if first_open is None:
            first_open = idx

        # Try getBackendName / description where available
        name = ""
        try:
            name = cap.getBackendName().lower()
        except AttributeError:
            pass

        cap.release()

        if any(k in name for k in _LOGITECH_KEYWORDS):
            return idx

    return None     # caller falls back to index 0


def _is_logitech(vid: str, pid: str, name: str) -> bool:
    if vid == _LOGITECH_C930E_VID:
        return True
    if vid == _LOGITECH_VID:
        return True
    name_lower = name.lower()
    return any(k in name_lower for k in _LOGITECH_KEYWORDS)


# --------------------------------------------------------------------------- #
#  Result data-classes returned by scan()                                      #
# --------------------------------------------------------------------------- #

@dataclass
class Point:
    x: float
    y: float


@dataclass
class BallData:
    label: str
    position: Point


@dataclass
class RobotData:
    label: str
    position: Point
    heading: Optional[float]   # degrees; None if undeterminable
    keypoints: list[Point]
    corners: list[Point]       # [TL, TR, BR, BL] in arena-cm


@dataclass
class CrossData:
    label: str
    corners: list[Point]       # [TL, TR, BR, BL] in arena-cm


@dataclass
class ScanResult:
    """
    Everything the third party needs.
    All coordinates are centimetres relative to the arena's bottom-left (0, 0).
    """
    arena_width_cm: float
    arena_height_cm: float
    goal_a: list[Point]        # [top-left, bottom-right] corners in cm
    goal_b: list[Point]
    robot: Optional[RobotData]
    cross: Optional[CrossData]
    balls: list[BallData]

    # -- Serialisation helpers --

    def to_dict(self) -> dict:
        def pt(p: Point) -> dict:
            return {"x": p.x, "y": p.y}

        return {
            "arena": {
                "width_cm":  self.arena_width_cm,
                "height_cm": self.arena_height_cm,
            },
            "goals": {
                "A": [pt(p) for p in self.goal_a],
                "B": [pt(p) for p in self.goal_b],
            },
            "robot": (
                {
                    "label":     self.robot.label,
                    "x":         self.robot.position.x,
                    "y":         self.robot.position.y,
                    "heading":   self.robot.heading,
                    "keypoints": [pt(k) for k in self.robot.keypoints],
                    "corners":   [pt(c) for c in self.robot.corners],
                }
                if self.robot else {}
            ),
            "cross": (
                {
                    "label":   self.cross.label,
                    "corners": [pt(c) for c in self.cross.corners],
                }
                if self.cross else {}
            ),
            "balls": [
                {"label": b.label, "x": b.position.x, "y": b.position.y}
                for b in self.balls
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# --------------------------------------------------------------------------- #
#  ArenaTracker                                                                #
# --------------------------------------------------------------------------- #

# Sentinel: pass camera_index=AUTO_DETECT to trigger auto-detection
AUTO_DETECT = -1

import threading as _threading


class ArenaTracker:
    """
    Owns the camera, YOLO model, and all calibration state.

    Singleton: only one instance (and one camera handle) ever exists.
    Calling ``ArenaTracker()`` a second time returns the same object, so
    two threads or two modules can share the tracker without fighting over
    /dev/videoN (which causes errno=16 "Device or resource busy").

    Pass ``camera_index=AUTO_DETECT`` (the default) to let the tracker
    automatically locate the Logitech C930e on Linux, macOS, or Windows.
    Pass an explicit integer to pin a specific /dev/videoN or DirectShow index.
    """

    # --- Singleton bookkeeping (class-level) ---
    _instance:  "Optional[ArenaTracker]" = None
    _init_lock: "_threading.Lock"        = _threading.Lock()

    def __new__(cls, **kwargs):
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialised = False
            else:
                print("[ArenaTracker] Reusing existing singleton instance.")
        return cls._instance

    def __init__(
        self,
        *,
        model_path:      str   = "runs/pose/train2/weights/best.pt",
        camera_index:    int   = AUTO_DETECT,
        arena_w_cm:      float = 167.0,
        arena_h_cm:      float = 121.5,
        config_file:     str   = "image_recon/arena_config.json",
        calib_file:      str   = "image_recon/camera_calib.npz",
        front_kp_index:  int   = 0,
        back_kp_index:   int   = 1,
        detection_conf:  float = 0.25,
    ) -> None:
        # __init__ is called every time ArenaTracker() is invoked, even when
        # __new__ returns the existing singleton.  Guard against re-init.
        if self._initialised:
            return
        self._initialised = True

        # -- Config --
        self._model_path   = model_path
        self._camera_index = camera_index   # may be AUTO_DETECT (-1)
        self._arena_w      = arena_w_cm
        self._arena_h      = arena_h_cm
        self._config_file  = config_file
        self._calib_file   = calib_file
        self._front_kp     = front_kp_index
        self._back_kp      = back_kp_index
        self._conf         = detection_conf

        # -- Arena calibration state --
        self._corners:    list[tuple[int, int]] = []
        self._goal_a_pts: list[tuple[int, int]] = []
        self._goal_b_pts: list[tuple[int, int]] = []
        self._setup_step: str = "CORNERS"

        # -- Runtime state (populated by start()) --
        self._model:      Optional[YOLO]             = None
        self._cap:        Optional[cv2.VideoCapture] = None
        self._M:          Optional[np.ndarray]       = None
        self._M_inv:      Optional[np.ndarray]       = None
        self._cam_mtx:    Optional[np.ndarray]       = None
        self._cam_dist:   Optional[np.ndarray]       = None
        self._running:    bool                       = False
        self._resolved_index: int                    = 0   # filled in by _open_camera
        self._scan_lock:  _threading.Lock            = _threading.Lock()

    # ------------------------------------------------------------------ #
    #  Public lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """
        Open the camera, load calibration, and warm up the model.

        Raises RuntimeError if no calibration file exists – run
        `ArenaTracker().run()` once to produce one.
        """
        if self._running:
            return

        os.makedirs(os.path.dirname(self._config_file) or ".", exist_ok=True)

        if not self._load_calibration():
            raise RuntimeError(
                "No arena calibration found. "
                "Run `ArenaTracker().run()` interactively once to calibrate, "
                "then call start() again."
            )

        self._M, self._M_inv = self._compute_perspective()
        self._model           = YOLO(self._model_path)
        self._cam_mtx, self._cam_dist = self._load_camera_calibration()

        self._cap = self._open_camera()

        # Discard first frames so auto-exposure settles
        for _ in range(5):
            self._cap.read()

        self._running = True
        print(
            f"[ArenaTracker] Ready – camera index {self._resolved_index}, "
            f"arena {self._arena_w} × {self._arena_h} cm"
        )

    def scan(self) -> ScanResult:
        """
        Capture one frame from the camera and return a ScanResult.
        Must call start() first.  Thread-safe: concurrent calls queue.
        """
        if not self._running:
            raise RuntimeError("Call start() before scan().")
        with self._scan_lock:
            frame = self._grab_frame()
            return self._process_frame(frame)

    def stop(self) -> None:
        """Release the camera and mark the tracker as stopped."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._running = False
        print("[ArenaTracker] Stopped.")

    # Context-manager support
    def __enter__(self) -> "ArenaTracker":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------ #
    #  Camera opening                                                      #
    # ------------------------------------------------------------------ #

    def _open_camera(self) -> cv2.VideoCapture:
        """
        Resolve the camera index (auto-detect if needed) and open it with
        the best available backend for the current OS.
        """
        system = platform.system()
        backend = {
            "Linux":   cv2.CAP_V4L2,
            "Darwin":  cv2.CAP_AVFOUNDATION,
            "Windows": cv2.CAP_DSHOW,
        }.get(system, cv2.CAP_ANY)

        if self._camera_index == AUTO_DETECT:
            idx = find_logitech_c930e()
        else:
            idx = self._camera_index

        self._resolved_index = idx
        cap = cv2.VideoCapture(idx, backend)

        if not cap.isOpened():
            # Try CAP_ANY as a final fallback before giving up
            cap.release()
            cap = cv2.VideoCapture(idx, cv2.CAP_ANY)

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {idx} on {system}. "
                "Check that the Logitech C930e is plugged in and not "
                "claimed by another process."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # C930e supports 30fps at 1080p; request explicitly
        cap.set(cv2.CAP_PROP_FPS, 30)

        # Use MJPEG to get full 1280×720 over USB 2.0 bandwidth
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        return cap

    # ------------------------------------------------------------------ #
    #  Interactive debug loop                                              #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """
        Full interactive OpenCV window.  Use this once to produce the
        calibration file, then use start()/scan() from your code.

        Keys:  s – single scan   c – continuous overlay
               v – save frames   r – re-calibrate   q – quit
        """
        model = YOLO(self._model_path)
        cap   = self._open_camera()
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera (index {self._resolved_index}).")
            return

        os.makedirs(os.path.dirname(self._config_file) or ".", exist_ok=True)

        if not self._load_calibration():
            if not self._setup_arena(cap):
                print("Setup cancelled.")
                cap.release()
                return

        self._compute_perspective()
        cam_mtx, cam_dist = self._load_camera_calibration()
        print(f"[ArenaTracker] Arena locked: {self._arena_w} × {self._arena_h} cm")

        win = "Live Preview"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)

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
                print(f"-> Continuous: {'ON' if continuous else 'OFF'}")
            elif key == ord("r"):
                cv2.destroyWindow(win)
                if self._setup_arena(cap):
                    self._compute_perspective()
                    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(win, 1280, 720)
                else:
                    break

            preview = frame.copy()

            if continuous or key == ord("s"):
                result = self._process_frame(frame, model=model)
                vis    = self._render_debug_frame(frame, result, model)

                if key == ord("s"):
                    print("\n--- SCAN ---")
                    print(result.to_json())
                    cv2.imwrite("image_recon/latest_scan.jpg", vis)
                    if show_visuals:
                        w2 = "Result – any key to resume"
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
    #  Frame acquisition                                                   #
    # ------------------------------------------------------------------ #

    def _grab_frame(self) -> np.ndarray:
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera.")
        if self._cam_mtx is not None:
            frame = cv2.undistort(frame, self._cam_mtx, self._cam_dist, None, self._cam_mtx)
        return frame

    # ------------------------------------------------------------------ #
    #  Detection pipeline                                                  #
    # ------------------------------------------------------------------ #

    def _process_frame(
        self,
        frame: np.ndarray,
        *,
        model: Optional[YOLO] = None,
    ) -> ScanResult:
        """Run YOLO on one frame and return a structured ScanResult."""
        model = model or self._model
        M     = self._M
        M_inv = self._M_inv

        # Black out everything outside the arena polygon
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if len(self._corners) == 4:
            cv2.fillPoly(mask, [np.array(self._corners, dtype=np.int32)], 255)
        masked = cv2.bitwise_and(frame, frame, mask=mask)

        results    = model(masked, verbose=False, conf=self._conf)[0]
        detections = self._parse_detections(results, model, M)
        return self._build_result(detections)

    def _parse_detections(
        self, results, model: YOLO, M: np.ndarray
    ) -> list[dict]:
        out = []
        if results.boxes is None:
            return out

        for i in range(len(results.boxes)):
            cls   = int(results.boxes[i].cls[0].item())
            label = model.names[cls]

            x1, y1, x2, y2 = results.boxes[i].xyxy[0].tolist()
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            ax, ay = self._to_cm(cx, cy, M)

            # Drop detections that land outside the arena
            if not (0 <= ax <= self._arena_w and 0 <= ay <= self._arena_h):
                continue

            keypoints, heading = self._extract_pose(results, i, label, M)

            out.append({
                "label":      label,
                "cx": cx,     "cy": cy,
                "ax": ax,     "ay": ay,
                "corners_cm": [
                    self._to_cm(x1, y1, M),  # TL
                    self._to_cm(x2, y1, M),  # TR
                    self._to_cm(x2, y2, M),  # BR
                    self._to_cm(x1, y2, M),  # BL
                ],
                "keypoints": keypoints,
                "heading":   heading,
            })

        return out

    def _extract_pose(
        self, results, index: int, label: str, M: np.ndarray
    ) -> tuple[list[dict], Optional[float]]:
        keypoints: list[dict]    = []
        heading: Optional[float] = None

        if results.keypoints is None or len(results.keypoints.xy) <= index:
            return keypoints, heading

        for kx, ky in results.keypoints.xy[index].cpu().numpy():
            if kx > 0 and ky > 0:
                kax, kay = self._to_cm(kx, ky, M)
                keypoints.append({"x": kax, "y": kay, "px": int(kx), "py": int(ky)})

        if (
            "robot" in label.lower()
            and len(keypoints) > max(self._front_kp, self._back_kp)
        ):
            try:
                front = keypoints[self._front_kp]
                back  = keypoints[self._back_kp]
                heading = round(
                    math.degrees(math.atan2(
                        front["y"] - back["y"],
                        front["x"] - back["x"],
                    )),
                    1,
                )
            except IndexError:
                pass

        return keypoints, heading

    # ------------------------------------------------------------------ #
    #  Result builder                                                      #
    # ------------------------------------------------------------------ #

    def _build_result(self, detections: list[dict]) -> ScanResult:
        robot: Optional[RobotData] = None
        cross: Optional[CrossData] = None
        balls: list[BallData]      = []

        for d in detections:
            lbl     = d["label"].lower()
            corners = [Point(x, y) for x, y in d["corners_cm"]]

            if "ball" in lbl:
                balls.append(BallData(d["label"], Point(d["ax"], d["ay"])))

            elif "robot" in lbl:
                robot = RobotData(
                    label     = d["label"],
                    position  = Point(d["ax"], d["ay"]),
                    heading   = d["heading"],
                    keypoints = [Point(kp["x"], kp["y"]) for kp in d["keypoints"]],
                    corners   = corners,
                )

            else:
                cross = CrossData(label=d["label"], corners=corners)

        return ScanResult(
            arena_width_cm  = self._arena_w,
            arena_height_cm = self._arena_h,
            goal_a = [Point(*self._to_cm(*pt, self._M)) for pt in self._goal_a_pts],
            goal_b = [Point(*self._to_cm(*pt, self._M)) for pt in self._goal_b_pts],
            robot  = robot,
            cross  = cross,
            balls  = balls,
        )

    # ------------------------------------------------------------------ #
    #  Coordinate helpers                                                  #
    # ------------------------------------------------------------------ #

    def _to_cm(self, x: float, y: float, M: np.ndarray) -> tuple[float, float]:
        pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
        tx, ty = cv2.perspectiveTransform(pt, M)[0][0]
        return round(float(tx), 1), round(float(ty), 1)

    def _compute_perspective(self) -> tuple[np.ndarray, np.ndarray]:
        src = np.array(self._corners, dtype=np.float32)
        dst = np.array([
            [0,             0            ],
            [self._arena_w, 0            ],
            [self._arena_w, self._arena_h],
            [0,             self._arena_h],
        ], dtype=np.float32)
        self._M     = cv2.getPerspectiveTransform(src, dst)
        self._M_inv = cv2.getPerspectiveTransform(dst, src)
        return self._M, self._M_inv

    # ------------------------------------------------------------------ #
    #  Calibration I/O                                                     #
    # ------------------------------------------------------------------ #

    def _load_calibration(self) -> bool:
        if not os.path.exists(self._config_file):
            return False
        try:
            with open(self._config_file) as f:
                data = json.load(f)
            self._corners    = [tuple(p) for p in data.get("corners",    [])]
            self._goal_a_pts = [tuple(p) for p in data.get("goal_a_pts", [])]
            self._goal_b_pts = [tuple(p) for p in data.get("goal_b_pts", [])]
            if len(self._corners) == 4:
                print("[ArenaTracker] Arena calibration loaded.")
                return True
        except Exception as exc:
            print(f"[WARNING] Could not load calibration: {exc}")
        return False

    def _save_calibration(self) -> None:
        with open(self._config_file, "w") as f:
            json.dump(
                {
                    "corners":    self._corners,
                    "goal_a_pts": self._goal_a_pts,
                    "goal_b_pts": self._goal_b_pts,
                },
                f, indent=4,
            )
        print("[ArenaTracker] Calibration saved.")

    def _load_camera_calibration(
        self,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if os.path.exists(self._calib_file):
            with np.load(self._calib_file) as data:
                print("[ArenaTracker] Lens calibration loaded.")
                return data["mtx"], data["dist"]
        print("[WARNING] No lens calibration – image won't be undistorted.")
        return None, None

    # ------------------------------------------------------------------ #
    #  Interactive setup wizard                                            #
    # ------------------------------------------------------------------ #

    def _setup_arena(self, cap: cv2.VideoCapture) -> bool:
        self._corners.clear()
        self._goal_a_pts.clear()
        self._goal_b_pts.clear()
        self._setup_step = "CORNERS"

        print("\n--- ARENA SETUP ---")
        for _ in range(5):
            cap.read()

        win = "Arena Setup"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)
        cv2.setMouseCallback(win, self._handle_mouse)

        _STEPS = {
            "CORNERS": lambda: (f"1. Click 4 corners (BL BR TR TL): {len(self._corners)}/4", (0, 255, 255)),
            "GOAL_A":  lambda: (f"2. Goal A – TL then BR: {len(self._goal_a_pts)}/2",         (255, 150, 0)),
            "GOAL_B":  lambda: (f"3. Goal B – TL then BR: {len(self._goal_b_pts)}/2",         (0, 0, 255)),
            "DONE":    lambda: ("Done! ENTER to save | R to reset",                            (0, 255, 0)),
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
            self._draw_goal_on_frame(disp, self._goal_a_pts, (255, 150, 0), "Goal A")
            self._draw_goal_on_frame(disp, self._goal_b_pts, (0, 0, 255),   "Goal B")

            cv2.imshow(win, disp)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("r"):
                self._corners.clear()
                self._goal_a_pts.clear()
                self._goal_b_pts.clear()
                self._setup_step = "CORNERS"
                print("--- Reset ---")
            elif key == 13 and self._setup_step == "DONE":
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
            print(f"  Corner {len(self._corners)}/4: ({x}, {y})")
            if len(self._corners) == 4:
                self._setup_step = "GOAL_A"
                print("-> Click TL + BR of Goal A")
        elif self._setup_step == "GOAL_A" and len(self._goal_a_pts) < 2:
            self._goal_a_pts.append((x, y))
            print(f"  Goal A {len(self._goal_a_pts)}/2: ({x}, {y})")
            if len(self._goal_a_pts) == 2:
                self._setup_step = "GOAL_B"
                print("-> Click TL + BR of Goal B")
        elif self._setup_step == "GOAL_B" and len(self._goal_b_pts) < 2:
            self._goal_b_pts.append((x, y))
            print(f"  Goal B {len(self._goal_b_pts)}/2: ({x}, {y})")
            if len(self._goal_b_pts) == 2:
                self._setup_step = "DONE"
                print("-> Press ENTER to save.")

    # ------------------------------------------------------------------ #
    #  Visualisation (debug / run() only)                                  #
    # ------------------------------------------------------------------ #

    def _render_debug_frame(
        self,
        frame: np.ndarray,
        result: ScanResult,
        model: YOLO,
    ) -> np.ndarray:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if len(self._corners) == 4:
            cv2.fillPoly(mask, [np.array(self._corners, dtype=np.int32)], 255)
        yolo_res = model(cv2.bitwise_and(frame, frame, mask=mask), verbose=False, conf=self._conf)[0]
        vis = yolo_res.plot()
        vis = self._draw_arena_overlay(vis, self._M_inv)

        # Robot heading arrow
        if result.robot and result.robot.heading is not None:
            kps = result.robot.keypoints
            if len(kps) > max(self._front_kp, self._back_kp):
                def cm_to_px(kp: Point) -> tuple[int, int]:
                    pt = np.array([[[kp.x, kp.y]]], dtype=np.float32)
                    ox, oy = cv2.perspectiveTransform(pt, self._M_inv)[0][0]
                    return int(ox), int(oy)

                cv2.arrowedLine(
                    vis,
                    cm_to_px(kps[self._back_kp]),
                    cm_to_px(kps[self._front_kp]),
                    (0, 255, 255), 3, tipLength=0.3,
                )

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
        pts: list[tuple[int, int]],
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
        img: np.ndarray,
        text: str,
        pos: tuple[int, int],
        font_scale: float,
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 3)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,     thickness)


# --------------------------------------------------------------------------- #
#  Entry point                                                                 #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    ArenaTracker().run()