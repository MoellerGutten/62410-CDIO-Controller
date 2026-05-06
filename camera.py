import re
import subprocess
import platform
from typing import Optional
import os
import cv2

# --------------------------------------------------------------------------- #
#  Camera auto-detection                                                      #
# --------------------------------------------------------------------------- #

# USB vendor:product IDs for the Logitech C930e
_LOGITECH_C930E_VID = "046d"
_LOGITECH_C930E_PID = "0843"

# Fallback: any Logitech webcam (vendor ID only)
_LOGITECH_VID = "046d"

# Keywords used as a last resort when USB IDs cannot be read
_LOGITECH_KEYWORDS = ("c930", "logitech", "logi")


def find_logitech_c930e(max_index: int = 10) -> int:
    system = platform.system()

    try:
        index = _usb_enumerate(system)
        if index is not None:
            print(f"[CameraDetect] Found Logitech C930e via USB enumeration → index {index}")
            return index
    except Exception as exc:
        print(f"[CameraDetect] USB enumeration skipped ({exc})")

    index = _scan_by_name(max_index)
    if index is not None:
        print(f"[CameraDetect] Found Logitech camera by name scan → index {index}")
        return index

    print("[CameraDetect] Could not identify C930e; falling back to index 0.")
    return 0


# ── Linux ──────────────────────────────────────────────────────────────────

def _linux_find_camera() -> Optional[int]:
    v4l_root = "/sys/class/video4linux"
    if not os.path.isdir(v4l_root):
        return None

    exact_match: list[int]    = []
    vendor_match: list[int]   = []

    for node in sorted(os.listdir(v4l_root)):
        match = re.match(r"video(\d+)$", node)
        if not match:
            continue
        cam_idx = int(match.group(1))

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
    path = device_path
    for _ in range(6):
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
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPCameraDataType"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode(errors="replace")
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    matching_names: list[str] = []
    current_name  = ""
    current_vid   = ""
    current_pid   = ""

    for line in out.splitlines():
        line = line.strip()
        if line.endswith(":") and not line.startswith(" ") and ":" not in line[:-1]:
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

    target = matching_names[0].lower()
    for idx in range(10):
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            continue
        cap.release()
        return matching_names.index(matching_names[0])

    return None


# ── Windows ────────────────────────────────────────────────────────────────

def _windows_find_camera() -> Optional[int]:
    """Enumerate DirectShow devices — same ordering as OpenCV on Windows."""
    from pygrabber.dshow_graph import FilterGraph
    try:
        # pygrabber uses DirectShow, same device order as cv2.VideoCapture
        graph = FilterGraph()
        devices = graph.get_input_devices()  # list of names, index == CV index
        for idx, name in enumerate(devices):
            name_lower = name.lower()
            if any(k in name_lower for k in _LOGITECH_KEYWORDS):
                return idx
        return None
    except ImportError:
        pass

    # Fallback: WMI — less reliable ordering but better than nothing
    ps_script = (
        "Get-WmiObject Win32_PnPEntity | "
        "Where-Object { $_.DeviceID -match 'VID_046D' } | "
        "Select-Object -ExpandProperty Name"
    )
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps_script],
            stderr=subprocess.DEVNULL, timeout=10,
        ).decode(errors="replace")
        names = [l.strip().lower() for l in out.splitlines() if l.strip()]
        if names:
            print(f"[CameraDetect] WMI found: {names} — assuming index 0")
            return 0  # best guess without DirectShow
    except Exception:
        pass

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
    system = platform.system()
    backends = (
        [cv2.CAP_MSMF] if system == "Windows"
        else [cv2.CAP_V4L2]   if system == "Linux"
        else [cv2.CAP_AVFOUNDATION] if system == "Darwin"
        else [cv2.CAP_ANY]
    )

    first_open: Optional[int] = None

    for idx in range(max_index):
        cap = None
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                break
            cap.release()
            cap = None

        if cap is None or not cap.isOpened():
            if cap:
                cap.release()
            continue

        if first_open is None:
            first_open = idx

        name = ""
        try:
            name = cap.getBackendName().lower()
        except AttributeError:
            pass

        cap.release()

        if any(k in name for k in _LOGITECH_KEYWORDS):
            return idx

    return None


def _is_logitech(vid: str, pid: str, name: str) -> bool:
    if vid == _LOGITECH_C930E_VID:
        return True
    if vid == _LOGITECH_VID:
        return True
    name_lower = name.lower()
    return any(k in name_lower for k in _LOGITECH_KEYWORDS)