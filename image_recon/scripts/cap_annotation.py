"""
capture_annotation_images.py
=============================
Captures undistorted frames from the arena camera for YOLO annotation.

Two modes
---------
  Manual  (default) – press SPACE to save a frame, Q to quit.
  Auto              – pass --auto 200 to collect 200 frames automatically
                      with a configurable interval between each capture.

Saved images land in  output_dir/  (default: annotation_captures/)
and are named  frame_0001.jpg, frame_0002.jpg, …

Usage examples
--------------
  # Manual – you decide when each shot looks good
  python capture_annotation_images.py

  # Auto – 200 frames, one every 2 seconds
  python capture_annotation_images.py --auto 200 --interval 2.0

  # Different camera or output folder
  python capture_annotation_images.py --camera 1 --out my_dataset/raw
"""

import argparse
import os
import time

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
#  Calibration paths  (must match arena_tracker.py)                           #
# --------------------------------------------------------------------------- #
CALIB_FILE = "image_recon/camera_calib.npz"
ARENA_CONFIG = "image_recon/arena_config.json"


def load_lens_calibration():
    if not os.path.exists(CALIB_FILE):
        print("[WARNING] No lens calibration found – images will NOT be undistorted.")
        return None, None
    with np.load(CALIB_FILE) as data:
        print("[INFO] Lens calibration loaded – fisheye correction active.")
        return data["mtx"], data["dist"]


def undistort(frame, mtx, dist):
    if mtx is None:
        return frame
    return cv2.undistort(frame, mtx, dist, None, mtx)


def draw_hud(frame, count, target, mode, interval=None):
    """Burn a small status overlay onto the preview (not onto saved images)."""
    h, w = frame.shape[:2]
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 48), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.5, frame, 0.5, 0, frame)

    if mode == "manual":
        msg = f"MANUAL  |  Saved: {count}  |  SPACE=capture  Q=quit"
    else:
        msg = f"AUTO  |  {count}/{target}  |  every {interval}s  |  Q=quit"

    cv2.putText(frame, msg, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame


def save_frame(frame, out_dir, count):
    path = os.path.join(out_dir, f"frame_{count:04d}.jpg")
    cv2.imwrite(path, frame)
    print(f"  Saved {path}")
    return count + 1


def run_manual(cap, mtx, dist, out_dir):
    print("\n[Manual mode]  SPACE = save frame   Q = quit\n")
    count = 1

    while True:
        ret, raw = cap.read()
        if not ret:
            continue

        clean  = undistort(raw, mtx, dist)        # what gets saved
        preview = clean.copy()
        draw_hud(preview, count - 1, None, "manual")
        cv2.imshow("Capture – Manual", preview)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            count = save_frame(clean, out_dir, count)

    print(f"\nDone – {count - 1} images saved to '{out_dir}/'")


def run_auto(cap, mtx, dist, out_dir, target, interval):
    print(f"\n[Auto mode]  Capturing {target} frames every {interval}s   Q = quit early\n")
    count   = 1
    last_t  = time.time() - interval  # capture immediately on first loop

    while count <= target:
        ret, raw = cap.read()
        if not ret:
            continue

        clean   = undistort(raw, mtx, dist)
        preview = clean.copy()
        draw_hud(preview, count - 1, target, "auto", interval)
        cv2.imshow("Capture – Auto", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Quit early.")
            break

        now = time.time()
        if now - last_t >= interval:
            count  = save_frame(clean, out_dir, count)
            last_t = now

    print(f"\nDone – {count - 1} images saved to '{out_dir}/'")


def main():
    parser = argparse.ArgumentParser(description="Capture undistorted annotation images.")
    parser.add_argument("--camera",   type=int,   default=0,                    help="Camera index (default 0)")
    parser.add_argument("--out",      type=str,   default="annotation_captures", help="Output directory")
    parser.add_argument("--auto",     type=int,   default=None,                  help="Auto-capture N frames")
    parser.add_argument("--interval", type=float, default=1.0,                   help="Seconds between auto captures (default 1.0)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    mtx, dist = load_lens_calibration()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Let auto-exposure settle
    print("Warming up camera...")
    for _ in range(10):
        cap.read()

    try:
        if args.auto is not None:
            run_auto(cap, mtx, dist, args.out, args.auto, args.interval)
        else:
            run_manual(cap, mtx, dist, args.out)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()