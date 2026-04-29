import cv2
import numpy as np
import os

# --- CONFIG ---
CAMERA_INDEX = 0
CB_W = 10  # Number of INTERNAL corners horizontally
CB_H = 7  # Number of INTERNAL corners vertically
CALIB_FILE = "image_recon/camera_calib.npz"

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    objp = np.zeros((CB_W * CB_H, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CB_W, 0:CB_H].T.reshape(-1, 2)

    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane

    print("--- CAMERA CALIBRATION ---")
    print("1. Hold a printed checkerboard flat in front of the camera.")
    print("2. Move it around to different angles and edges of the screen.")
    print("3. Press 'c' to capture a frame (try to get 15-20 good ones).")
    print("4. Press ENTER to calculate and save the calibration.")
    print("5. Press 'q' to quit without saving.")

    captured_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret_cb, corners = cv2.findChessboardCorners(gray, (CB_W, CB_H), None)

        if ret_cb:
            cv2.drawChessboardCorners(display, (CB_W, CB_H), corners, ret_cb)
            msg_color = (0, 255, 0)
        else:
            msg_color = (0, 0, 255)

        cv2.putText(display, f"Captured: {captured_frames}/20", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, msg_color, 2)
        cv2.imshow('Calibration', display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c') and ret_cb:
            # Refine corner locations for better accuracy
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            objpoints.append(objp)
            imgpoints.append(corners2)
            captured_frames += 1
            print(f"Captured frame {captured_frames}!")

        elif key == 13 and captured_frames > 5: # ENTER key
            print("\nCalculating calibration... this might take a few seconds.")
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            
            os.makedirs("image_recon", exist_ok=True)
            np.savez(CALIB_FILE, mtx=mtx, dist=dist)
            print(f"Calibration saved to {CALIB_FILE}!")
            break

        elif key == ord('q'):
            print("Cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()