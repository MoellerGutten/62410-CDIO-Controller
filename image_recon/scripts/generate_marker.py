import cv2
import numpy as np

def create_marker():
    # Load the 4x4 dictionary we specified in the tracker
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_size_px = 500  # High resolution for crisp printing

    # Generate the marker (Handles both new and old OpenCV versions)
    try:
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, 0, marker_size_px)
    except AttributeError:
        marker_img = cv2.aruco.drawMarker(aruco_dict, 0, marker_size_px)

    # ArUco NEEDS a white border to be detected. Let's add a 50px white border.
    bordered_marker = cv2.copyMakeBorder(
        marker_img, 
        50, 50, 50, 50, 
        cv2.BORDER_CONSTANT, 
        value=[255, 255, 255]
    )

    # Save to your folder
    filename = "aruco_id0_robot.jpg"
    cv2.imwrite(filename, bordered_marker)
    print(f"✅ Saved marker to {filename}. Print this out and keep it flat!")

if __name__ == "__main__":
    create_marker()