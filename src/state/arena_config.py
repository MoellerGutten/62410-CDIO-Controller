from dataclasses import dataclass


@dataclass(frozen=True)
class ArenaConfig:
    # ------------------------------------------------------------------ #
    #  Fysiske mål på banen                                              #
    # ------------------------------------------------------------------ #
    width_cm:  float = 167.0
    height_cm: float = 121.5

    # ------------------------------------------------------------------ #
    #  YOLO-model                                                        #
    # ------------------------------------------------------------------ #
    model_path:     str   = "runs/pose/train17/weights/best.pt"
    detection_conf: float = 0.50

    # ------------------------------------------------------------------ #
    #  Kamera                                                            #
    # ------------------------------------------------------------------ #
    camera_index:  int = -1       # -1 = auto-detect
    frame_width:   int = 1280
    frame_height:  int = 720
    frame_fps:     int = 30

    # ------------------------------------------------------------------ #
    # Kamera position og aruco højde                                     #
    # ------------------------------------------------------------------ #
    camera_height: int = 147
    camera_x:      int = 88
    camera_y:      int = 54
    aruco_height:  int = 12

    # ------------------------------------------------------------------ #
    #  Filer                                                             #
    # ------------------------------------------------------------------ #
    arena_config_file: str = "image_recon/arena_config.json"
    camera_calib_file: str = "image_recon/camera_calib.npz"

    # ------------------------------------------------------------------ #
    #  ArUco + ArUco offset                                              #
    # ------------------------------------------------------------------ #
    aruco_target_id: int = 0
    aruco_offset_x: float = -12
    aruco_offset_y: float = 0