from dataclasses import dataclass


@dataclass(frozen=True)
class ArenaConfig:
    # ------------------------------------------------------------------ #
    #  Fysiske mål på banen (mål dem én gang med målebånd)               #
    # ------------------------------------------------------------------ #
    width_cm:  float = 167.0
    height_cm: float = 121.5

    # ------------------------------------------------------------------ #
    #  YOLO-model                                                          #
    # ------------------------------------------------------------------ #
    model_path:     str   = "runs/pose/train7/weights/best.pt"
    detection_conf: float = 0.30

    # ------------------------------------------------------------------ #
    #  Kamera                                                              #
    # ------------------------------------------------------------------ #
    camera_index:  int = -1       # -1 = auto-detect
    frame_width:   int = 1280
    frame_height:  int = 720
    frame_fps:     int = 30

    # ------------------------------------------------------------------ #
    #  Filer                                                               #
    # ------------------------------------------------------------------ #
    arena_config_file: str = "image_recon/arena_config.json"
    camera_calib_file: str = "image_recon/camera_calib.npz"

    # ------------------------------------------------------------------ #
    #  ArUco                                                               #
    # ------------------------------------------------------------------ #
    aruco_target_id: int = 0