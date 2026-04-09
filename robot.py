"""
Robot client and coordinate mapping. Encapsulates Niryo API calls,
camera image acquisition, and relative-to-absolute coordinate transforms.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
from pyniryo import NiryoRobot, PoseObject

from config import (
    LIMIT_1,
    LIMIT_2,
    LIMIT_3,
    LIMIT_4,
    SCANNING_POSITION,
)


def pose_from_tuple(p: Tuple[float, float, float, float, float, float]) -> PoseObject:
    """Convenience wrapper to convert 6-tuple to Niryo pose object."""
    return PoseObject(*p)


class NiryoVisionPicker:
    """Integrated robot driver combining hardware access, vision state, and grasping logic."""

    def __init__(self, ip: str):
        self.ip = ip
        self.robot: Optional[NiryoRobot] = None

        self.filter_color = "ANY"   # Color predicate: ANY | RED | GREEN | BLUE
        self.filter_shape = "ANY"   # Shape predicate: ANY | CIRCLE | SQUARE
        self.selected_index: Optional[int] = None

        self.pick_roll = SCANNING_POSITION[3]
        self.pick_pitch = SCANNING_POSITION[4]
        self.pick_yaw = SCANNING_POSITION[5]

        # Workspace bounds inferred from fiducial detection (TL, TR, BR, BL)
        self.workspace_corners: Optional[np.ndarray] = None

    def connect(self) -> None:
        """Initialize Niryo robot: establish connection, run calibration, auto-detect gripper."""
        print(f"Connecting to robot: {self.ip}")
        self.robot = NiryoRobot(self.ip)
        # Clear stale collision flag from prior session to enable movement
        try:
            self.robot.clear_collision_detected()
        except Exception:
            pass
        self.robot.calibrate_auto()
        print("Robot calibrated")

        try:
            self.robot.update_tool()
            print("Gripper detected")
        except Exception as e:
            print(f"[WARN] Could not auto-detect gripper: {e}")

    def move_scan(self) -> None:
        """Navigate to observation position with camera aligned to workspace."""
        self.robot.move(pose_from_tuple(SCANNING_POSITION))

    def move_home(self) -> None:
        """Return to home position."""
        self.robot.move_to_home_pose()

    def clear_collision(self) -> None:
        """Reset collision flag and re-sync hardware state. Must remove obstruction first!"""
        if self.robot is None:
            raise RuntimeError("Robot not connected")
        print("[COLLISION] Clearing collision flag and re-calibrating...")
        self.robot.clear_collision_detected()
        self.robot.calibrate_auto()
        print("[COLLISION] Collision state cleared. Robot ready.")

    def safe_shutdown(self) -> None:
        """Graceful shutdown: release gripper, return home, disconnect hardware."""
        if self.robot is None:
            return

        print("Graceful shutdown...")
        try:
            try:
                self.robot.release_with_tool()
            except Exception:
                pass
            try:
                self.move_home()
            except Exception as e:
                print(f"[WARN] Could not move home during shutdown: {e}")
        finally:
            self.robot.close_connection()
            self.robot = None
            print("Robot desconectado")

    def capture_frame(self) -> np.ndarray:
        """Acquire image from onboard camera and decompress."""
        img_compressed = self.robot.get_img_compressed()
        frame = cv2.imdecode(np.frombuffer(img_compressed, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("No se pudo decodificar la imagen de la camara")
        return frame


def relative_to_robot_xy(x_rel: float, y_rel: float) -> Tuple[float, float]:
    """Convert normalized workspace coordinates [0..1]x[0..1] to robot Cartesian coordinates.
    Accounts for camera/robot frame mirroring via bilinear interpolation on corner calibration."""
    # Camera frame X-axis is horizontally flipped w.r.t. robot coordinates.
    # Image-left maps to robot-right; image-right maps to robot-left.
    # See LIMIT_* for physical corner mapping reference.
    p_tl = np.array(LIMIT_4[:2], dtype=float)   # img-TL -> robot far-right
    p_bl = np.array(LIMIT_3[:2], dtype=float)   # img-BL -> robot near-right
    p_tr = np.array(LIMIT_1[:2], dtype=float)   # img-TR -> robot far-left
    p_br = np.array(LIMIT_2[:2], dtype=float)   # img-BR -> robot near-left

    top = p_tl * (1.0 - x_rel) + p_tr * x_rel
    bottom = p_bl * (1.0 - x_rel) + p_br * x_rel
    xy = top * (1.0 - y_rel) + bottom * y_rel

    return float(xy[0]), float(xy[1])
