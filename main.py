"""
Core vision-based robot control application. Real-time detection of colored parts,
interactive filtering and selection, automated pick-and-place orchestration.

System Design:
  - Workspace calibration via fiducial marker detection (HoughCircles)
  - HSV-based color segmentation with contour analysis for part localization
  - Shape classification using circularity and polygon approximation metrics
  - Perspective transform: pixel->relative->robot Cartesian coordinates
  - Interactive terminal control with terminal-based command dispatcher
"""

import queue
import sys
import threading

import cv2
import numpy as np

from config import ROBOT_IP
from robot import NiryoVisionPicker
from ui import (
    apply_filters,
    draw_hud,
    draw_objects,
    draw_workspace_overlay,
    input_worker,
    print_terminal_help,
    process_command,
)
from vision import detect_objects, detect_workspace_from_dianas, fallback_workspace_corners


def run() -> None:
    """Main event loop: camera acquisition, vision pipeline, visualization, command processing."""
    picker = NiryoVisionPicker(ROBOT_IP)
    cmd_queue: "queue.Queue[str]" = queue.Queue()

    input_thread = threading.Thread(target=input_worker, args=(cmd_queue,), daemon=True)

    try:
        picker.connect()
        picker.move_scan()

        try:
            picker.robot.release_with_tool()
        except Exception:
            pass

        print_terminal_help()
        input_thread.start()

        cv2.namedWindow("Niryo Vision Menu", cv2.WINDOW_NORMAL)

        running = True
        while running:
            frame = picker.capture_frame()

            # 1) Detect workspace via fiducial markers with exponential smoothing
            detected = detect_workspace_from_dianas(frame)
            detected_live = detected is not None

            if detected is not None:
                if picker.workspace_corners is None:
                    picker.workspace_corners = detected
                else:
                    # IIR filter to suppress jitter
                    picker.workspace_corners = 0.75 * picker.workspace_corners + 0.25 * detected

            if picker.workspace_corners is None:
                picker.workspace_corners = fallback_workspace_corners()

            workspace_corners = picker.workspace_corners.astype(np.float32)

            # 2) Detect parts within workspace bounds, apply active filters
            all_objects = detect_objects(frame, workspace_corners)
            visible_objects = apply_filters(all_objects, picker.filter_color, picker.filter_shape)

            if picker.selected_index is not None and (
                picker.selected_index < 1 or picker.selected_index > len(visible_objects)
            ):
                picker.selected_index = None

            # 3) Render annotations and HUD
            draw_workspace_overlay(frame, workspace_corners, detected_live)
            draw_objects(frame, visible_objects, picker.selected_index)
            draw_hud(frame, picker, visible_objects)

            cv2.imshow("Niryo Vision Menu", frame)
            cv2.waitKey(1)

            # 4) Dispatch terminal commands (non-blocking queue drain)
            while not cmd_queue.empty():
                cmd_line = cmd_queue.get_nowait()
                running = process_command(cmd_line, picker, visible_objects, workspace_corners)
                if not running:
                    break

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()
        picker.safe_shutdown()


if __name__ == "__main__":
    run()
