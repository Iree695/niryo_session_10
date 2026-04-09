"""
Dice capture pipeline for Niryo robot. Acquires images at optimal zoom level,
detects individual dice, and persists labeled samples for neural network training.

Supports three data collection workflows:
  python capture.py              : Manual labeling per sample 
  python capture.py --raw        : Preserves raw frames for batch annotation
  python capture.py --label J    : Bulk capture with predetermined class

Camera positioning automatically adjusts resolution based on capture depth.
"""

import argparse
import os
import sys
import time
from typing import List, Tuple
import cv2
import numpy as np

from config import DATASET_DIR, DICE_FACES, ROBOT_IP, SCANNING_POSITION
from robot import NiryoVisionPicker
from vision import detect_workspace_from_dianas, fallback_workspace_corners


RAW_DIR = os.path.join(DATASET_DIR, "raw")
CROP_PADDING = 15  # border expansion for robustness in augmentation

# Morphological constraints to distinguish dice from workspace artifacts
DICE_MIN_AREA = 400
DICE_MAX_AREA = 8000

# Move closer to workspace for magnified view. Maintains scan-referenced XY alignment
# while reducing working distance from 0.23m to 0.18m, yielding 2-2.5x pixel density.
# Fiducial visibility degradation is acceptable during capture phase.
CAPTURE_POSITION = (
    SCANNING_POSITION[0],   # x
    SCANNING_POSITION[1],   # y
    0.18,                   # z — lower than scan (0.23)
    SCANNING_POSITION[3],   # roll
    SCANNING_POSITION[4],   # pitch
    SCANNING_POSITION[5],   # yaw
)


def setup_dirs(label: str = None) -> str:
    """Ensures label-specific subdirectory exists in dataset hierarchy."""
    if label:
        target = os.path.join(DATASET_DIR, label)
    else:
        target = RAW_DIR
    os.makedirs(target, exist_ok=True)
    return target


def generate_filename(prefix: str = "capture") -> str:
    """Generates a unique name based on timestamp."""
    ts = int(time.time() * 1000)
    return f"{prefix}_{ts}.png"


def extract_dice_crops(frame: np.ndarray, workspace_corners: np.ndarray) -> List[Tuple[np.ndarray, tuple]]:
    """Detects dice in the frame and returns list of (crop, bounding_rect).
    Strategy: mask the interior of the workspace, use Canny to detect
    edges (works with light dice on light background) and close contours
    with morphology to obtain solid regions.
    """
    h_frame, w_frame = frame.shape[:2]

    # 1) Mask of the workspace interior (excludes dark border)
    ws_mask = np.zeros((h_frame, w_frame), dtype=np.uint8)
    ws_poly = workspace_corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(ws_mask, [ws_poly], 255)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    ws_mask = cv2.erode(ws_mask, erode_kernel, iterations=1)

    # 2) Canny on masked grayscale — detects edges independently of brightness
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
    edges = cv2.Canny(blurred, 30, 100)

    # Apply workspace mask
    edges = cv2.bitwise_and(edges, ws_mask)

    # 3) Dilate + close to join edges into solid contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.dilate(edges, kernel, iterations=2)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crops = []

    # Exclusion radius around each fiducial (pixels)
    DIANA_EXCLUSION_RADIUS = 50

    for contour in contours:
        area = cv2.contourArea(contour)
        if not (DICE_MIN_AREA <= area <= DICE_MAX_AREA):
            continue

        # Centroid of contour
        m = cv2.moments(contour)
        if m["m00"] == 0:
            continue
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])

        # Skip if close to any fiducial
        is_diana = False
        for corner in workspace_corners:
            dx = cx - corner[0]
            dy = cy - corner[1]
            if (dx * dx + dy * dy) ** 0.5 < DIANA_EXCLUSION_RADIUS:
                is_diana = True
                break
        if is_diana:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Crop with padding
        x1 = max(0, x - CROP_PADDING)
        y1 = max(0, y - CROP_PADDING)
        x2 = min(w_frame, x + w + CROP_PADDING)
        y2 = min(h_frame, y + h + CROP_PADDING)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crops.append((crop, (x1, y1, x2 - x1, y2 - y1)))

    return crops


def draw_detections(display: np.ndarray, crops: List[Tuple[np.ndarray, tuple]]) -> None:
    """Draws numbered bounding boxes over detected dice."""
    for i, (_, (x, y, w, h)) in enumerate(crops, start=1):
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(display, str(i), (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def run_capture(raw_mode: bool = False, fixed_label: str = None, low: bool = False) -> None:
    picker = NiryoVisionPicker(ROBOT_IP)
    workspace_corners = None

    try:
        picker.connect()

        if low:
            # First go to normal scan to detect workspace fiducials
            picker.move_scan()
            frame_for_ws = picker.capture_frame()
            ws_detected = detect_workspace_from_dianas(frame_for_ws)
            if ws_detected is not None:
                workspace_corners = ws_detected
                print("[CAPTURE] Fiducials detected from scan, saving for reference")
            else:
                workspace_corners = fallback_workspace_corners()
                print("[CAPTURE] Fiducials not detected, using fallback")

            # Now lower to the capture pose (closer, better resolution)
            from pyniryo import PoseObject
            picker.robot.move(PoseObject(*CAPTURE_POSITION))
            print(f"[CAPTURE] Low pose Z={CAPTURE_POSITION[2]}m (better resolution)")
        else:
            picker.move_scan()

        # Ensure gripper is open to not obstruct the view
        try:
            picker.robot.release_with_tool()
        except Exception:
            pass

        cv2.namedWindow("Captura Niryo", cv2.WINDOW_NORMAL)

        count = 0
        print("\n=== Capture Mode ===")
        print("  SPACE    -> capture detected dice crops")
        print("  f        -> save complete frame (reference)")
        print("  q / ESC  -> exit")
        if raw_mode:
            setup_dirs()
            print(f"  Raw mode: complete frames in {RAW_DIR}")
        elif fixed_label:
            if fixed_label not in DICE_FACES:
                print(f"  [WARN] Label '{fixed_label}' not in DICE_FACES {DICE_FACES}")
            setup_dirs(fixed_label)
            print(f"  Fixed label: all crops as '{fixed_label}'")
        else:
            for face in DICE_FACES:
                setup_dirs(face)
            print(f"  Valid faces: {DICE_FACES}")
            print("  After capturing, label is requested via terminal for each crop")
        print()

        while True:
            frame = picker.capture_frame()

            # Detect workspace
            detected = detect_workspace_from_dianas(frame)
            if detected is not None:
                if workspace_corners is None:
                    workspace_corners = detected
                else:
                    workspace_corners = 0.75 * workspace_corners + 0.25 * detected

            if workspace_corners is None:
                workspace_corners = fallback_workspace_corners()

            ws = workspace_corners.astype(np.float32)

            # Detect dice
            crops = extract_dice_crops(frame, ws)

            # Display with detections
            display = frame.copy()
            draw_detections(display, crops)
            ws_status = "WS: fiducials" if detected is not None else "WS: fallback"
            cv2.putText(display, f"Dice: {len(crops)} | Photos: {count} | {ws_status} | SPACE=capture | q=exit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            cv2.imshow("Captura Niryo", display)
            key = cv2.waitKey(100) & 0xFF

            if key == ord("q") or key == 27:
                break

            # 'f' = save complete frame as reference
            if key == ord("f"):
                ref_dir = setup_dirs("_frames")
                filepath = os.path.join(ref_dir, generate_filename("frame"))
                cv2.imwrite(filepath, frame)
                print(f"  Complete frame saved: {filepath}")

            # SPACE = capture crops
            if key == ord(" "):
                if not crops:
                    print("  No dice detected in workspace")
                    continue

                if raw_mode:
                    # Raw mode: save complete frame
                    filepath = os.path.join(RAW_DIR, generate_filename())
                    cv2.imwrite(filepath, frame)
                    count += 1
                    print(f"  [{count}] Frame saved: {filepath}")

                elif fixed_label:
                    # Save all crops with the same label
                    target_dir = setup_dirs(fixed_label)
                    for i, (crop, _) in enumerate(crops, start=1):
                        filepath = os.path.join(target_dir, generate_filename(fixed_label))
                        cv2.imwrite(filepath, crop)
                        count += 1
                        print(f"  [{count}] Crop {i} -> {fixed_label}: {filepath}")

                else:
                    # Interactive mode: label each crop
                    for i, (crop, (x, y, w, h)) in enumerate(crops, start=1):
                        # Show enlarged crop
                        crop_big = cv2.resize(crop, (200, 200), interpolation=cv2.INTER_NEAREST)
                        cv2.imshow("Crop", crop_big)
                        cv2.waitKey(1)

                        label = input(
                            f"  Crop {i}/{len(crops)} ({w}x{h}px) "
                            f"label ({'/'.join(DICE_FACES)}, s=skip): "
                        ).strip().upper()

                        if label == "S":
                            print("    Skipped")
                            continue
                        if label not in DICE_FACES:
                            print(f"    '{label}' not valid, skipping")
                            continue

                        target_dir = setup_dirs(label)
                        filepath = os.path.join(target_dir, generate_filename(label))
                        cv2.imwrite(filepath, crop)
                        count += 1
                        print(f"    [{count}] Saved: {filepath}")

                    try:
                        cv2.destroyWindow("Crop")
                    except Exception:
                        pass

        # Final summary
        print(f"\nTotal crops saved: {count}")
        print("Dataset summary:")
        for face in DICE_FACES:
            folder = os.path.join(DATASET_DIR, face)
            if os.path.isdir(folder):
                n = len([f for f in os.listdir(folder) if f.endswith(".png")])
                if n > 0:
                    print(f"  {face}: {n} imagenes")

    except KeyboardInterrupt:
        print("\nInterrumpido")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()
        picker.safe_shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Captura de fotos desde el robot Niryo")
    parser.add_argument("--raw", action="store_true",
                        help="Guardar frames completos sin etiquetar en dataset/raw/")
    parser.add_argument("--label", type=str, default=None,
                        help="Etiqueta fija para todos los crops (ej: --label J)")
    parser.add_argument("--low", action="store_true",
                        help="Bajar camara a Z=0.18m para mejor resolucion de crops")
    args = parser.parse_args()

    run_capture(
        raw_mode=args.raw,
        fixed_label=args.label.upper() if args.label else None,
        low=args.low,
    )
