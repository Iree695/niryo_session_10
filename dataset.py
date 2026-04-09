"""
Dataset management utilities for training image curation.
Handles directory structure, sample persistence, and class balance assessment.
"""

import os
import time
from typing import List

import cv2
import numpy as np

from config import DATASET_DIR, DICE_FACES


def ensure_dataset_dirs(dataset_dir: str = DATASET_DIR) -> None:
    """Create standard dataset folder hierarchy, one subdirectory per class.

    Estructura resultante:
        dataset/
            9/
            10/
            J/
            Q/
            K/
            A/
    """
    for face in DICE_FACES:
        path = os.path.join(dataset_dir, face)
        os.makedirs(path, exist_ok=True)
    print(f"[DATASET] Directorios creados en {dataset_dir}")


def save_crop(crop: np.ndarray, label: str, dataset_dir: str = DATASET_DIR) -> str:
    """Persist labeled dice crop image with millisecond timestamp filename.
    
    Raises:
        ValueError: if label not in valid dice faces.
    """
    if label not in DICE_FACES:
        raise ValueError(f"Label '{label}' no valido. Opciones: {DICE_FACES}")

    folder = os.path.join(dataset_dir, label)
    os.makedirs(folder, exist_ok=True)

    timestamp = int(time.time() * 1000)
    filename = f"{label}_{timestamp}.png"
    filepath = os.path.join(folder, filename)

    cv2.imwrite(filepath, crop)
    return filepath


def get_dataset_summary(dataset_dir: str = DATASET_DIR) -> dict:
    """Report sample statistics across class labels. Useful for imbalance detection."""
    summary = {}
    for face in DICE_FACES:
        folder = os.path.join(dataset_dir, face)
        if os.path.isdir(folder):
            count = len([f for f in os.listdir(folder) if f.endswith(".png")])
        else:
            count = 0
        summary[face] = count
    return summary


def capture_dataset_mode(picker, detect_fn, workspace_corners: np.ndarray) -> None:
    """Interactive capture loop: acquire frame, detect objects, request user labels."""
    ensure_dataset_dirs()

    print("\n[DATASET] Modo captura activado")
    print(f"  Caras validas: {DICE_FACES}")
    print("  Escribe la cara del dado para etiquetar, 's' para skip, 'q' para salir")

    frame = picker.capture_frame()
    objects = detect_fn(frame, workspace_corners)

    if not objects:
        print("[DATASET] No se detectaron objetos. Asegurate de que hay dados en el workspace.")
        return

    for i, obj in enumerate(objects, start=1):
        x, y, w, h = obj["bounding_rect"]
        crop = frame[y:y+h, x:x+w]

        cv2.imshow("Crop para etiquetar", crop)
        cv2.waitKey(1)

        label = input(f"  [{i}/{len(objects)}] Etiqueta para este dado (cara): ").strip().upper()

        if label == "Q":
            print("[DATASET] Saliendo del modo captura")
            break
        if label == "S":
            print("  Saltado")
            continue
        if label not in DICE_FACES:
            print(f"  Label '{label}' no valido, saltando")
            continue

        filepath = save_crop(crop, label)
        print(f"  Guardado: {filepath}")

    cv2.destroyWindow("Crop para etiquetar")

    summary = get_dataset_summary()
    print("\n[DATASET] Resumen actual:")
    for face, count in summary.items():
        print(f"  {face}: {count} imagenes")
