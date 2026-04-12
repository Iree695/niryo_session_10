"""
Configuracion centralizada del sistema Niryo Poker Dice Vision.

Contiene todos los parametros de calibracion del robot, deteccion de vision
y constantes para los nuevos modulos de clasificacion CNN y evaluacion de jugadas.
"""

# ============================================================================
# HARDWARE CONFIGURATION
# ============================================================================

ROBOT_IP = "172.16.125.35"
# Reachable workspace pose with camera aligned to work surface
SCANNING_POSITION = (0, -0.148, 0.206, -2.556, 1.523, 2.082)

# Cartesian workspace limits in robot frame (calibrated via pointer probe).
# Mapping: image-space (TL, TR, BR, BL) to robot coordinates (physical corners)
LIMIT_1 = (-0.08, -0.329, 0.129, -0.523, 1.531, -2.056)  # TL
LIMIT_2 = (-0.082, -0.163, 0.129, 3.046, 1.511, 1.506)  # BL
LIMIT_3 = (0.082, -0.153, 0.129, -2.957, 1.538, 1.745)   # TR
LIMIT_4 = (0.082, -0.328, 0.129, 2.289, 1.529, 0.68)   # BR

# Gripper approach and contact depths (meters) for precision grasping
APPROACH_Z = 0.164
PICK_Z = 0.131

# ============================================================================
# VISION PROCESSING
# ============================================================================

# Pixel-space workspace bounds (used when fiducial markers are not detected)
FALLBACK_WORKSPACE_PIXEL_BOUNDS = {
    "x_min": 100,
    "x_max": 540,
    "y_min": 80,
    "y_max": 400,
}

# Morphological constraints for object detection in contours
MIN_AREA = 1000
MAX_AREA = 150000
EPSILON_FACTOR = 0.03

# HSV color segmentation ranges for object classification.
# Red spans the hue wrap-around boundary (0-10, 160-180 in HSV)
COLOR_RANGES = {
    "RED": [
        {"lower": (0, 70, 50), "upper": (10, 255, 255)},
        {"lower": (160, 70, 50), "upper": (180, 255, 255)},
    ],
    "GREEN": [{"lower": (35, 70, 50), "upper": (85, 255, 255)}],
    "BLUE": [{"lower": (85, 60, 50), "upper": (130, 255, 255)}],
}

# ============================================================================
# NEURAL NETWORK
# ============================================================================

DICE_FACES = ["9", "10", "J", "Q", "K", "A"]
ONNX_MODEL_PATH = "dice_classifier.onnx"
DATASET_DIR = "dataset/"
CNN_INPUT_SIZE = (64, 64)
CNN_CONFIDENCE_THRESHOLD = 0.7
