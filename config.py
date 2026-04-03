# config.py
# Centralized Configuration Parameters for Smart Board Capture

# --- Vision Thresholds ---
SSIM_THRESHOLD = 0.85
REVERT_THRESHOLD = 0.90
STABILITY_THRESHOLD = 0.98

# --- Timing Constants ---
POLL_INTERVAL = 0.5
STABILITY_TIME = 4.0

# --- Tesseract OCR Settings ---
TESSERACT_CONFIG = '--oem 3 --psm 6'

# --- Enhancement Settings ---
CLIPPING_LIMIT = 2.0
