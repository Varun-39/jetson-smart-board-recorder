import cv2
import numpy as np
import config

class BoardProcessor:
    def __init__(self):
        pass

    def detect_and_warp(self, frame):
        """Finds the largest 4-point contour and warps to 1280x720."""
        # Grayscale -> GaussianBlur -> Canny
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Approximate to 4 points
        if contours:
            # Sort contours by area in descending order
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for cnt in contours[:5]:
                # Mathematical Vision Fix: Use convexHull before approxPolyDP
                hull = cv2.convexHull(cnt)
                peri = cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
                if len(approx) == 4:
                    return self._warp_perspective(frame, approx)
        
        # Safety Fallback: Return original frame if no valid board detected
        print("[WARNING] Could not detect a valid 4-point board contour. Falling back to raw frame.")
        return frame

    def _warp_perspective(self, frame, approx):
        pts = approx.reshape(4, 2)
        
        # Ordering points
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # Top-left
        rect[2] = pts[np.argmax(s)] # Bottom-right
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # Top-right
        rect[3] = pts[np.argmax(diff)] # Bottom-left
        
        # Destination canvas
        dst = np.array([
            [0, 0],
            [1280 - 1, 0],
            [1280 - 1, 720 - 1],
            [0, 720 - 1]
        ], dtype="float32")
        
        # Transform (using cv2.INTER_CUBIC to preserve text edges)
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(frame, M, (1280, 720), flags=cv2.INTER_CUBIC)
        return warped

    def enhance_for_reading(self, warped_frame, clip_limit=config.CLIPPING_LIMIT):
        """Enhances contrast for reading using LAB CLAHE."""
        # Convert to LAB
        lab = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge back and convert to BGR
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return enhanced
