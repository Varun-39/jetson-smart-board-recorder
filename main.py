import cv2
import time
import os
import glob
from skimage.metrics import structural_similarity as ssim

def get_next_filename(capture_dir="captures"):
    """Finds the next sequential page number based on existing files."""
    os.makedirs(capture_dir, exist_ok=True)
    existing = glob.glob(os.path.join(capture_dir, "page_*.jpg"))
    if not existing:
         return 1
    nums = []
    for f in existing:
        try:
             name = os.path.basename(f)
             num_str = name.split('_')[1].split('.')[0]
             nums.append(int(num_str))
        except ValueError:
             pass
    return max(nums) + 1 if nums else 1

def preprocess(frame):
    """Resize to 640x360 and convert to grayscale for fast SSIM."""
    small = cv2.resize(frame, (640, 360))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return gray

def main():
    # 1. INITIALIZE (GStreamer pipeline for Jetson Nano)
    pipeline = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=21/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink drop=true sync=false"
    )
    print("Opening CSI camera...")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("Error: Could not open CSI camera. Verify nvarguscamerasrc is working.")
        # Fallback to standard webcam for local testing if needed
        # cap = cv2.VideoCapture(0)
        return

    capture_dir = "captures"
    page_num = get_next_filename(capture_dir)

    print("Warming up camera...")
    for _ in range(15):
        cap.read()
        time.sleep(0.05)
    
    # 2. REFERENCE
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab initial frame.")
        return
        
    reference_frame = preprocess(frame)
    previous_frame = reference_frame.copy()

    # State variables
    state = "IDLE"
    last_check_time = time.time()
    stability_start_time = None
    saved_ref_ssim = 1.0
    saved_delta_ssim = 1.0

    print("Started Smart Board Change Detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        current_time = time.time()
        
        # 3. LOOP - Process every 0.5 seconds
        if current_time - last_check_time >= 0.5:
            current_frame_small = preprocess(frame)
            
            # SSIM between 'current_frame' and 'reference_frame'
            ref_ssim, _ = ssim(current_frame_small, reference_frame, full=True)
            
            # Monitor 'delta_SSIM' (SSIM between Frame_t and Frame_t-1)
            delta_ssim, _ = ssim(current_frame_small, previous_frame, full=True)
            
            # CHANGE DETECTED LOGIC
            if ref_ssim < 0.85:
                if state == "IDLE":
                    print(f"[CHANGE] Detected (SSIM={ref_ssim:.2f}) — waiting for stability...")
                    state = "CHANGE DETECTED"

                # STABILITY LOGIC
                if delta_ssim > 0.98:
                    if state == "CHANGE DETECTED":
                        state = "STABILIZING"
                        stability_start_time = current_time
                    elif state == "STABILIZING":
                        elapsed_stable = current_time - stability_start_time
                        
                        # STABLE FOR 4 SECONDS -> SAVE
                        if elapsed_stable >= 4.0:
                            filename = os.path.join(capture_dir, f"page_{page_num:03d}.jpg")
                            cv2.imwrite(filename, frame)
                            print(f"[SAVED] {os.path.basename(filename)}")
                            
                            # Update reference frame
                            reference_frame = current_frame_small.copy()
                            page_num += 1
                            
                            # Reset timers
                            state = "IDLE"
                            stability_start_time = None
                else:
                    # Scene is moving, reset stability timer
                    if state == "STABILIZING":
                        state = "CHANGE DETECTED"
                        stability_start_time = None
            else:
                # REVERT LOGIC
                if ref_ssim >= 0.90 and state != "IDLE":
                    print(f"[RESET] Scene reverted — debounce reset")
                    state = "IDLE"
                    stability_start_time = None

            saved_ref_ssim = ref_ssim
            saved_delta_ssim = delta_ssim
            previous_frame = current_frame_small.copy()
            last_check_time = current_time

        # Update HUD
        cv2.putText(frame, f"State: {state}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if state != "IDLE" else (0, 255, 0), 2)
        cv2.putText(frame, f"Ref SSIM: {saved_ref_ssim:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Delta SSIM: {saved_delta_ssim:.2f}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if state == "STABILIZING" and stability_start_time:
            countdown = max(0, 4.0 - (time.time() - stability_start_time))
            cv2.putText(frame, f"Saving in: {countdown:.1f}s", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

        cv2.imshow("Smart Board Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
