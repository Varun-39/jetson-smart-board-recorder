import cv2
import time
import os
import glob
import multiprocessing
import config
import numpy as np
from skimage.metrics import structural_similarity as ssim

from vision_utils import BoardProcessor, CameraThread
from ocr_engine import OCRWorker
from app import run_flask
try:
    from trt_engine import TensorRTInference
except ImportError:
    TensorRTInference = None
    print("[WARNING] Could not import TensorRT environment. Testing locally without AI tracking.")

def get_next_filename(capture_dir="captures"):
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
    small = cv2.resize(frame, (640, 360))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return gray

def get_person_bboxes(yolo_output, conf_thres=0.5):
    # Standard YOLOv8 output shape [1, 84, 8400]
    preds = yolo_output[0] 
    person_confs = preds[4, :]
    valid_indices = np.where(person_confs > conf_thres)[0]
    
    bboxes = []
    for idx in valid_indices:
        x, y, w, h = preds[0:4, idx]
        # Map 640x640 back to 1280x720 frame
        sx = 1280 / 640.0
        sy = 720 / 640.0
        x1 = int((x - w/2) * sx)
        y1 = int((y - h/2) * sy)
        x2 = int((x + w/2) * sx)
        y2 = int((y + h/2) * sy)
        bboxes.append( (x1, y1, x2, y2) )
    return bboxes

def check_intersection(person_bbox, board_points):
    if board_points is None:
        return False
    bx, by, bw, bh = cv2.boundingRect(board_points)
    board_rect = (bx, by, bx+bw, by+bh)
    
    px1, py1, px2, py2 = person_bbox
    intersect_x0 = max(board_rect[0], px1)
    intersect_y0 = max(board_rect[1], py1)
    intersect_x1 = min(board_rect[2], px2)
    intersect_y1 = min(board_rect[3], py2)
    
    return (intersect_x1 > intersect_x0 and intersect_y1 > intersect_y0)

def main():
    processor = BoardProcessor()
    ocr_worker = OCRWorker()
    ocr_worker.start()

    print("Launching Web Dashboard on http://localhost:5000")
    web_process = multiprocessing.Process(target=run_flask, daemon=True)
    web_process.start()

    # Load TRT Engine
    if os.path.exists("yolov8n.engine"):
        trt_engine = TensorRTInference("yolov8n.engine")
    else:
        trt_engine = None
        print("[WARNING] yolov8n.engine not found. Run export_trt.sh first. Person detection disabled.")

    pipeline = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=21/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink drop=true sync=false"
    )
    print("Opening CSI camera asynchronously...")
    cap = CameraThread(pipeline)

    capture_dir = "captures"
    page_num = get_next_filename(capture_dir)

    print("Warming up camera...")
    time.sleep(2)
    
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to grab initial frame.")
        return
        
    reference_frame = preprocess(frame)
    previous_frame = reference_frame.copy()

    state = "IDLE"
    last_check_time = time.time()
    stability_start_time = None
    saved_ref_ssim = 1.0

    print("Started Smart Board Recorder (Hybrid Logic). Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.1)
            continue
            
        current_time = time.time()
        
        if current_time - last_check_time >= config.POLL_INTERVAL:
            current_frame_small = preprocess(frame)
            
            # TRT Inference
            person_intersecting = False
            if trt_engine:
                # Prepare image for YOLO (640x640)
                yolo_img = cv2.resize(frame, (640, 640))
                yolo_img = yolo_img.transpose((2, 0, 1)).astype(np.float32) / 255.0
                yolo_img = np.expand_dims(yolo_img, axis=0)
                
                preds = trt_engine.infer(yolo_img)
                bboxes = get_person_bboxes(preds, conf_thres=0.5)
                
                # Approximate board contour
                _, board_approx = processor.detect_and_warp(frame)
                
                for bbox in bboxes:
                    if check_intersection(bbox, board_approx):
                        person_intersecting = True
                        break
                        
                # Draw bboxes for debug
                for bbox in bboxes:
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

            # Hybrid Check Logic
            if person_intersecting:
                state = "STATE_WRITING"
                stability_start_time = None
            else:
                ref_ssim, _ = ssim(current_frame_small, reference_frame, full=True)
                saved_ref_ssim = ref_ssim
                delta_ssim, _ = ssim(current_frame_small, previous_frame, full=True)

                if state == "STATE_WRITING" or state == "STATE_STABILIZING":
                    if state == "STATE_WRITING":
                        state = "STATE_STABILIZING"
                        stability_start_time = current_time
                    
                    if delta_ssim > config.STABILITY_THRESHOLD:
                        elapsed = current_time - stability_start_time
                        if elapsed >= 3.0: 
                            # Stable for 3 seconds outside ROI
                            state = "STATE_TRIGGER"
                    else:
                        stability_start_time = current_time
                        
                elif state == "STATE_TRIGGER":
                    print("[STATE] Triggering OCR Capture!")
                    filename = os.path.join(capture_dir, f"page_{page_num:03d}.jpg")
                    warp_frame, _ = processor.detect_and_warp(frame)
                    if warp_frame is not None:
                        enhanced = processor.enhance_for_reading(warp_frame)
                        cv2.imwrite(filename, enhanced)
                        # Zero disk tuple
                        ocr_worker.add_to_queue((filename, enhanced))
                        
                        reference_frame = current_frame_small.copy()
                        page_num += 1
                    
                    state = "IDLE"
                    stability_start_time = None

            previous_frame = current_frame_small.copy()
            last_check_time = current_time

        # Update HUD
        cv2.putText(frame, f"State: {state}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if state != "IDLE" else (0, 255, 0), 2)
        cv2.putText(frame, f"Ref SSIM: {saved_ref_ssim:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow("Smart Board Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Shutting down OCR thread...")
    ocr_worker.add_to_queue(None)
    ocr_worker.join()

    cap.release()
    cv2.destroyAllWindows()
    
    print("Shutting down Web Dashboard...")
    if web_process.is_alive():
        web_process.terminate()
        web_process.join()
    print("System safely shut down.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
