import threading
import queue
import time
import os
import cv2
import easyocr

class OCRWorker(threading.Thread):
    def __init__(self):
        super().__init__()
        self.q = queue.Queue()
        self.daemon = True # Thread terminates when main program exits
        self.active_tasks = 0
        self.lock = threading.Lock()
        
        # Initialize EasyOCR reader (will be instantiated on first run or in thread)
        self.reader = None

    def add_to_queue(self, task):
        """Enqueue (image_path, image_np) to be processed, or None to shutdown."""
        self.q.put(task)

    def is_processing(self):
        """Check if OCR is currently running or if there are images in the queue."""
        with self.lock:
            return self.active_tasks > 0 or not self.q.empty()

    def run(self):
        """Singleton loop that waits for images and processes them sequentially."""
        print("[OCR] Initializing EasyOCR Engine (FP16)...")
        # Initialize EasyOCR reader here to ensure it's in the same thread
        self.reader = easyocr.Reader(['en'], gpu=True)
        print("[OCR] EasyOCR Engine Ready.")
        
        while True:
            task = self.q.get()
            
            # Sentinel check for graceful shutdown
            if task is None:
                self.q.task_done()
                break
                
            image_path, image_np = task
            
            with self.lock:
                self.active_tasks += 1
                
            try:
                import sqlite3
                
                # 1. Start Zero-Disk OCR directly from numpy array
                # easyocr readtext returns a list of tuples (bbox, text, prob)
                results = self.reader.readtext(image_np)
                
                text = "\n".join([res[1] for res in results])
                
                # Calculate average confidence for the DB
                prob = sum([res[2] for res in results]) / len(results) if len(results) > 0 else 0.0
                
                # Check for blank OCR reads before writing
                if not text.strip():
                    print(f"[OCR WARNING] No text detected for {image_path}")
                else:
                    # Replace standard file writing with an SQLite INSERT
                    conn = sqlite3.connect('app.db', timeout=15.0)
                    c = conn.cursor()
                    c.execute("""
                        INSERT INTO board_captures (image_path, raw_text, confidence_score) 
                        VALUES (?, ?, ?)
                    """, (image_path, text, prob))
                    conn.commit()
                    conn.close()
                    print(f"[OCR] Processed and logged text to SQLite for {image_path}")
                    
            except Exception as e:
                print(f"[OCR ERROR] Failed to process {image_path}: {e}")
            finally:
                with self.lock:
                    self.active_tasks -= 1
                self.q.task_done()
