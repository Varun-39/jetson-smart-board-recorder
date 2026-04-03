import threading
import queue
import time
import os
import pytesseract

class OCRWorker(threading.Thread):
    def __init__(self):
        super().__init__()
        self.q = queue.Queue()
        self.daemon = True # Thread terminates when main program exits
        self.active_tasks = 0

    def add_to_queue(self, image_path):
        """Enqueue an image to be processed by Tesseract."""
        self.q.put(image_path)

    def is_processing(self):
        """Check if OCR is currently running or if there are images in the queue."""
        return self.active_tasks > 0 or not self.q.empty()

    def run(self):
        """Singleton loop that waits for images and processes them sequentially."""
        while True:
            image_path = self.q.get()
            if image_path is None:
                break
                
            self.active_tasks += 1
            try:
                # Process the image with pytesseract
                text = pytesseract.image_to_string(image_path)
                
                # Save to matching .txt file
                base_name = os.path.splitext(image_path)[0]
                txt_path = f"{base_name}.txt"
                
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                print(f"[OCR] Background processed and saved text to {os.path.basename(txt_path)}")
            except Exception as e:
                print(f"[OCR ERROR] Failed to process {image_path}: {e}")
            finally:
                self.active_tasks -= 1
                self.q.task_done()
