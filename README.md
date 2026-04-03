# 🧠 Smart Board Capture (SBC) & Search System

**Author:** Varun Bothra (Batch 2X15)  
**Target Architecture:** NVIDIA Jetson Nano (4GB)

An Edge-AI system that leverages lightweight Computer Vision to automatically monitor whiteboard activity, intelligently detect contextual changes, rectify spatial perspective, dynamically enhance text readability, and provide a seamless, searchable local web dashboard for archiving classroom semantics without cloud dependencies.

---

## 🏗 High-Level Architecture

The SBC pipeline is meticulously engineered for continuous, real-time edge processing:

1. **Hardware Acquisition (`GStreamer`)**: High-framerate ingestion of NV12 visual data natively via the CSI hardware interface.
2. **Context Stability (`SSIM`)**: Constant tracking of continuous structural deviations over time.
3. **Spatial Rectification (`Homography`)**: Canny edge geometry mapping dynamically flattens the board.
4. **Contrast Engineering (`CLAHE`)**: LAB space luminance isolation strips shadows and pronounces text.
5. **Background OCR (`Tesseract Queue`)**: Thread-safe background execution parses semantic data sequentially.
6. **Web Dashboard (`Flask + Vanilla JS`)**: A decoupled daemon process serves interactive JSON mappings asynchronously to connected clients.

---

## ✨ Key Features

- **Intelligent Change Detection**: Contextual tracking via Structural Similarity (SSIM) effectively mitigates "teacher occlusion" by demanding a 4-second continuous stability window before concluding an interaction.
- **Real-time Rectification**: Autonomous edge extraction and perspective-warping transformations guarantee a perfectly flat, top-down 1280x720 capture regardless of camera placement.
- **Asynchronous OCR**: A strict singleton Queue prevents memory deadlocking, feeding PyTesseract images exactly one at a time while safely buffering heavy capture bursts.
- **NVIDIA-Themed Web UI**: A dark-mode Dashboard equipped with live Vanilla JS DOM polling, robust inline full-text search, and a one-click PDF Compiler.

---

## ⚡ Performance Optimization

Given the restrictive 4GB unified memory limit of the Jetson architecture, absolute efficiency was prioritized:
- **SSIM Downscaling**: `skimage.metrics` evaluations occur on aggressively downscaled 640x360 greyscale matrices, driving computational requirements down exponentially while maintaining mathematical integrity.
- **Subprocess Isolation**: The Web UI functions out-of-core over a separate `multiprocessing` architecture, entirely decoupled from the Python Global Interpreter Lock (GIL) tying up the primary OpenCV vision loop.
- **Interpolation Tuning**: Used computationally inexpensive but structurally sharp `cv2.INTER_CUBIC` metrics specifically targeting typographic geometries over heavier alternatives.

---

## 🧰 Hardware Requirements

- **Processor**: NVIDIA Jetson Nano (4GB variant explicitly recommended for concurrent CV/OCR workloads).
- **Vision Interface**: IMX219 (or compliant) CSI Camera module.
- **Power Delivery**: 5V/4A Barrel Jack Power Supply (to prevent brown-outs under heavy GPU inference loads).

---

## ⚙️ Software Installation

### System Dependencies
The system relies on localized binaries to circumvent cloud latencies:
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr libtesseract-dev
```

### Python Dependencies
Using a virtual environment is recommended to guarantee version compatibility:
```bash
pip install opencv-python flask pytesseract fpdf2 scikit-image
```

---

## 🚀 How to Run

1. Clone or transition into the repository directory.
2. Initialize the application matrix:
   ```bash
   python3 main.py
   ```
3. Connect a device to the identical subnet, open a web browser, and access the dashboard:
   ```text
   http://<jetson-ip>:5000/
   ```
4. Press `q` within the primary camera HUD to trigger a safe system teardown and terminate all threading artifacts securely.

---

## 📂 Project Structure

```text
├── .gsd/                 # Generative specification tracking records
├── captures/             # Automated payload destination (.jpg, .txt)
├── app.py                # Flask WSGI architecture & PDF exporter
├── main.py               # Application entrypoint & OpenCV runtime
├── ocr_engine.py         # Thread-safe Singleton for PyTesseract
├── README.md             # Theoretical baseline and documentation
├── vision_utils.py       # BoardProcessor algorithms (WARP/CLAHE)
└── templates/
    └── index.html        # Front-end NV-themed Dashboard
```

---

## 🔮 Future Scope

- **LMS Integration**: Seamless APIs designed to hook endpoints into Canvas or Google Classroom systems at lecture termination.
- **Deep Learning Tracking**: Introducing YOLOv8 TensorRT engines targeting hands/markers to implement precise spatial gesture triggers, superseding passive 4s SSIM continuity evaluations.
