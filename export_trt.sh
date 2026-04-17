#!/bin/bash

# export_trt.sh
# Standard Jetson script to export ONNX to TRT with swap verification

echo "Checking swap memory before TensorRT Engine compilation..."
SWAP_TOTAL=$(free -m | awk '/Swap/ {print $2}')

if [ "$SWAP_TOTAL" -lt 4000 ]; then
    echo "================================================================"
    echo " [ERROR] Insufficient Swap Space (${SWAP_TOTAL}MB) Detected"
    echo "================================================================"
    echo "Compiling a YOLO model on a 4GB Jetson Nano frequently triggers"
    echo "Out of Memory (OOM) crashes without an expanded swap space."
    echo ""
    echo "Please configure at least 4GB (preferably 8GB) of swap."
    echo "Run the following commands as root:"
    echo "  sudo fallocate -l 8G /swapfile"
    echo "  sudo chmod 600 /swapfile"
    echo "  sudo mkswap /swapfile"
    echo "  sudo swapon /swapfile"
    echo ""
    echo "Then re-run this script."
    exit 1
fi

echo "Swap space is sufficient (${SWAP_TOTAL}MB)."
echo "Starting TensorRT Engine Compilation..."

# Using Jetson native trtexec
/usr/src/tensorrt/bin/trtexec --onnx=yolov8n.onnx \
                              --saveEngine=yolov8n.engine \
                              --fp16 \
                              --workspace=2048

echo "Compilation Complete! The engine is saved as yolov8n.engine."
