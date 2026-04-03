# SPEC.md — Project Specification

> **Status**: `FINALIZED`

## Vision
A Jetson Nano-optimized smart board change detector that autonomously saves clean JPEG snapshots after recognizing stable, written content without requiring manual intervention, web servers, or cloud dependencies.

## Goals
1. Continuously monitor a board via CSI camera and detect meaningful changes using SSIM.
2. Ensure the board is stable (user has stopped writing/moved away) for 4 continuous seconds before capturing.
3. Save stable board states sequentially to local disk (`captures/page_NNN.jpg`).
4. Optimize for Jetson Nano by downscaling frames for processing while preserving original resolution captures.

## Non-Goals (Out of Scope)
- OCR (Optical Character Recognition)
- Flask or any web server integration
- Perspective warping or auto-cropping
- Cloud sync

## Users
- Educators or presenters who want a hands-free, automated way to digitize their physical whiteboard or blackboard notes as they teach.

## Constraints
- **Hardware**: Jetson Nano (requires specific GStreamer pipeline for CSI camera `nvarguscamerasrc`).
- **Software**: Single `main.py` Python script using OpenCV and optionally scikit-image.
- **Performance**: Must remain lightweight to prevent CPU throttling on edge hardware (resize processing frames to 640x360).

## Success Criteria
- [ ] Opens camera and displays live preview window with SSIM and Status text overlay.
- [ ] Successfully detects initial change against reference frame (SSIM < 0.85).
- [ ] Accurately determines scene stability (delta_SSIM > 0.98 between consecutive frames) for 4 seconds.
- [ ] Saves full-resolution captured frame when stable, updates reference, and increments counter.
- [ ] Resets timer gracefully if scene destabilizes before the 4-second mark.
- [ ] Cleanly quits and releases hardware on 'q' press.
