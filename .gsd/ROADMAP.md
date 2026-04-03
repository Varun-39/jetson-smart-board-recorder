# ROADMAP.md

> **Current Phase**: Phase 1
> **Milestone**: v1.0

## Must-Haves (from SPEC)
- [ ] GStreamer Jetson Nano pipeline initialization
- [ ] 0.5s loop with SSIM comparison (vs reference and vs previous)
- [ ] 4-second stability timer logic
- [ ] Sequential image saving to `captures/`
- [ ] HUD overlay (status text and SSIM) and clean exit

## Phases

### Phase 1: Core Loop Implementation
**Status**: ⬜ Not Started
**Objective**: Write the `main.py` script containing the computer vision loop, stability tracking, and file saving.
**Requirements**: REQ-01, REQ-02, REQ-03, REQ-04

### Phase 2: Hardware Testing (User)
**Status**: ⬜ Not Started
**Objective**: User runs `main.py` on the Jetson Nano physically and verifies camera pipeline and stability timeouts.
