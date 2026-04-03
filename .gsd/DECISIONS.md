# DECISIONS.md
> Architecture Decision Records (ADRs)

| ID | Date | Context | Decision | Consequences |
|----|------|---------|----------|--------------|
| 001 | 2026-04-03 | Core CV loop on Jetson Nano | Use SSIM with 640x360 downscaling | Reduces CPU usage on Nano, prevents throttling while allowing HD 1280x720 saves |
| 002 | 2026-04-03 | Camera interface | GStreamer `nvarguscamerasrc` | Standard robust pipeline for CSI cameras on Jetson |
| 003 | 2026-04-03 | Stability check | `delta_SSIM > 0.98` for 4 seconds | Prevents capturing frames while users are actively blocking the board |
