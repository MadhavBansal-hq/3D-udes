# Namma Space: phone video to walkable 3D web scene

Techfest 2026-27 (IIT Bombay), Round 1. One smartphone video of a room becomes a photorealistic
Gaussian-splat scene you can walk through in a browser with WASD and the mouse. One command, one
laptop GPU, 8 to 17 minutes, no manual modelling, all free and open-source tools.

```
video ─► sharp keyframes ─► ALIKED + LightGlue ─► COLMAP global SfM ─┬─► outlier filter ─► undistort ─► Brush splat ─► web viewer
                                                    pose spikes? ─────┘ (incremental SfM fallback)
```

## Setup (Windows, NVIDIA GPU)

1. Download into `tools/`:
   - [COLMAP 4.2.1](https://github.com/colmap/colmap/releases/tag/4.2.1) `colmap-x64-windows-cuda.zip` → `tools/colmap/`
   - [Brush v0.3.0](https://github.com/ArthurBrussee/brush/releases/tag/v0.3.0) `brush-app-x86_64-pc-windows-msvc.zip` → `tools/brush/`
2. Python environment (3.13). The `nvidia-*` wheels give COLMAP's ONNX models GPU access without a CUDA toolkit:
   ```
   py -3.13 -m venv .venv
   .venv\Scripts\pip install numpy opencv-python-headless nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12
   ```

## Run

```
.venv\Scripts\python pipeline\run.py data\room\video.mp4 --fps 4 --max-side 1280
```

Output: `data/room/splat/splat.ply` and `data/room/splat/cameras.json`. Each stage caches its result;
`--force` redoes everything. `--steps 7000` gives a faster preview, `--mapper global|incremental`
overrides the automatic pose check, `--from images` takes a folder of photos instead of a video.

## View

```
.venv\Scripts\python -m http.server 8080
```

Open `http://localhost:8080/viewer/index.html?splat=../data/room/splat/splat.ply`.

| Key | Action |
| --- | --- |
| Click, then mouse | Look (or drag / arrow keys where pointer lock is unavailable) |
| W A S D | Move |
| Q / E | Down / up |
| Scroll | Walking speed |
| `[` / `]` | Jump between capture camera positions |

## Capture SOP (short form)

Lock exposure and focus, lights on, nobody in the room, mirrors covered. Walk slowly in two laps
(perimeter facing in, middle facing out), about 1.5 m from walls. Never pivot on the spot; walk an arc.
Don't fill the frame with blank wall. Copy the original file by USB or cloud drive, never via a
messaging app.

## Optional: gsplat comparison trainer

`pipeline/shims/fused_ssim.py` is a PyTorch stand-in for the `fused-ssim` CUDA extension so gsplat's
`simple_trainer.py` runs on Windows without nvcc (Python 3.10, torch 2.4.1+cu124, gsplat 1.5.3 wheel;
put `pipeline/shims` on `PYTHONPATH`).

## Credits

COLMAP (BSD), ALIKED (BSD-3-Clause), LightGlue (Apache-2.0), ONNX Runtime (MIT), Brush (Apache-2.0),
gsplat (Apache-2.0), OpenCV (Apache-2.0), NumPy (BSD-3-Clause), Spark (MIT), three.js (MIT).
