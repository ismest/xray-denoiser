# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py

# Run tests
python test_simple.py
```

## Architecture

**Entry point**: `main.py` launches a PyQt5 GUI application (`DenoiseApp` in `denoise_app.py`).

**Core modules**:
- `image_processor.py` - Central `ImageProcessor` class managing image loading, denoising, super-resolution, and saving
- `denoise_algorithms.py` - Denoising functions (`hybrid_denoise`, `adaptive_denoise`, `non_local_means_denoise`, `bilateral_filter_denoise`, `wavelet_denoise`) with multi-depth support (uint8/uint16/float)
- `super_resolution.py` - Super-resolution reconstruction (bicubic, lanczos, edge-preserving upscaling with contrast enhancement)
- `neural_denoise.py` - Optional ONNX-based neural denoiser with fallback to bilateral filter
- `metrics.py` - PSNR, SSIM, MSE calculation with multi-depth normalization

**Key patterns**:
- All denoising functions normalize images to float64 [0,1] before processing, then denormalize to original dtype
- Processing runs in a background `QThread` to keep GUI responsive
- Each algorithm has fallback chains (NLM → Bilateral → Gaussian → original) to handle edge cases

## Dependencies

- PyQt5 (GUI), OpenCV (image I/O), scikit-image (denoising algorithms), NumPy, SciPy
- Optional: onnxruntime (neural denoising), PyInstaller (building executables)

## Building Executable

```bash
python build_executable.py
```

Creates standalone executable via PyInstaller in `dist/` directory.
