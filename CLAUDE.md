# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application (multi-page GUI)
python main.py

# Run tests
python test_simple.py
```

## Current Version: v3.2.10 (stable-v1)

## Architecture v2.0 - Multi-Page Design

**Entry point**: `main.py` launches a PyQt5 GUI application with three pages.

### Pages

1. **图片预处理 (Preprocess Page)** - `preprocess_page.py`
   - **Step 1: 噪音提取** - Extract noise profile from single X-ray image
   - **Step 2: 数据集生成** - Generate synthetic noisy/clean patch pairs using extracted noise parameters
   - Noise model: Poisson(λ) + AWGN(σ) + Gaussian Blur(σ=1)
   - Output: train/test/validation splits with metadata

2. **算法训练 (Training Page)** - `training_page.py`
   - Load preprocessed datasets
   - Train neural denoising models (PyTorch)
   - Export models to ONNX format
   - Real-time training monitoring

3. **降噪与超分辨率 (Denoise & SR Page)** - `denoise_app.py`
   - Two-step workflow: Denoise -> Super-Resolution
   - Multiple denoising algorithms
   - Quality metrics (PSNR, SSIM, MSE)

### Core modules:
- `main_window.py` - Main window with sidebar navigation and QStackedWidget
- `image_processor.py` - Central `ImageProcessor` class managing image loading, denoising, super-resolution, and saving
- `denoise_algorithms.py` - Denoising functions (`hybrid_denoise`, `adaptive_denoise`, `non_local_means_denoise`, `bilateral_filter_denoise`, `wavelet_denoise`, `bm3d_denoise`, `anisotropic_diffusion_denoise`, `iterative_reconstruction_denoise`) with multi-depth support (uint8/uint16/float)
- `super_resolution.py` - Super-resolution reconstruction (bicubic, lanczos, edge-preserving upscaling with contrast enhancement)
- `neural_denoise.py` - Optional ONNX-based neural denoiser with fallback to bilateral filter
- `metrics.py` - PSNR, SSIM, MSE calculation with multi-depth normalization
- `training_page.py` - PyTorch-based neural network training with ONNX export
- `preprocess_page.py` - Two-step workflow: noise extraction from single image, then synthetic dataset generation

**Key patterns**:
- All denoising functions normalize images to float64 [0,1] before processing, then denormalize to original dtype
- Processing runs in a background `QThread` to keep GUI responsive
- Each algorithm has fallback chains (NLM → Bilateral → Gaussian → original) to handle edge cases
- Training uses patch-based processing with data augmentation

## Dependencies

- PyQt5 (GUI), OpenCV (image I/O), scikit-image (denoising algorithms), NumPy, SciPy
- Optional: onnxruntime (neural denoising), PyInstaller (building executables)

## Building Executable

```bash
python build_executable.py
```

Creates standalone executable via PyInstaller in `dist/` directory.
