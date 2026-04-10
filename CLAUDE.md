# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application (multi-page GUI)
python main.py

# Run lightweight tests (no OpenCV needed)
python test_simple.py
```

## Architecture - Multi-Page Design

**Entry point**: `main.py` → `main_window.py` (`MainWindow`) hosts three pages in a `QStackedWidget` with a dark sidebar for navigation.

### Pages

| File | Chinese name | Purpose |
|------|-------------|---------|
| `preprocess_page.py` | 图片预处理 | Step 1: noise extraction from a single X-ray; Step 2: synthetic noisy/clean patch-pair dataset generation |
| `noise2void_page.py` | Noise2Void | Self-supervised training directly from a single noisy image (no clean reference needed) |
| `denoise_app.py` | 降噪与超分辨率 | Two-step workflow: Denoise → Super-Resolution, with quality metrics |
| `training_page.py` | 算法训练 | Supervised training from a preprocessed dataset; exports ONNX |

### Core modules

- `image_processor.py` — `ImageProcessor` class: load/denoise/super-resolve/save; orchestrates `denoise_algorithms`, `neural_denoise`, `super_resolution`, `metrics`
- `denoise_algorithms.py` — all classical denoising functions; every function normalizes to float64 [0,1] then denormalizes back to the original dtype
- `super_resolution.py` — bicubic, lanczos, edge-preserving upscaling
- `neural_denoise.py` — `NeuralDenoiser`: ONNX inference wrapper; falls back to bilateral filter if ONNX unavailable
- `metrics.py` — PSNR, SSIM, MSE with multi-depth normalization
- `algorithm_config.py` — loads/saves `algorithm_config.json`; `get_denoise_algorithms()` / `get_sr_algorithms()` dynamically append trained models found on disk
- `algorithm_editor_dialog.py` — `DenoiseAlgorithmEditor` / `SRAlgorithmEditor` dialogs for enabling/disabling/renaming algorithms at runtime

### Trained model storage

After training, models are saved under:
```
integrated_model/
  denoise/<YYYYMMDD_HHMMSS>/
    best_denoiser.pth   # PyTorch weights
    denoiser.onnx       # ONNX export (used at inference)
    model_ready.marker  # presence = model is complete
  super_resolution/<YYYYMMDD_HHMMSS>/
    ...same structure...
```
`algorithm_config.py::_scan_integrated_models()` discovers these at runtime and surfaces them as selectable algorithms in the UI. `algorithm_config.json` persists user-level enable/disable and rename state for both built-in and trained algorithms.

### Key patterns

- **Chinese path workaround**: image loading uses `np.fromfile(path, dtype=np.uint8)` + `cv2.imdecode(...)` instead of `cv2.imread()` to handle non-ASCII paths.
- **Background processing**: every long-running operation runs in a `QThread` subclass that emits `progress(int, str)` and `finished(bool, str)` signals.
- **Fallback chains**: classical denoising degrades gracefully — NLM → Bilateral → Gaussian → original — when image is too small or a library is missing.
- **Dataset structure**: `NoiseDataset` in `training_page.py` supports both the current layout (`train/noisy`, `train/clean`) and the legacy `noisy_patches`/`clean_patches` layout.

## Dependencies

- **Required**: PyQt5, OpenCV, scikit-image, NumPy, SciPy, matplotlib
- **Training**: PyTorch (`TORCH_AVAILABLE` flag guards all imports)
- **Inference**: onnxruntime (`ONNX_AVAILABLE` flag)
- **Optional**: Pillow (better format support in `image_processor.py`), PyInstaller (executable build)

## Building Executable

```bash
python build_executable.py
```

Creates standalone executable via PyInstaller in `dist/`.
