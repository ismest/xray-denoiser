# Installation and Setup Guide

## Quick Start (Simple Mode)

The application includes a simplified version that works with minimal dependencies:

1. **Prerequisites**: Python 3.6+ with PyQt5
2. **Install PyQt5**:
   ```bash
   pip install PyQt5
   ```
3. **Run the simple version**:
   ```bash
   python simple_denoise.py
   ```

## Full Feature Mode

For advanced denoising algorithms and PSNR/SSIM metrics, install additional dependencies:

### Method 1: Using requirements.txt
```bash
pip install -r requirements.txt
python main.py
```

### Method 2: Manual installation
```bash
# Install core dependencies
pip install PyQt5 opencv-python scikit-image numpy scipy

# Run the full application
python main.py
```

## Building Executables

To create standalone executables that don't require Python installation:

### Prerequisites
```bash
pip install pyinstaller
```

### Build Process
```bash
# For simple version (recommended for compatibility)
python build_executable.py

# Or manually:
pyinstaller --onefile --windowed --name "X-ray-Denoiser" simple_denoise.py
```

## Platform-Specific Notes

### Windows
- Executables will be `.exe` files
- May require Visual C++ Redistributables for OpenCV

### macOS
- Executables will be `.app` bundles
- May need to grant permissions for file access

### Linux
- Executables are plain binary files
- Ensure required system libraries are installed

## Troubleshooting

### Common Issues

1. **"ModuleNotFoundError: No module named 'cv2'"**
   - Solution: Install OpenCV: `pip install opencv-python`

2. **"ImportError: DLL load failed" (Windows)**
   - Solution: Install Visual C++ Redistributables or use the simple version

3. **Application crashes on large images**
   - Solution: Use smaller images or increase system memory

4. **PyQt5 installation issues**
   - Solution: Try `pip install PyQt5==5.15.4` for better compatibility

### Fallback Strategy

If you encounter dependency issues, always use `simple_denoise.py` which only requires PyQt5 and provides basic functionality.

## System Requirements

- **Minimum**: Python 3.6, 2GB RAM, 100MB disk space
- **Recommended**: Python 3.8+, 4GB RAM, SSD storage
- **Supported OS**: Windows 7+, macOS 10.12+, Linux (most distributions)

## Testing Your Installation

1. Run the application
2. Load any image file (JPEG, PNG, BMP)
3. Apply denoising
4. Save the result

If all steps work, your installation is successful!

## Updating Dependencies

To update to the latest versions:
```bash
pip install --upgrade -r requirements.txt
```