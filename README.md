# X-ray Image Denoiser & Super-Resolution

A cross-platform desktop application for denoising X-ray images with super-resolution reconstruction capabilities.

## Features

- **Two-Step Workflow**: Denoise → Super-Resolution for optimal image enhancement
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Multiple denoising algorithms**: Hybrid approach, Non-local Means, Bilateral, Wavelet, and Neural Network
- **Super-Resolution**: Multiple upscaling methods (Bicubic, Lanczos, Edge-Preserving)
- **Quantitative metrics**: PSNR, SSIM, MSE for both denoising and super-resolution quality
- **User-friendly GUI**: Tab-based interface with real-time preview
- **Multiple format support**: JPEG, PNG, BMP, TIFF, and other common image formats

## Installation

### Prerequisites

- Python 3.6 or higher
- pip package manager

### From Source

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python main.py
   ```

### Building Executable

To create a standalone executable that doesn't require Python installation:

1. Install PyInstaller:
   ```bash
   pip install pyinstaller
   ```
2. Build the executable:
   ```bash
   pyinstaller --onefile --windowed --name "X-ray Denoiser" main.py
   ```
3. The executable will be created in the `dist/` directory

## Usage

### Two-Step Workflow

#### Step 1: Denoising (降噪处理)

1. **Load Image**: Click "📁 加载图像" and select your X-ray image file
2. **Select Algorithm**: Choose from available denoising algorithms:
   - **Hybrid (Recommended)**: Combines NLM + Bilateral for optimal results
   - **Non-local Means**: Effective for preserving fine details
   - **Bilateral Filter**: Good for edge preservation
   - **Wavelet**: Frequency-domain denoising approach
   - **Neural Network**: Deep learning based denoising (if available)
3. **Adjust Parameters**: Fine-tune algorithm-specific parameters
4. **Execute Denoising**: Click "▶ 执行降噪" to process

#### Step 2: Super-Resolution (超分辨率重构)

After denoising is complete:

1. **Select Method**: Choose upscaling method:
   - **Bicubic**: Standard interpolation
   - **Lanczos (Recommended)**: High-quality interpolation
   - **Edge-Preserving**: Enhanced edge detail preservation
2. **Select Scale**: Choose magnification factor (1.5x, 2.0x, 3.0x, 4.0x)
3. **Enhancement Options**:
   - Edge Enhancement: Improves edge clarity
   - Contrast Enhancement: CLAHE-based contrast improvement
4. **Execute SR**: Click "🔍 执行超分辨率" to process

### Viewing Results

- Use the tab interface to switch between:
  - **原始图像**: Original loaded image
  - **降噪结果**: Denoised image
  - **超分辨率结果**: Super-resolution image

### Saving Results

- **💾 保存降噪图**: Save the denoised image
- **💾 保存 SR 图**: Save the super-resolution image

## Metrics Explanation

- **PSNR (Peak Signal-to-Noise Ratio)**: Measured in decibels (dB). Higher values indicate better quality. Typical values range from 20-50 dB.
- **SSIM (Structural Similarity Index)**: Ranges from 0 to 1. Values closer to 1 indicate higher similarity to the original structure.

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)
- And other formats supported by OpenCV

## System Requirements

- **RAM**: 2GB minimum (4GB recommended for large images)
- **Storage**: 100MB free space
- **Display**: 1280x720 resolution or higher recommended

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Dependencies

- PyQt5: Cross-platform GUI framework
- OpenCV: Computer vision and image processing
- scikit-image: Image processing algorithms
- NumPy: Numerical computing
- SciPy: Scientific computing

## Troubleshooting

### Common Issues

1. **"Failed to load image"**: Ensure the image file is not corrupted and is in a supported format
2. **Slow processing**: Large images may take longer to process. Consider resizing very large images first
3. **Missing dependencies**: Make sure all requirements are installed using `pip install -r requirements.txt`

### Performance Tips

- For very large images (>4000x4000 pixels), consider preprocessing to reduce size
- The Hybrid algorithm provides the best balance of quality and performance
- Close other applications when processing large batches of images

## Development

To contribute or modify the application:

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Contact

For questions or support, please open an issue on the GitHub repository.