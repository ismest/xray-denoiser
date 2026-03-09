# X-ray Image Denoiser

A cross-platform desktop application for denoising X-ray images with quantitative quality metrics.

## Features

- **Cross-platform**: Works on Windows, macOS, and Linux
- **Multiple denoising algorithms**: Hybrid approach combining non-local means, bilateral filtering, and wavelet denoising
- **Quantitative metrics**: PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index)
- **User-friendly GUI**: Easy to use interface with real-time preview
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

1. **Load Image**: Click "Load X-ray Image" and select your X-ray image file
2. **Select Algorithm**: Choose from available denoising algorithms:
   - **Hybrid (Recommended)**: Combines multiple methods for optimal results
   - **Non-local Means**: Effective for preserving fine details
   - **Bilateral Filter**: Good for edge preservation
   - **Wavelet**: Frequency-domain denoising approach
3. **Process Image**: Click "Denoise Image" to apply the selected algorithm
4. **View Results**: Compare original and denoised images side by side
5. **Check Metrics**: View PSNR and SSIM values in the results panel
6. **Save Result**: Save the denoised image to your preferred location

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