#!/usr/bin/env python3
"""
X-ray Image Denoiser - Main Application Entry Point

This application provides a cross-platform GUI for denoising X-ray images
using hybrid denoising algorithms and provides quantitative metrics (PSNR/SSIM).
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from denoise_app import DenoiseApp
from PyQt5.QtWidgets import QApplication

def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    # Create and show the main window
    window = DenoiseApp()
    window.show()
    
    # Run the application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()