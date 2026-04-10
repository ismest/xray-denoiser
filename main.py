#!/usr/bin/env python3
"""
X-ray Image Denoiser - Main Application Entry Point

This application provides a multi-page GUI for:
1. Image Preprocessing - Noise extraction and dataset construction
2. Algorithm Training - Neural network model training
3. Denoising & Super-Resolution - Image processing workflow
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_window import MainWindow
from denoise_sr_page import DenoiseSRApp  # For backward compatibility (standalone mode)
from PyQt5.QtWidgets import QApplication


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)

    # Use the new multi-page main window
    window = MainWindow()
    window.show()

    # Run the application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
