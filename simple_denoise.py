#!/usr/bin/env python3
"""
Simple X-ray Image Denoiser - Fallback version with basic dependencies
"""

import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class SimpleDenoiseApp(QMainWindow):
    """
    Simplified version of the denoiser that works with basic PyQt5 only.
    Uses simple Gaussian blur as a fallback denoising method.
    """
    
    def __init__(self):
        super().__init__()
        self.original_image = None
        self.denoised_image = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle('X-ray Image Denoiser (Simple)')
        self.setGeometry(100, 100, 1000, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        button_layout.addWidget(self.load_btn)
        
        self.process_btn = QPushButton("Apply Simple Denoise")
        self.process_btn.clicked.connect(self.simple_denoise)
        self.process_btn.setEnabled(False)
        button_layout.addWidget(self.process_btn)
        
        self.save_btn = QPushButton("Save Result")
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)
        button_layout.addWidget(self.save_btn)
        
        main_layout.addLayout(button_layout)
        
        # Image display
        image_layout = QHBoxLayout()
        self.original_label = QLabel("Original Image")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumHeight(400)
        self.original_label.setStyleSheet("border: 1px solid gray;")
        image_layout.addWidget(self.original_label)
        
        self.denoised_label = QLabel("Denoised Image")
        self.denoised_label.setAlignment(Qt.AlignCenter)
        self.denoised_label.setMinimumHeight(400)
        self.denoised_label.setStyleSheet("border: 1px solid gray;")
        image_layout.addWidget(self.denoised_label)
        
        main_layout.addLayout(image_layout)
        
        self.statusBar().showMessage('Ready - Simple mode (Gaussian blur only)')
        
    def load_image(self):
        """Load an image file."""
        try:
            from PyQt5.QtWidgets import QFileDialog
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Image", "", 
                "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
            )
            
            if file_path:
                self.original_pixmap = QPixmap(file_path)
                if self.original_pixmap.isNull():
                    raise Exception("Failed to load image")
                
                scaled_original = self.original_pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.original_label.setPixmap(scaled_original)
                self.original_label.setText("")
                self.process_btn.setEnabled(True)
                self.save_btn.setEnabled(False)
                self.statusBar().showMessage(f'Loaded: {os.path.basename(file_path)}')
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
            
    def simple_denoise(self):
        """Apply simple Gaussian blur denoising."""
        try:
            from PyQt5.QtGui import QPainter, QPen
            from PyQt5.QtCore import QRect
            
            # Create a blurred version by scaling down and up
            if hasattr(self, 'original_pixmap'):
                # Simple approach: create a slightly blurred version
                width = self.original_pixmap.width()
                height = self.original_pixmap.height()
                
                # Scale down then up to create blur effect
                small_pixmap = self.original_pixmap.scaled(
                    max(1, width // 4), max(1, height // 4), 
                    Qt.IgnoreAspectRatio, Qt.SmoothTransformation
                )
                self.denoised_pixmap = small_pixmap.scaled(
                    width, height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
                )
                
                scaled_denoised = self.denoised_pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.denoised_label.setPixmap(scaled_denoised)
                self.denoised_label.setText("")
                self.save_btn.setEnabled(True)
                self.statusBar().showMessage('Simple denoising applied (Gaussian blur approximation)')
                
                # Show simple metrics message
                QMessageBox.information(self, "Info", 
                    "Simple mode active.\nFor full features (PSNR/SSIM metrics and advanced algorithms),\n"
                    "install OpenCV, scikit-image, and NumPy:\n\n"
                    "pip install opencv-python scikit-image numpy scipy")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Denoising failed: {str(e)}")
            
    def save_result(self):
        """Save the denoised image."""
        try:
            if hasattr(self, 'denoised_pixmap'):
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "Save Denoised Image", "", 
                    "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
                )
                
                if file_path:
                    success = self.denoised_pixmap.save(file_path)
                    if success:
                        QMessageBox.information(self, "Success", "Image saved successfully!")
                        self.statusBar().showMessage(f'Saved: {os.path.basename(file_path)}')
                    else:
                        QMessageBox.critical(self, "Error", "Failed to save image.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed: {str(e)}")

def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    window = SimpleDenoiseApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()