import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QProgressBar, QGroupBox,
                             QComboBox, QTextEdit, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
from image_processor import ImageProcessor
import numpy as np

class DenoiseApp(QMainWindow):
    """
    Main GUI application for X-ray image denoising.
    """
    
    def __init__(self):
        super().__init__()
        self.processor = ImageProcessor()
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle('X-ray Image Denoiser')
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # Create image display area
        image_display = self.create_image_display()
        main_layout.addWidget(image_display)
        
        # Create results panel
        results_panel = self.create_results_panel()
        main_layout.addWidget(results_panel)
        
        # Create status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage('Ready')
        
    def create_control_panel(self):
        """Create the control panel with buttons and options."""
        control_group = QGroupBox("Controls")
        control_layout = QHBoxLayout()
        
        # Load image button
        self.load_btn = QPushButton("Load X-ray Image")
        self.load_btn.clicked.connect(self.load_image)
        control_layout.addWidget(self.load_btn)
        
        # Algorithm selection
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["Hybrid (Recommended)", "Non-local Means", "Bilateral Filter", "Wavelet"])
        control_layout.addWidget(QLabel("Algorithm:"))
        control_layout.addWidget(self.algorithm_combo)
        
        # Process button
        self.process_btn = QPushButton("Denoise Image")
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setEnabled(False)
        control_layout.addWidget(self.process_btn)
        
        # Save button
        self.save_btn = QPushButton("Save Result")
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)
        control_layout.addWidget(self.save_btn)
        
        control_group.setLayout(control_layout)
        return control_group
        
    def create_image_display(self):
        """Create the image display area."""
        display_group = QGroupBox("Image Preview")
        display_layout = QHBoxLayout()
        
        # Original image label
        self.original_label = QLabel("Original Image")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumHeight(400)
        self.original_label.setStyleSheet("border: 1px solid gray;")
        display_layout.addWidget(self.original_label)
        
        # Denoised image label
        self.denoised_label = QLabel("Denoised Image")
        self.denoised_label.setAlignment(Qt.AlignCenter)
        self.denoised_label.setMinimumHeight(400)
        self.denoised_label.setStyleSheet("border: 1px solid gray;")
        display_layout.addWidget(self.denoised_label)
        
        display_group.setLayout(display_layout)
        return display_group
        
    def create_results_panel(self):
        """Create the results panel for metrics display."""
        results_group = QGroupBox("Denoising Results")
        results_layout = QVBoxLayout()
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        results_layout.addWidget(self.progress_bar)
        
        # Metrics display
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMaximumHeight(100)
        results_layout.addWidget(self.metrics_text)
        
        results_group.setLayout(results_layout)
        return results_group
        
    def load_image(self):
        """Load an image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open X-ray Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;All Files (*)"
        )
        
        if file_path:
            self.status_bar.showMessage(f'Loading {os.path.basename(file_path)}...')
            success = self.processor.load_image(file_path)
            if success:
                self.display_original_image()
                self.process_btn.setEnabled(True)
                self.save_btn.setEnabled(False)
                self.metrics_text.clear()
                self.status_bar.showMessage('Image loaded successfully')
            else:
                QMessageBox.critical(self, "Error", "Failed to load image file.")
                self.status_bar.showMessage('Failed to load image')
                
    def display_original_image(self):
        """Display the original image in the GUI."""
        original_img = self.processor.get_original_image()
        if original_img is not None:
            # Convert to RGB if needed
            if len(original_img.shape) == 3:
                rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            else:
                rgb_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
            
            # Create QImage
            height, width, channel = rgb_img.shape
            bytes_per_line = 3 * width
            q_img = QImage(rgb_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # Create and set pixmap
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(500, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.original_label.setPixmap(scaled_pixmap)
            self.original_label.setText("")
            
    def process_image(self):
        """Process the loaded image with selected algorithm."""
        algorithm_map = {
            "Hybrid (Recommended)": "hybrid",
            "Non-local Means": "nlm",
            "Bilateral Filter": "bilateral",
            "Wavelet": "wavelet"
        }
        
        selected_algorithm = self.algorithm_combo.currentText()
        method = algorithm_map.get(selected_algorithm, "hybrid")
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage('Processing image...')
        
        # Use QTimer to allow UI updates
        QTimer.singleShot(100, lambda: self._process_image_thread(method))
        
    def _process_image_thread(self, method):
        """Internal method to process image (simulates threading)."""
        try:
            success = self.processor.process_image(method)
            if success:
                self.display_denoised_image()
                self.calculate_and_display_metrics()
                self.save_btn.setEnabled(True)
                self.status_bar.showMessage('Image processed successfully')
            else:
                QMessageBox.critical(self, "Error", "Failed to process image.")
                self.status_bar.showMessage('Failed to process image')
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Processing failed: {str(e)}")
            self.status_bar.showMessage('Processing failed')
        finally:
            self.progress_bar.setVisible(False)
            
    def display_denoised_image(self):
        """Display the denoised image in the GUI."""
        denoised_img = self.processor.get_denoised_image()
        if denoised_img is not None:
            # Ensure proper shape for display
            if len(denoised_img.shape) == 2:
                rgb_img = cv2.cvtColor(denoised_img, cv2.COLOR_GRAY2RGB)
            else:
                rgb_img = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2RGB)
            
            # Create QImage
            height, width, channel = rgb_img.shape
            bytes_per_line = 3 * width
            q_img = QImage(rgb_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # Create and set pixmap
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(500, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.denoised_label.setPixmap(scaled_pixmap)
            self.denoised_label.setText("")
            
    def calculate_and_display_metrics(self):
        """Calculate and display denoising metrics."""
        metrics = self.processor.calculate_metrics()
        if metrics:
            psnr = metrics.get('psnr', 0)
            ssim_val = metrics.get('ssim', 0)
            
            metrics_text = f"Peak Signal-to-Noise Ratio (PSNR): {psnr:.2f} dB\n"
            metrics_text += f"Structural Similarity Index (SSIM): {ssim_val:.4f}"
            
            self.metrics_text.setText(metrics_text)
            
    def save_result(self):
        """Save the denoised image."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Denoised Image", "", 
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)"
        )
        
        if file_path:
            self.status_bar.showMessage('Saving result...')
            success = self.processor.save_result(file_path)
            if success:
                self.status_bar.showMessage('Result saved successfully')
                QMessageBox.information(self, "Success", "Denoised image saved successfully!")
            else:
                QMessageBox.critical(self, "Error", "Failed to save image.")
                self.status_bar.showMessage('Failed to save result')

# Fix the cv2 import issue by importing it here
try:
    import cv2
except ImportError:
    pass