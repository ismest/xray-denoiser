import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QProgressBar, QGroupBox,
                             QComboBox, QTextEdit, QMessageBox, QSlider, QSpinBox, 
                             QDoubleSpinBox, QTabWidget, QCheckBox, QGridLayout)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from image_processor import ImageProcessor
import numpy as np


class ProcessingThread(QThread):
    """Background thread for image processing."""
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(int, str)  # Added stage message

    def __init__(self, processor, method, params):
        super().__init__()
        self.processor = processor
        self.method = method
        self.params = params

    def run(self):
        try:
            # Stage 1: Initialize processing
            self.progress.emit(10, "Initializing...")

            # Stage 2: Start processing
            self.progress.emit(30, f"Applying {self.method} denoising...")

            # Stage 3: Processing in progress
            self.progress.emit(50, "Processing image...")

            # Stage 4: Finalizing
            self.progress.emit(80, "Finalizing...")

            success = self.processor.process_image(self.method, **self.params)

            # Stage 5: Complete
            self.progress.emit(100, "Complete")

            if success:
                self.finished.emit(True, "Processing completed successfully")
            else:
                self.finished.emit(False, "Processing failed - no result returned")
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.finished.emit(False, f"{str(e)}\n\n{error_details}")


class DenoiseApp(QMainWindow):
    """
    Main GUI application for X-ray image denoising.
    Enhanced with parameter controls and batch processing support.
    """
    
    def __init__(self):
        super().__init__()
        self.processor = ImageProcessor()
        self.processing_thread = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle('X-ray Image Denoiser - Enhanced')
        self.setGeometry(100, 100, 1400, 900)
        self.setWindowIcon(self.style().standardIcon(self.style().SP_ComputerIcon))
        
        # Apply modern stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QWidget {
                font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
                font-size: 12px;
            }
            QTabWidget::pane {
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                background: white;
                margin: 0px;
            }
            QTabBar::tab {
                background: #f5f5f5;
                border: 1px solid #e0e0e0;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: white;
                border-bottom-color: white;
                font-weight: bold;
                color: #1976d2;
            }
            QGroupBox {
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                margin-top: 1ex;
                font-weight: bold;
                color: #333333;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                top: -8px;
                padding: 0 4px;
            }
            QLabel {
                color: #333333;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QPushButton:disabled {
                background-color: #bdbdbd;
                color: #757575;
            }
            QPushButton#saveButton {
                background-color: #4CAF50;
            }
            QPushButton#saveButton:hover {
                background-color: #388E3C;
            }
            QPushButton#processButton {
                background-color: #4CAF50;
            }
            QPushButton#processButton:hover {
                background-color: #388E3C;
            }
            QComboBox {
                padding: 6px;
                border: 1px solid #bdbdbd;
                border-radius: 4px;
                background-color: white;
                min-height: 28px;
            }
            QSpinBox, QDoubleSpinBox {
                padding: 6px;
                border: 1px solid #bdbdbd;
                border-radius: 4px;
                background-color: white;
                min-height: 28px;
            }
            QTextEdit {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                background-color: white;
                padding: 8px;
            }
            QProgressBar {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
            QStatusBar {
                background-color: #f5f5f5;
                color: #666666;
                border-top: 1px solid #e0e0e0;
            }
        """)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget for controls
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Basic controls tab
        basic_tab = self.create_basic_tab()
        self.tab_widget.addTab(basic_tab, "Basic")
        
        # Advanced controls tab
        advanced_tab = self.create_advanced_tab()
        self.tab_widget.addTab(advanced_tab, "Advanced")
        
        # Create image display area
        image_display = self.create_image_display()
        main_layout.addWidget(image_display)
        
        # Create results panel
        results_panel = self.create_results_panel()
        main_layout.addWidget(results_panel)
        
        # Create status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage('Ready - Load an image to begin')
        
    def create_basic_tab(self):
        """Create the basic controls tab."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)
        
        control_group = QGroupBox("Quick Controls")
        control_layout = QHBoxLayout()
        control_layout.setSpacing(12)
        
        # Load image button
        self.load_btn = QPushButton("Load X-ray Image")
        self.load_btn.setObjectName("loadButton")
        self.load_btn.clicked.connect(self.load_image)
        control_layout.addWidget(self.load_btn)
        
        # Algorithm selection
        algorithm_label = QLabel("Algorithm:")
        algorithm_label.setStyleSheet("font-weight: bold;")
        control_layout.addWidget(algorithm_label)
        self.algorithm_combo = QComboBox()
        self.update_algorithm_list()
        control_layout.addWidget(self.algorithm_combo)
        
        # Strength selection
        strength_label = QLabel("Strength:")
        strength_label.setStyleSheet("font-weight: bold;")
        control_layout.addWidget(strength_label)
        self.strength_combo = QComboBox()
        self.strength_combo.addItems(["Low", "Medium", "High"])
        self.strength_combo.setCurrentIndex(1)
        control_layout.addWidget(self.strength_combo)
        
        # Process button
        self.process_btn = QPushButton("Denoise Image")
        self.process_btn.setObjectName("processButton")
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setEnabled(False)
        control_layout.addWidget(self.process_btn)
        
        # Save button
        self.save_btn = QPushButton("Save Result")
        self.save_btn.setObjectName("saveButton")
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)
        control_layout.addWidget(self.save_btn)
        
        control_layout.addStretch()  # Add stretch to push controls to left
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        return tab
    
    def create_advanced_tab(self):
        """Create the advanced controls tab."""
        tab = QWidget()
        layout = QGridLayout(tab)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # NLM Parameters
        self.nlm_group = QGroupBox("Non-Local Means Parameters")
        nlm_layout = QVBoxLayout()
        nlm_layout.setSpacing(8)

        nlm_layout.addWidget(QLabel("Filter Strength (h):"))
        self.nlm_h_spin = QSpinBox()
        self.nlm_h_spin.setRange(1, 50)
        self.nlm_h_spin.setValue(10)
        nlm_layout.addWidget(self.nlm_h_spin)

        nlm_layout.addWidget(QLabel("Patch Size:"))
        self.nlm_patch_spin = QSpinBox()
        self.nlm_patch_spin.setRange(3, 15)
        self.nlm_patch_spin.setValue(7)
        nlm_layout.addWidget(self.nlm_patch_spin)

        self.nlm_group.setLayout(nlm_layout)
        layout.addWidget(self.nlm_group, 0, 0)

        # Bilateral Parameters
        self.bilateral_group = QGroupBox("Bilateral Filter Parameters")
        bilateral_layout = QVBoxLayout()
        bilateral_layout.setSpacing(8)

        bilateral_layout.addWidget(QLabel("Diameter (d):"))
        self.bilateral_d_spin = QSpinBox()
        self.bilateral_d_spin.setRange(3, 25)
        self.bilateral_d_spin.setValue(9)
        bilateral_layout.addWidget(self.bilateral_d_spin)

        bilateral_layout.addWidget(QLabel("Sigma Color:"))
        self.bilateral_color_spin = QSpinBox()
        self.bilateral_color_spin.setRange(10, 200)
        self.bilateral_color_spin.setValue(75)
        bilateral_layout.addWidget(self.bilateral_color_spin)

        self.bilateral_group.setLayout(bilateral_layout)
        layout.addWidget(self.bilateral_group, 0, 1)

        # Neural Network Parameters
        self.neural_group = QGroupBox("Neural Network Parameters")
        neural_layout = QVBoxLayout()
        neural_layout.setSpacing(8)

        neural_layout.addWidget(QLabel("Patch Size:"))
        self.neural_patch_spin = QSpinBox()
        self.neural_patch_spin.setRange(64, 512)
        self.neural_patch_spin.setValue(256)
        neural_layout.addWidget(self.neural_patch_spin)

        neural_layout.addWidget(QLabel("Stride:"))
        self.neural_stride_spin = QSpinBox()
        self.neural_stride_spin.setRange(32, 256)
        self.neural_stride_spin.setValue(128)
        neural_layout.addWidget(self.neural_stride_spin)

        self.neural_group.setLayout(neural_layout)
        layout.addWidget(self.neural_group, 0, 2)

        # Image Info
        info_group = QGroupBox("Image Information")
        info_layout = QVBoxLayout()
        info_layout.setSpacing(8)
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(180)
        self.info_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                background-color: #f9f9f9;
                padding: 10px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
            }
        """)
        info_layout.addWidget(self.info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group, 1, 0, 1, 3)

        # Initialize parameter panel visibility
        self.update_parameter_panel()

        return tab
        
    def create_image_display(self):
        """Create the image display area."""
        display_group = QGroupBox("Image Preview")
        display_layout = QHBoxLayout()
        
        # Original image label
        original_container = QVBoxLayout()
        original_title = QLabel("Original Image")
        original_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #333333; padding: 8px;")
        original_container.addWidget(original_title)
        self.original_label = QLabel("No image loaded")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumHeight(400)
        self.original_label.setStyleSheet("""
            border: 2px dashed #bdbdbd;
            background-color: #fafafa;
            border-radius: 8px;
            color: #666666;
            font-style: italic;
            padding: 20px;
        """)
        original_container.addWidget(self.original_label)
        display_layout.addLayout(original_container)
        
        # Denoised image label
        denoised_container = QVBoxLayout()
        denoised_title = QLabel("Denoised Image")
        denoised_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #333333; padding: 8px;")
        denoised_container.addWidget(denoised_title)
        self.denoised_label = QLabel("Process an image to see result")
        self.denoised_label.setAlignment(Qt.AlignCenter)
        self.denoised_label.setMinimumHeight(400)
        self.denoised_label.setStyleSheet("""
            border: 2px dashed #bdbdbd;
            background-color: #fafafa;
            border-radius: 8px;
            color: #666666;
            font-style: italic;
            padding: 20px;
        """)
        denoised_container.addWidget(self.denoised_label)
        display_layout.addLayout(denoised_container)
        
        display_group.setLayout(display_layout)
        return display_group
        
    def create_results_panel(self):
        """Create the results panel for metrics display."""
        results_group = QGroupBox("Denoising Results")
        results_layout = QVBoxLayout()
        
        # Progress bar with better styling
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                text-align: center;
                height: 24px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        results_layout.addWidget(self.progress_bar)
        
        # Metrics display with better styling
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMaximumHeight(120)
        self.metrics_text.setText("Metrics will appear here after processing")
        self.metrics_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                background-color: #f9f9f9;
                padding: 12px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                color: #333333;
            }
        """)
        results_layout.addWidget(self.metrics_text)
        
        results_group.setLayout(results_layout)
        return results_group
    
    def update_algorithm_list(self):
        """Update the algorithm dropdown with available methods."""
        self.algorithm_combo.clear()
        methods = self.processor.get_supported_methods()
        for key, name in methods:
            self.algorithm_combo.addItem(name, key)

        # Connect algorithm change to parameter panel update
        self.algorithm_combo.currentIndexChanged.connect(self.update_parameter_panel)

    def update_parameter_panel(self):
        """Update parameter panel visibility based on selected algorithm."""
        method = self.algorithm_combo.currentData()

        # Hide all parameter groups first
        for group in [self.nlm_group, self.bilateral_group, self.neural_group]:
            group.setVisible(False)

        # Show only relevant parameters
        if method == 'nlm':
            self.nlm_group.setVisible(True)
        elif method == 'bilateral':
            self.bilateral_group.setVisible(True)
        elif method == 'neural':
            self.neural_group.setVisible(True)
        # For hybrid and other methods, no specific parameters shown
        
    def load_image(self):
        """Load an image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open X-ray Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;All Files (*);;DICOM (*.dcm)"
        )
        
        if file_path:
            self.status_bar.showMessage(f'Loading {os.path.basename(file_path)}...')
            success = self.processor.load_image(file_path)
            if success:
                self.display_original_image()
                self.process_btn.setEnabled(True)
                self.save_btn.setEnabled(False)
                self.metrics_text.clear()
                self.denoised_label.setText("Process an image to see result")
                self.denoised_label.setPixmap(QPixmap())
                self.status_bar.showMessage('Image loaded successfully')
                
                # Update info display
                info = self.processor.get_image_info()
                info_text = f"""Filename: {info.get('filename', 'N/A')}
Shape: {info.get('shape', 'N/A')}
Data Type: {info.get('dtype', 'N/A')}
Bit Depth: {info.get('depth', 'N/A')} bits
Size: {info.get('size_mb', 0):.2f} MB"""
                self.info_text.setText(info_text)
            else:
                QMessageBox.critical(self, "Error", "Failed to load image file.")
                self.status_bar.showMessage('Failed to load image')
                
    def display_original_image(self):
        """Display the original image in the GUI."""
        try:
            import cv2
            original_img = self.processor.get_original_image()
            if original_img is not None:
                # Convert to RGB for display
                if len(original_img.shape) == 3:
                    if original_img.shape[2] == 3:
                        rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                    else:
                        gray = original_img[:,:,0]
                        rgb_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                else:
                    rgb_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
                
                # Convert to uint8 for display
                if rgb_img.dtype != np.uint8:
                    if rgb_img.dtype == np.uint16:
                        rgb_img = (rgb_img.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
                    else:
                        rgb_img = (rgb_img / rgb_img.max() * 255).astype(np.uint8)
                
                # Create QImage
                height, width, channel = rgb_img.shape
                bytes_per_line = 3 * width
                q_img = QImage(rgb_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                
                # Create and set pixmap
                pixmap = QPixmap.fromImage(q_img)
                scaled_pixmap = pixmap.scaled(600, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.original_label.setPixmap(scaled_pixmap)
                self.original_label.setText("")
        except Exception as e:
            print(f"Display error: {e}")
            self.original_label.setText("Error displaying image")
            
    def process_image(self):
        """Process the loaded image with selected algorithm."""
        # Get algorithm
        method = self.algorithm_combo.currentData()

        # Collect parameters based on method
        params = {}

        if method == 'hybrid':
            # Hybrid uses strength parameter
            strength_map = {"Low": "low", "Medium": "medium", "High": "high"}
            params['strength'] = strength_map.get(self.strength_combo.currentText(), "medium")

        elif method == 'nlm':
            # NLM uses h and patch_size
            params['h'] = self.nlm_h_spin.value()
            params['patch_size'] = self.nlm_patch_spin.value()

        elif method == 'bilateral':
            # Bilateral uses d, sigma_color, sigma_space
            params['d'] = self.bilateral_d_spin.value()
            params['sigma_color'] = self.bilateral_color_spin.value()
            params['sigma_space'] = self.bilateral_color_spin.value()

        elif method == 'neural':
            # Neural uses patch_size and stride
            params['patch_size'] = self.neural_patch_spin.value()
            params['stride'] = self.neural_stride_spin.value()

        elif method in ['wavelet', 'gaussian', 'nlm', 'bilateral']:
            # These methods don't need additional parameters from UI
            pass

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage(f'Processing with {method} algorithm...')
        self.process_btn.setEnabled(False)

        # Run processing in background thread
        self.processing_thread = ProcessingThread(self.processor, method, params)
        self.processing_thread.progress.connect(self.update_progress)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.start()
    
    def update_progress(self, value, stage_message=""):
        """Update progress bar with stage information."""
        self.progress_bar.setValue(value)
        if stage_message:
            self.status_bar.showMessage(stage_message)
    
    def processing_finished(self, success, message):
        """Handle processing completion."""
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)

        if success:
            self.display_denoised_image()
            self.calculate_and_display_metrics()
            self.save_btn.setEnabled(True)
            self.status_bar.showMessage('Image processed successfully')
        else:
            # Show detailed error in a scrollable dialog
            error_box = QMessageBox(self)
            error_box.setIcon(QMessageBox.Critical)
            error_box.setWindowTitle("Processing Error")
            error_box.setText("Processing failed with the following error:")
            error_box.setInformativeText(message)
            error_box.setDetailedText(message)
            error_box.setStandardButtons(QMessageBox.Ok)
            error_box.setSizeGripEnabled(True)
            error_box.setStyleSheet("""
                QMessageBox {
                    background-color: #fafafa;
                }
                QLabel {
                    color: #333333;
                    min-width: 400px;
                }
                QTextEdit {
                    min-width: 600px;
                    min-height: 300px;
                    font-family: 'Consolas', 'Courier New', monospace;
                    font-size: 11px;
                }
            """)
            error_box.exec_()
            self.status_bar.showMessage('Processing failed - see error dialog for details')
            
    def display_denoised_image(self):
        """Display the denoised image in the GUI."""
        try:
            import cv2
            denoised_img = self.processor.get_denoised_image()
            if denoised_img is not None:
                # Convert to RGB for display
                if len(denoised_img.shape) == 2:
                    rgb_img = cv2.cvtColor(denoised_img, cv2.COLOR_GRAY2RGB)
                else:
                    rgb_img = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2RGB)
                
                # Convert to uint8 for display
                if rgb_img.dtype != np.uint8:
                    if rgb_img.dtype == np.uint16:
                        rgb_img = (rgb_img.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
                    else:
                        rgb_img = (rgb_img / rgb_img.max() * 255).astype(np.uint8)
                
                # Create QImage
                height, width, channel = rgb_img.shape
                bytes_per_line = 3 * width
                q_img = QImage(rgb_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                
                # Create and set pixmap
                pixmap = QPixmap.fromImage(q_img)
                scaled_pixmap = pixmap.scaled(600, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.denoised_label.setPixmap(scaled_pixmap)
                self.denoised_label.setText("")
        except Exception as e:
            print(f"Display error: {e}")
            self.denoised_label.setText("Error displaying result")
            
    def calculate_and_display_metrics(self):
        """Calculate and display denoising metrics."""
        metrics = self.processor.calculate_metrics()
        if metrics:
            psnr = metrics.get('psnr', 0)
            ssim_val = metrics.get('ssim', 0)
            mse = metrics.get('mse', 0)
            
            metrics_text = f"Peak Signal-to-Noise Ratio (PSNR): {psnr:.2f} dB\n"
            metrics_text += f"Structural Similarity Index (SSIM): {ssim_val:.4f}\n"
            metrics_text += f"Mean Squared Error (MSE): {mse:.6f}"
            
            self.metrics_text.setText(metrics_text)
            
    def save_result(self):
        """Save the denoised image."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Denoised Image", "", 
            "PNG Files (*.png);;TIFF Files (*.tiff *.tif);;JPEG Files (*.jpg *.jpeg);;All Files (*)"
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
    
    def closeEvent(self, event):
        """Handle window close event."""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.wait(1000)
        event.accept()


# Fix the cv2 import issue by importing it here
try:
    import cv2
except ImportError:
    pass
