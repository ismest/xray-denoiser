import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QProgressBar, QGroupBox,
                             QComboBox, QTextEdit, QMessageBox, QSlider, QSpinBox,
                             QDoubleSpinBox, QTabWidget, QCheckBox, QGridLayout, QFormLayout)
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
            self.progress.emit(10, "正在初始化...")

            # Stage 2: Start processing
            self.progress.emit(30, f"正在应用 {self.method} 降噪...")

            # Stage 3: Processing in progress
            self.progress.emit(50, "正在处理图像...")

            # Stage 4: Finalizing
            self.progress.emit(80, "正在完成...")

            success = self.processor.process_image(self.method, **self.params)

            # Stage 5: Complete
            self.progress.emit(100, "完成")

            if success:
                self.finished.emit(True, "处理成功完成")
            else:
                self.finished.emit(False, "处理失败 - 未返回结果")
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
        self.setWindowTitle('X 射线图像降噪器 - 增强版')
        self.setGeometry(100, 100, 1400, 900)
        self.setWindowIcon(self.style().standardIcon(self.style().SP_ComputerIcon))
        
        # Apply modern stylesheet with optimized fonts and spacing
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QWidget {
                font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
                font-size: 28px;
            }
            QTabWidget::pane {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                background: white;
                margin: 0px;
            }
            QTabBar::tab {
                background: #f5f5f5;
                border: 1px solid #e0e0e0;
                padding: 16px 32px;
                margin-right: 4px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-size: 28px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: white;
                border-bottom-color: white;
                color: #1976d2;
            }
            QGroupBox {
                border: 2px solid #e0e0e0;
                border-radius: 12px;
                margin-top: 15px;
                font-weight: bold;
                font-size: 30px;
                color: #333333;
                padding-top: 30px;
                padding-left: 15px;
                padding-right: 15px;
                padding-bottom: 15px;
                background-color: #fafafa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                top: 0px;
                padding: 0 8px;
                color: #1976d2;
            }
            QLabel {
                color: #333333;
                font-size: 28px;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 16px 32px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 28px;
                min-height: 56px;
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
                padding: 12px;
                border: 1px solid #bdbdbd;
                border-radius: 8px;
                background-color: white;
                min-height: 56px;
                font-size: 28px;
            }
            QSpinBox, QDoubleSpinBox {
                padding: 12px;
                border: 1px solid #bdbdbd;
                border-radius: 8px;
                background-color: white;
                min-height: 56px;
                font-size: 28px;
            }
            QTextEdit {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                background-color: white;
                padding: 16px;
                font-size: 26px;
            }
            QProgressBar {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                text-align: center;
                height: 40px;
                font-weight: bold;
                font-size: 24px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 5px;
            }
            QStatusBar {
                background-color: #f5f5f5;
                color: #666666;
                border-top: 1px solid #e0e0e0;
                font-size: 24px;
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
        self.tab_widget.addTab(basic_tab, "基本")
        
        # Advanced controls tab
        advanced_tab = self.create_advanced_tab()
        self.tab_widget.addTab(advanced_tab, "高级")
        
        # Create image display area
        image_display = self.create_image_display()
        main_layout.addWidget(image_display)
        
        # Create results panel
        results_panel = self.create_results_panel()
        main_layout.addWidget(results_panel)
        
        # Create status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage('就绪 - 加载图像开始')
        
    def create_basic_tab(self):
        """Create the basic controls tab with optimized layout."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # File operations group
        file_group = QGroupBox("文件操作")
        file_layout = QHBoxLayout()
        file_layout.setSpacing(15)

        # Load image button
        self.load_btn = QPushButton("加载 X 光图像")
        self.load_btn.setObjectName("loadButton")
        self.load_btn.clicked.connect(self.load_image)
        file_layout.addWidget(self.load_btn)

        # Save button
        self.save_btn = QPushButton("保存结果")
        self.save_btn.setObjectName("saveButton")
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)
        file_layout.addWidget(self.save_btn)

        file_layout.addStretch()
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Algorithm selection group
        algorithm_group = QGroupBox("降噪算法")
        algorithm_layout = QHBoxLayout()
        algorithm_layout.setSpacing(15)

        # Algorithm selection
        algorithm_label = QLabel("算法:")
        algorithm_label.setStyleSheet("font-weight: bold; font-size: 28px;")
        algorithm_layout.addWidget(algorithm_label)
        self.algorithm_combo = QComboBox()
        self.update_algorithm_list()
        self.algorithm_combo.setMinimumWidth(200)
        algorithm_layout.addWidget(self.algorithm_combo)

        # Strength selection
        strength_label = QLabel("强度:")
        strength_label.setStyleSheet("font-weight: bold; font-size: 28px;")
        strength_layout = QHBoxLayout()
        strength_layout.setSpacing(10)
        strength_layout.addWidget(strength_label)
        self.strength_combo = QComboBox()
        self.strength_combo.addItems(["低", "中", "高"])
        self.strength_combo.setCurrentIndex(1)
        self.strength_combo.setMinimumWidth(120)
        strength_layout.addWidget(self.strength_combo)
        strength_layout.addStretch()
        algorithm_layout.addLayout(strength_layout)

        algorithm_group.setLayout(algorithm_layout)
        layout.addWidget(algorithm_group)

        # Process button group
        process_group = QGroupBox("处理")
        process_layout = QHBoxLayout()
        process_layout.setSpacing(15)

        # Process button
        self.process_btn = QPushButton("开始降噪")
        self.process_btn.setObjectName("processButton")
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setEnabled(False)
        self.process_btn.setMinimumHeight(56)
        process_layout.addWidget(self.process_btn)

        process_group.setLayout(process_layout)
        layout.addWidget(process_group)

        layout.addStretch()
        return tab
    
    def create_advanced_tab(self):
        """Create the advanced controls tab with optimized layout."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Parameters container
        params_container = QHBoxLayout()
        params_container.setSpacing(20)

        # NLM Parameters
        self.nlm_group = QGroupBox("非局部均值参数")
        nlm_layout = QFormLayout()
        nlm_layout.setSpacing(12)
        nlm_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.nlm_h_spin = QSpinBox()
        self.nlm_h_spin.setRange(1, 50)
        self.nlm_h_spin.setValue(10)
        self.nlm_h_spin.setStyleSheet("padding: 12px;")
        nlm_layout.addRow("滤波强度 (h):", self.nlm_h_spin)

        self.nlm_patch_spin = QSpinBox()
        self.nlm_patch_spin.setRange(3, 15)
        self.nlm_patch_spin.setValue(7)
        self.nlm_patch_spin.setStyleSheet("padding: 12px;")
        nlm_layout.addRow("块大小:", self.nlm_patch_spin)

        self.nlm_group.setLayout(nlm_layout)
        params_container.addWidget(self.nlm_group)

        # Bilateral Parameters
        self.bilateral_group = QGroupBox("双边滤波参数")
        bilateral_layout = QFormLayout()
        bilateral_layout.setSpacing(12)
        bilateral_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.bilateral_d_spin = QSpinBox()
        self.bilateral_d_spin.setRange(3, 25)
        self.bilateral_d_spin.setValue(9)
        self.bilateral_d_spin.setStyleSheet("padding: 12px;")
        bilateral_layout.addRow("直径 (d):", self.bilateral_d_spin)

        self.bilateral_color_spin = QSpinBox()
        self.bilateral_color_spin.setRange(10, 200)
        self.bilateral_color_spin.setValue(75)
        self.bilateral_color_spin.setStyleSheet("padding: 12px;")
        bilateral_layout.addRow("颜色标准差:", self.bilateral_color_spin)

        self.bilateral_group.setLayout(bilateral_layout)
        params_container.addWidget(self.bilateral_group)

        # Neural Network Parameters
        self.neural_group = QGroupBox("神经网络参数")
        neural_layout = QFormLayout()
        neural_layout.setSpacing(12)
        neural_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.neural_patch_spin = QSpinBox()
        self.neural_patch_spin.setRange(64, 512)
        self.neural_patch_spin.setValue(256)
        self.neural_patch_spin.setStyleSheet("padding: 12px;")
        neural_layout.addRow("块大小:", self.neural_patch_spin)

        self.neural_stride_spin = QSpinBox()
        self.neural_stride_spin.setRange(32, 256)
        self.neural_stride_spin.setValue(128)
        self.neural_stride_spin.setStyleSheet("padding: 12px;")
        neural_layout.addRow("步长:", self.neural_stride_spin)

        self.neural_group.setLayout(neural_layout)
        params_container.addWidget(self.neural_group)

        layout.addLayout(params_container)

        # Image Info
        info_group = QGroupBox("图像信息")
        info_layout = QVBoxLayout()
        info_layout.setSpacing(12)
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(200)
        self.info_text.setMinimumHeight(150)
        info_layout.addWidget(self.info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Initialize parameter panel visibility
        self.update_parameter_panel()

        return tab
        
    def create_image_display(self):
        """Create the image display area with optimized layout."""
        display_group = QGroupBox("图像预览")
        display_layout = QHBoxLayout()
        display_layout.setSpacing(20)

        # Original image label
        original_container = QVBoxLayout()
        original_container.setSpacing(10)
        original_title = QLabel("原始图像")
        original_title.setStyleSheet("font-weight: bold; font-size: 32px; color: #333333; padding: 10px; qproperty-alignment: AlignCenter;")
        original_container.addWidget(original_title)
        self.original_label = QLabel("未加载图像")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumHeight(500)
        self.original_label.setStyleSheet("""
            border: 2px dashed #bdbdbd;
            background-color: #fafafa;
            border-radius: 8px;
            color: #666666;
            font-style: italic;
            font-size: 26px;
            padding: 20px;
        """)
        original_container.addWidget(self.original_label)
        display_layout.addLayout(original_container)

        # Denoised image label
        denoised_container = QVBoxLayout()
        denoised_container.setSpacing(10)
        denoised_title = QLabel("降噪后图像")
        denoised_title.setStyleSheet("font-weight: bold; font-size: 32px; color: #333333; padding: 10px; qproperty-alignment: AlignCenter;")
        denoised_container.addWidget(denoised_title)
        self.denoised_label = QLabel("处理图像以查看结果")
        self.denoised_label.setAlignment(Qt.AlignCenter)
        self.denoised_label.setMinimumHeight(500)
        self.denoised_label.setStyleSheet("""
            border: 2px dashed #bdbdbd;
            background-color: #fafafa;
            border-radius: 8px;
            color: #666666;
            font-style: italic;
            font-size: 26px;
            padding: 20px;
        """)
        denoised_container.addWidget(self.denoised_label)
        display_layout.addLayout(denoised_container)

        display_group.setLayout(display_layout)
        return display_group
        
    def create_results_panel(self):
        """Create the results panel for metrics display with optimized layout."""
        results_group = QGroupBox("降噪结果")
        results_layout = QVBoxLayout()
        results_layout.setSpacing(15)

        # Progress bar with better styling
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                text-align: center;
                height: 40px;
                font-weight: bold;
                font-size: 24px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 5px;
            }
        """)
        results_layout.addWidget(self.progress_bar)

        # Metrics display with better styling
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMaximumHeight(180)
        self.metrics_text.setMinimumHeight(150)
        self.metrics_text.setText("处理后显示指标")
        self.metrics_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                background-color: #f9f9f9;
                padding: 16px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 26px;
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
            self, "打开 X 光图像", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;所有文件 (*);;DICOM (*.dcm)"
        )

        if file_path:
            self.status_bar.showMessage(f'正在加载 {os.path.basename(file_path)}...')
            success = self.processor.load_image(file_path)
            if success:
                self.display_original_image()
                self.process_btn.setEnabled(True)
                self.save_btn.setEnabled(False)
                self.metrics_text.clear()
                self.denoised_label.setText("处理图像以查看结果")
                self.denoised_label.setPixmap(QPixmap())
                self.status_bar.showMessage('图像加载成功')

                # Update info display
                info = self.processor.get_image_info()
                info_text = f"""文件名：{info.get('filename', 'N/A')}
形状：{info.get('shape', 'N/A')}
数据类型：{info.get('dtype', 'N/A')}
位深度：{info.get('depth', 'N/A')} 位
大小：{info.get('size_mb', 0):.2f} MB"""
                self.info_text.setText(info_text)
            else:
                QMessageBox.critical(self, "错误", "加载图像文件失败。")
                self.status_bar.showMessage('加载图像失败')
                
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
        self.status_bar.showMessage(f'正在使用 {method} 算法处理...')
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
            self.status_bar.showMessage('图像处理成功')
        else:
            # Show detailed error in a scrollable dialog
            error_box = QMessageBox(self)
            error_box.setIcon(QMessageBox.Critical)
            error_box.setWindowTitle("处理错误")
            error_box.setText("处理失败，错误信息如下:")
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
                    font-size: 28px;
                }
                QTextEdit {
                    min-width: 600px;
                    min-height: 300px;
                    font-family: 'Consolas', 'Courier New', monospace;
                    font-size: 26px;
                }
            """)
            error_box.exec_()
            self.status_bar.showMessage('处理失败 - 查看错误对话框了解详情')
            
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
            self, "保存降噪图像", "",
            "PNG 文件 (*.png);;TIFF 文件 (*.tiff *.tif);;JPEG 文件 (*.jpg *.jpeg);;所有文件 (*)"
        )

        if file_path:
            self.status_bar.showMessage('正在保存结果...')
            success = self.processor.save_result(file_path)
            if success:
                self.status_bar.showMessage('结果保存成功')
                QMessageBox.information(self, "成功", "降噪图像保存成功!")
            else:
                QMessageBox.critical(self, "错误", "保存图像失败。")
                self.status_bar.showMessage('保存结果失败')
    
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
