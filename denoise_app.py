import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QProgressBar,
                             QComboBox, QTextEdit, QMessageBox, QSpinBox,
                             QDoubleSpinBox, QGridLayout, QFormLayout, QScrollArea,
                             QSizePolicy, QFrame)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from image_processor import ImageProcessor
import numpy as np


class ProcessingThread(QThread):
    """Background thread for image processing."""
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(int, str)

    def __init__(self, processor, method, params):
        super().__init__()
        self.processor = processor
        self.method = method
        self.params = params

    def run(self):
        try:
            self.progress.emit(10, "正在初始化...")
            self.progress.emit(30, f"正在应用 {self.method} 降噪...")
            self.progress.emit(50, "正在处理图像...")
            self.progress.emit(80, "正在完成...")
            success = self.processor.process_image(self.method, **self.params)
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
    """Main GUI application for X-ray image denoising."""

    def __init__(self):
        super().__init__()
        self.processor = ImageProcessor()
        self.processing_thread = None
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle('X 射线图像降噪器')
        self.setMinimumSize(1280, 800)
        self.resize(1400, 900)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f2f5;
            }
            QWidget {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                font-size: 14px;
            }
            QFrame#card, QFrame#controlPanel {
                background-color: white;
                border-radius: 12px;
            }
            QLabel#sectionTitle {
                font-weight: 600;
                font-size: 15px;
                color: #1a1a1a;
                padding: 8px 0 8px 12px;
                border-left: 4px solid #1976d2;
                margin-bottom: 12px;
            }
            QPushButton {
                background-color: #1976d2;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #1565c0; }
            QPushButton:pressed { background-color: #0d47a1; }
            QPushButton:disabled { background-color: #e0e0e0; color: #9e9e9e; }
            QPushButton#saveButton, QPushButton#processButton {
                background-color: #2e7d32;
            }
            QPushButton#saveButton:hover, QPushButton#processButton:hover {
                background-color: #256628;
            }
            QComboBox {
                padding: 10px 14px;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                background-color: white;
                min-height: 40px;
            }
            QComboBox:hover { border: 1px solid #1976d2; }
            QComboBox::drop-down { border: none; padding-right: 10px; }
            QComboBox::down-arrow {
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #666;
            }
            QComboBox QAbstractItemView {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                selection-background-color: #e3f2fd;
                selection-color: #1976d2;
                padding: 4px;
            }
            QComboBox QAbstractItemView::item {
                min-height: 36px;
                padding: 6px 10px;
                border-radius: 4px;
            }
            QSpinBox, QDoubleSpinBox {
                padding: 10px 14px;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                background-color: white;
                min-height: 40px;
            }
            QSpinBox:hover, QDoubleSpinBox:hover { border: 1px solid #1976d2; }
            QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                border: none;
                width: 24px;
                background: #f5f5f5;
                border-radius: 6px;
                margin: 2px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover,
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
                background: #e0e0e0;
            }
            QTextEdit {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                background-color: #fafafa;
                padding: 12px;
                font-size: 13px;
                font-family: 'Consolas', 'Courier New', monospace;
            }
            QTextEdit:focus { border: 1px solid #1976d2; }
            QProgressBar {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                text-align: center;
                height: 32px;
                font-weight: 600;
                font-size: 13px;
                background-color: #f5f5f5;
            }
            QProgressBar::chunk {
                background-color: #2e7d32;
                border-radius: 7px;
            }
            QStatusBar {
                background-color: white;
                color: #666666;
                border-top: 1px solid #e0e0e0;
                font-size: 13px;
            }
            QScrollBar:vertical {
                background: #f5f5f5;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle {
                background: #c0c0c0;
                border-radius: 4px;
            }
            QScrollBar::handle:hover { background: #a0a0a0; }
            QScrollBar::add-line, QScrollBar::sub-line,
            QScrollBar::add-page, QScrollBar::sub-page {
                height: 0; width: 0; background: none;
            }
            QLabel#imageBox {
                border: 2px dashed #e0e0e0;
                border-radius: 12px;
                background-color: #fafafa;
                color: #999;
                font-style: italic;
            }
            QLabel#imageTitle {
                font-weight: 600;
                font-size: 15px;
                color: #1a1a1a;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # Left panel - Controls
        left_panel = QFrame()
        left_panel.setObjectName("controlPanel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(16)
        left_layout.setContentsMargins(20, 20, 20, 20)

        title_label = QLabel("控制面板")
        title_label.setObjectName("sectionTitle")
        left_layout.addWidget(title_label)

        # File operations
        file_label = QLabel("文件操作")
        file_label.setObjectName("sectionTitle")
        file_label.setCursor(Qt.PointingHandCursor)
        left_layout.addWidget(file_label)

        file_layout = QHBoxLayout()
        file_layout.setSpacing(12)
        self.load_btn = QPushButton("📁 加载 X 光图像")
        self.load_btn.clicked.connect(self.load_image)
        file_layout.addWidget(self.load_btn)

        self.save_btn = QPushButton("💾 保存结果")
        self.save_btn.setObjectName("saveButton")
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)
        file_layout.addWidget(self.save_btn)
        left_layout.addLayout(file_layout)

        # Algorithm selection
        algo_label = QLabel("降噪算法")
        algo_label.setObjectName("sectionTitle")
        algo_label.setCursor(Qt.PointingHandCursor)
        left_layout.addWidget(algo_label)

        algo_layout = QHBoxLayout()
        algo_layout.setSpacing(12)
        algo_label2 = QLabel("算法")
        algo_label2.setStyleSheet("color: #666;")
        algo_layout.addWidget(algo_label2)

        self.algorithm_combo = QComboBox()
        self.update_algorithm_list()
        self.algorithm_combo.setMinimumWidth(200)
        algo_layout.addWidget(self.algorithm_combo)

        algo_layout.addSpacing(20)
        strength_label = QLabel("强度")
        strength_label.setStyleSheet("color: #666;")
        algo_layout.addWidget(strength_label)

        self.strength_combo = QComboBox()
        self.strength_combo.addItems(["低", "中", "高"])
        self.strength_combo.setCurrentIndex(1)
        self.strength_combo.setMinimumWidth(70)
        algo_layout.addWidget(self.strength_combo)
        algo_layout.addStretch()
        left_layout.addLayout(algo_layout)

        # Parameters
        params_label = QLabel("参数设置")
        params_label.setObjectName("sectionTitle")
        params_label.setCursor(Qt.PointingHandCursor)
        left_layout.addWidget(params_label)

        params_container = QHBoxLayout()
        params_container.setSpacing(12)

        self.nlm_group = QWidget()
        nlm_layout = QFormLayout()
        nlm_layout.setSpacing(8)
        self.nlm_h_spin = QSpinBox()
        self.nlm_h_spin.setRange(1, 50)
        self.nlm_h_spin.setValue(10)
        nlm_layout.addRow("滤波强度 h:", self.nlm_h_spin)
        self.nlm_patch_spin = QSpinBox()
        self.nlm_patch_spin.setRange(3, 15)
        self.nlm_patch_spin.setValue(7)
        nlm_layout.addRow("块大小:", self.nlm_patch_spin)
        self.nlm_group.setLayout(nlm_layout)
        params_container.addWidget(self.nlm_group)

        self.bilateral_group = QWidget()
        bilateral_layout = QFormLayout()
        bilateral_layout.setSpacing(8)
        self.bilateral_d_spin = QSpinBox()
        self.bilateral_d_spin.setRange(3, 25)
        self.bilateral_d_spin.setValue(9)
        bilateral_layout.addRow("直径 d:", self.bilateral_d_spin)
        self.bilateral_color_spin = QSpinBox()
        self.bilateral_color_spin.setRange(10, 200)
        self.bilateral_color_spin.setValue(75)
        bilateral_layout.addRow("颜色标准差:", self.bilateral_color_spin)
        self.bilateral_group.setLayout(bilateral_layout)
        params_container.addWidget(self.bilateral_group)

        self.neural_group = QWidget()
        neural_layout = QFormLayout()
        neural_layout.setSpacing(8)
        self.neural_patch_spin = QSpinBox()
        self.neural_patch_spin.setRange(64, 512)
        self.neural_patch_spin.setValue(256)
        neural_layout.addRow("块大小:", self.neural_patch_spin)
        self.neural_stride_spin = QSpinBox()
        self.neural_stride_spin.setRange(32, 256)
        self.neural_stride_spin.setValue(128)
        neural_layout.addRow("步长:", self.neural_stride_spin)
        self.neural_group.setLayout(neural_layout)
        params_container.addWidget(self.neural_group)

        left_layout.addLayout(params_container)

        # Image info
        info_label = QLabel("图像信息")
        info_label.setObjectName("sectionTitle")
        info_label.setCursor(Qt.PointingHandCursor)
        left_layout.addWidget(info_label)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(100)
        self.info_text.setMinimumHeight(80)
        left_layout.addWidget(self.info_text)

        # Process button
        self.process_btn = QPushButton("▶ 开始降噪")
        self.process_btn.setObjectName("processButton")
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setEnabled(False)
        self.process_btn.setMinimumHeight(50)
        self.process_btn.setStyleSheet("font-size: 16px; padding: 16px;")
        left_layout.addWidget(self.process_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        left_layout.addStretch()

        # Right panel - Image display
        right_panel = QFrame()
        right_panel.setObjectName("card")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(16)
        right_layout.setContentsMargins(20, 20, 20, 20)

        images_layout = QHBoxLayout()
        images_layout.setSpacing(16)

        original_container = QVBoxLayout()
        original_title = QLabel("原始图像")
        original_title.setObjectName("imageTitle")
        original_title.setAlignment(Qt.AlignCenter)
        original_container.addWidget(original_title)

        self.original_label = QLabel("未加载图像")
        self.original_label.setObjectName("imageBox")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(400, 350)
        self.original_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        original_container.addWidget(self.original_label, 1)
        images_layout.addLayout(original_container, 1)

        denoised_container = QVBoxLayout()
        denoised_title = QLabel("降噪后图像")
        denoised_title.setObjectName("imageTitle")
        denoised_title.setAlignment(Qt.AlignCenter)
        denoised_container.addWidget(denoised_title)

        self.denoised_label = QLabel("处理图像以查看结果")
        self.denoised_label.setObjectName("imageBox")
        self.denoised_label.setAlignment(Qt.AlignCenter)
        self.denoised_label.setMinimumSize(400, 350)
        self.denoised_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        denoised_container.addWidget(self.denoised_label, 1)
        images_layout.addLayout(denoised_container, 1)

        right_layout.addLayout(images_layout, 1)

        results_label = QLabel("评估指标")
        results_label.setObjectName("sectionTitle")
        results_label.setCursor(Qt.PointingHandCursor)
        right_layout.addWidget(results_label)

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMaximumHeight(80)
        self.metrics_text.setMinimumHeight(60)
        self.metrics_text.setText("处理后显示 PSNR、SSIM、MSE 指标")
        right_layout.addWidget(self.metrics_text)
        right_layout.addStretch()

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)

        self.status_bar = self.statusBar()
        self.status_bar.showMessage('就绪 - 加载图像开始')
        self.update_parameter_panel()

    def update_algorithm_list(self):
        """Update the algorithm dropdown with available methods."""
        self.algorithm_combo.clear()
        methods = self.processor.get_supported_methods()
        for key, name in methods:
            self.algorithm_combo.addItem(name, key)
        self.algorithm_combo.currentIndexChanged.connect(self.update_parameter_panel)

    def update_parameter_panel(self):
        """Update parameter panel visibility based on selected algorithm."""
        method = self.algorithm_combo.currentData()
        self.nlm_group.setVisible(method == 'nlm')
        self.bilateral_group.setVisible(method == 'bilateral')
        self.neural_group.setVisible(method == 'neural')

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
                info = self.processor.get_image_info()
                self.info_text.setText(f"文件名：{info.get('filename', 'N/A')}\n形状：{info.get('shape', 'N/A')}\n数据类型：{info.get('dtype', 'N/A')}\n位深度：{info.get('depth', 'N/A')} 位")
            else:
                QMessageBox.critical(self, "错误", "加载图像文件失败。")

    def display_original_image(self):
        """Display the original image."""
        import cv2
        try:
            original_img = self.processor.get_original_image()
            if original_img is not None:
                rgb_img = self._convert_to_rgb(original_img)
                height, width = rgb_img.shape[:2]
                bytes_per_line = 3 * width
                q_img = QImage(rgb_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                scaled_pixmap = pixmap.scaled(self.original_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.original_label.setPixmap(scaled_pixmap)
                self.original_label.setText("")
        except Exception as e:
            print(f"Display error: {e}")
            self.original_label.setText("Error displaying image")

    def _convert_to_rgb(self, img):
        """Convert image to RGB for display."""
        import cv2
        if len(img.shape) == 3:
            if img.shape[2] == 3:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                gray = img[:, :, 0]
                rgb_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if rgb_img.dtype != np.uint8:
            if rgb_img.dtype == np.uint16:
                rgb_img = (rgb_img.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
            else:
                rgb_img = (rgb_img / rgb_img.max() * 255).astype(np.uint8)
        return rgb_img

    def process_image(self):
        """Process the loaded image."""
        method = self.algorithm_combo.currentData()
        params = {}

        if method == 'hybrid':
            strength_map = {"低": "low", "中": "medium", "高": "high"}
            params['strength'] = strength_map.get(self.strength_combo.currentText(), "medium")
        elif method == 'nlm':
            params['h'] = self.nlm_h_spin.value()
            params['patch_size'] = self.nlm_patch_spin.value()
        elif method == 'bilateral':
            params['d'] = self.bilateral_d_spin.value()
            params['sigma_color'] = self.bilateral_color_spin.value()
            params['sigma_space'] = self.bilateral_color_spin.value()
        elif method == 'neural':
            params['patch_size'] = self.neural_patch_spin.value()
            params['stride'] = self.neural_stride_spin.value()

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage(f'正在使用 {method} 算法处理...')
        self.process_btn.setEnabled(False)

        self.processing_thread = ProcessingThread(self.processor, method, params)
        self.processing_thread.progress.connect(self.update_progress)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.start()

    def update_progress(self, value, stage_message=""):
        """Update progress bar."""
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
            QMessageBox.critical(self, "处理错误", message)

    def display_denoised_image(self):
        """Display the denoised image."""
        import cv2
        try:
            denoised_img = self.processor.get_denoised_image()
            if denoised_img is not None:
                rgb_img = self._convert_to_rgb(denoised_img)
                height, width = rgb_img.shape[:2]
                bytes_per_line = 3 * width
                q_img = QImage(rgb_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                scaled_pixmap = pixmap.scaled(self.denoised_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.denoised_label.setPixmap(scaled_pixmap)
                self.denoised_label.setText("")
        except Exception as e:
            print(f"Display error: {e}")
            self.denoised_label.setText("Error displaying result")

    def calculate_and_display_metrics(self):
        """Calculate and display metrics."""
        metrics = self.processor.calculate_metrics()
        if metrics:
            psnr = metrics.get('psnr', 0)
            ssim_val = metrics.get('ssim', 0)
            mse = metrics.get('mse', 0)
            self.metrics_text.setText(f"PSNR: {psnr:.2f} dB\nSSIM: {ssim_val:.4f}\nMSE: {mse:.6f}")

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

    def closeEvent(self, event):
        """Handle window close event."""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.wait(1000)
        event.accept()


try:
    import cv2
except ImportError:
    pass
