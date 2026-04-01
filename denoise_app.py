"""
Main GUI application for X-ray image denoising and super-resolution.
Two-step workflow: Denoise -> Super-Resolution
"""

import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QProgressBar,
                             QComboBox, QTextEdit, QMessageBox, QSpinBox, QCheckBox,
                             QDoubleSpinBox, QGridLayout, QFormLayout, QScrollArea,
                             QSizePolicy, QFrame, QTabWidget, QGroupBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from image_processor import ImageProcessor
from super_resolution import get_supported_sr_methods
import numpy as np


class ProcessingThread(QThread):
    """Background thread for image processing."""
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(int, str)

    def __init__(self, processor, step, method=None, sr_params=None, denoise_params=None):
        super().__init__()
        self.processor = processor
        self.step = step  # 'denoise' or 'sr'
        self.method = method
        self.sr_params = sr_params or {}
        self.denoise_params = denoise_params or {}

    def run(self):
        try:
            if self.step == 'denoise':
                self.progress.emit(10, "正在初始化...")
                self.progress.emit(30, f"正在应用 {self.method} 降噪...")
                self.progress.emit(50, "正在处理图像...")
                self.progress.emit(80, "正在完成...")
                success = self.processor.process_image(self.method, **self.denoise_params)
                self.progress.emit(100, "降噪完成")
                if success:
                    self.finished.emit(True, "降噪处理成功完成")
                else:
                    self.finished.emit(False, "降噪处理失败 - 未返回结果")
            elif self.step == 'sr':
                self.progress.emit(10, "正在初始化超分辨率...")
                self.progress.emit(30, f"正在应用 {self.sr_params.get('method', 'lanczos')} 算法...")
                self.progress.emit(50, f"正在放大 {self.sr_params.get('scale', 2.0)}x...")
                self.progress.emit(80, "正在增强细节...")
                success = self.processor.apply_super_resolution(**self.sr_params)
                self.progress.emit(100, "超分辨率完成")
                if success:
                    self.finished.emit(True, "超分辨率处理成功完成")
                else:
                    self.finished.emit(False, "超分辨率处理失败")
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
        self.setWindowTitle('X 射线图像降噪与超分辨率重构系统')
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #f1f5f9;
            }
            QWidget {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
                font-size: 13px;
                color: #333;
            }
            QFrame#card, QFrame#controlPanel {
                background-color: white;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
            }
            QLabel#sectionTitle {
                font-weight: 600;
                font-size: 14px;
                color: #1e293b;
                padding: 8px 0 8px 12px;
                border-left: 4px solid #3b82f6;
                background-color: #f8fafc;
                border-radius: 4px;
            }
            QLabel#stepTitle {
                font-weight: 700;
                font-size: 16px;
                color: #0f172a;
                padding: 12px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3b82f6, stop:1 #8b5cf6);
                color: white;
                border-radius: 8px;
            }
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-weight: 500;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #2563eb; }
            QPushButton:pressed { background-color: #1d4ed8; }
            QPushButton:disabled { background-color: #cbd5e1; color: #94a3b8; }
            QPushButton#primaryBtn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #16a34a, stop:1 #059669);
                font-size: 15px;
                font-weight: 600;
                padding: 14px 24px;
            }
            QPushButton#primaryBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #15803d, stop:1 #047857);
            }
            QPushButton#primaryBtn:disabled {
                background: #cbd5e1;
            }
            QPushButton#secondaryBtn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8b5cf6, stop:1 #7c3aed);
                font-size: 15px;
                font-weight: 600;
                padding: 14px 24px;
            }
            QPushButton#secondaryBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7c3aed, stop:1 #6d28d9);
            }
            QPushButton#secondaryBtn:disabled {
                background: #cbd5e1;
            }
            QPushButton#saveBtn {
                background-color: #0891b2;
                font-weight: 600;
            }
            QPushButton#saveBtn:hover { background-color: #0e7490; }
            QPushButton#saveBtn:disabled { background-color: #cbd5e1; }
            QComboBox {
                padding: 8px 12px;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                background-color: white;
                min-height: 36px;
            }
            QComboBox:hover { border: 1px solid #3b82f6; }
            QComboBox:focus { border: 1px solid #3b82f6; }
            QComboBox::drop-down { border: none; padding-right: 8px; }
            QComboBox::down-arrow {
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #64748b;
            }
            QSpinBox, QDoubleSpinBox {
                padding: 8px 12px;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                background-color: white;
                min-height: 36px;
            }
            QSpinBox:hover, QDoubleSpinBox:hover { border: 1px solid #3b82f6; }
            QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                border: none;
                width: 24px;
                background: #f1f5f9;
                border-radius: 6px;
                margin: 2px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover,
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
                background: #e2e8f0;
            }
            QTextEdit {
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                background-color: #f8fafc;
                padding: 8px;
                font-size: 12px;
                font-family: 'Consolas', 'Courier New', 'Monaco', monospace;
            }
            QTextEdit:focus { border: 1px solid #3b82f6; }
            QProgressBar {
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                text-align: center;
                height: 28px;
                font-weight: 600;
                font-size: 13px;
                background-color: #f8fafc;
                color: #475569;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3b82f6, stop:1 #8b5cf6);
                border-radius: 7px;
            }
            QStatusBar {
                background-color: white;
                color: #64748b;
                border-top: 1px solid #e2e8f0;
                font-size: 12px;
            }
            QScrollBar:vertical {
                background: #f1f5f9;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle {
                background: #cbd5e1;
                border-radius: 4px;
            }
            QScrollBar::handle:hover { background: #94a3b8; }
            QScrollBar::add-line, QScrollBar::sub-line,
            QScrollBar::add-page, QScrollBar::sub-page {
                height: 0; width: 0; background: none;
            }
            QLabel#imageBox {
                border: 2px dashed #cbd5e1;
                border-radius: 12px;
                background-color: #f8fafc;
                color: #94a3b8;
                font-style: italic;
                font-size: 14px;
            }
            QLabel#imageTitle {
                font-weight: 600;
                font-size: 14px;
                color: #1e293b;
                padding: 8px;
                background-color: #f1f5f9;
                border-radius: 8px;
            }
            QGroupBox {
                font-weight: 600;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                color: #475569;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # Left panel - Controls
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)

        # Right panel - Image display
        right_panel = self.create_display_panel()
        main_layout.addWidget(right_panel, 2)

        self.status_bar = self.statusBar()
        self.status_bar.showMessage('就绪 - 加载图像开始')

        self.update_parameter_panel()

    def create_control_panel(self):
        """Create the left control panel with two steps."""
        panel = QFrame()
        panel.setObjectName("controlPanel")
        layout = QVBoxLayout(panel)
        layout.setSpacing(14)
        layout.setContentsMargins(16, 16, 16, 16)

        # File operations
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(10)

        self.load_btn = QPushButton("📁 加载图像")
        self.load_btn.clicked.connect(self.load_image)
        self.load_btn.setMinimumHeight(42)
        file_layout.addWidget(self.load_btn)

        save_layout = QHBoxLayout()
        save_layout.setSpacing(10)
        self.save_denoise_btn = QPushButton("💾 保存降噪图")
        self.save_denoise_btn.setObjectName("saveBtn")
        self.save_denoise_btn.clicked.connect(lambda: self.save_result(use_sr=False))
        self.save_denoise_btn.setEnabled(False)
        self.save_denoise_btn.setMinimumHeight(38)
        save_layout.addWidget(self.save_denoise_btn)

        self.save_sr_btn = QPushButton("💾 保存 SR 图")
        self.save_sr_btn.setObjectName("saveBtn")
        self.save_sr_btn.clicked.connect(lambda: self.save_result(use_sr=True))
        self.save_sr_btn.setEnabled(False)
        self.save_sr_btn.setMinimumHeight(38)
        save_layout.addWidget(self.save_sr_btn)
        file_layout.addLayout(save_layout)

        layout.addWidget(file_group)

        # Step 1: Denoising
        step1_group = QGroupBox("步骤 1: 降噪处理")
        step1_layout = QVBoxLayout(step1_group)
        step1_layout.setSpacing(12)

        # Algorithm selection
        algo_layout = QFormLayout()
        algo_layout.setSpacing(10)

        self.algorithm_combo = QComboBox()
        self.update_algorithm_list()
        self.algorithm_combo.setMinimumHeight(36)
        algo_layout.addRow("算法:", self.algorithm_combo)

        self.strength_combo = QComboBox()
        self.strength_combo.addItems(["低", "中", "高"])
        self.strength_combo.setCurrentIndex(1)
        self.strength_combo.setMinimumHeight(36)
        algo_layout.addRow("强度:", self.strength_combo)

        step1_layout.addLayout(algo_layout)

        # Denoising parameters
        self.denoise_params_widget = self.create_denoise_params()
        step1_layout.addWidget(self.denoise_params_widget)

        self.denoise_btn = QPushButton("▶ 执行降噪")
        self.denoise_btn.setObjectName("primaryBtn")
        self.denoise_btn.clicked.connect(self.denoise_image)
        self.denoise_btn.setEnabled(False)
        self.denoise_btn.setMinimumHeight(48)
        step1_layout.addWidget(self.denoise_btn)

        self.denoise_progress = QProgressBar()
        self.denoise_progress.setVisible(False)
        step1_layout.addWidget(self.denoise_progress)

        self.denoise_status = QLabel("等待处理...")
        self.denoise_status.setStyleSheet("color: #64748b; font-size: 12px;")
        step1_layout.addWidget(self.denoise_status)

        layout.addWidget(step1_group)

        # Step 2: Super-Resolution
        step2_group = QGroupBox("步骤 2: 超分辨率重构")
        step2_layout = QVBoxLayout(step2_group)
        step2_layout.setSpacing(12)

        # SR parameters
        sr_layout = QFormLayout()
        sr_layout.setSpacing(10)

        self.sr_method_combo = QComboBox()
        for key, name in get_supported_sr_methods():
            self.sr_method_combo.addItem(name, key)
        self.sr_method_combo.setCurrentIndex(1)  # Lanczos default
        self.sr_method_combo.setMinimumHeight(36)
        sr_layout.addRow("插值方法:", self.sr_method_combo)

        self.sr_scale_combo = QComboBox()
        self.sr_scale_combo.addItems(["1.5x", "2.0x", "3.0x", "4.0x"])
        self.sr_scale_combo.setCurrentIndex(1)  # 2.0x default
        self.sr_scale_combo.setMinimumHeight(36)
        sr_layout.addRow("放大倍数:", self.sr_scale_combo)

        step2_layout.addLayout(sr_layout)

        self.sr_btn = QPushButton("🔍 执行超分辨率")
        self.sr_btn.setObjectName("secondaryBtn")
        self.sr_btn.clicked.connect(self.apply_sr)
        self.sr_btn.setEnabled(False)
        self.sr_btn.setMinimumHeight(48)
        step2_layout.addWidget(self.sr_btn)

        self.sr_progress = QProgressBar()
        self.sr_progress.setVisible(False)
        step2_layout.addWidget(self.sr_progress)

        self.sr_status = QLabel("等待降噪完成...")
        self.sr_status.setStyleSheet("color: #64748b; font-size: 12px;")
        step2_layout.addWidget(self.sr_status)

        layout.addWidget(step2_group)

        # Image info
        info_group = QGroupBox("图像信息")
        info_layout = QVBoxLayout(info_group)
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(90)
        self.info_text.setMinimumHeight(80)
        info_layout.addWidget(self.info_text)
        layout.addWidget(info_group)

        layout.addStretch()

        return panel

    def create_denoise_params(self):
        """Create denoising parameter widgets."""
        widget = QFrame()
        widget.setStyleSheet("QFrame { background-color: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0; }")
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        # NLM parameters
        self.nlm_group = QGroupBox("Non-local Means 参数")
        nlm_layout = QHBoxLayout()
        nlm_layout.setSpacing(10)

        self.nlm_h_spin = QSpinBox()
        self.nlm_h_spin.setRange(1, 50)
        self.nlm_h_spin.setValue(10)
        self.nlm_h_spin.setMinimumHeight(32)
        nlm_layout.addWidget(QLabel("滤波强度:"))
        nlm_layout.addWidget(self.nlm_h_spin)

        self.nlm_patch_spin = QSpinBox()
        self.nlm_patch_spin.setRange(3, 15)
        self.nlm_patch_spin.setValue(7)
        self.nlm_patch_spin.setMinimumHeight(32)
        nlm_layout.addWidget(QLabel("块大小:"))
        nlm_layout.addWidget(self.nlm_patch_spin)

        self.nlm_group.setLayout(nlm_layout)
        layout.addWidget(self.nlm_group)

        # Bilateral parameters
        self.bilateral_group = QGroupBox("Bilateral 参数")
        bilateral_layout = QHBoxLayout()
        bilateral_layout.setSpacing(10)

        self.bilateral_d_spin = QSpinBox()
        self.bilateral_d_spin.setRange(3, 25)
        self.bilateral_d_spin.setValue(9)
        self.bilateral_d_spin.setMinimumHeight(32)
        bilateral_layout.addWidget(QLabel("直径:"))
        bilateral_layout.addWidget(self.bilateral_d_spin)

        self.bilateral_color_spin = QSpinBox()
        self.bilateral_color_spin.setRange(10, 200)
        self.bilateral_color_spin.setValue(75)
        self.bilateral_color_spin.setMinimumHeight(32)
        bilateral_layout.addWidget(QLabel("强度:"))
        bilateral_layout.addWidget(self.bilateral_color_spin)

        self.bilateral_group.setLayout(bilateral_layout)
        layout.addWidget(self.bilateral_group)

        # Neural parameters
        self.neural_group = QGroupBox("神经网络参数")
        neural_layout = QHBoxLayout()
        neural_layout.setSpacing(10)

        self.neural_patch_spin = QSpinBox()
        self.neural_patch_spin.setRange(64, 512)
        self.neural_patch_spin.setValue(256)
        self.neural_patch_spin.setMinimumHeight(32)
        neural_layout.addWidget(QLabel("块大小:"))
        neural_layout.addWidget(self.neural_patch_spin)

        self.neural_stride_spin = QSpinBox()
        self.neural_stride_spin.setRange(32, 256)
        self.neural_stride_spin.setValue(128)
        self.neural_stride_spin.setMinimumHeight(32)
        neural_layout.addWidget(QLabel("步长:"))
        neural_layout.addWidget(self.neural_stride_spin)

        self.neural_group.setLayout(neural_layout)
        layout.addWidget(self.neural_group)

        return widget

    def create_display_panel(self):
        """Create the right display panel."""
        panel = QFrame()
        panel.setObjectName("card")
        layout = QVBoxLayout(panel)
        layout.setSpacing(14)
        layout.setContentsMargins(16, 16, 16, 16)

        # Tabs for different image views
        self.image_tabs = QTabWidget()
        self.image_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                background-color: #f8fafc;
            }
            QTabBar::tab {
                background-color: #e2e8f0;
                padding: 10px 20px;
                border-radius: 6px;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background-color: white;
                font-weight: 600;
            }
            QTabBar::tab:hover {
                background-color: #cbd5e1;
            }
        """)

        # Original image tab
        original_tab = QWidget()
        original_layout = QVBoxLayout(original_tab)
        original_layout.setContentsMargins(12, 12, 12, 12)
        self.original_label = QLabel("未加载图像")
        self.original_label.setObjectName("imageBox")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(500, 400)
        original_layout.addWidget(self.original_label)
        self.image_tabs.addTab(original_tab, "原始图像")

        # Denoised image tab
        denoised_tab = QWidget()
        denoised_layout = QVBoxLayout(denoised_tab)
        denoised_layout.setContentsMargins(12, 12, 12, 12)
        self.denoised_label = QLabel("降噪后显示")
        self.denoised_label.setObjectName("imageBox")
        self.denoised_label.setAlignment(Qt.AlignCenter)
        self.denoised_label.setMinimumSize(500, 400)
        denoised_layout.addWidget(self.denoised_label)
        self.image_tabs.addTab(denoised_tab, "降噪结果")

        # SR image tab
        sr_tab = QWidget()
        sr_layout = QVBoxLayout(sr_tab)
        sr_layout.setContentsMargins(12, 12, 12, 12)
        self.sr_label = QLabel("超分辨率后显示")
        self.sr_label.setObjectName("imageBox")
        self.sr_label.setAlignment(Qt.AlignCenter)
        self.sr_label.setMinimumSize(500, 400)
        sr_layout.addWidget(self.sr_label)
        self.image_tabs.addTab(sr_tab, "超分辨率结果")

        layout.addWidget(self.image_tabs, 1)

        # Metrics section
        metrics_group = QGroupBox("评估指标")
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(15)

        # Denoising metrics
        denoise_metrics_widget = QWidget()
        dm_layout = QVBoxLayout(denoise_metrics_widget)
        dm_layout.setContentsMargins(0, 0, 0, 0)
        dm_layout.addWidget(QLabel("<b>降噪质量</b>"))
        self.denoise_metrics_text = QTextEdit()
        self.denoise_metrics_text.setReadOnly(True)
        self.denoise_metrics_text.setMaximumHeight(70)
        self.denoise_metrics_text.setText("降噪后显示 PSNR、SSIM、MSE 指标")
        dm_layout.addWidget(self.denoise_metrics_text)
        metrics_layout.addWidget(denoise_metrics_widget, 1)

        # SR metrics
        sr_metrics_widget = QWidget()
        smr_layout = QVBoxLayout(sr_metrics_widget)
        smr_layout.setContentsMargins(0, 0, 0, 0)
        smr_layout.addWidget(QLabel("<b>超分辨率质量</b>"))
        self.sr_metrics_text = QTextEdit()
        self.sr_metrics_text.setReadOnly(True)
        self.sr_metrics_text.setMaximumHeight(70)
        self.sr_metrics_text.setText("超分辨率后显示指标")
        smr_layout.addWidget(self.sr_metrics_text)
        metrics_layout.addWidget(sr_metrics_widget, 1)

        metrics_layout.setStretch(0, 1)
        metrics_layout.setStretch(1, 1)

        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)

        return panel

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
                self.denoise_btn.setEnabled(True)
                self.save_denoise_btn.setEnabled(False)
                self.save_sr_btn.setEnabled(False)
                self.denoise_metrics_text.setText("降噪后显示 PSNR、SSIM、MSE 指标")
                self.sr_metrics_text.setText("超分辨率后显示指标")
                self.denoised_label.setText("降噪后显示")
                self.denoised_label.setPixmap(QPixmap())
                self.sr_label.setText("超分辨率后显示")
                self.sr_label.setPixmap(QPixmap())
                self.status_bar.showMessage('图像加载成功')
                info = self.processor.get_image_info()
                self.info_text.setText(
                    f"文件名：{info.get('filename', 'N/A')}\n"
                    f"形状：{info.get('shape', 'N/A')}\n"
                    f"数据类型：{info.get('dtype', 'N/A')}\n"
                    f"位深度：{info.get('depth', 'N/A')} 位"
                )
                self.sr_btn.setEnabled(False)
                self.sr_status.setText("等待降噪完成...")
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
        # Handle images with alpha channel (4 channels)
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                # Remove alpha channel, keep only BGR
                bgr_img = img[:, :, :3]
                rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            elif img.shape[2] == 3:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                # Single channel or other
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

    def denoise_image(self):
        """Step 1: Denoise the loaded image."""
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

        self.denoise_progress.setVisible(True)
        self.denoise_progress.setValue(0)
        self.denoise_btn.setEnabled(False)
        self.denoise_status.setText("正在降噪...")

        self.processing_thread = ProcessingThread(
            self.processor, step='denoise', method=method, denoise_params=params
        )
        self.processing_thread.progress.connect(self.update_denoise_progress)
        self.processing_thread.finished.connect(self.denoise_finished)
        self.processing_thread.start()

    def update_denoise_progress(self, value, stage_message=""):
        """Update denoise progress bar."""
        self.denoise_progress.setValue(value)
        if stage_message:
            self.denoise_status.setText(stage_message)

    def denoise_finished(self, success, message):
        """Handle denoising completion."""
        self.denoise_progress.setVisible(False)
        self.denoise_btn.setEnabled(True)
        if success:
            self.display_denoised_image()
            self.calculate_and_display_denoise_metrics()
            self.save_denoise_btn.setEnabled(True)
            self.sr_btn.setEnabled(True)
            self.denoise_status.setText("降噪完成 ✓")
            self.status_bar.showMessage('降噪处理成功')
        else:
            QMessageBox.critical(self, "处理错误", message)
            self.denoise_status.setText("降噪失败")

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
                self.image_tabs.setCurrentIndex(1)
        except Exception as e:
            print(f"Display error: {e}")
            self.denoised_label.setText("Error displaying result")

    def calculate_and_display_denoise_metrics(self):
        """Calculate and display denoising metrics."""
        metrics = self.processor.calculate_metrics()
        if metrics:
            psnr = metrics.get('psnr', 0)
            ssim_val = metrics.get('ssim', 0)
            mse = metrics.get('mse', 0)
            self.denoise_metrics_text.setText(f"PSNR: {psnr:.2f} dB\nSSIM: {ssim_val:.4f}\nMSE: {mse:.6f}")

    def apply_sr(self):
        """Step 2: Apply super-resolution to denoised image."""
        scale_map = {"1.5x": 1.5, "2.0x": 2.0, "3.0x": 3.0, "4.0x": 4.0}
        scale = scale_map.get(self.sr_scale_combo.currentText(), 2.0)
        method = self.sr_method_combo.currentData()

        sr_params = {
            'scale': scale,
            'method': method,
            'enhance_edges': True,
            'enhance_contrast': True
        }

        self.sr_progress.setVisible(True)
        self.sr_progress.setValue(0)
        self.sr_btn.setEnabled(False)
        self.sr_status.setText("正在超分辨率...")

        self.processing_thread = ProcessingThread(
            self.processor, step='sr', sr_params=sr_params
        )
        self.processing_thread.progress.connect(self.update_sr_progress)
        self.processing_thread.finished.connect(self.sr_finished)
        self.processing_thread.start()

    def update_sr_progress(self, value, stage_message=""):
        """Update SR progress bar."""
        self.sr_progress.setValue(value)
        if stage_message:
            self.sr_status.setText(stage_message)

    def sr_finished(self, success, message):
        """Handle super-resolution completion."""
        self.sr_progress.setVisible(False)
        self.sr_btn.setEnabled(True)
        if success:
            self.display_sr_image()
            self.calculate_and_display_sr_metrics()
            self.save_sr_btn.setEnabled(True)
            self.sr_status.setText("超分辨率完成 ✓")
            self.status_bar.showMessage('超分辨率处理成功')
        else:
            QMessageBox.critical(self, "处理错误", message)
            self.sr_status.setText("超分辨率失败")

    def display_sr_image(self):
        """Display the super-resolution image."""
        import cv2
        try:
            sr_img = self.processor.get_sr_image()
            if sr_img is not None:
                rgb_img = self._convert_to_rgb(sr_img)
                height, width = rgb_img.shape[:2]
                bytes_per_line = 3 * width
                q_img = QImage(rgb_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                scaled_pixmap = pixmap.scaled(self.sr_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.sr_label.setPixmap(scaled_pixmap)
                self.sr_label.setText("")
                self.image_tabs.setCurrentIndex(2)
        except Exception as e:
            print(f"Display error: {e}")
            self.sr_label.setText("Error displaying result")

    def calculate_and_display_sr_metrics(self):
        """Calculate and display SR metrics."""
        metrics = self.processor.get_sr_metrics()
        if metrics:
            psnr = metrics.get('psnr', 0)
            ssim_val = metrics.get('ssim', 0)
            mse = metrics.get('mse', 0)
            self.sr_metrics_text.setText(f"PSNR: {psnr:.2f} dB\nSSIM: {ssim_val:.4f}\nMSE: {mse:.6f}")

    def save_result(self, use_sr: bool = False):
        """Save the processed image."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "",
            "PNG 文件 (*.png);;TIFF 文件 (*.tiff *.tif);;JPEG 文件 (*.jpg *.jpeg);;所有文件 (*)"
        )
        if file_path:
            self.status_bar.showMessage('正在保存结果...')
            success = self.processor.save_result(file_path, use_sr=use_sr)
            if success:
                self.status_bar.showMessage('结果保存成功')
                QMessageBox.information(self, "成功", "图像保存成功!")
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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DenoiseApp()
    window.show()
    sys.exit(app.exec_())
