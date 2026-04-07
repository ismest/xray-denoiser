"""
Main GUI application for X-ray image denoising and super-resolution.
Two-step workflow: Denoise -> Super-Resolution

Refactored to support both QMainWindow (standalone) and QWidget (embedded) modes.
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


class DenoiseWidget(QWidget):
    """Denoising and Super-Resolution widget (can be embedded in main window)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.processor = ImageProcessor()
        self.processing_thread = None
        self.parent_window = parent
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        self.setStyleSheet("""
            QWidget {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
                font-size: 15px;
                color: #1e293b;
            }
            QFrame#card, QFrame#controlPanel {
                background-color: white;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
            }
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 15px;
            }
            QPushButton:hover { background-color: #2563eb; }
            QPushButton:pressed { background-color: #1d4ed8; }
            QPushButton:disabled { background-color: #cbd5e1; color: #94a3b8; }
            QPushButton#primaryBtn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #16a34a, stop:1 #059669);
                font-size: 16px;
                font-weight: 600;
                padding: 16px 28px;
            }
            QPushButton#primaryBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #15803d, stop:1 #047857);
            }
            QPushButton#primaryBtn:disabled { background: #cbd5e1; }
            QPushButton#secondaryBtn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8b5cf6, stop:1 #7c3aed);
                font-size: 16px;
                font-weight: 600;
                padding: 16px 28px;
            }
            QPushButton#secondaryBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7c3aed, stop:1 #6d28d9);
            }
            QPushButton#secondaryBtn:disabled { background: #cbd5e1; }
            QPushButton#saveBtn {
                background-color: #0891b2;
                font-weight: 600;
                font-size: 16px;
            }
            QPushButton#saveBtn:hover { background-color: #0e7490; }
            QPushButton#saveBtn:disabled { background-color: #cbd5e1; }
            QComboBox {
                padding: 10px 14px;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                background-color: white;
                min-height: 44px;
                font-size: 15px;
            }
            QComboBox:hover { border: 1px solid #3b82f6; }
            QTextEdit {
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                background-color: #f8fafc;
                padding: 10px;
                font-size: 13px;
                font-family: 'Consolas', 'Courier New', 'Monaco', monospace;
            }
            QProgressBar {
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                text-align: center;
                height: 36px;
                font-weight: 600;
                font-size: 15px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3b82f6, stop:1 #8b5cf6);
                border-radius: 7px;
            }
            QLabel#imageBox {
                border: 2px dashed #cbd5e1;
                border-radius: 12px;
                background-color: #f8fafc;
                color: #94a3b8;
                font-style: italic;
                font-size: 16px;
            }
            QGroupBox {
                font-weight: 600;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                margin-top: 16px;
                padding-top: 16px;
                font-size: 16px;
            }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # 页面标题 - 与其他页面保持一致
        self.title_label = QLabel("降噪与超分辨率 - 图像处理工作流")
        self.title_label.setObjectName("pageTitle")
        self.title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: 600;
            color: #1e293b;
            padding: 14px 18px;
            border-left: 4px solid #0ea5e9;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #f0f9ff, stop:1 transparent);
            border-radius: 8px;
        """)
        main_layout.addWidget(self.title_label)

        # 内容区域（水平布局）
        content_layout = QHBoxLayout()
        content_layout.setSpacing(16)

        # Left panel - Controls (with scroll support)
        left_panel = self.create_control_panel()
        left_scroll = QScrollArea()
        left_scroll.setWidget(left_panel)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFrameStyle(QFrame.NoFrame)
        left_scroll.setMinimumWidth(280)
        content_layout.addWidget(left_scroll, 1)

        # Right panel - Image display
        right_panel = self.create_display_panel()
        content_layout.addWidget(right_panel, 2)

        main_layout.addLayout(content_layout)
        self.update_parameter_panel()

    def create_control_panel(self):
        """Create the left control panel."""
        panel = QFrame()
        panel.setObjectName("controlPanel")
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        # 1. 加载图像
        file_group = QGroupBox("1. 加载图像")
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(8)

        self.load_btn = QPushButton("加载")
        self.load_btn.clicked.connect(self.load_image)
        self.load_btn.setMinimumHeight(36)
        file_layout.addWidget(self.load_btn)

        save_layout = QHBoxLayout()
        save_layout.setSpacing(8)
        self.save_denoise_btn = QPushButton("💾 保存降噪图")
        self.save_denoise_btn.setObjectName("saveBtn")
        self.save_denoise_btn.clicked.connect(lambda: self.save_result(use_sr=False))
        self.save_denoise_btn.setEnabled(False)
        self.save_denoise_btn.setMinimumHeight(34)
        save_layout.addWidget(self.save_denoise_btn)

        self.save_sr_btn = QPushButton("💾 保存 SR 图")
        self.save_sr_btn.setObjectName("saveBtn")
        self.save_sr_btn.clicked.connect(lambda: self.save_result(use_sr=True))
        self.save_sr_btn.setEnabled(False)
        self.save_sr_btn.setMinimumHeight(34)
        save_layout.addWidget(self.save_sr_btn)
        file_layout.addLayout(save_layout)

        # 图像信息
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(90)
        self.info_text.setMinimumHeight(80)
        self.info_text.setPlaceholderText("加载图像后显示信息...")
        file_layout.addWidget(self.info_text)

        layout.addWidget(file_group)

        # 2. 降噪处理
        step1_group = QGroupBox("2. 降噪处理")
        step1_layout = QVBoxLayout(step1_group)
        step1_layout.setSpacing(8)

        algo_layout = QFormLayout()
        algo_layout.setSpacing(8)

        self.algorithm_combo = QComboBox()
        self.update_algorithm_list()
        self.algorithm_combo.setMinimumHeight(32)
        algo_layout.addRow("算法:", self.algorithm_combo)

        self.strength_combo = QComboBox()
        self.strength_combo.addItems(["低", "中", "高"])
        self.strength_combo.setCurrentIndex(1)
        self.strength_combo.setMinimumHeight(32)
        algo_layout.addRow("强度:", self.strength_combo)

        step1_layout.addLayout(algo_layout)

        self.denoise_params_widget = self.create_denoise_params()
        step1_layout.addWidget(self.denoise_params_widget)

        self.denoise_btn = QPushButton("▶ 执行降噪")
        self.denoise_btn.setObjectName("primaryBtn")
        self.denoise_btn.clicked.connect(self.denoise_image)
        self.denoise_btn.setEnabled(False)
        self.denoise_btn.setMinimumHeight(40)
        step1_layout.addWidget(self.denoise_btn)

        self.denoise_progress = QProgressBar()
        self.denoise_progress.setVisible(False)
        step1_layout.addWidget(self.denoise_progress)

        self.denoise_status = QLabel("等待处理...")
        self.denoise_status.setStyleSheet("color: #64748b; font-size: 15px;")
        step1_layout.addWidget(self.denoise_status)

        layout.addWidget(step1_group)

        # 3. 超分辨率重构
        step2_group = QGroupBox("3. 超分辨率重构")
        step2_layout = QVBoxLayout(step2_group)
        step2_layout.setSpacing(10)

        sr_layout = QFormLayout()
        sr_layout.setSpacing(10)

        self.sr_method_combo = QComboBox()
        for key, name in get_supported_sr_methods():
            self.sr_method_combo.addItem(name, key)
        self.sr_method_combo.setCurrentIndex(1)
        self.sr_method_combo.setMinimumHeight(44)
        sr_layout.addRow("插值方法:", self.sr_method_combo)

        self.sr_scale_combo = QComboBox()
        self.sr_scale_combo.addItems(["1.5x", "2.0x", "3.0x", "4.0x"])
        self.sr_scale_combo.setCurrentIndex(1)
        self.sr_scale_combo.setMinimumHeight(44)
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
        self.sr_status.setStyleSheet("color: #64748b; font-size: 15px;")
        step2_layout.addWidget(self.sr_status)

        layout.addWidget(step2_group)

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

        # BM3D parameters
        self.bm3d_group = QGroupBox("BM3D 参数")
        bm3d_layout = QHBoxLayout()
        bm3d_layout.setSpacing(10)
        self.bm3d_sigma_spin = QSpinBox()
        self.bm3d_sigma_spin.setRange(1, 100)
        self.bm3d_sigma_spin.setValue(20)
        self.bm3d_sigma_spin.setMinimumHeight(32)
        bm3d_layout.addWidget(QLabel("噪声标准差:"))
        bm3d_layout.addWidget(self.bm3d_sigma_spin)
        self.bm3d_group.setLayout(bm3d_layout)
        layout.addWidget(self.bm3d_group)

        # Anisotropic Diffusion parameters
        self.aniso_group = QGroupBox("各向异性扩散参数")
        aniso_layout = QHBoxLayout()
        aniso_layout.setSpacing(10)
        self.aniso_iter_spin = QSpinBox()
        self.aniso_iter_spin.setRange(1, 50)
        self.aniso_iter_spin.setValue(10)
        self.aniso_iter_spin.setMinimumHeight(32)
        aniso_layout.addWidget(QLabel("迭代次数:"))
        aniso_layout.addWidget(self.aniso_iter_spin)

        self.aniso_kappa_spin = QSpinBox()
        self.aniso_kappa_spin.setRange(10, 200)
        self.aniso_kappa_spin.setValue(50)
        self.aniso_kappa_spin.setMinimumHeight(32)
        aniso_layout.addWidget(QLabel("梯度阈值:"))
        aniso_layout.addWidget(self.aniso_kappa_spin)
        self.aniso_group.setLayout(aniso_layout)
        layout.addWidget(self.aniso_group)

        # Iterative Reconstruction parameters
        self.iter_group = QGroupBox("迭代重建参数")
        iter_layout = QHBoxLayout()
        iter_layout.setSpacing(10)
        self.iter_recon_iter_spin = QSpinBox()
        self.iter_recon_iter_spin.setRange(1, 20)
        self.iter_recon_iter_spin.setValue(5)
        self.iter_recon_iter_spin.setMinimumHeight(32)
        iter_layout.addWidget(QLabel("迭代次数:"))
        iter_layout.addWidget(self.iter_recon_iter_spin)

        self.iter_reg_spin = QDoubleSpinBox()
        self.iter_reg_spin.setRange(0.01, 1.0)
        self.iter_reg_spin.setSingleStep(0.01)
        self.iter_reg_spin.setValue(0.1)
        self.iter_reg_spin.setMinimumHeight(32)
        iter_layout.addWidget(QLabel("正则化强度:"))
        iter_layout.addWidget(self.iter_reg_spin)

        self.iter_method_combo = QComboBox()
        self.iter_method_combo.addItem("Total Variation", "tv")
        self.iter_method_combo.addItem("Tikhonov", "tikhonov")
        self.iter_method_combo.setMinimumHeight(32)
        iter_layout.addWidget(QLabel("方法:"))
        iter_layout.addWidget(self.iter_method_combo)
        self.iter_group.setLayout(iter_layout)
        layout.addWidget(self.iter_group)

        return widget

    def create_display_panel(self):
        """Create the right display panel."""
        panel = QFrame()
        panel.setObjectName("card")
        layout = QVBoxLayout(panel)
        layout.setSpacing(14)
        layout.setContentsMargins(16, 16, 16, 16)

        self.image_tabs = QTabWidget()
        self.image_tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #e2e8f0; border-radius: 8px; background-color: #f8fafc; }
            QTabBar::tab { background-color: #e2e8f0; padding: 10px 20px; border-radius: 6px; margin-right: 4px; }
            QTabBar::tab:selected { background-color: white; font-weight: 600; }
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
        """Update the algorithm dropdown."""
        self.algorithm_combo.clear()
        methods = self.processor.get_supported_methods()
        for key, name in methods:
            self.algorithm_combo.addItem(name, key)
        self.algorithm_combo.currentIndexChanged.connect(self.update_parameter_panel)

    def update_parameter_panel(self):
        """Update parameter panel visibility."""
        method = self.algorithm_combo.currentData()
        self.nlm_group.setVisible(method == 'nlm')
        self.bilateral_group.setVisible(method == 'bilateral')
        self.neural_group.setVisible(method in ['neural', 'trained_neural_denoise'])
        self.bm3d_group.setVisible(method == 'bm3d')
        self.aniso_group.setVisible(method == 'anisotropic')
        self.iter_group.setVisible(method == 'iterative')

    def load_image(self):
        """Load an image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开 X 光图像", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;所有文件 (*);;DICOM (*.dcm)"
        )
        if file_path:
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
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                rgb_img = img[:, :, :3]
            elif img.shape[2] == 3:
                mean_r = np.mean(img[:, :, 0])
                mean_b = np.mean(img[:, :, 2])
                rgb_img = img.copy()
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
        elif method == 'trained_neural_denoise':
            # Use trained neural model with same parameters
            params['patch_size'] = self.neural_patch_spin.value()
            params['stride'] = self.neural_stride_spin.value()
        elif method == 'bm3d':
            params['sigma'] = self.bm3d_sigma_spin.value()
        elif method == 'anisotropic':
            params['niter'] = self.aniso_iter_spin.value()
            params['kappa'] = self.aniso_kappa_spin.value()
        elif method == 'iterative':
            params['niter'] = self.iter_recon_iter_spin.value()
            params['regularization'] = self.iter_reg_spin.value()
            params['recon_method'] = self.iter_method_combo.currentData()

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
        else:
            QMessageBox.critical(self, "处理错误", message)
            self.denoise_status.setText("降噪失败")

    def display_denoised_image(self):
        """Display the denoised image."""
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
        else:
            QMessageBox.critical(self, "处理错误", message)
            self.sr_status.setText("超分辨率失败")

    def display_sr_image(self):
        """Display the super-resolution image."""
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
            resolution_info = metrics.get('resolution', {})
            if 'lr_size' in resolution_info:
                lr_w, lr_h = resolution_info['lr_size']
                sr_w, sr_h = resolution_info['sr_size']
                expected_w, expected_h = resolution_info.get('expected_sr_size', (sr_w, sr_h))
                actual_scale_w = resolution_info.get('actual_scale_w', 0)
                actual_scale_h = resolution_info.get('actual_scale_h', 0)

                resolution_text = (
                    f"分辨率验证:\n"
                    f"  原始尺寸：{lr_w} x {lr_h}\n"
                    f"  SR 尺寸：{sr_w} x {sr_h}\n"
                    f"  期望尺寸：{expected_w} x {expected_h}\n"
                    f"  实际倍数：{actual_scale_w:.2f}x (宽), {actual_scale_h:.2f}x (高)"
                )
            else:
                sr_w, sr_h = resolution_info.get('sr_size', (0, 0))
                resolution_text = f"SR 尺寸：{sr_w} x {sr_h}"

            sharpness = metrics.get('sharpness', 0)
            edge_strength = metrics.get('edge_strength', 0)
            entropy = metrics.get('entropy', 0)

            quality_text = (
                f"{resolution_text}\n\n"
                f"图像质量指标:\n"
                f"  清晰度：{sharpness:.2f}\n"
                f"  边缘强度：{edge_strength:.4f}\n"
                f"  纹理熵：{entropy:.2f}"
            )

            self.sr_metrics_text.setText(quality_text)

    def save_result(self, use_sr: bool = False):
        """Save the processed image."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "",
            "PNG 文件 (*.png);;TIFF 文件 (*.tiff *.tif);;JPEG 文件 (*.jpg *.jpeg);;所有文件 (*)"
        )
        if file_path:
            success = self.processor.save_result(file_path, use_sr=use_sr)
            if success:
                QMessageBox.information(self, "成功", "图像保存成功!")
            else:
                QMessageBox.critical(self, "错误", "保存图像失败。")

    def closeEvent(self, event):
        """Handle window close event."""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.wait(1000)
        event.accept()

    def apply_medical_style(self):
        """应用 Medical Minimalism 风格"""
        # 按钮样式
        for btn in self.findChildren(QPushButton):
            if "加载" in btn.text():
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #f1f5f9;
                        color: #475569;
                        border: 1px solid #cbd5e1;
                        padding: 14px 28px;
                        border-radius: 8px;
                        font-weight: 600;
                        font-size: 16px;
                    }
                    QPushButton:hover {
                        background-color: #e2e8f0;
                        border-color: #0ea5e9;
                    }
                    QPushButton:pressed {
                        background-color: #cbd5e1;
                    }
                """)
            elif "保存" in btn.text():
                btn.setStyleSheet("""
                    QPushButton#saveBtn {
                        background-color: #0891b2;
                        color: white;
                        border: none;
                        padding: 14px 28px;
                        border-radius: 8px;
                        font-weight: 600;
                        font-size: 16px;
                    }
                    QPushButton#saveBtn:hover {
                        background-color: #0e7490;
                    }
                    QPushButton#saveBtn:disabled {
                        background: #cbd5e1;
                        color: #94a3b8;
                    }
                """)
            elif "降噪" in btn.text():
                btn.setStyleSheet("""
                    QPushButton#primaryBtn {
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #16a34a, stop:1 #059669);
                        color: white;
                        border: none;
                        padding: 12px 24px;
                        border-radius: 8px;
                        font-weight: 600;
                        font-size: 16px;
                    }
                    QPushButton#primaryBtn:hover {
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #15803d, stop:1 #047857);
                    }
                    QPushButton#primaryBtn:disabled {
                        background: #cbd5e1;
                        color: #94a3b8;
                    }
                """)
            elif "超分辨率" in btn.text():
                btn.setStyleSheet("""
                    QPushButton#secondaryBtn {
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0ea5e9, stop:1 #0284c7);
                        color: white;
                        border: none;
                        padding: 16px 32px;
                        border-radius: 8px;
                        font-weight: 600;
                        font-size: 16px;
                    }
                    QPushButton#secondaryBtn:hover {
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #0369a1);
                    }
                    QPushButton#secondaryBtn:disabled {
                        background: #cbd5e1;
                        color: #94a3b8;
                    }
                """)

        # 进度条
        for progress in self.findChildren(QProgressBar):
            progress.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #e2e8f0;
                    border-radius: 8px;
                    text-align: center;
                    height: 36px;
                    font-weight: 600;
                    font-size: 15px;
                    background-color: #f8fafc;
                    color: #475569;
                }
                QProgressBar::chunk {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0ea5e9, stop:1 #0284c7);
                    border-radius: 7px;
                }
            """)

        # GroupBox 样式
        for group in self.findChildren(QGroupBox):
            group.setStyleSheet("""
                QGroupBox {
                    font-weight: 600;
                    color: #475569;
                    border: 1px solid #e2e8f0;
                    border-radius: 8px;
                    margin-top: 16px;
                    padding-top: 16px;
                    font-size: 16px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 12px;
                    padding: 0 8px;
                    color: #0ea5e9;
                    font-weight: 600;
                    font-size: 16px;
                }
            """)

        # ComboBox
        for combo in self.findChildren(QComboBox):
            combo.setStyleSheet("""
                QComboBox {
                    padding: 10px 14px;
                    border: 1px solid #cbd5e1;
                    border-radius: 8px;
                    background-color: white;
                    min-height: 44px;
                    color: #1e293b;
                    font-size: 15px;
                }
                QComboBox:hover {
                    border-color: #0ea5e9;
                }
                QComboBox:focus {
                    border-color: #0ea5e9;
                }
                QComboBox::drop-down {
                    border: none;
                    padding-right: 8px;
                }
                QComboBox::down-arrow {
                    border-left: 5px solid transparent;
                    border-right: 5px solid transparent;
                    border-top: 6px solid #64748b;
                }
            """)

        # SpinBox
        for spin in self.findChildren(QSpinBox):
            spin.setStyleSheet("""
                QSpinBox {
                    padding: 10px 14px;
                    border: 1px solid #cbd5e1;
                    border-radius: 8px;
                    background-color: white;
                    min-height: 40px;
                    color: #1e293b;
                    font-size: 15px;
                }
                QSpinBox:hover {
                    border-color: #0ea5e9;
                }
                QSpinBox:focus {
                    border-color: #0ea5e9;
                }
                QSpinBox::up-button, QSpinBox::down-button {
                    border: none;
                    width: 32px;
                    background: #f1f5f9;
                    border-radius: 6px;
                    margin: 2px;
                }
                QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                    background: #e2e8f0;
                }
            """)

        # 文本框
        for text in self.findChildren(QTextEdit):
            text.setStyleSheet("""
                QTextEdit {
                    font-family: 'Consolas', 'Courier New', monospace;
                    font-size: 13px;
                    background-color: #f8fafc;
                    border: 1px solid #e2e8f0;
                    border-radius: 8px;
                    padding: 10px;
                    color: #475569;
                }
                QTextEdit:focus {
                    border-color: #0ea5e9;
                }
            """)

        # 页面标题样式
        title_label = self.findChild(QLabel, "pageTitle")
        if title_label:
            title_label.setStyleSheet("""
                font-size: 24px;
                font-weight: 600;
                color: #1e293b;
                padding: 14px 18px;
                border-left: 4px solid #0ea5e9;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #f0f9ff, stop:1 transparent);
                border-radius: 8px;
            """)

        # 图像显示框
        for label in self.findChildren(QLabel):
            if label.objectName() == "imageBox":
                label.setStyleSheet("""
                    QLabel#imageBox {
                        border: 2px dashed #cbd5e1;
                        border-radius: 12px;
                        background-color: #f8fafc;
                        color: #94a3b8;
                        font-style: italic;
                        font-size: 16px;
                    }
                    QLabel#imageBox:hover {
                        border-color: #0ea5e9;
                        background-color: #f0f9ff;
                    }
                """)

        # 选项卡样式
        for tabs in self.findChildren(QTabWidget):
            tabs.setStyleSheet("""
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
                    color: #64748b;
                }
                QTabBar::tab:selected {
                    background-color: white;
                    font-weight: 600;
                    color: #0ea5e9;
                }
                QTabBar::tab:hover {
                    background-color: #cbd5e1;
                }
            """)


# For backward compatibility - standalone QMainWindow version
class DenoiseApp(QMainWindow):
    """Standalone main window version for backward compatibility."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('X 射线图像降噪与超分辨率重构系统')
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)
        self.setStyleSheet("QMainWindow { background-color: #f1f5f9; }")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QHBoxLayout(central_widget)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)

        self.denoise_widget = DenoiseWidget(parent=self)
        layout.addWidget(self.denoise_widget)

        self.status_bar = self.statusBar()
        self.status_bar.showMessage('就绪 - 请加载图像')

    def closeEvent(self, event):
        if hasattr(self, 'denoise_widget') and self.denoise_widget.processing_thread:
            self.denoise_widget.processing_thread.wait(1000)
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
