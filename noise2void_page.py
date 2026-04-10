"""
Noise2Void 自监督训练页面
通过单张噪声图像训练降噪神经网络
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QProgressBar, QGroupBox,
                             QTextEdit, QMessageBox, QSpinBox, QDoubleSpinBox,
                             QFrame, QScrollArea, QFormLayout, QComboBox,
                             QGridLayout, QSplitter)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

import cv2


# 医疗极简主义设计令牌
class DesignTokens:
    """UI 设计令牌 - Medical Minimalism 风格"""
    PRIMARY_500 = "#0ea5e9"
    PRIMARY_600 = "#0284c7"
    PRIMARY_700 = "#0369a1"
    BACKGROUND = "#f8fafc"
    SURFACE = "#ffffff"
    BORDER = "#e2e8f0"
    TEXT_PRIMARY = "#1e293b"
    TEXT_SECONDARY = "#64748b"
    TEXT_MUTED = "#94a3b8"
    SUCCESS = "#10b981"
    WARNING = "#f59e0b"
    ERROR = "#ef4444"
    SPACING_8 = 8
    SPACING_10 = 10
    SPACING_12 = 12
    SPACING_14 = 14
    SPACING_16 = 16
    RADIUS_MEDIUM = 8
    RADIUS_LARGE = 12


class Noise2VoidTrainingThread(QThread):
    """Noise2Void 训练线程"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)
    log_signal = pyqtSignal(str)
    epoch_end = pyqtSignal(int, dict)

    def __init__(self, image_path, output_dir, params):
        super().__init__()
        self.image_path = image_path
        self.output_dir = output_dir
        self.params = params

    def _detect_hardware(self):
        """检测硬件资源并返回设备信息和预估时间"""
        # 检测 GPU
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            hardware_info = f"GPU: {device_name} ({gpu_memory:.1f} GB)"
            # GPU 训练时间估算（基于经验值）
            estimated_time_per_epoch = 3  # 秒
        else:
            hardware_info = f"CPU: {torch.get_num_threads()} 线程"
            # CPU 训练时间估算（基于经验值，约慢 10 倍）
            estimated_time_per_epoch = 30  # 秒

        epochs = self.params.get('epochs', 50)
        estimated_total = epochs * estimated_time_per_epoch

        return hardware_info, device_name if torch.cuda.is_available() else "CPU", estimated_total

    def run(self):
        """执行 Noise2Void 训练"""
        import time
        try:
            # 记录训练开始时间
            start_time = time.time()

            # 检测硬件资源
            hardware_info, device_type, estimated_time = self._detect_hardware()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            self.progress.emit(10, f"硬件检测：{hardware_info}")
            if estimated_time > 0:
                mins = estimated_time // 60
                secs = estimated_time % 60
                self.progress.emit(15, f"预计训练时间：约{mins}分{secs}秒")
            self.log_signal.emit(f"使用设备：{device}")

            self.progress.emit(20, "正在加载图像...")

            # 加载图像
            img = self._load_image(self.image_path)
            if img is None:
                self.finished.emit(False, "无法加载图像")
                return

            self.progress.emit(20, "正在准备训练数据...")

            # 确保输出目录存在
            os.makedirs(self.output_dir, exist_ok=True)

            # 准备 N2V 训练数据
            train_data = self._prepare_n2v_data(img)

            self.progress.emit(30, "正在初始化网络...")

            # 创建 N2V 网络
            net = Noise2VoidNet().to(device)
            optimizer = torch.optim.Adam(net.parameters(), lr=self.params.get('lr', 0.001))
            criterion = nn.MSELoss()

            # 训练循环
            epochs = self.params.get('epochs', 50)
            batch_size = self.params.get('batch_size', 16)
            patch_size = self.params.get('patch_size', 64)

            self.progress.emit(40, "开始训练...")

            best_loss = float('inf')

            for epoch in range(epochs):
                if self.isInterruptionRequested():
                    self.progress.emit(0, "训练已停止")
                    self.finished.emit(False, "用户停止训练")
                    return

                # 训练一个 epoch
                net.train()
                epoch_loss = 0
                num_batches = 0

                for i in range(0, len(train_data) - batch_size, batch_size):
                    if self.isInterruptionRequested():
                        self.progress.emit(0, "训练已停止")
                        self.finished.emit(False, "用户停止训练")
                        return

                    batch = np.array(train_data[i:i+batch_size])

                    # N2V 盲点训练
                    masked_input, target = self._n2v_mask(batch)

                    optimizer.zero_grad()
                    output = net(torch.from_numpy(masked_input).float().to(device).unsqueeze(1))
                    target = torch.from_numpy(target).float().to(device).unsqueeze(1)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                avg_loss = epoch_loss / max(num_batches, 1)

                # 保存最佳模型
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    model_path = os.path.join(self.output_dir, 'noise2void_model.pth')
                    torch.save(net.state_dict(), model_path)

                progress = 40 + int((epoch + 1) / epochs * 55)
                self.progress.emit(progress, f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
                self.log_signal.emit(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
                self.epoch_end.emit(epoch + 1, {'loss': avg_loss, 'best_loss': best_loss})

            # 保存配置
            self.progress.emit(95, "正在保存模型...")

            config = {
                'algorithm': 'Noise2Void',
                'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'params': self.params,
                'source_image': self.image_path,
                'best_loss': best_loss
            }
            config_path = os.path.join(self.output_dir, 'n2v_config.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

            # 创建 marker 文件
            marker_path = os.path.join(self.output_dir, 'model_ready.marker')
            with open(marker_path, 'w', encoding='utf-8') as f:
                f.write(f"Integrated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Algorithm: Noise2Void (Self-Supervised)\n")
                f.write(f"Source: {self.image_path}\n")

            # 计算实际训练时间
            end_time = time.time()
            actual_time = int(end_time - start_time)
            actual_mins = actual_time // 60
            actual_secs = actual_time % 60

            self.progress.emit(100, "训练完成")
            self.progress.emit(100, f"实际训练时间：{actual_mins}分{actual_secs}秒")
            self.finished.emit(True, f"模型已保存到：{self.output_dir}")

        except Exception as e:
            import traceback
            error_msg = f"训练失败：{str(e)}\n\n{traceback.format_exc()}"
            self.finished.emit(False, error_msg)

    def _load_image(self, path):
        """加载图像"""
        try:
            import cv2
            # 使用 imdecode 读取中文路径文件
            img_data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            img = img.astype('float64') / 255.0
            return img
        except Exception as e:
            print(f"Failed to load image: {e}")
            return None

    def _prepare_n2v_data(self, img):
        """准备 N2V 训练数据"""
        patch_size = self.params.get('patch_size', 64)
        num_patches = self.params.get('num_patches', 10000)

        h, w = img.shape
        patches = []

        for _ in range(num_patches):
            y = np.random.randint(0, h - patch_size)
            x = np.random.randint(0, w - patch_size)
            patch = img[y:y+patch_size, x:x+patch_size]
            patches.append(patch)

        return patches

    def _n2v_mask(self, batch):
        """N2V 盲点掩码"""
        mask_prob = 0.1
        mask = np.random.rand(*batch.shape) > mask_prob
        masked_input = batch * mask
        return masked_input, batch


class Noise2VoidNet(nn.Module):
    """Noise2Void 网络架构"""

    def __init__(self, depth=3, features=32):
        super().__init__()
        self.depth = depth

        # 编码器
        layers = []
        in_channels = 1
        for i in range(depth):
            out_channels = features * (2 ** i)
            layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            if i < depth - 1:
                layers.append(nn.MaxPool2d(2))
            in_channels = out_channels

        self.encoder = nn.Sequential(*layers)

        # 解码器
        decoder_layers = []
        for i in range(depth - 2, -1, -1):
            in_channels = features * (2 ** (i + 1))
            out_channels = features * (2 ** i)
            decoder_layers.append(nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2))
            decoder_layers.append(nn.ReLU(inplace=True))
            decoder_layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            decoder_layers.append(nn.ReLU(inplace=True))

        self.decoder = nn.Sequential(*decoder_layers)

        # 输出层
        self.output = nn.Conv2d(features, 1, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.sigmoid(self.output(x))


class Noise2VoidPage(QWidget):
    """Noise2Void 自监督训练页面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.training_thread = None
        self.is_training = False
        self.train_history = {'epoch': [], 'loss': [], 'best_loss': []}
        self.image_path = None
        self.output_dir = None
        self.init_ui()

    def init_ui(self):
        """初始化界面 - 与算法训练页面保持一致的布局"""
        layout = QVBoxLayout(self)
        layout.setSpacing(DesignTokens.SPACING_12)
        layout.setContentsMargins(DesignTokens.SPACING_12, DesignTokens.SPACING_12,
                                   DesignTokens.SPACING_12, DesignTokens.SPACING_12)

        # 标题 - 与算法训练页面一致的风格
        title = QLabel("Noise2Void 自监督训练 - 单张噪声图像训练")
        title.setStyleSheet(f"""
            font-size: 24px;
            font-weight: 600;
            color: {DesignTokens.TEXT_PRIMARY};
            padding: 14px 18px;
            border-left: 4px solid {DesignTokens.PRIMARY_500};
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #f0f9ff, stop:1 transparent);
            border-radius: {DesignTokens.RADIUS_MEDIUM}px;
        """)
        layout.addWidget(title)

        # 提示信息
        info_text = QLabel("""
<b>Noise2Void</b> 是一种自监督深度学习去噪算法，只需要单张噪声图像即可训练。
<b>原理</b>：使用"盲点"网络架构，训练时预测每个像素的值时不使用该像素本身，从而实现无需干净图像的去噪训练。
<b>优势</b>：无需成对的噪声/干净图像 · 单张图像即可训练 · 适用于 X 射线等难以获取干净图像的领域
""")
        info_text.setStyleSheet(f"""
            QLabel {{
                background-color: #f0f9ff;
                color: {DesignTokens.TEXT_SECONDARY};
                padding: 12px;
                border-radius: {DesignTokens.RADIUS_MEDIUM}px;
                border-left: 4px solid {DesignTokens.PRIMARY_500};
                font-size: 14px;
                line-height: 1.6;
            }}
        """)
        info_text.setWordWrap(True)
        layout.addWidget(info_text)

        # 主内容区域 - 左右布局
        main_layout = QHBoxLayout()
        main_layout.setSpacing(DesignTokens.SPACING_16)

        # 左侧 - 训练配置（带滚动支持）
        left_panel = self._create_config_panel()
        left_scroll = QScrollArea()
        left_scroll.setWidget(left_panel)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFrameStyle(QFrame.NoFrame)
        left_scroll.setMinimumWidth(300)
        main_layout.addWidget(left_scroll, 1)

        # 右侧 - 训练监控
        right_panel = self._create_monitor_panel()
        main_layout.addWidget(right_panel, 2)

        layout.addLayout(main_layout)

        self.apply_medical_style()

    def _create_config_panel(self):
        """创建配置面板 - 与算法训练页面一致的风格"""
        panel = QFrame()
        panel.setStyleSheet(f"""
            QFrame {{
                background-color: {DesignTokens.SURFACE};
                border-radius: {DesignTokens.RADIUS_LARGE}px;
                border: 1px solid {DesignTokens.BORDER};
            }}
        """)
        layout = QVBoxLayout(panel)
        layout.setSpacing(DesignTokens.SPACING_12)
        layout.setContentsMargins(DesignTokens.SPACING_14, DesignTokens.SPACING_14,
                                   DesignTokens.SPACING_14, DesignTokens.SPACING_14)

        # 1. 加载噪声图像
        image_group = QGroupBox("1. 加载噪声图像")
        image_layout = QVBoxLayout(image_group)
        image_layout.setSpacing(DesignTokens.SPACING_8)
        image_layout.setContentsMargins(10, 10, 10, 10)

        # 图像预览（最上方）
        self.image_preview_label = QLabel()
        self.image_preview_label.setFixedHeight(220)
        self.image_preview_label.setStyleSheet(f"""
            QLabel {{
                background-color: #f8fafc;
                border: 2px dashed {DesignTokens.BORDER};
                border-radius: {DesignTokens.RADIUS_MEDIUM}px;
            }}
        """)
        self.image_preview_label.setAlignment(Qt.AlignCenter)
        self.image_preview_label.setText("图像预览")
        image_layout.addWidget(self.image_preview_label)

        # 加载按钮（样式与降噪与超分页面一致）
        self.load_image_btn = QPushButton("加载")
        self.load_image_btn.setObjectName("loadBtn")
        self.load_image_btn.clicked.connect(self.load_image)
        self.load_image_btn.setMinimumHeight(40)
        self.load_image_btn.setStyleSheet(f"""
            QPushButton#loadBtn {{
                background-color: #f1f5f9;
                color: {DesignTokens.TEXT_SECONDARY};
                border: 1px solid {DesignTokens.BORDER};
                padding: 10px 20px;
                border-radius: {DesignTokens.RADIUS_MEDIUM}px;
                font-weight: 600;
                font-size: 16px;
            }}
            QPushButton#loadBtn:hover {{
                background-color: #e2e8f0;
                border-color: {DesignTokens.PRIMARY_500};
            }}
        """)
        image_layout.addWidget(self.load_image_btn)

        # 文件信息显示
        self.image_info_label = QLabel("未加载图像")
        self.image_info_label.setStyleSheet(f"color: {DesignTokens.TEXT_MUTED}; font-size: 14px;")
        self.image_info_label.setWordWrap(True)
        image_layout.addWidget(self.image_info_label)

        layout.addWidget(image_group)

        # 2. 训练参数配置
        params_group = QGroupBox("2. 训练参数")
        params_layout = QFormLayout(params_group)
        params_layout.setSpacing(DesignTokens.SPACING_10)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 500)
        self.epochs_spin.setValue(50)
        self.epochs_spin.setMinimumHeight(40)
        self.epochs_spin.setStyleSheet("font-size: 15px; padding: 10px 14px;")
        params_layout.addRow("训练轮数 (Epochs):", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(16)
        self.batch_spin.setMinimumHeight(40)
        self.batch_spin.setStyleSheet("font-size: 15px; padding: 10px 14px;")
        params_layout.addRow("批次大小 (Batch Size):", self.batch_spin)

        self.patch_spin = QSpinBox()
        self.patch_spin.setRange(32, 256)
        self.patch_spin.setValue(64)
        self.patch_spin.setMinimumHeight(40)
        self.patch_spin.setStyleSheet("font-size: 15px; padding: 10px 14px;")
        params_layout.addRow("块大小 (Patch Size):", self.patch_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.01)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setMinimumHeight(40)
        self.lr_spin.setStyleSheet("font-size: 15px; padding: 10px 14px;")
        params_layout.addRow("学习率 (Learning Rate):", self.lr_spin)

        layout.addWidget(params_group)

        # 3. 模型输出和集成
        output_group = QGroupBox("3. 模型输出")
        output_layout = QVBoxLayout(output_group)
        output_layout.setSpacing(DesignTokens.SPACING_8)
        output_layout.setContentsMargins(10, 10, 10, 10)

        self.output_dir_edit = QTextEdit()
        self.output_dir_edit.setReadOnly(True)
        self.output_dir_edit.setMaximumHeight(40)
        self.output_dir_edit.setPlaceholderText("选择模型输出目录...")
        output_layout.addWidget(self.output_dir_edit)

        self.browse_output_btn = QPushButton("选择模型输出目录")
        self.browse_output_btn.clicked.connect(self.browse_output)
        self.browse_output_btn.setMinimumHeight(40)
        self.browse_output_btn.setStyleSheet(self._get_button_style())
        output_layout.addWidget(self.browse_output_btn)

        # 训练控制按钮
        self.train_btn = QPushButton("▶ 开始训练")
        self.train_btn.setObjectName("primaryBtn")
        self.train_btn.setMinimumHeight(48)
        self.train_btn.clicked.connect(self.start_training)
        self.train_btn.setEnabled(False)
        self.train_btn.setStyleSheet(f"""
            QPushButton#primaryBtn {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {DesignTokens.PRIMARY_500}, stop:1 {DesignTokens.PRIMARY_600});
                color: white;
                font-size: 16px;
                font-weight: 600;
                padding: 14px 28px;
                border-radius: {DesignTokens.RADIUS_LARGE}px;
            }}
            QPushButton#primaryBtn:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {DesignTokens.PRIMARY_600}, stop:1 {DesignTokens.PRIMARY_700});
            }}
            QPushButton#primaryBtn:disabled {{
                background: {DesignTokens.BORDER};
                color: {DesignTokens.TEXT_MUTED};
            }}
        """)
        output_layout.addWidget(self.train_btn)

        self.stop_btn = QPushButton("⏹ 停止训练")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setMinimumHeight(48)
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setStyleSheet(f"""
            QPushButton#stopBtn {{
                background-color: #dc2626;
                color: white;
                font-size: 16px;
                font-weight: 600;
                padding: 14px 28px;
                border-radius: {DesignTokens.RADIUS_LARGE}px;
            }}
            QPushButton#stopBtn:hover {{
                background-color: #b91c1c;
            }}
            QPushButton#stopBtn:disabled {{
                background-color: {DesignTokens.BORDER};
                color: {DesignTokens.TEXT_MUTED};
            }}
        """)
        self.stop_btn.setEnabled(False)
        output_layout.addWidget(self.stop_btn)

        layout.addWidget(output_group)

        # 4. 模型集成
        integrate_group = QGroupBox("4. 模型集成")
        integrate_layout = QVBoxLayout(integrate_group)
        integrate_layout.setSpacing(DesignTokens.SPACING_10)

        # 模型类型选择
        model_type_layout = QHBoxLayout()
        model_type_label = QLabel("模型类型:")
        model_type_label.setStyleSheet("font-size: 15px; font-weight: 500; color: #475569;")
        model_type_layout.addWidget(model_type_label)

        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["降噪模型 (Denoiser)", "超分辨率模型 (Super-Resolution)"])
        self.model_type_combo.setMinimumHeight(40)
        self.model_type_combo.setStyleSheet("""
            QComboBox {
                font-size: 15px;
                padding: 10px 14px;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
            }
        """)
        model_type_layout.addWidget(self.model_type_combo, 1)
        integrate_layout.addLayout(model_type_layout)

        # 集成按钮
        self.integrate_model_btn = QPushButton("添加")
        self.integrate_model_btn.setObjectName("integrateModelBtn")
        self.integrate_model_btn.setMinimumHeight(48)
        self.integrate_model_btn.clicked.connect(self.integrate_model)
        self.integrate_model_btn.setEnabled(False)
        self.integrate_model_btn.setStyleSheet(f"""
            QPushButton#integrateModelBtn {{
                background-color: {DesignTokens.PRIMARY_500};
                color: white;
                font-size: 16px;
                font-weight: 600;
                padding: 14px 28px;
                border-radius: {DesignTokens.RADIUS_LARGE}px;
            }}
            QPushButton#integrateModelBtn:hover {{
                background-color: {DesignTokens.PRIMARY_600};
            }}
            QPushButton#integrateModelBtn:disabled {{
                background-color: {DesignTokens.BORDER};
            }}
        """)
        integrate_layout.addWidget(self.integrate_model_btn)

        layout.addWidget(integrate_group)

        layout.addStretch()
        return panel

    def _create_monitor_panel(self):
        """创建监控面板 - 与算法训练页面一致的风格"""
        panel = QFrame()
        panel.setStyleSheet(f"""
            QFrame {{
                background-color: {DesignTokens.SURFACE};
                border-radius: {DesignTokens.RADIUS_LARGE}px;
                border: 1px solid {DesignTokens.BORDER};
            }}
        """)
        layout = QVBoxLayout(panel)
        layout.setSpacing(DesignTokens.SPACING_12)
        layout.setContentsMargins(DesignTokens.SPACING_14, DesignTokens.SPACING_14,
                                   DesignTokens.SPACING_14, DesignTokens.SPACING_14)

        # 训练进度
        progress_group = QGroupBox("训练进度")
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setSpacing(DesignTokens.SPACING_8)
        progress_layout.setContentsMargins(10, 10, 10, 10)

        self.progress = QProgressBar()
        self.progress.setMinimumHeight(36)
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {DesignTokens.BORDER};
                border-radius: 8px;
                text-align: center;
                font-weight: 600;
                font-size: 15px;
                background-color: #f8fafc;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {DesignTokens.PRIMARY_500}, stop:1 {DesignTokens.PRIMARY_600});
                border-radius: 7px;
            }}
        """)
        progress_layout.addWidget(self.progress)

        self.status_label = QLabel("准备就绪")
        self.status_label.setStyleSheet(f"color: {DesignTokens.TEXT_SECONDARY}; font-size: 15px;")
        progress_layout.addWidget(self.status_label)

        layout.addWidget(progress_group)

        # 训练日志
        log_group = QGroupBox("训练日志")
        log_layout = QVBoxLayout(log_group)
        log_layout.setSpacing(0)
        log_layout.setContentsMargins(0, 0, 0, 0)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("训练日志将显示在这里...")
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 16px;
                background-color: #f8fafc;
                border: none;
                padding: 10px;
            }}
        """)
        self.log_text.setMinimumHeight(250)
        log_layout.addWidget(self.log_text)

        layout.addWidget(log_group)

        # Loss 曲线图
        chart_group = QGroupBox("训练 Loss 曲线")
        chart_layout = QVBoxLayout(chart_group)
        chart_layout.setSpacing(0)
        chart_layout.setContentsMargins(0, 0, 0, 0)

        # 创建 matplotlib 图表
        self.loss_figure = Figure(figsize=(10, 8), dpi=100)
        self.loss_canvas = FigureCanvasQTAgg(self.loss_figure)
        self.loss_canvas.setMinimumHeight(300)
        self.loss_canvas.setStyleSheet("background-color: white;")

        # 初始化图表
        self.loss_ax = self.loss_figure.add_subplot(111)
        self.loss_ax.set_xlabel('Epoch')
        self.loss_ax.set_ylabel('Loss')
        self.loss_ax.set_title('Training Loss')
        self.loss_ax.grid(True, alpha=0.3)
        self.loss_line = None

        chart_layout.addWidget(self.loss_canvas)
        layout.addWidget(chart_group)

        return panel

    def _get_button_style(self):
        """获取按钮样式"""
        return f"""
            QPushButton {{
                background-color: #f1f5f9;
                color: {DesignTokens.TEXT_SECONDARY};
                border: 1px solid {DesignTokens.BORDER};
                padding: 12px 24px;
                border-radius: {DesignTokens.RADIUS_MEDIUM}px;
                font-weight: 600;
                font-size: 16px;
            }}
            QPushButton:hover {{
                background-color: #e2e8f0;
                border-color: {DesignTokens.PRIMARY_500};
            }}
        """

    def load_image(self):
        """加载噪声图像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择噪声图像", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;所有文件 (*)"
        )
        if file_path:
            self.image_path = file_path

            # 显示图像信息
            try:
                # 使用 imdecode 读取中文路径文件
                # 使用 IMREAD_ANYDEPTH 保留原始位深度（8 位或 16 位）
                img_data = np.fromfile(file_path, dtype=np.uint8)
                img = cv2.imdecode(img_data, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
                if img is not None:
                    h, w = img.shape
                    dtype = str(img.dtype)
                    bit_depth = "8 位" if dtype == "uint8" else "16 位" if dtype == "uint16" else dtype
                    info = f"文件名：{os.path.basename(file_path)}\n形状：({h}, {w})\n数据类型：{dtype}\n位深度：{bit_depth}"
                    self.image_info_label.setText(info)
                    self.image_info_label.setStyleSheet(f"color: {DesignTokens.TEXT_PRIMARY}; font-size: 14px;")

                    # 显示图像预览
                    self._show_image_preview(img)

                    # 检查是否可以启用开始训练按钮
                    self._check_can_train()
                else:
                    self.image_info_label.setText(f"无法加载图像：{os.path.basename(file_path)}")
                    self.image_info_label.setStyleSheet(f"color: {DesignTokens.ERROR}; font-size: 14px;")
            except Exception as e:
                self.image_info_label.setText(f"加载失败：{str(e)}")
                self.image_info_label.setStyleSheet(f"color: {DesignTokens.ERROR}; font-size: 14px;")

    def _check_can_train(self):
        """检查是否可以开始训练（图像和输出目录都已选择）"""
        if hasattr(self, 'image_path') and self.image_path and \
           hasattr(self, 'output_dir') and self.output_dir:
            self.train_btn.setEnabled(True)
        else:
            self.train_btn.setEnabled(False)

    def _show_image_preview(self, img):
        """显示图像预览"""
        try:
            # 归一化到 0-255
            if img.dtype == np.uint16:
                img_display = (img.astype(np.float64) / 65535.0 * 255).astype(np.uint8)
            else:
                img_display = img

            # 创建 QPixmap 用于显示
            h, w = img_display.shape
            bytes_per_line = w
            qimg = QImage(img_display.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimg)

            # 缩放到适合显示的大小
            scaled_pixmap = pixmap.scaled(
                self.image_preview_label.width() - 20,
                self.image_preview_label.height() - 20,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_preview_label.setPixmap(scaled_pixmap)
            self.image_preview_label.setText("")
        except Exception as e:
            self.image_preview_label.setText(f"预览失败：{str(e)}")

    def browse_output(self):
        """选择输出目录"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择模型输出目录", "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if dir_path:
            self.output_dir = dir_path
            self.output_dir_edit.setText(dir_path)
            self.output_dir_edit.setStyleSheet(f"color: {DesignTokens.SUCCESS}; font-size: 14px;")
            # 检查是否可以启用开始训练按钮
            self._check_can_train()

    def start_training(self):
        """开始训练"""
        if not hasattr(self, 'image_path') or not self.image_path:
            QMessageBox.warning(self, "警告", "请先加载噪声图像")
            return

        if not hasattr(self, 'output_dir') or not self.output_dir:
            QMessageBox.warning(self, "警告", "请先选择模型输出目录")
            return

        # 创建带时间戳的子目录（与 DenseNet 训练页面逻辑一致）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(self.output_dir, f'noise2void_{timestamp}')
        self.output_dir = output_dir  # 更新为实际训练目录
        self.output_dir_edit.setText(output_dir)

        # 获取参数
        params = {
            'epochs': self.epochs_spin.value(),
            'batch_size': self.batch_spin.value(),
            'patch_size': self.patch_spin.value(),
            'lr': self.lr_spin.value(),
            'num_patches': 10000
        }

        self.is_training = True
        self.train_history = {'epoch': [], 'loss': [], 'best_loss': []}

        # 禁用/启用按钮
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setValue(0)

        # 创建并启动训练线程
        self.training_thread = Noise2VoidTrainingThread(
            self.image_path, output_dir, params
        )
        self.training_thread.progress.connect(self.update_progress)
        self.training_thread.finished.connect(self.training_finished)
        self.training_thread.log_signal.connect(self.append_log)
        self.training_thread.epoch_end.connect(self.update_epoch_stats)
        self.training_thread.start()

        self.status_label.setText("正在训练...")
        self.append_log(f"开始训练 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.append_log(f"参数：epochs={params['epochs']}, batch_size={params['batch_size']}, patch_size={params['patch_size']}, lr={params['lr']}")

    def stop_training(self):
        """停止训练"""
        if not self.is_training:
            return

        if not hasattr(self, 'training_thread') or not self.training_thread.isRunning():
            self.is_training = False
            self.train_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            return

        self.training_thread.requestInterruption()
        self.append_log("正在停止训练...")
        self.status_label.setText("正在停止...")

        # 等待线程结束（最多 10 秒）
        finished = self.training_thread.wait(10000)

        # 无论是否超时，都重置状态
        self.is_training = False
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setValue(0)
        self.status_label.setText("训练已停止")

        if finished:
            self.append_log("训练已停止")
        else:
            self.append_log("警告：线程未能在预期时间内停止")

    def update_progress(self, value, message):
        """更新进度"""
        self.progress.setValue(value)
        self.status_label.setText(message)
        # 将硬件检测、预计时间和实际训练时间信息添加到日志
        if "硬件检测" in message or "预计训练时间" in message or "实际训练时间" in message:
            self.log_text.append(message)

    def update_epoch_stats(self, epoch, stats):
        """更新 epoch 统计"""
        self.train_history['epoch'].append(epoch)
        self.train_history['loss'].append(stats.get('loss', 0))

        self.status_label.setText(
            f"Epoch: {epoch} | Loss: {stats.get('loss', 0):.6f} | Best Loss: {stats.get('best_loss', 0):.6f}"
        )

        # 更新 Loss 曲线图
        self.update_loss_chart()

    def update_loss_chart(self):
        """更新 Loss 曲线图"""
        try:
            self.loss_ax.clear()
            self.loss_ax.set_xlabel('Epoch')
            self.loss_ax.set_ylabel('Loss')
            self.loss_ax.set_title('Training Loss')
            self.loss_ax.grid(True, alpha=0.3)

            if len(self.train_history['epoch']) > 0:
                self.loss_ax.plot(
                    self.train_history['epoch'],
                    self.train_history['loss'],
                    'b-',
                    linewidth=2,
                    label='Training Loss'
                )
                self.loss_ax.legend()

            self.loss_canvas.draw()
        except Exception as e:
            print(f"Failed to update loss chart: {e}")

    def training_finished(self, success, message):
        """训练完成"""
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.is_training = False

        if success:
            self.status_label.setText("训练完成")
            self.status_label.setStyleSheet(f"color: {DesignTokens.SUCCESS}; font-size: 15px; font-weight: 600;")
            self.append_log(f"训练成功完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # 启用集成按钮
            self.integrate_model_btn.setEnabled(True)
        else:
            self.status_label.setText("训练失败")
            self.status_label.setStyleSheet(f"color: {DesignTokens.ERROR}; font-size: 15px;")
            self.append_log(f"训练失败：{message}")
            QMessageBox.warning(self, "训练失败", message)

    def integrate_model(self):
        """集成模型到算法列表"""
        if not hasattr(self, 'output_dir') or not self.output_dir:
            QMessageBox.warning(self, "警告", "请先选择模型输出目录")
            return

        if not os.path.isdir(self.output_dir):
            QMessageBox.warning(self, "警告", f"输出目录不存在：\n{self.output_dir}")
            return

        # 检查模型文件
        model_files = [f for f in os.listdir(self.output_dir) if f.endswith(('.pth', '.onnx', '.json'))]
        if not model_files:
            # 尝试查找默认文件名
            default_files = ['noise2void_model.pth', 'n2v_config.json', 'model_ready.marker']
            for fname in default_files:
                if os.path.exists(os.path.join(self.output_dir, fname)):
                    model_files.append(fname)
            if not model_files:
                QMessageBox.warning(self, "警告",
                    f"没有找到模型文件\n输出目录：{self.output_dir}\n请确认训练已完成")
                return

        try:
            import shutil

            # 根据模型类型决定目标目录
            model_type = self.model_type_combo.currentText()
            if "降噪" in model_type:
                base_dir = os.path.join(os.path.dirname(__file__), 'integrated_model', 'denoise')
            else:
                base_dir = os.path.join(os.path.dirname(__file__), 'integrated_model', 'super_resolution')

            # 创建带时间戳的子目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target_dir = os.path.join(base_dir, timestamp)
            os.makedirs(target_dir, exist_ok=True)

            # 复制模型文件
            copied_count = 0
            for filename in model_files:
                src = os.path.join(self.output_dir, filename)
                dst = os.path.join(target_dir, filename)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                    copied_count += 1

            # 创建集成标记文件
            marker_path = os.path.join(target_dir, 'model_ready.marker')
            with open(marker_path, 'w', encoding='utf-8') as f:
                f.write(f"Integrated at {datetime.now().isoformat()}\nModel Type: {model_type}")

            # 同步到 JSON 配置（使管理对话框能识别此模型）
            from algorithm_config import add_algorithm
            label = 'Denoise' if '降噪' in model_type else 'Super-Resolution'
            prefix = 'trained_neural_denoise' if '降噪' in model_type else 'trained_sr'
            algo_key = f"{prefix}_{timestamp}"
            algo_name = f"Trained Neural {label} [{timestamp}]"
            add_algorithm(
                'denoise' if '降噪' in model_type else 'super_resolution',
                algo_key, algo_name, enabled=True
            )

            self.append_log(f"✓ 模型已集成到：{target_dir}（复制了 {copied_count} 个文件）")
            self.integrate_model_btn.setEnabled(False)

            QMessageBox.information(self, "成功",
                f"模型已集成到{'降噪' if '降噪' in model_type else '超分辨率'}算法\n"
                f"复制了 {copied_count} 个文件\n目录：{target_dir}")

            # 通知主窗口刷新降噪页面的算法列表
            self._refresh_denoise_algorithm_list()

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            QMessageBox.critical(self, "错误", f"集成失败：{str(e)}\n\n{error_detail}")
            self.append_log(f"✗ 集成失败：{str(e)}")

    def _refresh_denoise_algorithm_list(self):
        """通知主窗口刷新降噪和超分算法列表。"""
        try:
            main_win = self.window()
            if main_win and hasattr(main_win, 'denoise_widget'):
                main_win.denoise_widget.update_algorithm_list()
                main_win.denoise_widget._update_sr_algorithm_list()
                self.append_log("✓ 算法列表已刷新")
            else:
                self.append_log(f"⚠ 无法找到主窗口，窗口类型: {type(main_win).__name__}")
        except Exception as e:
            self.append_log(f"⚠ 刷新算法列表失败：{e}")

    def append_log(self, message):
        """添加日志"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.append(f"[{timestamp}] {message}")
        # 自动滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def apply_medical_style(self):
        """应用 Medical Minimalism 风格"""
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {DesignTokens.BACKGROUND};
            }}
            QGroupBox {{
                font-weight: 600;
                font-size: 16px;
                color: #475569;
                margin-top: 12px;
                padding-top: 12px;
                border: 1px solid {DesignTokens.BORDER};
                border-radius: {DesignTokens.RADIUS_LARGE}px;
                background-color: {DesignTokens.SURFACE};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 8px;
                color: {DesignTokens.PRIMARY_500};
                font-weight: 600;
                font-size: 16px;
            }}
        """)
