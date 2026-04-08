"""
Noise2Void 自监督训练页面
通过单张噪声图像训练降噪神经网络
"""

import sys
import os
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QProgressBar, QGroupBox,
                             QTextEdit, QMessageBox, QSpinBox, QDoubleSpinBox,
                             QFrame, QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal


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

    def run(self):
        """执行 Noise2Void 训练"""
        try:
            self.progress.emit(10, "正在加载图像...")

            # 加载图像
            img = self._load_image(self.image_path)
            if img is None:
                self.finished.emit(False, "无法加载图像")
                return

            self.progress.emit(20, "正在准备训练数据...")

            # 准备 N2V 训练数据
            train_data = self._prepare_n2v_data(img)

            self.progress.emit(30, "正在初始化网络...")

            # 创建 N2V 网络
            net = Noise2VoidNet()
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
                    self.finished.emit(False, "训练已取消")
                    return

                # 训练一个 epoch
                net.train()
                epoch_loss = 0
                num_batches = 0

                for i in range(0, len(train_data) - batch_size, batch_size):
                    batch = np.array(train_data[i:i+batch_size])

                    # N2V 盲点训练
                    masked_input, target = self._n2v_mask(batch)

                    optimizer.zero_grad()
                    output = net(torch.from_numpy(masked_input).float().unsqueeze(1))
                    loss = criterion(output, torch.from_numpy(target).float().unsqueeze(1))
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
            os.makedirs(self.output_dir, exist_ok=True)

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

            self.progress.emit(100, "训练完成")
            self.finished.emit(True, f"模型已保存到：{self.output_dir}")

        except Exception as e:
            import traceback
            error_msg = f"训练失败：{str(e)}\n\n{traceback.format_exc()}"
            self.finished.emit(False, error_msg)

    def _load_image(self, path):
        """加载图像"""
        try:
            import cv2
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            img = img.astype('float64') / 255.0
            return img
        except:
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

        self.image_path_edit = QTextEdit()
        self.image_path_edit.setReadOnly(True)
        self.image_path_edit.setMaximumHeight(40)
        self.image_path_edit.setPlaceholderText("选择单张噪声图像进行训练...")
        image_layout.addWidget(self.image_path_edit)

        self.load_image_btn = QPushButton("选择噪声图像")
        self.load_image_btn.clicked.connect(self.load_image)
        self.load_image_btn.setMinimumHeight(40)
        self.load_image_btn.setStyleSheet(self._get_button_style())
        image_layout.addWidget(self.load_image_btn)

        self.image_info_label = QLabel("未加载图像")
        self.image_info_label.setStyleSheet(f"color: {DesignTokens.TEXT_MUTED}; font-size: 14px;")
        image_layout.addWidget(self.image_info_label)

        layout.addWidget(image_group)

        # 2. 训练参数配置
        params_group = QGroupBox("2. 训练参数配置")
        params_layout = QVBoxLayout(params_group)
        params_layout.setSpacing(DesignTokens.SPACING_10)

        # 训练轮数
        epochs_layout = QHBoxLayout()
        epochs_label = QLabel("训练轮数 (Epochs):")
        epochs_label.setStyleSheet(f"font-size: 14px; color: {DesignTokens.TEXT_SECONDARY};")
        epochs_layout.addWidget(epochs_label)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 500)
        self.epochs_spin.setValue(50)
        self.epochs_spin.setMinimumHeight(40)
        self.epochs_spin.setStyleSheet("font-size: 15px; padding: 10px 14px;")
        epochs_layout.addWidget(self.epochs_spin)
        params_layout.addLayout(epochs_layout)

        # 批次大小
        batch_layout = QHBoxLayout()
        batch_label = QLabel("批次大小 (Batch Size):")
        batch_label.setStyleSheet(f"font-size: 14px; color: {DesignTokens.TEXT_SECONDARY};")
        batch_layout.addWidget(batch_label)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(16)
        self.batch_spin.setMinimumHeight(40)
        self.batch_spin.setStyleSheet("font-size: 15px; padding: 10px 14px;")
        batch_layout.addWidget(self.batch_spin)
        params_layout.addLayout(batch_layout)

        # 块大小
        patch_layout = QHBoxLayout()
        patch_label = QLabel("块大小 (Patch Size):")
        patch_label.setStyleSheet(f"font-size: 14px; color: {DesignTokens.TEXT_SECONDARY};")
        patch_layout.addWidget(patch_label)
        self.patch_spin = QSpinBox()
        self.patch_spin.setRange(32, 256)
        self.patch_spin.setValue(64)
        self.patch_spin.setMinimumHeight(40)
        self.patch_spin.setStyleSheet("font-size: 15px; padding: 10px 14px;")
        patch_layout.addWidget(self.patch_spin)
        params_layout.addLayout(patch_layout)

        # 学习率
        lr_layout = QHBoxLayout()
        lr_label = QLabel("学习率:")
        lr_label.setStyleSheet(f"font-size: 14px; color: {DesignTokens.TEXT_SECONDARY};")
        lr_layout.addWidget(lr_label)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.01)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setMinimumHeight(40)
        self.lr_spin.setStyleSheet("font-size: 15px; padding: 10px 14px;")
        lr_layout.addWidget(self.lr_spin)
        params_layout.addLayout(lr_layout)

        layout.addWidget(params_group)

        # 3. 模型输出目录
        output_group = QGroupBox("3. 模型输出目录")
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

        layout.addWidget(output_group)

        # 4. 开始训练按钮
        train_group = QGroupBox("4. 开始训练")
        train_layout = QVBoxLayout(train_group)
        train_layout.setSpacing(DesignTokens.SPACING_12)

        self.train_btn = QPushButton("▶ 开始训练")
        self.train_btn.setObjectName("primaryBtn")
        self.train_btn.setMinimumHeight(48)
        self.train_btn.clicked.connect(self.start_training)
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
        train_layout.addWidget(self.train_btn)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: #f1f5f9;
                border: 1px solid {DesignTokens.BORDER};
                border-radius: {DesignTokens.RADIUS_MEDIUM}px;
                text-align: center;
                font-size: 14px;
                color: {DesignTokens.TEXT_SECONDARY};
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {DesignTokens.SUCCESS}, stop:1 #059669);
                border-radius: {DesignTokens.RADIUS_MEDIUM}px;
            }}
        """)
        train_layout.addWidget(self.progress_bar)

        # 状态标签
        self.status_label = QLabel("等待开始训练...")
        self.status_label.setStyleSheet(f"color: {DesignTokens.TEXT_SECONDARY}; font-size: 14px;")
        train_layout.addWidget(self.status_label)

        layout.addWidget(train_group)

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

        # 训练日志
        log_group = QGroupBox("训练日志")
        log_layout = QVBoxLayout(log_group)
        log_layout.setSpacing(DesignTokens.SPACING_8)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(300)
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 13px;
                background-color: #1e293b;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: {DesignTokens.RADIUS_MEDIUM}px;
                padding: 12px;
            }}
        """)
        log_layout.addWidget(self.log_text)

        layout.addWidget(log_group)

        # 训练统计
        stats_group = QGroupBox("训练统计")
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setSpacing(DesignTokens.SPACING_8)

        self.stats_label = QLabel("等待训练开始...")
        self.stats_label.setStyleSheet(f"color: {DesignTokens.TEXT_SECONDARY}; font-size: 14px;")
        stats_layout.addWidget(self.stats_label)

        layout.addWidget(stats_group)

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
                font-size: 15px;
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
            self.image_path_edit.setText(file_path)

            # 显示图像信息
            try:
                import cv2
                # 使用 imdecode 读取中文路径文件
                img_data = np.fromfile(file_path, dtype=np.uint8)
                img = cv2.imdecode(img_data, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    h, w = img.shape
                    dtype = str(img.dtype)
                    bit_depth = "8 位" if dtype == "uint8" else "16 位" if dtype == "uint16" else dtype
                    info = f"文件名：{os.path.basename(file_path)} | 形状：({h}, {w}) | 数据类型：{dtype} | 位深度：{bit_depth}"
                    self.image_info_label.setText(info)
                    self.image_info_label.setStyleSheet(f"color: {DesignTokens.SUCCESS}; font-size: 14px;")
                else:
                    self.image_info_label.setText(f"无法加载图像：{os.path.basename(file_path)}")
                    self.image_info_label.setStyleSheet(f"color: {DesignTokens.ERROR}; font-size: 14px;")
            except Exception as e:
                self.image_info_label.setText(f"加载失败：{str(e)}")
                self.image_info_label.setStyleSheet(f"color: {DesignTokens.ERROR}; font-size: 14px;")

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

    def start_training(self):
        """开始训练"""
        if not hasattr(self, 'image_path') or not self.image_path:
            QMessageBox.warning(self, "警告", "请先加载噪声图像")
            return

        if not hasattr(self, 'output_dir') or not self.output_dir:
            QMessageBox.warning(self, "警告", "请先选择模型输出目录")
            return

        # 获取参数
        params = {
            'epochs': self.epochs_spin.value(),
            'batch_size': self.batch_spin.value(),
            'patch_size': self.patch_spin.value(),
            'lr': self.lr_spin.value(),
            'num_patches': 10000
        }

        # 确认
        reply = QMessageBox.question(
            self, "确认开始训练",
            f"确定要开始 Noise2Void 训练吗？\n\n"
            f"<b>参数配置:</b><br>"
            f"- Epochs: {params['epochs']}<br>"
            f"- Batch Size: {params['batch_size']}<br>"
            f"- Patch Size: {params['patch_size']}<br>"
            f"- Learning Rate: {params['lr']}<br><br>"
            f"<b>预计训练时间：</b>5-30 分钟（取决于 GPU）",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # 禁用按钮
        self.train_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.is_training = True

        # 创建并启动训练线程
        self.training_thread = Noise2VoidTrainingThread(
            self.image_path, self.output_dir, params
        )
        self.training_thread.progress.connect(self.update_progress)
        self.training_thread.finished.connect(self.training_finished)
        self.training_thread.log_signal.connect(self.append_log)
        self.training_thread.epoch_end.connect(self.update_epoch_stats)
        self.training_thread.start()

        self.status_label.setText("正在训练...")
        self.status_label.setStyleSheet(f"color: {DesignTokens.TEXT_SECONDARY}; font-size: 14px;")
        self.append_log(f"开始训练 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.append_log(f"参数：{params}")

    def update_progress(self, value, message):
        """更新进度"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def update_epoch_stats(self, epoch, stats):
        """更新 epoch 统计"""
        self.train_history['epoch'].append(epoch)
        self.train_history['loss'].append(stats.get('loss', 0))
        self.train_history['best_loss'].append(stats.get('best_loss', 0))

        self.stats_label.setText(
            f"Epoch: {epoch} | "
            f"Loss: {stats.get('loss', 0):.6f} | "
            f"Best Loss: {stats.get('best_loss', 0):.6f}"
        )
        self.stats_label.setStyleSheet(f"color: {DesignTokens.TEXT_PRIMARY}; font-size: 14px; font-weight: 600;")

    def training_finished(self, success, message):
        """训练完成"""
        self.train_btn.setEnabled(True)
        self.is_training = False

        if success:
            self.status_label.setText("训练完成")
            self.status_label.setStyleSheet(f"color: {DesignTokens.SUCCESS}; font-size: 14px; font-weight: 600;")
            self.append_log(f"训练成功完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # 询问是否集成到算法列表
            reply = QMessageBox.question(
                self, "训练完成",
                f"{message}\n\n"
                f"是否要将训练的模型集成到降噪与超分辨率页面的算法列表中？",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.integrate_model()
        else:
            self.status_label.setText("训练失败")
            self.status_label.setStyleSheet(f"color: #ef4444; font-size: 14px;")
            self.append_log(f"训练失败：{message}")
            QMessageBox.warning(self, "训练失败", message)

    def integrate_model(self):
        """集成模型到算法列表"""
        try:
            from algorithm_config import add_algorithm
            import shutil

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            algo_key = f"trained_neural_denoise_n2v_{timestamp}"
            algo_name = f"Noise2Void (Self-Supervised) [{timestamp}]"

            # 复制模型到 integrated_model/denoise 目录
            target_base = os.path.join(os.path.dirname(__file__), 'integrated_model', 'denoise', timestamp)
            os.makedirs(target_base, exist_ok=True)

            # 复制模型文件
            for item in os.listdir(self.output_dir):
                src = os.path.join(self.output_dir, item)
                dst = os.path.join(target_base, item)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)

            # 添加到算法配置
            if add_algorithm('denoise', algo_key, algo_name, enabled=True):
                QMessageBox.information(self, "成功", f"模型已集成到算法列表:\n{algo_name}")
                self.append_log(f"模型已集成：{algo_name}")
            else:
                QMessageBox.warning(self, "警告", "添加到算法列表失败")

        except Exception as e:
            QMessageBox.warning(self, "错误", f"集成失败：{str(e)}")

    def append_log(self, message):
        """添加日志"""
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    def apply_medical_style(self):
        """应用 Medical Minimalism 风格"""
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {DesignTokens.BACKGROUND};
            }}
            QGroupBox {{
                font-weight: 600;
                font-size: 16px;
                color: {DesignTokens.TEXT_SECONDARY};
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
            }}
        """)
