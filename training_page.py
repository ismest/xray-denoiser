"""
算法训练页面 - 神经网络训练用于降噪和超分辨率
"""

import sys
import os
import json
from datetime import datetime
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QProgressBar, QComboBox,
                             QSpinBox, QDoubleSpinBox, QFormLayout, QGroupBox,
                             QFrame, QMessageBox, QTextEdit, QGridLayout,
                             QCheckBox, QTabWidget, QScrollArea)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import cv2

# 尝试导入训练相关的库
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Training features will be limited.")


class SimpleDenoiserNet(nn.Module):
    """简单的降噪神经网络模型。"""

    def __init__(self, in_channels=1, out_channels=1, features=64, num_blocks=8):
        super().__init__()
        layers = []

        # 输入层
        layers.append(nn.Conv2d(in_channels, features, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))

        # 残差块
        for _ in range(num_blocks):
            layers.append(nn.Conv2d(features, features, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(features, features, 3, padding=1))

        # 输出层
        layers.append(nn.Conv2d(features, out_channels, 3, padding=1))

        self.network = nn.Sequential(*layers)

        # 残差连接
        self.skip_connect = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        identity = self.skip_connect(x)
        out = self.network(x)
        return out + identity


class NoiseDataset(Dataset):
    """噪音数据集类。"""

    def __init__(self, dataset_dir, patch_size=64, augment=True):
        self.dataset_dir = dataset_dir
        self.patch_size = patch_size
        self.augment = augment

        # 支持新结构（图片预处理生成的格式）
        self.noisy_dir = os.path.join(dataset_dir, 'train', 'noisy')
        self.clean_dir = os.path.join(dataset_dir, 'train', 'clean')

        # 如果新结构不存在，尝试旧结构（兼容旧版本）
        if not os.path.isdir(self.noisy_dir) or not os.path.isdir(self.clean_dir):
            self.noisy_dir = os.path.join(dataset_dir, 'noisy_patches')
            self.clean_dir = os.path.join(dataset_dir, 'clean_patches')

        # 检查目录是否存在
        if not os.path.isdir(self.noisy_dir) or not os.path.isdir(self.clean_dir):
            self.noisy_files = []
            self.clean_files = []
            self.pairs = []
            return

        self.noisy_files = sorted([
            os.path.join(self.noisy_dir, f)
            for f in os.listdir(self.noisy_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

        self.clean_files = sorted([
            os.path.join(self.clean_dir, f)
            for f in os.listdir(self.clean_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

        # 匹配文件对
        self.pairs = []
        noisy_basenames = {os.path.basename(f): f for f in self.noisy_files}
        for cf in self.clean_files:
            base = os.path.basename(cf)
            # 新结构：文件名相同
            if base in noisy_basenames:
                self.pairs.append((noisy_basenames[base], cf))
            # 旧结构：文件名带 noisy_ 前缀
            elif f'noisy_{base}' in noisy_basenames:
                self.pairs.append((noisy_basenames[f'noisy_{base}'], cf))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx]

        # 加载图像
        noisy_img = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
        clean_img = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)

        if noisy_img is None or clean_img is None:
            # 返回默认值
            return torch.zeros(1, self.patch_size, self.patch_size), \
                   torch.zeros(1, self.patch_size, self.patch_size)

        # 归一化到 [0, 1]
        noisy_norm = noisy_img.astype(np.float32) / 255.0
        clean_norm = clean_img.astype(np.float32) / 255.0

        # 随机裁剪 patch
        h, w = noisy_norm.shape
        if h > self.patch_size and w > self.patch_size:
            y = np.random.randint(0, h - self.patch_size)
            x = np.random.randint(0, w - self.patch_size)
            noisy_patch = noisy_norm[y:y+self.patch_size, x:x+self.patch_size]
            clean_patch = clean_norm[y:y+self.patch_size, x:x+self.patch_size]
        else:
            # 图像太小，直接 resize
            noisy_patch = cv2.resize(noisy_norm, (self.patch_size, self.patch_size))
            clean_patch = cv2.resize(clean_norm, (self.patch_size, self.patch_size))

        # 数据增强
        if self.augment:
            k = np.random.randint(0, 4)
            noisy_patch = np.rot90(noisy_patch, k).copy()
            clean_patch = np.rot90(clean_patch, k).copy()

            if np.random.rand() > 0.5:
                noisy_patch = np.fliplr(noisy_patch).copy()
                clean_patch = np.fliplr(clean_patch).copy()

        # 转换为 tensor (C, H, W)
        noisy_tensor = torch.from_numpy(noisy_patch).unsqueeze(0).float()
        clean_tensor = torch.from_numpy(clean_patch).unsqueeze(0).float()

        return noisy_tensor, clean_tensor


class TrainingThread(QThread):
    """后台训练线程。"""
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(int, str)
    epoch_end = pyqtSignal(int, dict)

    def __init__(self, dataset_dir, output_dir, epochs=50, batch_size=16,
                 learning_rate=0.001, patch_size=64, model_type='denoiser'):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patch_size = patch_size
        self.model_type = model_type

    def run(self):
        if not TORCH_AVAILABLE:
            self.finished.emit(False, "PyTorch 不可用，无法进行训练")
            return

        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.progress.emit(10, f"使用设备：{device}")

            # 创建输出目录
            os.makedirs(self.output_dir, exist_ok=True)
            model_dir = os.path.join(self.output_dir, 'models')
            os.makedirs(model_dir, exist_ok=True)

            # 加载数据集
            self.progress.emit(20, "正在加载数据集...")
            dataset = NoiseDataset(self.dataset_dir, patch_size=self.patch_size)

            if len(dataset) == 0:
                self.finished.emit(False, "数据集中没有找到有效的图像对")
                return

            # 划分训练/验证集
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size

            train_dataset, val_dataset = torch.utils.data.dataset.random_split(
                dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                      shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                    shuffle=False, num_workers=0)

            self.progress.emit(30, f"数据集：{train_size} 训练样本，{val_size} 验证样本")

            # 创建模型
            model = SimpleDenoiserNet(in_channels=1, features=64, num_blocks=8).to(device)

            # 损失函数和优化器
            criterion = nn.L1Loss()
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )

            # 训练循环
            best_val_loss = float('inf')

            for epoch in range(self.epochs):
                if not self.isRunning():
                    break

                # 训练阶段
                model.train()
                train_loss = 0.0

                for batch_idx, (noisy, clean) in enumerate(train_loader):
                    noisy = noisy.to(device)
                    clean = clean.to(device)

                    optimizer.zero_grad()
                    output = model(noisy)
                    loss = criterion(output, clean)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                train_loss /= len(train_loader)

                # 验证阶段
                model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for noisy, clean in val_loader:
                        noisy = noisy.to(device)
                        clean = clean.to(device)
                        output = model(noisy)
                        loss = criterion(output, clean)
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                scheduler.step(val_loss)

                # 更新进度
                progress_pct = 30 + int((epoch + 1) / self.epochs * 60)
                self.progress.emit(progress_pct, f"Epoch {epoch+1}/{self.epochs}")
                self.epoch_end.emit(epoch + 1, {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'lr': optimizer.param_groups[0]['lr']
                })

                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model_path = os.path.join(model_dir, 'best_denoiser.pth')
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                    }, model_path)

            # 导出为 ONNX
            self.progress.emit(95, "正在导出 ONNX 模型...")
            model.eval()
            dummy_input = torch.randn(1, 1, self.patch_size, self.patch_size).to(device)

            try:
                onnx_path = os.path.join(self.output_dir, 'models', 'denoiser.onnx')
                torch.onnx.export(
                    model, dummy_input, onnx_path,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch', 2: 'height', 3: 'width'},
                        'output': {0: 'batch', 2: 'height', 3: 'width'}
                    },
                    opset_version=11
                )
            except Exception as onnx_error:
                # ONNX 导出失败，继续保存配置
                print(f"ONNX 导出失败：{onnx_error}")
                self.progress.emit(95, f"ONNX 导出失败：{str(onnx_error)}")

            # 保存训练配置
            config = {
                'model_type': self.model_type,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'patch_size': self.patch_size,
                'best_val_loss': best_val_loss,
                'device': str(device)
            }
            config_path = os.path.join(self.output_dir, 'train_config.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            self.progress.emit(100, "训练完成")
            self.finished.emit(True, f"训练完成！模型已保存到：{model_dir}")

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.finished.emit(False, f"训练失败：{str(e)}\n\n{error_details}")


class TrainingPage(QWidget):
    """算法训练页面。"""

    def __init__(self):
        super().__init__()
        self.is_training = False
        self.current_epoch = 0
        self.train_history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'lr': []}
        self.init_ui()

    def init_ui(self):
        """初始化用户界面。"""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        # 标题
        title = QLabel("算法训练 - 神经网络模型训练")
        title.setStyleSheet("""
            font-size: 24px;
            font-weight: 600;
            color: #1e293b;
            padding: 14px 18px;
            border-left: 4px solid #0ea5e9;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #f0f9ff, stop:1 transparent);
            border-radius: 8px;
        """)
        layout.addWidget(title)

        # 提示信息
        if not TORCH_AVAILABLE:
            warning = QLabel("⚠ PyTorch 未安装，训练功能不可用。请安装 PyTorch: pip install torch")
            warning.setStyleSheet("""
                QLabel {
                    background-color: #fef3c7;
                    color: #92400e;
                    padding: 12px;
                    border-radius: 8px;
                    border-left: 4px solid #f59e0b;
                }
            """)
            layout.addWidget(warning)

        # 主内容区域
        main_layout = QHBoxLayout()
        main_layout.setSpacing(16)

        # 左侧 - 训练配置（带滚动支持）
        left_panel = self._create_config_panel()
        left_scroll = QScrollArea()
        left_scroll.setWidget(left_panel)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFrameStyle(QFrame.NoFrame)
        left_scroll.setMinimumWidth(280)
        main_layout.addWidget(left_scroll, 1)

        # 右侧 - 训练监控
        right_panel = self._create_monitor_panel()
        main_layout.addWidget(right_panel, 2)

        layout.addLayout(main_layout)

    def _create_config_panel(self):
        """创建配置面板。"""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
            }
        """)
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)
        layout.setContentsMargins(14, 14, 14, 14)

        # 1. 数据集选择
        data_group = QGroupBox("1. 数据集目录")
        data_layout = QVBoxLayout(data_group)
        data_layout.setSpacing(10)

        self.dataset_path_edit = QTextEdit()
        self.dataset_path_edit.setReadOnly(True)
        self.dataset_path_edit.setMaximumHeight(50)
        self.dataset_path_edit.setPlaceholderText("选择预处理生成的数据集目录...")
        data_layout.addWidget(self.dataset_path_edit)

        browse_btn = QPushButton("选择数据集目录")
        browse_btn.clicked.connect(self.browse_dataset)
        browse_btn.setMinimumHeight(44)
        data_layout.addWidget(browse_btn)

        layout.addWidget(data_group)

        # 2. 训练参数
        param_group = QGroupBox("2. 训练参数")
        param_layout = QFormLayout(param_group)
        param_layout.setSpacing(10)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 500)
        self.epochs_spin.setValue(50)
        self.epochs_spin.setMinimumHeight(40)
        self.epochs_spin.setStyleSheet("font-size: 15px; padding: 10px 14px;")
        param_layout.addRow("训练轮数 (Epochs):", self.epochs_spin)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 128)
        self.batch_size_spin.setValue(16)
        self.batch_size_spin.setMinimumHeight(40)
        self.batch_size_spin.setStyleSheet("font-size: 15px; padding: 10px 14px;")
        param_layout.addRow("批次大小 (Batch Size):", self.batch_size_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setMinimumHeight(40)
        self.lr_spin.setStyleSheet("font-size: 15px; padding: 10px 14px;")
        param_layout.addRow("学习率 (Learning Rate):", self.lr_spin)

        self.patch_size_spin = QSpinBox()
        self.patch_size_spin.setRange(32, 256)
        self.patch_size_spin.setValue(64)
        self.patch_size_spin.setMinimumHeight(40)
        self.patch_size_spin.setStyleSheet("font-size: 15px; padding: 10px 14px;")
        param_layout.addRow("块大小 (Patch Size):", self.patch_size_spin)

        layout.addWidget(param_group)

        # 3. 输出设置
        output_group = QGroupBox("3. 模型输出")
        output_layout = QVBoxLayout(output_group)
        output_layout.setSpacing(10)

        self.output_path_edit = QTextEdit()
        self.output_path_edit.setReadOnly(True)
        self.output_path_edit.setMaximumHeight(50)
        self.output_path_edit.setPlaceholderText("模型保存目录...")
        output_layout.addWidget(self.output_path_edit)

        self.browse_output_btn = QPushButton("选择输出目录")
        self.browse_output_btn.clicked.connect(self.browse_output)
        self.browse_output_btn.setMinimumHeight(44)
        output_layout.addWidget(self.browse_output_btn)

        # 训练按钮
        self.train_btn = QPushButton("开始训练")
        self.train_btn.setObjectName("primaryBtn")
        self.train_btn.setStyleSheet("""
            QPushButton#primaryBtn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #16a34a, stop:1 #059669);
                font-size: 16px;
                font-weight: 600;
                padding: 16px 32px;
                border-radius: 8px;
            }
            QPushButton#primaryBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #15803d, stop:1 #047857);
            }
            QPushButton#primaryBtn:disabled {
                background: #cbd5e1;
            }
        """)
        self.train_btn.clicked.connect(self.start_training)
        self.train_btn.setEnabled(False)
        self.train_btn.setMinimumHeight(48)
        output_layout.addWidget(self.train_btn)

        # 停止按钮
        self.stop_btn = QPushButton("停止训练")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc2626;
                color: white;
                font-size: 16px;
                font-weight: 600;
                padding: 14px 28px;
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #b91c1c; }
            QPushButton:disabled { background-color: #cbd5e1; }
        """)
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMinimumHeight(48)
        output_layout.addWidget(self.stop_btn)

        layout.addWidget(output_group)

        # 4. 模型集成
        integrate_group = QGroupBox("4. 模型集成")
        integrate_layout = QVBoxLayout(integrate_group)
        integrate_layout.setSpacing(10)

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
        self.integrate_model_btn.setStyleSheet("""
            QPushButton#integrateModelBtn {
                background-color: #0ea5e9;
                color: white;
                font-size: 16px;
                font-weight: 600;
                padding: 16px 32px;
                border-radius: 8px;
            }
            QPushButton#integrateModelBtn:hover {
                background-color: #0284c7;
            }
            QPushButton#integrateModelBtn:disabled {
                background-color: #cbd5e1;
            }
        """)
        self.integrate_model_btn.clicked.connect(self.integrate_model)
        self.integrate_model_btn.setEnabled(False)
        integrate_layout.addWidget(self.integrate_model_btn)

        layout.addWidget(integrate_group)

        layout.addStretch()
        return panel

    def _create_monitor_panel(self):
        """创建训练监控面板。"""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
                padding: 20px;
            }
        """)
        layout = QVBoxLayout(panel)
        layout.setSpacing(16)
        layout.setContentsMargins(0, 0, 0, 0)

        # 进度显示
        progress_group = QGroupBox("训练进度")
        progress_layout = QVBoxLayout(progress_group)

        self.progress = QProgressBar()
        self.progress.setMinimumHeight(36)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                text-align: center;
                font-weight: 600;
                font-size: 15px;
                background-color: #f8fafc;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3b82f6, stop:1 #8b5cf6);
                border-radius: 7px;
            }
        """)
        progress_layout.addWidget(self.progress)

        self.status_label = QLabel("准备就绪")
        self.status_label.setStyleSheet("color: #64748b; font-size: 15px;")
        progress_layout.addWidget(self.status_label)

        layout.addWidget(progress_group)

        # 日志输出
        log_group = QGroupBox("训练日志")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("训练日志将显示在这里...")
        self.log_text.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 16px;
                background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        log_layout.addWidget(self.log_text)

        layout.addWidget(log_group)

        # Loss 曲线图
        chart_group = QGroupBox("训练 Loss 曲线")
        chart_layout = QVBoxLayout(chart_group)

        # 创建 matplotlib 图表
        self.loss_figure = Figure(figsize=(5, 4), dpi=100)
        self.loss_canvas = FigureCanvasQTAgg(self.loss_figure)
        self.loss_canvas.setMinimumHeight(300)
        self.loss_canvas.setStyleSheet("background-color: white;")

        # 初始化图表
        self.loss_ax = self.loss_figure.add_subplot(111)
        self.loss_ax.set_xlabel('Epoch')
        self.loss_ax.set_ylabel('Loss')
        self.loss_ax.set_title('Training and Validation Loss')
        self.loss_ax.grid(True, alpha=0.3)
        self.train_loss_line = None
        self.val_loss_line = None

        chart_layout.addWidget(self.loss_canvas)
        layout.addWidget(chart_group)

        return panel

    def browse_dataset(self):
        """选择数据集目录。"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择数据集目录", ""
        )

        if dir_path:
            self.dataset_path_edit.setText(dir_path)
            # 验证目录结构
            if self._validate_dataset(dir_path):
                self.train_btn.setEnabled(True)
                self.log_text.append(f"✓ 数据集目录有效：{dir_path}")
            else:
                self.log_text.append(f"⚠ 数据集目录可能不完整，请确认包含 train/clean 和 train/noisy 子目录")

    def _validate_dataset(self, dataset_dir):
        """验证数据集目录结构。"""
        # 检查新结构（图片预处理生成的格式）
        train_clean = os.path.join(dataset_dir, 'train', 'clean')
        train_noisy = os.path.join(dataset_dir, 'train', 'noisy')

        # 检查旧结构（兼容旧版本）
        noisy_patches = os.path.join(dataset_dir, 'noisy_patches')
        clean_patches = os.path.join(dataset_dir, 'clean_patches')

        # 新结构：检查 train/clean 和 train/noisy 是否存在
        if os.path.isdir(train_clean) and os.path.isdir(train_noisy):
            return True

        # 旧结构：检查 noisy_patches 和 clean_patches 是否存在
        if os.path.isdir(noisy_patches) and os.path.isdir(clean_patches):
            return True

        return False

    def browse_output(self):
        """选择模型输出目录。"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择模型输出目录", ""
        )

        if dir_path:
            self.output_path_edit.setText(dir_path)

    def start_training(self):
        """开始训练。"""
        dataset_dir = self.dataset_path_edit.toPlainText().strip()
        if not dataset_dir:
            QMessageBox.warning(self, "警告", "请先选择数据集目录")
            return

        output_dir = self.output_path_edit.toPlainText().strip()
        if not output_dir:
            output_dir = os.path.join(os.path.dirname(__file__), 'trained_models')

        # 创建带时间戳的子目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(output_dir, f'trained_models_{timestamp}')
        self.output_path_edit.setText(output_dir)

        self.is_training = True
        self.train_history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'lr': []}
        self.current_epoch = 0

        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setValue(0)

        self.thread = TrainingThread(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_size_spin.value(),
            learning_rate=self.lr_spin.value(),
            patch_size=self.patch_size_spin.value(),
            model_type=self.model_type_combo.currentData() or 'denoiser'
        )

        self.thread.progress.connect(self.update_progress)
        self.thread.finished.connect(self.training_finished)
        self.thread.epoch_end.connect(self.update_epoch_metrics)
        self.thread.start()

        self.log_text.append("训练开始...")

    def stop_training(self):
        """停止训练。"""
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.requestInterruption()
            self.thread.wait(2000)
            self.log_text.append("训练已停止")
            self.training_finished(False, "用户停止训练")

    def update_progress(self, value, message):
        """更新进度。"""
        self.progress.setValue(value)
        self.status_label.setText(message)

    def update_epoch_metrics(self, epoch, metrics):
        """更新 Epoch 指标。"""
        self.current_epoch = epoch

        # 记录历史数据
        self.train_history['epoch'].append(epoch)
        self.train_history['train_loss'].append(metrics['train_loss'])
        self.train_history['val_loss'].append(metrics['val_loss'])
        self.train_history['lr'].append(metrics['lr'])

        self.log_text.append(
            f"Epoch {epoch}: train_loss={metrics['train_loss']:.6f}, "
            f"val_loss={metrics['val_loss']:.6f}, lr={metrics['lr']:.6f}"
        )

        # 更新图表
        self.update_loss_chart()

        # 自动滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_loss_chart(self):
        """更新 Loss 曲线图。"""
        self.loss_ax.clear()
        self.loss_ax.set_xlabel('Epoch')
        self.loss_ax.set_ylabel('Loss')
        self.loss_ax.set_title('Training and Validation Loss')
        self.loss_ax.grid(True, alpha=0.3)

        if len(self.train_history['epoch']) > 0:
            self.loss_ax.plot(
                self.train_history['epoch'],
                self.train_history['train_loss'],
                'b-',
                label='Train Loss',
                linewidth=2
            )
            self.loss_ax.plot(
                self.train_history['epoch'],
                self.train_history['val_loss'],
                'r-',
                label='Val Loss',
                linewidth=2
            )
            self.loss_ax.legend(loc='upper right')

        self.loss_canvas.draw()

    def training_finished(self, success, message):
        """训练完成回调。"""
        self.is_training = False
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        if success:
            QMessageBox.information(self, "成功", message)
            self.log_text.append(f"✓ {message}")

            # 启用集成按钮
            self.integrate_model_btn.setEnabled(True)
            self.log_text.append("✓ 训练完成，点击'添加'按钮可集成模型")
        else:
            QMessageBox.warning(self, "提示", message)
            self.log_text.append(f"✗ {message}")

        self.status_label.setText("训练结束")

    def integrate_model(self):
        """集成训练后的模型到对应算法。"""
        try:
            output_dir = self.output_path_edit.toPlainText().strip()
            if not output_dir:
                QMessageBox.warning(self, "警告", "没有找到输出目录")
                return

            # 查找模型文件
            model_dir = os.path.join(output_dir, 'models')
            if not os.path.isdir(model_dir):
                QMessageBox.warning(self, "警告", "没有找到模型文件目录")
                return

            # 根据模型类型决定目标目录
            model_type = self.model_type_combo.currentText()
            if "降噪" in model_type:
                target_dir = os.path.join(os.path.dirname(__file__), 'integrated_model', 'denoise')
            else:
                target_dir = os.path.join(os.path.dirname(__file__), 'integrated_model', 'super_resolution')

            os.makedirs(target_dir, exist_ok=True)

            # 复制模型文件
            import shutil
            copied_count = 0
            for filename in os.listdir(model_dir):
                if filename.endswith(('.pth', '.onnx', '.json')):
                    src = os.path.join(model_dir, filename)
                    dst = os.path.join(target_dir, filename)
                    shutil.copy2(src, dst)
                    copied_count += 1

            self.log_text.append(f"✓ 模型已集成到：{target_dir}")
            self.integrate_model_btn.setEnabled(False)

            # 创建集成标记文件
            marker_path = os.path.join(target_dir, 'model_ready.marker')
            with open(marker_path, 'w', encoding='utf-8') as f:
                f.write(f"Integrated at {datetime.now().isoformat()}\nModel Type: {model_type}")

            QMessageBox.information(self, "成功", f"模型已集成到{'降噪' if '降噪' in model_type else '超分辨率'}算法\n目录：{target_dir}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"集成失败：{str(e)}")
            self.log_text.append(f"✗ 集成失败：{str(e)}")

    def apply_medical_style(self):
        """应用 Medical Minimalism 风格"""
        # 主标题
        for label in self.findChildren(QLabel):
            if label.text().startswith("算法训练"):
                label.setStyleSheet("""
                    font-size: 24px;
                    font-weight: 600;
                    color: #1e293b;
                    padding: 14px 18px;
                    border-left: 4px solid #0ea5e9;
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #f0f9ff, stop:1 transparent);
                    border-radius: 8px;
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

        # 按钮样式
        for btn in self.findChildren(QPushButton):
            if "选择" in btn.text():
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
            elif "开始" in btn.text():
                btn.setStyleSheet("""
                    QPushButton {
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #16a34a, stop:1 #059669);
                        color: white;
                        border: none;
                        padding: 16px 32px;
                        border-radius: 8px;
                        font-weight: 600;
                        font-size: 16px;
                    }
                    QPushButton:hover {
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #15803d, stop:1 #047857);
                    }
                    QPushButton:disabled {
                        background: #cbd5e1;
                        color: #94a3b8;
                    }
                """)
            elif "停止" in btn.text():
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #dc2626;
                        color: white;
                        border: none;
                        padding: 14px 28px;
                        border-radius: 8px;
                        font-weight: 600;
                        font-size: 16px;
                    }
                    QPushButton:hover {
                        background-color: #b91c1c;
                    }
                    QPushButton:disabled {
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

        # 日志文本框
        for text in self.findChildren(QTextEdit):
            if text.isReadOnly():
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

        # SpinBox 和 DoubleSpinBox
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

        for spin in self.findChildren(QDoubleSpinBox):
            spin.setStyleSheet("""
                QDoubleSpinBox {
                    padding: 8px 12px;
                    border: 1px solid #cbd5e1;
                    border-radius: 8px;
                    background-color: white;
                    min-height: 32px;
                    color: #1e293b;
                }
                QDoubleSpinBox:hover {
                    border-color: #0ea5e9;
                }
                QDoubleSpinBox:focus {
                    border-color: #0ea5e9;
                }
                QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                    border: none;
                    width: 24px;
                    background: #f1f5f9;
                    border-radius: 6px;
                    margin: 2px;
                }
                QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
                    background: #e2e8f0;
                }
            """)
