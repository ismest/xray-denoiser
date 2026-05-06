"""
数据集生成页面 - 使用提取的噪声参数生成合成图像配对数据集
"""

import os
import json
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QProgressBar, QSpinBox,
                             QDoubleSpinBox, QFormLayout, QGroupBox, QFrame,
                             QMessageBox, QTextEdit, QSizePolicy, QLineEdit)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import cv2


class DatasetGenerationThread(QThread):
    """后台线程用于数据集生成 - 使用提取的噪声参数生成合成噪声图像对。"""
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(int, str)

    def __init__(self, noise_params, base_image_dir, output_dir, total_patches=10000,
                 patch_size=64, train_split=80, test_split=10, val_split=10):
        super().__init__()
        self.noise_params = noise_params
        self.base_image_dir = base_image_dir
        self.output_dir = output_dir
        self.total_patches = total_patches
        self.patch_size = patch_size
        self.train_split = train_split
        self.test_split = test_split
        self.val_split = val_split

    def run(self):
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(self.output_dir, f"dataset_{timestamp}")

            train_clean_dir = os.path.join(self.output_dir, 'train', 'clean')
            train_noisy_dir = os.path.join(self.output_dir, 'train', 'noisy')
            test_clean_dir = os.path.join(self.output_dir, 'test', 'clean')
            test_noisy_dir = os.path.join(self.output_dir, 'test', 'noisy')
            val_clean_dir = os.path.join(self.output_dir, 'val', 'clean')
            val_noisy_dir = os.path.join(self.output_dir, 'val', 'noisy')

            for d in [train_clean_dir, train_noisy_dir, test_clean_dir, test_noisy_dir,
                      val_clean_dir, val_noisy_dir]:
                os.makedirs(d, exist_ok=True)

            self.progress.emit(10, "正在加载基础图像...")

            base_images = self._load_base_images()

            if len(base_images) == 0:
                self.finished.emit(False, "未找到基础图像")
                return

            self.progress.emit(20, f"已加载 {len(base_images)} 张基础图像，开始生成数据集...")

            train_count = int(self.total_patches * self.train_split / 100)
            test_count = int(self.total_patches * self.test_split / 100)
            val_count = self.total_patches - train_count - test_count

            saved_count = {'train': 0, 'test': 0, 'val': 0}
            total_saved = 0

            for i in range(self.total_patches):
                if i < train_count:
                    split = 'train'
                    idx = saved_count['train']
                elif i < train_count + test_count:
                    split = 'test'
                    idx = saved_count['test']
                else:
                    split = 'val'
                    idx = saved_count['val']

                base_img = base_images[np.random.randint(0, len(base_images))]

                h, w = base_img.shape[:2]
                if h > self.patch_size and w > self.patch_size:
                    y = np.random.randint(0, h - self.patch_size)
                    x = np.random.randint(0, w - self.patch_size)
                    clean_patch = base_img[y:y+self.patch_size, x:x+self.patch_size]
                else:
                    clean_patch = cv2.resize(base_img, (self.patch_size, self.patch_size))

                noisy_patch = self._add_noise(clean_patch)

                if split == 'train':
                    clean_path = os.path.join(train_clean_dir, f'patch_{idx:05d}.png')
                    noisy_path = os.path.join(train_noisy_dir, f'patch_{idx:05d}.png')
                elif split == 'test':
                    clean_path = os.path.join(test_clean_dir, f'patch_{idx:05d}.png')
                    noisy_path = os.path.join(test_noisy_dir, f'patch_{idx:05d}.png')
                else:
                    clean_path = os.path.join(val_clean_dir, f'patch_{idx:05d}.png')
                    noisy_path = os.path.join(val_noisy_dir, f'patch_{idx:05d}.png')

                cv2.imwrite(clean_path, clean_patch)
                cv2.imwrite(noisy_path, noisy_patch)

                saved_count[split] += 1
                total_saved += 1

                if i % 100 == 0:
                    progress_val = 20 + int(70 * i / self.total_patches)
                    self.progress.emit(progress_val, f"已生成 {total_saved}/{self.total_patches} 个样本...")

            metadata = {
                'noise_model': 'Poisson + AWGN + Gaussian Blur',
                'noise_params': self.noise_params,
                'total_patches': self.total_patches,
                'train_count': saved_count['train'],
                'test_count': saved_count['test'],
                'val_count': saved_count['val'],
                'patch_size': self.patch_size,
                'splits': {
                    'train': self.train_split,
                    'test': self.test_split,
                    'val': self.val_split
                }
            }

            meta_path = os.path.join(self.output_dir, 'dataset_metadata.json')
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            self.progress.emit(100, f"数据集生成完成 - 共 {total_saved} 个样本")
            self.finished.emit(True, f"成功生成 {total_saved} 个样本到 {self.output_dir}")

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.finished.emit(False, f"数据集生成失败：{str(e)}\n\n{error_details}")

    def _load_base_images(self):
        """加载基础图像。"""
        images = []

        if self.base_image_dir and os.path.isfile(self.base_image_dir):
            img = cv2.imread(self.base_image_dir, cv2.IMREAD_UNCHANGED)
            if img is not None:
                images.append(img)

        if self.base_image_dir and os.path.isdir(self.base_image_dir):
            image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
            print(f"Loading images from: {self.base_image_dir}")
            try:
                for root, dirs, files in os.walk(self.base_image_dir):
                    for file in files:
                        ext = os.path.splitext(file)[1].lower()
                        if ext in image_extensions:
                            img_path = os.path.join(root, file)
                            img_data = np.fromfile(img_path, dtype=np.uint8)
                            if img_data is not None and len(img_data) > 0:
                                img = cv2.imdecode(img_data, cv2.IMREAD_UNCHANGED)
                                if img is not None:
                                    images.append(img)
            except Exception as e:
                print(f"Error loading images from {self.base_image_dir}: {e}")

            print(f"Loaded {len(images)} images")
        else:
            print(f"base_image_dir is not set or not a directory: {self.base_image_dir}")

        return images

    def _add_noise(self, clean_patch):
        """根据论文方法添加合成噪声：Poisson(λ) + AWGN(σ) + Gaussian Blur(σ=1)。"""
        poisson_lambda = self.noise_params.get('poisson_lambda', 10)
        awgn_sigma = self.noise_params.get('awgn_sigma', 0.05)
        blur_sigma = self.noise_params.get('gaussian_blur_sigma', 1.0)

        lambda_range = self.noise_params.get('poisson_lambda_range', None)
        awgn_range = self.noise_params.get('awgn_sigma_range', None)

        if lambda_range and isinstance(lambda_range, dict):
            poisson_lambda = np.random.uniform(lambda_range['min'], lambda_range['max'])
        if awgn_range and isinstance(awgn_range, dict):
            awgn_sigma = np.random.uniform(awgn_range['min'], awgn_range['max'])

        if poisson_lambda == 0 and awgn_sigma == 0 and blur_sigma == 0:
            return clean_patch.copy()

        if clean_patch.dtype == np.uint16:
            img_float = clean_patch.astype(np.float64) / 65535.0
        else:
            img_float = clean_patch.astype(np.float64) / 255.0

        if poisson_lambda > 0:
            scaled = img_float * poisson_lambda
            noisy = np.random.poisson(scaled).astype(np.float64) / poisson_lambda
        else:
            noisy = img_float.copy()

        if awgn_sigma > 0:
            noisy += np.random.normal(0, awgn_sigma, noisy.shape)

        if blur_sigma > 0:
            kernel_size = int(6 * blur_sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            noisy = cv2.GaussianBlur(noisy, (kernel_size, kernel_size), blur_sigma)

        noisy = np.clip(noisy, 0, 1)

        if clean_patch.dtype == np.uint16:
            return (noisy * 65535).astype(np.uint16)
        else:
            return (noisy * 255).astype(np.uint8)


class DatasetGenerationPage(QWidget):
    """数据集生成页面 - 使用噪声参数生成合成配对数据集。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.noise_params = None
        self.init_ui()

    def showEvent(self, event):
        """切换到本页面时自动加载最新的噪声参数。"""
        super().showEvent(event)
        self._auto_load_noise_params()

    def _auto_load_noise_params(self):
        """从 noise_profile_output/noise_params.json 自动填充噪声参数 spinbox。"""
        params_path = os.path.join(os.path.dirname(__file__), 'noise_profile_output', 'noise_params.json')
        if os.path.exists(params_path):
            try:
                with open(params_path, 'r') as f:
                    self.noise_params = json.load(f)
                self.poisson_lambda_spin.setValue(self.noise_params['poisson_lambda'])
                self.awgn_sigma_spin.setValue(self.noise_params['awgn_sigma'])
                self.blur_sigma_spin.setValue(self.noise_params.get('gaussian_blur_sigma', 1.0))
            except Exception:
                pass

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        title = QLabel("数据集生成")
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

        content = self._create_content()
        layout.addWidget(content)

    def _create_content(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        widget.setMinimumSize(1000, 700)

        # 左侧控制面板
        left_panel = QFrame()
        left_panel.setObjectName("step2ControlPanel")
        left_panel.setStyleSheet("""
            QFrame#step2ControlPanel {
                background-color: #f8fafc;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
                padding: 20px;
            }
        """)
        left_panel.setMinimumWidth(400)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(16)

        info_label = QLabel("使用步骤 1 提取的噪声参数，生成合成噪声图像与干净图像配对数据集")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #475569; font-size: 15px; font-weight: 500; padding: 8px;")
        left_layout.addWidget(info_label)

        # 噪声参数（可编辑）
        params_group = QGroupBox("已提取的噪声参数（可编辑）")
        params_group.setMinimumHeight(160)
        params_layout = QFormLayout(params_group)
        params_layout.setSpacing(12)

        self.poisson_lambda_spin = QDoubleSpinBox()
        self.poisson_lambda_spin.setRange(0, 200)
        self.poisson_lambda_spin.setValue(0)
        self.poisson_lambda_spin.setDecimals(2)
        self.poisson_lambda_spin.setMinimumHeight(40)
        self.poisson_lambda_spin.setStyleSheet("""
            QDoubleSpinBox {
                font-size: 15px;
                padding: 10px 14px;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
            }
        """)
        params_layout.addRow("Poisson λ:", self.poisson_lambda_spin)

        self.awgn_sigma_spin = QDoubleSpinBox()
        self.awgn_sigma_spin.setRange(0, 0.5)
        self.awgn_sigma_spin.setValue(0)
        self.awgn_sigma_spin.setDecimals(4)
        self.awgn_sigma_spin.setSingleStep(0.01)
        self.awgn_sigma_spin.setMinimumHeight(40)
        self.awgn_sigma_spin.setStyleSheet("""
            QDoubleSpinBox {
                font-size: 15px;
                padding: 10px 14px;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
            }
        """)
        params_layout.addRow("AWGN σ:", self.awgn_sigma_spin)

        self.blur_sigma_spin = QDoubleSpinBox()
        self.blur_sigma_spin.setRange(0, 5)
        self.blur_sigma_spin.setValue(0)
        self.blur_sigma_spin.setDecimals(2)
        self.blur_sigma_spin.setSingleStep(0.1)
        self.blur_sigma_spin.setMinimumHeight(40)
        self.blur_sigma_spin.setStyleSheet("""
            QDoubleSpinBox {
                font-size: 15px;
                padding: 10px 14px;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
            }
        """)
        params_layout.addRow("Gaussian Blur σ:", self.blur_sigma_spin)

        left_layout.addWidget(params_group)

        # 原始数据集导入
        dataset_import_group = QGroupBox("原始数据集导入")
        dataset_import_group.setMinimumHeight(180)
        import_layout = QVBoxLayout(dataset_import_group)
        import_layout.setSpacing(12)

        path_layout = QHBoxLayout()
        self.dataset_path_edit = QLineEdit()
        self.dataset_path_edit.setReadOnly(True)
        self.dataset_path_edit.setMinimumHeight(40)
        self.dataset_path_edit.setPlaceholderText("选择包含原始图像的文件夹")
        self.dataset_path_edit.setStyleSheet("""
            QLineEdit {
                font-size: 15px;
                padding: 10px 14px;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
            }
        """)
        path_layout.addWidget(self.dataset_path_edit, 1)

        self.browse_dataset_btn = QPushButton("浏览")
        self.browse_dataset_btn.setMinimumHeight(40)
        self.browse_dataset_btn.setMinimumWidth(80)
        self.browse_dataset_btn.setStyleSheet("""
            QPushButton {
                background-color: #f1f5f9;
                color: #475569;
                border: 1px solid #cbd5e1;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 15px;
            }
            QPushButton:hover {
                background-color: #e2e8f0;
                border-color: #0ea5e9;
            }
        """)
        self.browse_dataset_btn.clicked.connect(self.browse_dataset_dir)
        path_layout.addWidget(self.browse_dataset_btn)

        import_layout.addLayout(path_layout)

        self.image_count_label = QLabel("未选择数据集")
        self.image_count_label.setStyleSheet("""
            QLabel {
                color: #64748b;
                font-size: 15px;
                padding: 8px;
            }
        """)
        import_layout.addWidget(self.image_count_label)

        left_layout.addWidget(dataset_import_group)

        # 数据集配置
        dataset_group = QGroupBox("数据集配置")
        dataset_group.setMinimumHeight(200)
        dataset_layout = QFormLayout(dataset_group)
        dataset_layout.setSpacing(12)
        dataset_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        dataset_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        dataset_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)

        self.total_patches_spin = QSpinBox()
        self.total_patches_spin.setRange(0, 100000)
        self.total_patches_spin.setValue(0)
        self.total_patches_spin.setMinimumHeight(40)
        self.total_patches_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.total_patches_spin.setStyleSheet("QSpinBox { font-size: 15px; padding: 10px 14px; }")
        dataset_layout.addRow("总样本数:", self.total_patches_spin)

        self.train_split_spin = QSpinBox()
        self.train_split_spin.setRange(50, 90)
        self.train_split_spin.setValue(80)
        self.train_split_spin.setMinimumHeight(40)
        self.train_split_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.train_split_spin.setStyleSheet("QSpinBox { font-size: 15px; padding: 10px 14px; }")
        dataset_layout.addRow("训练集 (%):", self.train_split_spin)

        self.test_split_spin = QSpinBox()
        self.test_split_spin.setRange(5, 25)
        self.test_split_spin.setValue(10)
        self.test_split_spin.setMinimumHeight(40)
        self.test_split_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.test_split_spin.setStyleSheet("QSpinBox { font-size: 15px; padding: 10px 14px; }")
        dataset_layout.addRow("测试集 (%):", self.test_split_spin)

        self.val_split_spin = QSpinBox()
        self.val_split_spin.setRange(5, 25)
        self.val_split_spin.setValue(10)
        self.val_split_spin.setMinimumHeight(40)
        self.val_split_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.val_split_spin.setStyleSheet("QSpinBox { font-size: 15px; padding: 10px 14px; }")
        dataset_layout.addRow("验证集 (%):", self.val_split_spin)

        self.dataset_patch_size_spin = QSpinBox()
        self.dataset_patch_size_spin.setRange(32, 256)
        self.dataset_patch_size_spin.setValue(64)
        self.dataset_patch_size_spin.setMinimumHeight(40)
        self.dataset_patch_size_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.dataset_patch_size_spin.setStyleSheet("QSpinBox { font-size: 15px; padding: 10px 14px; }")
        dataset_layout.addRow("块大小:", self.dataset_patch_size_spin)

        left_layout.addWidget(dataset_group)

        # 生成数据集导出
        output_group = QGroupBox("生成数据集导出")
        output_group.setMinimumHeight(180)
        output_layout = QVBoxLayout(output_group)

        output_dir_layout = QHBoxLayout()
        self.dataset_output_edit = QLineEdit()
        self.dataset_output_edit.setReadOnly(False)
        self.dataset_output_edit.setMinimumHeight(40)
        self.dataset_output_edit.setPlaceholderText("使用步骤 1 输出目录")
        self.dataset_output_edit.setStyleSheet("""
            QLineEdit {
                font-size: 15px;
                padding: 10px 14px;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
            }
        """)
        output_dir_layout.addWidget(self.dataset_output_edit, 1)

        self.browse_output_btn = QPushButton("浏览")
        self.browse_output_btn.setMinimumHeight(40)
        self.browse_output_btn.setMinimumWidth(80)
        self.browse_output_btn.setStyleSheet("""
            QPushButton {
                background-color: #f1f5f9;
                color: #475569;
                border: 1px solid #cbd5e1;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 15px;
            }
            QPushButton:hover {
                background-color: #e2e8f0;
                border-color: #0ea5e9;
            }
        """)
        self.browse_output_btn.clicked.connect(self.browse_output_dir)
        output_dir_layout.addWidget(self.browse_output_btn)

        output_layout.addLayout(output_dir_layout)
        left_layout.addWidget(output_group)

        # 状态显示
        status_group = QGroupBox("数据集生成状态")
        status_layout = QVBoxLayout(status_group)

        self.generation_status_label = QLabel("状态：未开始")
        self.generation_status_label.setStyleSheet("""
            QLabel {
                color: #64748b;
                font-size: 15px;
                padding: 10px;
                background-color: #f8fafc;
                border-radius: 6px;
                border: 1px solid #e2e8f0;
            }
        """)
        status_layout.addWidget(self.generation_status_label)

        self.generation_detail_label = QLabel("请先导入原始数据集并设置参数")
        self.generation_detail_label.setStyleSheet("""
            QLabel {
                color: #94a3b8;
                font-size: 14px;
                padding: 8px;
            }
        """)
        status_layout.addWidget(self.generation_detail_label)

        left_layout.addWidget(status_group)

        # 执行按钮
        self.generate_btn = QPushButton("生成数据集")
        self.generate_btn.setObjectName("step2PrimaryBtn")
        self.generate_btn.setStyleSheet("""
            QPushButton#step2PrimaryBtn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10b981, stop:1 #059669);
                color: white;
                border: none;
                padding: 16px 32px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 16px;
            }
            QPushButton#step2PrimaryBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #059669, stop:1 #047857);
            }
            QPushButton#step2PrimaryBtn:disabled {
                background: #cbd5e1;
                color: #94a3b8;
            }
        """)
        self.generate_btn.clicked.connect(self.start_dataset_generation)
        self.generate_btn.setEnabled(False)
        self.generate_btn.setMinimumHeight(52)
        left_layout.addWidget(self.generate_btn)

        self.step2_progress = QProgressBar()
        self.step2_progress.setVisible(False)
        self.step2_progress.setStyleSheet("""
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
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10b981, stop:1 #059669);
                border-radius: 7px;
            }
        """)
        left_layout.addWidget(self.step2_progress)

        left_layout.addStretch()

        # 右侧面板
        right_panel = self._create_dataset_info_panel()
        right_panel.setMinimumWidth(500)
        layout.addWidget(left_panel, 1)
        layout.addWidget(right_panel, 2)

        return widget

    def _create_dataset_info_panel(self):
        """创建数据集信息面板（右侧）。"""
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
        layout.setSpacing(18)
        layout.setContentsMargins(0, 0, 0, 0)

        info_group = QGroupBox("数据集生成说明")
        info_group.setMinimumHeight(200)
        info_layout = QVBoxLayout(info_group)
        info_text = QLabel(
            "根据噪音提取页面获取的噪声参数，生成合成噪声图像与干净图像的配对数据集。\n\n"
            "噪声模型：Poisson(λ) + AWGN(σ) + Gaussian Blur(σ=1)\n\n"
            "输出结构：\n"
            "  - train/clean/ 和 train/noisy/\n"
            "  - test/clean/ 和 test/noisy/\n"
            "  - val/clean/ 和 val/noisy/\n"
            "  - dataset_metadata.json"
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet("color: #475569; font-size: 15px; line-height: 1.6;")
        info_layout.addWidget(info_text)
        layout.addWidget(info_group)

        # 数据集预览
        preview_group = QGroupBox("数据集预览")
        preview_group.setMinimumHeight(400)
        preview_layout = QVBoxLayout(preview_group)

        preview_images_layout = QHBoxLayout()
        preview_images_layout.setSpacing(10)

        self.preview_clean_label = QLabel("暂无预览")
        self.preview_clean_label.setObjectName("previewBox")
        self.preview_clean_label.setStyleSheet("""
            QLabel#previewBox {
                border: 2px dashed #cbd5e1;
                border-radius: 8px;
                background-color: #f8fafc;
                color: #94a3b8;
                font-style: italic;
                font-size: 14px;
            }
        """)
        self.preview_clean_label.setAlignment(Qt.AlignCenter)
        self.preview_clean_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_clean_label.setMinimumSize(300, 300)
        preview_images_layout.addWidget(self.preview_clean_label)

        self.preview_noisy_label = QLabel("暂无预览")
        self.preview_noisy_label.setObjectName("previewBox")
        self.preview_noisy_label.setStyleSheet("""
            QLabel#previewBox {
                border: 2px dashed #cbd5e1;
                border-radius: 8px;
                background-color: #f8fafc;
                color: #94a3b8;
                font-style: italic;
                font-size: 14px;
            }
        """)
        self.preview_noisy_label.setAlignment(Qt.AlignCenter)
        self.preview_noisy_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_noisy_label.setMinimumSize(300, 300)
        preview_images_layout.addWidget(self.preview_noisy_label)

        preview_layout.addLayout(preview_images_layout)

        self.preview_info_label = QLabel("请先生成数据集")
        self.preview_info_label.setStyleSheet("""
            QLabel {
                color: #64748b;
                font-size: 14px;
                padding: 8px;
            }
        """)
        self.preview_info_label.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(self.preview_info_label)

        self.refresh_preview_btn = QPushButton("刷新预览")
        self.refresh_preview_btn.setObjectName("refreshPreviewBtn")
        self.refresh_preview_btn.setStyleSheet("""
            QPushButton#refreshPreviewBtn {
                background-color: #f1f5f9;
                color: #475569;
                border: 1px solid #cbd5e1;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 14px;
            }
            QPushButton#refreshPreviewBtn:hover {
                background-color: #e2e8f0;
                border-color: #0ea5e9;
            }
            QPushButton#refreshPreviewBtn:disabled {
                background-color: #cbd5e1;
                color: #94a3b8;
            }
        """)
        self.refresh_preview_btn.clicked.connect(self.load_preview_pairs)
        self.refresh_preview_btn.setEnabled(False)
        preview_layout.addWidget(self.refresh_preview_btn)

        layout.addWidget(preview_group)

        log_group = QGroupBox("处理日志")
        log_layout = QVBoxLayout(log_group)
        self.dataset_log_text = QTextEdit()
        self.dataset_log_text.setReadOnly(True)
        self.dataset_log_text.setMaximumHeight(200)
        self.dataset_log_text.setPlaceholderText("数据集生成日志将显示在这里...")
        self.dataset_log_text.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', monospace;
                font-size: 13px;
                background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        log_layout.addWidget(self.dataset_log_text)
        layout.addWidget(log_group)

        return panel

    # ========== 数据集生成方法 ==========

    def get_current_noise_params(self):
        """从可编辑控件获取当前的噪声参数。"""
        noise_params = self.noise_params if self.noise_params else {}
        return {
            'poisson_lambda': self.poisson_lambda_spin.value(),
            'awgn_sigma': self.awgn_sigma_spin.value(),
            'gaussian_blur_sigma': self.blur_sigma_spin.value(),
            'estimated_noise_std': noise_params.get('estimated_noise_std', 0),
            'image_dtype': noise_params.get('image_dtype', 'uint8'),
            'image_shape': noise_params.get('image_shape', []),
            'box_coords': noise_params.get('box_coords', []),
            'lambda_estimates_count': noise_params.get('lambda_estimates_count', 0),
            'lambda_min': noise_params.get('lambda_min', 0),
            'lambda_max': noise_params.get('lambda_max', 0),
        }

    def browse_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择数据集输出目录", "")
        if dir_path:
            self.dataset_output_edit.setText(dir_path)
            dataset_path = self.dataset_path_edit.text().strip()
            if dataset_path:
                self.generate_btn.setEnabled(True)
                self._set_generation_status("idle", "已选择输出目录，可以生成数据集")

    def browse_dataset_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择原始数据集目录", "")
        if dir_path:
            self.dataset_path_edit.setText(dir_path)
            self.count_images_in_dir(dir_path)

    def count_images_in_dir(self, root_dir):
        """递归统计文件夹下的图片数量。"""
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        image_count = 0
        dir_count = 0

        try:
            for root, dirs, files in os.walk(root_dir):
                dir_count += 1
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in image_extensions:
                        image_count += 1

            if image_count > 0:
                self.image_count_label.setText(
                    f"已找到 {image_count:,} 张图片（{dir_count:,} 个文件夹）"
                )
                self.image_count_label.setStyleSheet("""
                    QLabel { color: #10b981; font-size: 15px; font-weight: 600; padding: 8px; }
                """)
                self.total_patches_spin.setValue(min(image_count, 10000))

                output_dir = self.dataset_output_edit.text().strip()
                if output_dir:
                    self.generate_btn.setEnabled(True)
                    self._set_generation_status("idle", f"已准备生成 {image_count:,} 个样本")
                else:
                    self.generate_btn.setEnabled(False)
                    self._set_generation_status("idle", "请选择输出目录")
            else:
                self.image_count_label.setText("未找到图片文件")
                self.image_count_label.setStyleSheet("""
                    QLabel { color: #ef4444; font-size: 15px; padding: 8px; }
                """)
                self.generate_btn.setEnabled(False)

        except Exception as e:
            self.image_count_label.setText(f"统计失败：{e}")
            self.image_count_label.setStyleSheet("""
                QLabel { color: #ef4444; font-size: 15px; padding: 8px; }
            """)
            self.generate_btn.setEnabled(False)

    def start_dataset_generation(self):
        """开始数据集生成。"""
        print("=== 开始数据集生成 ===")

        base_image_dir = self.dataset_path_edit.text().strip()
        if not base_image_dir:
            self._set_generation_status("error", "请先导入原始数据集")
            QMessageBox.warning(self, "警告", "请先导入原始数据集（点击'浏览'选择包含图像的文件夹）")
            return

        total_patches = self.total_patches_spin.value()
        if total_patches <= 0:
            self._set_generation_status("error", "总样本数必须大于 0")
            QMessageBox.warning(self, "警告", "总样本数必须大于 0，请先导入原始数据集或手动设置样本数")
            return

        poisson_lambda = self.poisson_lambda_spin.value()
        awgn_sigma = self.awgn_sigma_spin.value()

        if poisson_lambda == 0 and awgn_sigma == 0:
            reply = QMessageBox.question(
                self, "噪声参数为零",
                "当前噪声参数均为 0，生成的数据集将没有噪声。是否继续？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        current_params = self.get_current_noise_params()

        output_dir = self.dataset_output_edit.text().strip()
        if not output_dir:
            output_dir = os.path.join(os.path.dirname(__file__), 'generated_dataset')
            self.dataset_output_edit.setText(output_dir)

        patch_size = self.dataset_patch_size_spin.value()
        train_split = self.train_split_spin.value()
        test_split = self.test_split_spin.value()
        val_split = self.val_split_spin.value()

        if train_split + test_split + val_split != 100:
            self._set_generation_status("error", "数据集比例之和不等于 100%")
            QMessageBox.warning(self, "警告", "训练集 + 测试集 + 验证集比例必须等于 100%")
            return

        self._set_generation_status("processing", "正在生成数据集...")
        self.step2_progress.setVisible(True)
        self.step2_progress.setValue(0)
        self.generate_btn.setEnabled(False)
        self.dataset_log_text.append("开始生成数据集...")

        self.thread = DatasetGenerationThread(
            noise_params=current_params,
            base_image_dir=base_image_dir,
            output_dir=output_dir,
            total_patches=total_patches,
            patch_size=patch_size,
            train_split=train_split,
            test_split=test_split,
            val_split=val_split
        )
        self.thread.progress.connect(self.update_step2_progress)
        self.thread.finished.connect(self.dataset_generation_finished)
        self.thread.start()

    def update_step2_progress(self, value, message):
        self.step2_progress.setValue(value)
        self.dataset_log_text.append(message)
        self._set_generation_status("processing", message)

    def _set_generation_status(self, status, message):
        if status == "idle":
            self.generation_status_label.setText("状态：未开始")
            self.generation_status_label.setStyleSheet("""
                QLabel { color: #64748b; font-size: 15px; padding: 10px;
                         background-color: #f8fafc; border-radius: 6px; border: 1px solid #e2e8f0; }
            """)
            self.generation_detail_label.setText(message)
        elif status == "processing":
            self.generation_status_label.setText("状态：正在生成数据集...")
            self.generation_status_label.setStyleSheet("""
                QLabel { color: #0284c7; font-size: 15px; font-weight: 600; padding: 10px;
                         background-color: #e0f2fe; border-radius: 6px; border: 1px solid #7dd3fc; }
            """)
            self.generation_detail_label.setText(message)
        elif status == "success":
            self.generation_status_label.setText("状态：生成完成 ✓")
            self.generation_status_label.setStyleSheet("""
                QLabel { color: #047857; font-size: 15px; font-weight: 600; padding: 10px;
                         background-color: #d1fae5; border-radius: 6px; border: 1px solid #6ee7b7; }
            """)
            self.generation_detail_label.setText(message)
        elif status == "error":
            self.generation_status_label.setText("状态：生成失败 ✗")
            self.generation_status_label.setStyleSheet("""
                QLabel { color: #dc2626; font-size: 15px; font-weight: 600; padding: 10px;
                         background-color: #fee2e2; border-radius: 6px; border: 1px solid #fca5a5; }
            """)
            self.generation_detail_label.setText(message)

    def dataset_generation_finished(self, success, message):
        self.step2_progress.setVisible(False)
        self.generate_btn.setEnabled(True)

        if success:
            self._set_generation_status("success", message)
            QMessageBox.information(self, "成功", message)
            self.dataset_log_text.append(f"✓ {message}")
            self.refresh_preview_btn.setEnabled(True)
            self.load_preview_pairs()
        else:
            self._set_generation_status("error", message)
            QMessageBox.critical(self, "错误", message)
            self.dataset_log_text.append(f"✗ {message}")

    def load_preview_pairs(self):
        """从原始数据集随机加载一张图片，显示干净版本和加噪版本的对比。"""
        dataset_path = self.dataset_path_edit.text().strip()
        if not dataset_path:
            self.preview_info_label.setText("请先导入原始数据集")
            return

        poisson_lambda = self.poisson_lambda_spin.value()
        awgn_sigma = self.awgn_sigma_spin.value()
        blur_sigma = self.blur_sigma_spin.value()

        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        image_files = []
        try:
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in image_extensions:
                        image_files.append(os.path.join(root, file))
        except Exception as e:
            self.preview_info_label.setText(f"读取数据集失败：{e}")
            return

        if not image_files:
            self.preview_info_label.setText("数据集中未找到图片文件")
            return

        import random
        selected_image_path = random.choice(image_files)

        try:
            img_data = np.fromfile(selected_image_path, dtype=np.uint8)
            if img_data is None or len(img_data) == 0:
                self.preview_info_label.setText("无法读取图片")
                return

            clean_img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if clean_img is None:
                self.preview_info_label.setText("无法解码图片")
                return

            clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
            noisy_img = self._add_noise_for_preview(clean_img, poisson_lambda, awgn_sigma, blur_sigma)

            self._display_preview_image(clean_img, self.preview_clean_label)
            self._display_preview_image(noisy_img, self.preview_noisy_label)

            filename = os.path.basename(selected_image_path)
            self.preview_info_label.setText(
                f"原始图片：{filename}\n"
                f"Poisson λ={poisson_lambda:.1f}, AWGN σ={awgn_sigma:.4f}, Blur σ={blur_sigma:.1f}"
            )

        except Exception as e:
            self.preview_info_label.setText(f"加载预览失败：{e}")

    def _add_noise_for_preview(self, clean_img, poisson_lambda, awgn_sigma, blur_sigma):
        if poisson_lambda == 0 and awgn_sigma == 0 and blur_sigma == 0:
            return clean_img.copy()

        img_float = clean_img.astype(np.float64) / 255.0

        if poisson_lambda > 0:
            scaled = img_float * poisson_lambda
            noisy = np.random.poisson(scaled).astype(np.float64) / poisson_lambda
        else:
            noisy = img_float.copy()

        if awgn_sigma > 0:
            noisy += np.random.normal(0, awgn_sigma, noisy.shape)

        if blur_sigma > 0:
            kernel_size = int(6 * blur_sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            noisy = cv2.GaussianBlur(noisy, (kernel_size, kernel_size), blur_sigma)

        noisy = np.clip(noisy, 0, 1)
        return (noisy * 255).astype(np.uint8)

    def _display_preview_image(self, img, label):
        try:
            if img is None:
                label.setText("无法加载")
                label.setPixmap(QPixmap())
                return

            h, w = img.shape[:2]
            rgb_bytes = img.tobytes()
            q_img = QImage(rgb_bytes, w, h, w * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            label_rect = label.frameGeometry()
            avail_width = label_rect.width() - 20
            avail_height = label_rect.height() - 20

            if avail_width > 0 and avail_height > 0:
                scaled = pixmap.scaled(avail_width, avail_height,
                                       Qt.KeepAspectRatio, Qt.SmoothTransformation)
                label.setPixmap(scaled)
            else:
                label.setPixmap(pixmap)

            label.setText("")

        except Exception:
            label.setText("显示失败")

    def apply_medical_style(self):
        """应用 Medical Minimalism 风格。"""
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

        for btn in self.findChildren(QPushButton):
            if btn.objectName() == "step2PrimaryBtn" or "生成" in btn.text():
                btn.setStyleSheet("""
                    QPushButton#step2PrimaryBtn {
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10b981, stop:1 #059669);
                        color: white;
                        border: none;
                        padding: 16px 32px;
                        border-radius: 8px;
                        font-weight: 600;
                        font-size: 16px;
                    }
                    QPushButton#step2PrimaryBtn:hover {
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #059669, stop:1 #047857);
                    }
                    QPushButton#step2PrimaryBtn:disabled {
                        background: #cbd5e1;
                        color: #94a3b8;
                    }
                """)

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
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10b981, stop:1 #059669);
                    border-radius: 7px;
                }
            """)

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
                """)

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
                QSpinBox:hover { border-color: #0ea5e9; }
                QSpinBox:focus { border-color: #0ea5e9; }
                QSpinBox::up-button, QSpinBox::down-button {
                    border: none; width: 32px; background: #f1f5f9;
                    border-radius: 6px; margin: 2px;
                }
                QSpinBox::up-button:hover, QSpinBox::down-button:hover { background: #e2e8f0; }
            """)
