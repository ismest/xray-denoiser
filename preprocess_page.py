"""
图片预处理页面 - 用于提取噪音信息、构建训练数据集
"""

import sys
import os
import json
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QProgressBar, QComboBox,
                             QSpinBox, QDoubleSpinBox, QFormLayout, QGroupBox,
                             QFrame, QMessageBox, QTextEdit, QGridLayout,
                             QScrollArea, QSizePolicy)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import cv2


class NoiseExtractionThread(QThread):
    """后台线程用于噪音提取。"""
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(int, str)

    def __init__(self, image_pairs, output_dir, noise_method='difference', patch_size=64):
        super().__init__()
        self.image_pairs = image_pairs  # [(noisy_path, clean_path), ...]
        self.output_dir = output_dir
        self.noise_method = noise_method
        self.patch_size = patch_size

    def run(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'noisy_patches'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'clean_patches'), exist_ok=True)

            total_pairs = len(self.image_pairs)
            saved_patches = 0

            for idx, (noisy_path, clean_path) in enumerate(self.image_pairs):
                self.progress.emit(
                    int((idx / total_pairs) * 100),
                    f"正在处理 {os.path.basename(noisy_path)}..."
                )

                # 加载图像
                noisy_img = cv2.imread(noisy_path, cv2.IMREAD_UNCHANGED)
                if clean_path and os.path.exists(clean_path):
                    clean_img = cv2.imread(clean_path, cv2.IMREAD_UNCHANGED)
                else:
                    # 如果没有干净图像，使用降噪后的图像作为近似
                    clean_img = cv2.bilateralFilter(noisy_img, 9, 75, 75)

                if noisy_img is None or clean_img is None:
                    continue

                # 确保图像尺寸一致
                if noisy_img.shape != clean_img.shape:
                    clean_img = cv2.resize(clean_img, (noisy_img.shape[1], noisy_img.shape[0]))

                # 提取噪音图
                if self.noise_method == 'difference':
                    # 差分法：noise = noisy - clean
                    noisy_float = noisy_img.astype(np.float32)
                    clean_float = clean_img.astype(np.float32)
                    noise_map = noisy_float - clean_float
                elif self.noise_method == 'std':
                    # 局部标准差法
                    noise_map = self._extract_noise_std(noisy_img)
                else:
                    noise_map = noisy_img.astype(np.float32) - clean_img.astype(np.float32)

                # 保存噪音图
                noise_filename = f"noise_{idx:04d}.png"
                noise_path = os.path.join(self.output_dir, 'noise_maps', noise_filename)
                os.makedirs(os.path.join(self.output_dir, 'noise_maps'), exist_ok=True)

                # 归一化噪音图用于保存
                noise_normalized = self._normalize_for_save(noise_map)
                cv2.imwrite(noise_path, noise_normalized)

                # 保存对应的干净图像
                clean_filename = f"clean_{idx:04d}.png"
                clean_save_path = os.path.join(self.output_dir, 'clean_patches', clean_filename)
                if clean_img.dtype == np.uint16:
                    clean_save = (clean_img.astype(np.float64) / 65535.0 * 255).astype(np.uint8)
                else:
                    clean_save = clean_img
                cv2.imwrite(clean_save_path, clean_save)

                # 保存噪音图像
                noisy_filename = f"noisy_{idx:04d}.png"
                noisy_save_path = os.path.join(self.output_dir, 'noisy_patches', noisy_filename)
                if noisy_img.dtype == np.uint16:
                    noisy_save = (noisy_img.astype(np.float64) / 65535.0 * 255).astype(np.uint8)
                else:
                    noisy_save = noisy_img
                cv2.imwrite(noisy_save_path, noisy_save)

                saved_patches += 1

            # 保存数据集元数据
            metadata = {
                'total_pairs': total_pairs,
                'saved_patches': saved_patches,
                'patch_size': self.patch_size,
                'noise_method': self.noise_method,
                'image_size': noisy_img.shape[:2] if noisy_img is not None else None
            }

            meta_path = os.path.join(self.output_dir, 'dataset_metadata.json')
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            self.progress.emit(100, "噪音提取完成")
            self.finished.emit(True, f"成功保存 {saved_patches} 组样本到 {self.output_dir}")

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.finished.emit(False, f"噪音提取失败：{str(e)}\n\n{error_details}")

    def _extract_noise_std(self, image, block_size=8):
        """使用局部标准差法提取噪音。"""
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_float = gray.astype(np.float32)

        h, w = gray.shape
        noise_map = np.zeros_like(gray_float)

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray_float[y:y+block_size, x:x+block_size]
                noise_map[y:y+block_size, x:x+block_size] = np.std(block)

        return noise_map

    def _normalize_for_save(self, image):
        """归一化图像到 uint8 用于保存。"""
        img_min = image.min()
        img_max = image.max()

        if img_max - img_min < 1e-8:
            return np.zeros_like(image, dtype=np.uint8)

        normalized = (image - img_min) / (img_max - img_min) * 255
        return normalized.astype(np.uint8)


class PreprocessPage(QWidget):
    """图片预处理页面 - 噪音提取和数据集构建。"""

    def __init__(self):
        super().__init__()
        self.image_pairs = []  # [(noisy_path, clean_path), ...]
        self.current_file_index = 0
        self.is_processing = False
        self.init_ui()

    def init_ui(self):
        """初始化用户界面。"""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        # 标题
        title = QLabel("图片预处理 - 噪音提取与数据集构建")
        title.setStyleSheet("""
            font-size: 18px;
            font-weight: 600;
            color: #1e293b;
            padding: 8px 12px;
            border-left: 4px solid #0ea5e9;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #f0f9ff, stop:1 transparent);
            border-radius: 8px;
        """)
        layout.addWidget(title)

        # 主内容区域 - 左右分栏
        main_layout = QHBoxLayout()
        main_layout.setSpacing(16)

        # 左侧 - 控制面板
        left_panel = self._create_control_panel()
        main_layout.addWidget(left_panel, 1)

        # 右侧 - 图像显示
        right_panel = self._create_display_panel()
        main_layout.addWidget(right_panel, 2)

        layout.addLayout(main_layout)

    def _create_control_panel(self):
        """创建左侧控制面板。"""
        panel = QFrame()
        panel.setObjectName("controlPanel")
        panel.setStyleSheet("""
            QFrame#controlPanel {
                background-color: white;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
                padding: 16px;
            }
        """)
        layout = QVBoxLayout(panel)
        layout.setSpacing(14)
        layout.setContentsMargins(0, 0, 0, 0)

        # 1. 文件加载
        file_group = QGroupBox("1. 加载图像")
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(10)

        self.load_noisy_btn = QPushButton("加载带噪图像 (单张/批量)")
        self.load_noisy_btn.clicked.connect(self.load_noisy_images)
        self.load_noisy_btn.setMinimumHeight(42)
        file_layout.addWidget(self.load_noisy_btn)

        self.load_clean_btn = QPushButton("加载参考干净图像 (可选)")
        self.load_clean_btn.clicked.connect(self.load_clean_images)
        self.load_clean_btn.setMinimumHeight(42)
        file_layout.addWidget(self.load_clean_btn)

        self.file_count_label = QLabel("已加载：0 张图像")
        self.file_count_label.setStyleSheet("color: #64748b; font-size: 12px;")
        file_layout.addWidget(self.file_count_label)

        layout.addWidget(file_group)

        # 2. 噪音提取参数
        noise_group = QGroupBox("2. 噪音提取参数")
        noise_layout = QFormLayout(noise_group)
        noise_layout.setSpacing(10)

        self.noise_method_combo = QComboBox()
        self.noise_method_combo.addItems([
            "差分法 (noisy - clean)",
            "局部标准差法",
        ])
        self.noise_method_combo.setCurrentIndex(0)
        noise_layout.addRow("提取方法:", self.noise_method_combo)

        self.patch_size_spin = QSpinBox()
        self.patch_size_spin.setRange(16, 256)
        self.patch_size_spin.setValue(64)
        self.patch_size_spin.setMinimumHeight(32)
        noise_layout.addRow("块大小:", self.patch_size_spin)

        layout.addWidget(noise_group)

        # 3. 输出设置
        output_group = QGroupBox("3. 输出设置")
        output_layout = QVBoxLayout(output_group)
        output_layout.setSpacing(10)

        self.output_path_edit = QTextEdit()
        self.output_path_edit.setReadOnly(True)
        self.output_path_edit.setMaximumHeight(60)
        self.output_path_edit.setPlaceholderText("选择输出目录...")
        output_layout.addWidget(self.output_path_edit)

        self.browse_output_btn = QPushButton("选择输出目录")
        self.browse_output_btn.clicked.connect(self.browse_output)
        output_layout.addWidget(self.browse_output_btn)

        layout.addWidget(output_group)

        # 4. 执行按钮
        self.extract_btn = QPushButton("开始提取噪音")
        self.extract_btn.setObjectName("primaryBtn")
        self.extract_btn.setStyleSheet("""
            QPushButton#primaryBtn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3b82f6, stop:1 #8b5cf6);
                font-size: 15px;
                font-weight: 600;
                padding: 14px 24px;
                border-radius: 8px;
            }
            QPushButton#primaryBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2563eb, stop:1 #7c3aed);
            }
            QPushButton#primaryBtn:disabled {
                background: #cbd5e1;
            }
        """)
        self.extract_btn.clicked.connect(self.start_extraction)
        self.extract_btn.setEnabled(False)
        self.extract_btn.setMinimumHeight(48)
        layout.addWidget(self.extract_btn)

        # 进度条
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # 状态日志
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        self.log_text.setPlaceholderText("处理日志将显示在这里...")
        self.log_text.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        layout.addWidget(self.log_text)

        layout.addStretch()

        return panel

    def _create_display_panel(self):
        """创建右侧图像显示面板。"""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
                padding: 16px;
            }
        """)
        layout = QVBoxLayout(panel)
        layout.setSpacing(14)
        layout.setContentsMargins(0, 0, 0, 0)

        # 图像预览标签
        preview_group = QGroupBox("图像预览")
        preview_layout = QGridLayout(preview_group)
        preview_layout.setSpacing(12)

        # 带噪图像显示
        noisy_label = QLabel("带噪图像")
        noisy_label.setStyleSheet("font-weight: 600; color: #475569;")
        preview_layout.addWidget(noisy_label, 0, 0)

        self.noisy_image_label = QLabel("未加载图像")
        self.noisy_image_label.setObjectName("imageBox")
        self.noisy_image_label.setStyleSheet("""
            QLabel#imageBox {
                border: 2px dashed #cbd5e1;
                border-radius: 12px;
                background-color: #f8fafc;
                color: #94a3b8;
                font-style: italic;
                font-size: 14px;
                min-height: 200px;
                min-width: 200px;
            }
        """)
        self.noisy_image_label.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(self.noisy_image_label, 1, 0)

        # 干净图像显示
        clean_label = QLabel("参考干净图像")
        clean_label.setStyleSheet("font-weight: 600; color: #475569;")
        preview_layout.addWidget(clean_label, 0, 1)

        self.clean_image_label = QLabel("未加载")
        self.clean_image_label.setObjectName("imageBox")
        self.clean_image_label.setStyleSheet("""
            QLabel#imageBox {
                border: 2px dashed #cbd5e1;
                border-radius: 12px;
                background-color: #f8fafc;
                color: #94a3b8;
                font-style: italic;
                font-size: 14px;
                min-height: 200px;
                min-width: 200px;
            }
        """)
        self.clean_image_label.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(self.clean_image_label, 1, 1)

        # 噪音图显示
        noise_label = QLabel("提取的噪音图")
        noise_label.setStyleSheet("font-weight: 600; color: #475569;")
        preview_layout.addWidget(noise_label, 2, 0)

        self.noise_image_label = QLabel("等待提取...")
        self.noise_image_label.setObjectName("imageBox")
        self.noise_image_label.setStyleSheet("""
            QLabel#imageBox {
                border: 2px dashed #cbd5e1;
                border-radius: 12px;
                background-color: #f8fafc;
                color: #94a3b8;
                font-style: italic;
                font-size: 14px;
                min-height: 200px;
                min-width: 200px;
            }
        """)
        self.noise_image_label.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(self.noise_image_label, 3, 0)

        layout.addWidget(preview_group)

        # 图像信息
        info_group = QGroupBox("图像信息")
        info_layout = QVBoxLayout(info_group)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(100)
        self.info_text.setPlaceholderText("加载图像后显示信息...")
        info_layout.addWidget(self.info_text)

        layout.addWidget(info_group)

        return panel

    def load_noisy_images(self):
        """加载带噪图像。"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择带噪图像", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;所有文件 (*)"
        )

        if files:
            for f in files:
                self.image_pairs.append([f, None])

            self.file_count_label.setText(f"已加载：{len(self.image_pairs)} 张图像")
            self.extract_btn.setEnabled(len(self.image_pairs) > 0)

            # 显示第一张图像的预览
            if self.image_pairs:
                self._display_preview(files[0], self.noisy_image_label)
                self._update_info(files[0])

    def load_clean_images(self):
        """加载参考干净图像。"""
        if not self.image_pairs:
            QMessageBox.warning(self, "提示", "请先加载带噪图像")
            return

        files, _ = QFileDialog.getOpenFileNames(
            self, "选择参考干净图像", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;所有文件 (*)"
        )

        if files:
            # 按顺序匹配
            for i, f in enumerate(files):
                if i < len(self.image_pairs):
                    self.image_pairs[i][1] = f

            self.file_count_label.setText(f"已加载：{len(self.image_pairs)} 对图像")
            self.log_text.append(f"已加载 {len(files)} 张参考图像")

    def browse_output(self):
        """选择输出目录。"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择输出目录", ""
        )

        if dir_path:
            self.output_path_edit.setText(dir_path)

    def start_extraction(self):
        """开始噪音提取。"""
        if not self.image_pairs:
            QMessageBox.warning(self, "警告", "请先加载图像")
            return

        output_dir = self.output_path_edit.toPlainText().strip()
        if not output_dir:
            # 使用默认目录
            output_dir = os.path.join(os.path.dirname(__file__), 'dataset_output')
            self.output_path_edit.setText(output_dir)

        method_map = {
            0: 'difference',
            1: 'std'
        }
        noise_method = method_map.get(self.noise_method_combo.currentIndex(), 'difference')
        patch_size = self.patch_size_spin.value()

        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.extract_btn.setEnabled(False)
        self.log_text.append("开始噪音提取...")

        self.thread = NoiseExtractionThread(
            self.image_pairs,
            output_dir,
            noise_method,
            patch_size
        )
        self.thread.progress.connect(self.update_progress)
        self.thread.finished.connect(self.extraction_finished)
        self.thread.start()

    def update_progress(self, value, message):
        """更新进度。"""
        self.progress.setValue(value)
        self.log_text.append(message)

    def extraction_finished(self, success, message):
        """提取完成回调。"""
        self.progress.setVisible(False)
        self.extract_btn.setEnabled(True)

        if success:
            QMessageBox.information(self, "成功", message)
            self.log_text.append(f"✓ {message}")
        else:
            QMessageBox.critical(self, "错误", message)
            self.log_text.append(f"✗ {message}")

    def _display_preview(self, image_path, label):
        """显示图像预览。"""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                # 转换为 RGB 显示
                if len(img.shape) == 2:
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                elif img.shape[2] == 3:
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    rgb_img = img

                # 归一化到 uint8
                if rgb_img.dtype == np.uint16:
                    rgb_img = (rgb_img.astype(np.float64) / 65535.0 * 255).astype(np.uint8)

                h, w = rgb_img.shape[:2]
                rgb_bytes = rgb_img.tobytes()
                q_img = QImage(rgb_bytes, w, h, w * 3, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                scaled = pixmap.scaled(label.size() - 20, label.size().height() - 20,
                                       Qt.KeepAspectRatio, Qt.SmoothTransformation)
                label.setPixmap(scaled)
                label.setText("")
        except Exception as e:
            print(f"Display error: {e}")
            label.setText("预览失败")

    def _update_info(self, image_path):
        """更新图像信息显示。"""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                info = f"""文件名：{os.path.basename(image_path)}
形状：{img.shape}
数据类型：{img.dtype}
位深度：{img.dtype.itemsize * 8} 位
大小：{img.nbytes / (1024 * 1024):.2f} MB"""
                self.info_text.setText(info)
        except Exception as e:
            self.info_text.setText(f"读取图像信息失败：{e}")

    def apply_medical_style(self):
        """应用 Medical Minimalism 风格"""
        # 主标题
        for label in self.findChildren(QLabel):
            if label.text().startswith("图片预处理"):
                label.setStyleSheet("""
                    font-size: 20px;
                    font-weight: 600;
                    color: #1e293b;
                    padding: 12px;
                    border-left: 4px solid #0ea5e9;
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #f0f9ff, stop:1 transparent);
                    border-radius: 8px;
                """)

        # 控制面板
        for panel in self.findChildren(QFrame):
            if panel.objectName() == "controlPanel":
                panel.setStyleSheet("""
                    QFrame#controlPanel {
                        background-color: #ffffff;
                        border-radius: 12px;
                        border: 1px solid #e2e8f0;
                        padding: 16px;
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
                    font-size: 13px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 12px;
                    padding: 0 8px;
                    color: #0ea5e9;
                    font-weight: 600;
                }
            """)

        # 按钮样式
        for btn in self.findChildren(QPushButton):
            if "加载" in btn.text() or "选择" in btn.text():
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #f1f5f9;
                        color: #475569;
                        border: 1px solid #cbd5e1;
                        padding: 10px 20px;
                        border-radius: 8px;
                        font-weight: 500;
                        font-size: 13px;
                    }
                    QPushButton:hover {
                        background-color: #e2e8f0;
                        border-color: #0ea5e9;
                    }
                    QPushButton:pressed {
                        background-color: #cbd5e1;
                    }
                """)
            elif "开始" in btn.text() or btn.objectName() == "primaryBtn":
                btn.setStyleSheet("""
                    QPushButton#primaryBtn {
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0ea5e9, stop:1 #0284c7);
                        color: white;
                        border: none;
                        padding: 12px 24px;
                        border-radius: 8px;
                        font-weight: 600;
                        font-size: 14px;
                    }
                    QPushButton#primaryBtn:hover {
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #0369a1);
                    }
                    QPushButton#primaryBtn:disabled {
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
                    height: 28px;
                    font-weight: 600;
                    font-size: 13px;
                    background-color: #f8fafc;
                    color: #475569;
                }
                QProgressBar::chunk {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0ea5e9, stop:1 #0284c7);
                    border-radius: 7px;
                }
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
                        font-size: 14px;
                    }
                    QLabel#imageBox:hover {
                        border-color: #0ea5e9;
                        background-color: #f0f9ff;
                    }
                """)

        # 日志文本框
        for text in self.findChildren(QTextEdit):
            if text.isReadOnly():
                text.setStyleSheet("""
                    QTextEdit {
                        font-family: 'Consolas', 'Courier New', monospace;
                        font-size: 11px;
                        background-color: #f8fafc;
                        border: 1px solid #e2e8f0;
                        border-radius: 8px;
                        padding: 8px;
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
                    padding: 8px 12px;
                    border: 1px solid #cbd5e1;
                    border-radius: 8px;
                    background-color: white;
                    min-height: 36px;
                    color: #1e293b;
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
                    padding: 8px 12px;
                    border: 1px solid #cbd5e1;
                    border-radius: 8px;
                    background-color: white;
                    min-height: 32px;
                    color: #1e293b;
                }
                QSpinBox:hover {
                    border-color: #0ea5e9;
                }
                QSpinBox:focus {
                    border-color: #0ea5e9;
                }
                QSpinBox::up-button, QSpinBox::down-button {
                    border: none;
                    width: 24px;
                    background: #f1f5f9;
                    border-radius: 6px;
                    margin: 2px;
                }
                QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                    background: #e2e8f0;
                }
            """)
