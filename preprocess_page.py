"""
图片预处理页面 - 用于噪音提取和数据集构建
两步流程：1. 噪音提取 2. 生成训练/测试/验证集
"""

import sys
import os
import json
import glob
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QProgressBar, QComboBox,
                             QSpinBox, QDoubleSpinBox, QFormLayout, QGroupBox,
                             QFrame, QMessageBox, QTextEdit, QGridLayout,
                             QScrollArea, QSizePolicy, QTabWidget, QSplitter,
                             QApplication, QLineEdit)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import cv2


class NoiseExtractionThread(QThread):
    """后台线程用于噪音提取 - 从单张 X 光图像提取噪声特征。"""
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(int, str)

    def __init__(self, image_path, output_dir, extraction_method='noise_profile', patch_size=64):
        super().__init__()
        self.image_path = image_path
        self.output_dir = output_dir
        self.extraction_method = extraction_method
        self.patch_size = patch_size

    def run(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)

            self.progress.emit(10, "正在加载图像...")
            # 加载 X 光图像（使用 imdecode 避免中文路径问题）
            img_data = np.fromfile(self.image_path, dtype=np.uint8)
            if img_data is None or len(img_data) == 0:
                self.finished.emit(False, "无法读取图像文件")
                return

            noisy_img = cv2.imdecode(img_data, cv2.IMREAD_UNCHANGED)
            if noisy_img is None:
                self.finished.emit(False, "无法解码图像")
                return

            self.progress.emit(30, "正在分析噪声特征...")

            # 根据论文方法：噪声 = Poisson + AWGN + 高斯模糊
            # 估计噪声参数
            noise_params = self._estimate_noise_params(noisy_img)

            self.progress.emit(60, "正在提取噪声图...")

            # 提取噪声图（使用局部统计法）
            if self.extraction_method == 'local_std':
                noise_map = self._extract_noise_local_std(noisy_img)
            elif self.extraction_method == 'homogeneous_region':
                noise_map = self._extract_noise_homogeneous(noisy_img)
            else:
                noise_map = self._extract_noise_local_std(noisy_img)

            self.progress.emit(80, "正在保存结果...")

            # 保存噪声参数
            params_path = os.path.join(self.output_dir, 'noise_params.json')
            with open(params_path, 'w') as f:
                json.dump(noise_params, f, indent=2)

            # 保存噪声图
            noise_path = os.path.join(self.output_dir, 'noise_map.png')
            noise_normalized = self._normalize_for_save(noise_map)
            cv2.imwrite(noise_path, noise_normalized)

            # 保存原始图像副本
            src_path = os.path.join(self.output_dir, 'source_image.png')
            cv2.imwrite(src_path, noisy_img)

            self.progress.emit(100, "噪声提取完成")
            self.finished.emit(True, f"噪声提取完成 - Poisson λ={noise_params['poisson_lambda']:.1f}, AWGN σ={noise_params['awgn_sigma']:.2f}")

        except Exception as e:
            import traceback
            self.finished.emit(False, f"提取失败：{e}\n{traceback.format_exc()}")

    def _estimate_noise_params(self, image):
        """估计噪声参数（Poisson + AWGN）- 基于论文 Rev. Sci. Instrum. 95, 063508 (2024) 方法。

        使用分区域盒（box）方法估计噪声参数：
        1. 将图像分割成多个区域盒
        2. 在每个盒内归一化到最大值
        3. 使用泊松分布的方差/均值关系估计λ值
        4. AWGN 设为 5%（论文标准值）
        """
        # 转换到 float [0, 1]
        if image.dtype == np.uint16:
            img_float = image.astype(np.float64) / 65535.0
        elif image.dtype == np.uint8:
            img_float = image.astype(np.float64) / 255.0
        else:
            img_float = image.astype(np.float64)
            # 归一化到 [0, 1]
            img_float = (img_float - img_float.min()) / (img_float.max() - img_float.min())

        # 如果是彩色图像，转换为灰度
        if len(img_float.shape) == 3:
            img_gray = cv2.cvtColor(img_float, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_float

        # 论文方法：分区域盒估计
        # 将图像分割成 4x4 的区域盒
        h, w = img_gray.shape
        box_h, box_w = h // 4, w // 4

        lambda_estimates = []
        noise_std_estimates = []
        box_coords = []  # 保存有效区域盒的坐标

        for i in range(4):
            for j in range(4):
                y1, y2 = i * box_h, (i + 1) * box_h
                x1, x2 = j * box_w, (j + 1) * box_w

                # 跳过过小的区域
                if y2 > h or x2 > w:
                    continue

                box = img_gray[y1:y2, x1:x2]

                # 归一化到盒内最大值（论文方法）
                box_max = box.max()
                if box_max < 0.1:  # 跳过太暗的区域
                    continue

                # 记录有效区域盒坐标
                box_coords.append({'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)})

                box_normalized = box / box_max

                # 计算盒内统计量
                box_mean = np.mean(box_normalized)
                box_var = np.var(box_normalized)

                # 泊松噪声：variance ≈ mean / λ
                # 所以 λ ≈ mean / variance
                if box_var > 0 and box_mean > 0:
                    box_lambda = box_mean / box_var
                    # 限制λ在合理范围内（论文中λ≈5-20）
                    if 2 < box_lambda < 200:
                        lambda_estimates.append(box_lambda)

                # 计算局部方差用于估计 AWGN
                kernel_size = 5
                local_mean = cv2.GaussianBlur(box, (kernel_size, kernel_size), 0)
                local_var = cv2.GaussianBlur((box - local_mean) ** 2, (kernel_size, kernel_size), 0)

                # 取最小方差区域（平坦区域）用于估计 AWGN
                flat_threshold = np.percentile(local_var, 20)
                flat_mask = local_var < flat_threshold
                if np.sum(flat_mask) > 10:
                    noise_std = np.sqrt(np.mean(local_var[flat_mask]))
                    noise_std_estimates.append(noise_std)

        # 使用中位数作为最终估计（对异常值更鲁棒）
        if lambda_estimates:
            poisson_lambda = np.median(lambda_estimates)
        else:
            # 回退到全局估计
            global_mean = np.mean(img_gray)
            global_var = np.var(img_gray)
            if global_var > 0:
                poisson_lambda = global_mean / global_var
            else:
                poisson_lambda = 10

        if noise_std_estimates:
            noise_std = np.median(noise_std_estimates)
        else:
            noise_std = np.sqrt(np.mean(cv2.GaussianBlur((img_gray - cv2.GaussianBlur(img_gray, (5, 5), 0)) ** 2, (7, 7), 0)))

        # 根据论文：AWGN 约为 5%，Poisson λ 在 5-20 范围
        # 限制估计值在合理范围内
        poisson_lambda = max(5, min(100, poisson_lambda))

        # AWGN sigma = 5% （论文标准值）
        awgn_sigma = 0.05

        # 根据噪声标准差调整 AWGN 比例
        awgn_ratio = noise_std * 0.5  # 约 50% 的噪声标准差来自 AWGN
        awgn_sigma = max(0.02, min(0.1, awgn_ratio))

        return {
            'poisson_lambda': float(poisson_lambda),
            'awgn_sigma': float(awgn_sigma),
            'gaussian_blur_sigma': 1.0,  # 论文中固定值
            'estimated_noise_std': float(noise_std),
            'image_dtype': str(image.dtype),
            'image_shape': list(image.shape),
            'lambda_estimates_count': len(lambda_estimates),
            'lambda_min': float(np.min(lambda_estimates)) if lambda_estimates else 0,
            'lambda_max': float(np.max(lambda_estimates)) if lambda_estimates else 0,
            'box_coords': box_coords,  # 区域盒坐标列表
        }

    def _extract_noise_local_std(self, image):
        """使用局部标准差法提取噪声图。"""
        if image.dtype == np.uint16:
            img_float = image.astype(np.float64) / 65535.0
        else:
            img_float = image.astype(np.float64) / 255.0

        # 计算局部标准差
        kernel_size = 7
        mean = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), 0)
        squared_diff = (img_float - mean) ** 2
        local_std = np.sqrt(cv2.GaussianBlur(squared_diff, (kernel_size, kernel_size), 0))

        return local_std

    def _extract_noise_homogeneous(self, image):
        """从均匀区域提取噪声。"""
        if image.dtype == np.uint16:
            img_float = image.astype(np.float64) / 65535.0
        else:
            img_float = image.astype(np.float64) / 255.0

        # 找到强度变化最小的区域
        kernel_size = 15
        local_std = cv2.GaussianBlur((img_float - cv2.GaussianBlur(img_float, (5, 5), 0)) ** 2,
                                      (kernel_size, kernel_size), 0)
        local_std = np.sqrt(local_std)

        return local_std

    def _normalize_for_save(self, noise_map):
        """归一化噪声图用于保存。"""
        noise_min = np.min(noise_map)
        noise_max = np.max(noise_map)
        if noise_max - noise_min > 0:
            noise_normalized = (noise_map - noise_min) / (noise_max - noise_min)
        else:
            noise_normalized = noise_map - noise_min

        # 转换到 uint8
        return (noise_normalized * 255).astype(np.uint8)


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
            # 创建带时间戳的输出目录
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(self.output_dir, f"dataset_{timestamp}")

            # 创建输出目录结构
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

            # 使用步骤 1 的源图像作为基础（或使用自然图像数据集）
            base_images = self._load_base_images()

            if len(base_images) == 0:
                self.finished.emit(False, "未找到基础图像")
                return

            self.progress.emit(20, f"已加载 {len(base_images)} 张基础图像，开始生成数据集...")

            # 计算各集合数量
            train_count = int(self.total_patches * self.train_split / 100)
            test_count = int(self.total_patches * self.test_split / 100)
            val_count = self.total_patches - train_count - test_count

            saved_count = {'train': 0, 'test': 0, 'val': 0}
            total_saved = 0

            for i in range(self.total_patches):
                # 确定属于哪个集合
                if i < train_count:
                    split = 'train'
                    idx = saved_count['train']
                elif i < train_count + test_count:
                    split = 'test'
                    idx = saved_count['test']
                else:
                    split = 'val'
                    idx = saved_count['val']

                # 随机选择基础图像
                base_img = base_images[np.random.randint(0, len(base_images))]

                # 随机裁剪 patch
                h, w = base_img.shape[:2]
                if h > self.patch_size and w > self.patch_size:
                    y = np.random.randint(0, h - self.patch_size)
                    x = np.random.randint(0, w - self.patch_size)
                    clean_patch = base_img[y:y+self.patch_size, x:x+self.patch_size]
                else:
                    clean_patch = cv2.resize(base_img, (self.patch_size, self.patch_size))

                # 添加噪声（根据论文方法：Poisson + AWGN + Gaussian Blur）
                noisy_patch = self._add_noise(clean_patch)

                # 保存
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

            # 保存元数据
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

            # 保存生成的样本对路径（用于预览）
            preview_pairs = []
            for split in ['train', 'test', 'val']:
                for i in range(min(5, saved_count[split])):  # 每个集合保存 5 对
                    clean_path = os.path.join(self.output_dir, split, 'clean', f'{split}_{i:05d}.png')
                    noisy_path = os.path.join(self.output_dir, split, 'noisy', f'{split}_{i:05d}.png')
                    if os.path.exists(clean_path) and os.path.exists(noisy_path):
                        preview_pairs.append({'clean': clean_path, 'noisy': noisy_path, 'split': split, 'index': i})

            pairs_path = os.path.join(self.output_dir, 'preview_pairs.json')
            with open(pairs_path, 'w', encoding='utf-8') as f:
                json.dump(preview_pairs, f, indent=2, ensure_ascii=False)

            self.progress.emit(100, f"数据集生成完成 - 共 {total_saved} 个样本")
            self.finished.emit(True, f"成功生成 {total_saved} 个样本到 {self.output_dir}")

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.finished.emit(False, f"数据集生成失败：{str(e)}\n\n{error_details}")

    def _load_base_images(self):
        """加载基础图像（从步骤 1 的源图像或现有数据集）。"""
        images = []

        # 优先使用步骤 1 的源图像（单文件）
        if self.base_image_dir and os.path.isfile(self.base_image_dir):
            img = cv2.imread(self.base_image_dir, cv2.IMREAD_UNCHANGED)
            if img is not None:
                images.append(img)

        # 如果是目录，递归遍历所有子文件夹加载图像
        if self.base_image_dir and os.path.isdir(self.base_image_dir):
            image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
            print(f"Loading images from: {self.base_image_dir}")
            try:
                for root, dirs, files in os.walk(self.base_image_dir):
                    for file in files:
                        ext = os.path.splitext(file)[1].lower()
                        if ext in image_extensions:
                            img_path = os.path.join(root, file)
                            print(f"  Loading: {img_path}")
                            # 使用 imdecode 读取图片，避免中文路径问题
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
        # 获取噪声参数
        poisson_lambda = self.noise_params.get('poisson_lambda', 0)
        awgn_sigma = self.noise_params.get('awgn_sigma', 0)
        blur_sigma = self.noise_params.get('gaussian_blur_sigma', 0)

        # 如果所有噪声参数都为 0，直接返回原始图像（干净和噪声图像保持一致）
        if poisson_lambda == 0 and awgn_sigma == 0 and blur_sigma == 0:
            return clean_patch.copy()

        # 转换到 float [0, 1]
        if clean_patch.dtype == np.uint16:
            img_float = clean_patch.astype(np.float64) / 65535.0
        else:
            img_float = clean_patch.astype(np.float64) / 255.0

        # 1. Poisson 噪声
        if poisson_lambda > 0:
            # 缩放以匹配 Poisson 分布的典型范围
            scaled = img_float * poisson_lambda
            noisy = np.random.poisson(scaled).astype(np.float64) / poisson_lambda
        else:
            noisy = img_float.copy()

        # 2. AWGN
        if awgn_sigma > 0:
            noisy += np.random.normal(0, awgn_sigma, noisy.shape)

        # 3. Gaussian Blur
        if blur_sigma > 0:
            kernel_size = int(6 * blur_sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            noisy = cv2.GaussianBlur(noisy, (kernel_size, kernel_size), blur_sigma)

        # 裁剪到 [0, 1]
        noisy = np.clip(noisy, 0, 1)

        # 转换回原数据类型
        if clean_patch.dtype == np.uint16:
            return (noisy * 65535).astype(np.uint16)
        else:
            return (noisy * 255).astype(np.uint8)


class PreprocessPage(QWidget):
    """图片预处理页面 - 噪音提取和数据集构建（两步流程）。"""

    def __init__(self):
        super().__init__()
        self.noise_params = None  # Step 1 output: {'poisson_lambda', 'awgn_sigma', 'gaussian_blur_sigma'}
        self.source_image_path = None
        self.extraction_output_dir = None
        self.current_file_index = 0
        self.is_processing = False
        self.init_ui()

    def init_ui(self):
        """初始化用户界面 - 两步流程带标签页。"""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        # 标题（与其他页面保持一致，不加版本号）
        title = QLabel("图片预处理 - 噪音提取与数据集构建")
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

        # 标签页容器
        tabs_container = QFrame()
        tabs_container.setStyleSheet("""
            QFrame {
                background-color: transparent;
            }
        """)
        tabs_layout = QVBoxLayout(tabs_container)
        tabs_layout.setContentsMargins(0, 10, 0, 0)
        tabs_layout.setSpacing(10)

        # 使用标签页分隔两步
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f1f5f9;
                color: #64748b;
                padding: 16px 32px;
                margin-right: 6px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: 600;
                font-size: 16px;
                min-width: 150px;
            }
            QTabBar::tab:selected {
                background-color: white;
                color: #0ea5e9;
                border-bottom: 2px solid white;
            }
            QTabBar::tab:!selected {
                margin-top: 4px;
            }
            QTabBar::tab:hover:!selected {
                background-color: #e2e8f0;
            }
        """)
        self.tabs.setMinimumHeight(750)
        tabs_layout.addWidget(self.tabs)
        layout.addWidget(tabs_container)

        # Step 1: 噪音提取
        step1_widget = self._create_step1_widget()
        self.tabs.addTab(step1_widget, "1. 噪音提取")

        # Step 2: 数据集生成
        step2_widget = self._create_step2_widget()
        self.tabs.addTab(step2_widget, "2. 数据集生成")

        # 状态栏
        self.status_label = QLabel("就绪 - 请先完成步骤 1：噪音提取")
        self.status_label.setStyleSheet("""
            color: #64748b;
            font-size: 15px;
            font-weight: 500;
            padding: 12px 14px;
            background-color: #f8fafc;
            border-radius: 8px;
        """)
        layout.addWidget(self.status_label)

    def _create_step1_widget(self):
        """创建步骤 1：噪音提取界面。"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # 设置最小尺寸防止按钮重叠
        # 宽度：左侧面板 (400px) + 右侧面板 (500px) + 间距边距 = 约 1000px
        # 高度：内容高度约 750px + 边距 = 约 800px
        widget.setMinimumSize(1100, 800)

        # 左侧控制面板
        left_panel = QFrame()
        left_panel.setObjectName("step1ControlPanel")
        left_panel.setStyleSheet("""
            QFrame#step1ControlPanel {
                background-color: #f8fafc;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
                padding: 20px;
            }
        """)
        left_panel.setMinimumWidth(400)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(16)

        # 说明
        info_label = QLabel("从单张 X 光图像提取噪声特征，生成噪声参数配置文件（noise_params.json）")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #475569; font-size: 15px; font-weight: 500; padding: 8px;")
        left_layout.addWidget(info_label)

        # 1. 加载图像
        load_group = QGroupBox("1. 加载 X 光图像")
        load_group.setMinimumHeight(130)
        load_layout = QVBoxLayout(load_group)
        self.load_btn = QPushButton("选择图像文件")
        self.load_btn.clicked.connect(self.load_source_image)
        self.load_btn.setMinimumHeight(48)
        self.load_btn.setStyleSheet("""
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
        """)
        load_layout.addWidget(self.load_btn)

        self.source_info_label = QLabel("未加载图像")
        self.source_info_label.setStyleSheet("color: #94a3b8; font-size: 15px; font-weight: 500;")
        load_layout.addWidget(self.source_info_label)
        left_layout.addWidget(load_group)

        # 2. 提取参数
        param_group = QGroupBox("2. 提取参数")
        param_group.setMinimumHeight(150)
        param_layout = QFormLayout(param_group)
        param_layout.setSpacing(12)

        self.extraction_method_combo = QComboBox()
        self.extraction_method_combo.addItems(["局部标准差法", "均匀区域法"])
        self.extraction_method_combo.setMinimumHeight(40)
        self.extraction_method_combo.setStyleSheet("""
            QComboBox {
                font-size: 15px;
                padding: 10px 14px;
            }
        """)
        param_layout.addRow("提取方法:", self.extraction_method_combo)

        self.patch_size_spin = QSpinBox()
        self.patch_size_spin.setRange(16, 256)
        self.patch_size_spin.setValue(64)
        self.patch_size_spin.setMinimumHeight(40)
        self.patch_size_spin.setStyleSheet("""
            QSpinBox {
                font-size: 15px;
                padding: 10px 14px;
            }
        """)
        param_layout.addRow("块大小:", self.patch_size_spin)

        left_layout.addWidget(param_group)

        # 3. 执行按钮
        self.extract_btn = QPushButton("开始提取噪声")
        self.extract_btn.setObjectName("step1PrimaryBtn")
        self.extract_btn.setStyleSheet("""
            QPushButton#step1PrimaryBtn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0ea5e9, stop:1 #0284c7);
                color: white;
                border: none;
                padding: 16px 32px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 16px;
            }
            QPushButton#step1PrimaryBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #0369a1);
            }
            QPushButton#step1PrimaryBtn:disabled {
                background: #cbd5e1;
                color: #94a3b8;
            }
        """)
        self.extract_btn.clicked.connect(self.start_noise_extraction)
        self.extract_btn.setEnabled(False)
        self.extract_btn.setMinimumHeight(52)
        left_layout.addWidget(self.extract_btn)

        # 进度条
        self.step1_progress = QProgressBar()
        self.step1_progress.setVisible(False)
        self.step1_progress.setStyleSheet("""
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
        left_layout.addWidget(self.step1_progress)

        left_layout.addStretch()

        # 右侧图像显示
        right_panel = self._create_image_preview()
        right_panel.setMinimumWidth(500)
        layout.addWidget(left_panel, 1)
        layout.addWidget(right_panel, 2)

        return widget

    def _create_step2_widget(self):
        """创建步骤 2：数据集生成界面。"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # 设置最小尺寸防止按钮重叠
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

        # 说明
        info_label = QLabel("使用步骤 1 提取的噪声参数，生成合成噪声图像与干净图像配对数据集")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #475569; font-size: 15px; font-weight: 500; padding: 8px;")
        left_layout.addWidget(info_label)

        # 噪声参数显示（可编辑）
        params_group = QGroupBox("已提取的噪声参数（可编辑）")
        params_group.setMinimumHeight(160)
        params_layout = QFormLayout(params_group)
        params_layout.setSpacing(12)

        # Poisson λ
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

        # AWGN σ
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

        # Gaussian Blur σ
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

        # 数据集路径选择
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

        # 图片数量统计
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
        self.total_patches_spin.setStyleSheet("""
            QSpinBox {
                font-size: 15px;
                padding: 10px 14px;
            }
        """)
        dataset_layout.addRow("总样本数:", self.total_patches_spin)

        self.train_split_spin = QSpinBox()
        self.train_split_spin.setRange(50, 90)
        self.train_split_spin.setValue(80)
        self.train_split_spin.setMinimumHeight(40)
        self.train_split_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.train_split_spin.setStyleSheet("""
            QSpinBox {
                font-size: 15px;
                padding: 10px 14px;
            }
        """)
        dataset_layout.addRow("训练集 (%):", self.train_split_spin)

        self.test_split_spin = QSpinBox()
        self.test_split_spin.setRange(5, 25)
        self.test_split_spin.setValue(10)
        self.test_split_spin.setMinimumHeight(40)
        self.test_split_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.test_split_spin.setStyleSheet("""
            QSpinBox {
                font-size: 15px;
                padding: 10px 14px;
            }
        """)
        dataset_layout.addRow("测试集 (%):", self.test_split_spin)

        self.val_split_spin = QSpinBox()
        self.val_split_spin.setRange(5, 25)
        self.val_split_spin.setValue(10)
        self.val_split_spin.setMinimumHeight(40)
        self.val_split_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.val_split_spin.setStyleSheet("""
            QSpinBox {
                font-size: 15px;
                padding: 10px 14px;
            }
        """)
        dataset_layout.addRow("验证集 (%):", self.val_split_spin)

        left_layout.addWidget(dataset_group)

        # 生成数据集导出
        output_group = QGroupBox("生成数据集导出")
        output_group.setMinimumHeight(180)
        output_layout = QVBoxLayout(output_group)

        # 使用水平布局放置目录选择和浏览按钮
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

        # 数据集生成状态显示
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

        # 进度条
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

        # 右侧说明
        right_panel = self._create_dataset_info_panel()
        right_panel.setMinimumWidth(500)
        layout.addWidget(left_panel, 1)
        layout.addWidget(right_panel, 2)

        return widget

    def _create_image_preview(self):
        """创建图像预览面板（步骤 1 右侧）。"""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
                padding: 12px;
            }
        """)
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)

        # 图像预览
        preview_group = QGroupBox("源图像预览")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setSpacing(10)
        preview_layout.setContentsMargins(10, 10, 10, 10)
        # 根据 X 光图片尺寸（最大约 2333x1608）设定最小高度
        preview_group.setMinimumHeight(550)

        self.source_image_label = QLabel("未加载图像")
        self.source_image_label.setObjectName("imageBox")
        self.source_image_label.setStyleSheet("""
            QLabel#imageBox {
                border: 2px dashed #cbd5e1;
                border-radius: 12px;
                background-color: #f8fafc;
                color: #94a3b8;
                font-style: italic;
                font-size: 16px;
            }
        """)
        self.source_image_label.setAlignment(Qt.AlignCenter)
        self.source_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 根据图片尺寸设定最小预览区域
        self.source_image_label.setMinimumSize(500, 500)
        preview_layout.addWidget(self.source_image_label)

        layout.addWidget(preview_group)

        # 噪声参数显示
        params_group = QGroupBox("提取的噪声参数")
        params_group.setMinimumHeight(160)
        params_layout = QVBoxLayout(params_group)
        params_layout.setSpacing(8)
        params_layout.setContentsMargins(8, 8, 8, 8)
        self.extracted_params_text = QTextEdit()
        self.extracted_params_text.setReadOnly(True)
        self.extracted_params_text.setMaximumHeight(140)
        self.extracted_params_text.setMinimumHeight(120)
        self.extracted_params_text.setPlaceholderText("提取噪声参数后显示...")
        self.extracted_params_text.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', monospace;
                font-size: 14px;
                background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 8px;
                color: #475569;
            }
        """)
        params_layout.addWidget(self.extracted_params_text)
        layout.addWidget(params_group)

        return panel

    def _create_dataset_info_panel(self):
        """创建数据集信息面板（步骤 2 右侧）。"""
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

        # 说明
        info_group = QGroupBox("数据集生成说明")
        info_group.setMinimumHeight(200)
        info_layout = QVBoxLayout(info_group)
        info_text = QLabel(
            "根据步骤 1 提取的噪声参数，生成合成噪声图像与干净图像的配对数据集。\n\n"
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

        # 生成的数据集预览
        preview_group = QGroupBox("数据集预览")
        preview_group.setMinimumHeight(400)
        preview_layout = QVBoxLayout(preview_group)

        # 预览图像区域
        preview_images_layout = QHBoxLayout()
        preview_images_layout.setSpacing(10)

        # 干净图像
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

        # 噪声图像
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

        # 预览信息
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

        # 刷新按钮
        self.refresh_preview_btn = QPushButton("🔄 刷新预览")
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

        # 日志
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

    # ========== 步骤 1：噪音提取方法 ==========
    def load_source_image(self):
        """加载单张 X 光源图像用于噪音提取。"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择 X 光图像", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;所有文件 (*)"
        )

        if file_path:
            self.source_image_path = file_path
            self.source_info_label.setText(f"已加载：{os.path.basename(file_path)}")
            self.source_info_label.setStyleSheet("color: #10b981; font-size: 15px; font-weight: 500;")
            self.extract_btn.setEnabled(True)
            self._display_source_preview(file_path)

    def start_noise_extraction(self):
        """开始噪音提取。"""
        if not self.source_image_path:
            QMessageBox.warning(self, "警告", "请先加载图像")
            return

        # 使用默认输出目录（步骤 1 和步骤 2 共用）
        self.extraction_output_dir = os.path.join(os.path.dirname(__file__), 'noise_profile_output')

        extraction_method = 'local_std' if self.extraction_method_combo.currentIndex() == 0 else 'homogeneous_region'
        patch_size = self.patch_size_spin.value()

        self.step1_progress.setVisible(True)
        self.step1_progress.setValue(0)
        self.extract_btn.setEnabled(False)
        self.status_label.setText("正在提取噪声特征...")

        self.thread = NoiseExtractionThread(
            self.source_image_path,
            self.extraction_output_dir,
            extraction_method,
            patch_size
        )
        self.thread.progress.connect(self.update_step1_progress)
        self.thread.finished.connect(self.noise_extraction_finished)
        self.thread.start()

    def update_step1_progress(self, value, message):
        """更新步骤 1 进度。"""
        self.step1_progress.setValue(value)
        self.status_label.setText(message)

    def noise_extraction_finished(self, success, message):
        """噪音提取完成回调。"""
        self.step1_progress.setVisible(False)
        self.extract_btn.setEnabled(True)

        if success:
            QMessageBox.information(self, "成功", message)
            self.status_label.setText("步骤 1 完成 - 可以进行数据集生成")
            # 加载噪声参数
            self._load_noise_params()
            # 在预览图像上显示区域盒
            self._display_noise_boxes()
            # 启用步骤 2 按钮
            self.generate_btn.setEnabled(True)
            # 不再自动跳转，让用户手动切换
        else:
            QMessageBox.critical(self, "错误", message)
            self.status_label.setText("步骤 1 失败 - 请重试")

    def _load_noise_params(self):
        """加载提取的噪声参数到可编辑控件。"""
        if not self.extraction_output_dir:
            self.extraction_output_dir = os.path.join(os.path.dirname(__file__), 'noise_profile_output')

        params_path = os.path.join(self.extraction_output_dir, 'noise_params.json')

        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                self.noise_params = json.load(f)

            # 填充可编辑的噪声参数控件（步骤 2 页面）
            self.poisson_lambda_spin.setValue(self.noise_params['poisson_lambda'])
            self.awgn_sigma_spin.setValue(self.noise_params['awgn_sigma'])
            self.blur_sigma_spin.setValue(self.noise_params.get('gaussian_blur_sigma', 1.0))

            # 显示简要信息（步骤 1 页面右侧）
            self.extracted_params_text.setPlainText(
                f"Poisson λ = {self.noise_params['poisson_lambda']:.1f}\n"
                f"AWGN σ = {self.noise_params['awgn_sigma']:.4f}\n"
                f"Gaussian Blur σ = 1.0\n"
                f"(基于 {self.noise_params.get('lambda_estimates_count', 'N/A')} 个区域盒估计)"
            )

    def get_current_noise_params(self):
        """从可编辑控件获取当前的噪声参数。"""
        # 如果 noise_params 为 None，使用默认值
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
        """浏览选择输出目录。"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择数据集输出目录", ""
        )
        if dir_path:
            self.dataset_output_edit.setText(dir_path)
            # 检查是否已导入数据集
            dataset_path = self.dataset_path_edit.text().strip()
            if dataset_path:
                self.generate_btn.setEnabled(True)
                self._set_generation_status("idle", "已选择输出目录，可以生成数据集")

    def browse_dataset_dir(self):
        """浏览选择原始数据集目录。"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择原始数据集目录", ""
        )
        if dir_path:
            self.dataset_path_edit.setText(dir_path)
            # 统计图片数量
            self.count_images_in_dir(dir_path)

    def count_images_in_dir(self, root_dir):
        """递归统计文件夹下的图片数量（包括嵌套子文件夹）。"""
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

            # 更新显示
            if image_count > 0:
                self.image_count_label.setText(
                    f"已找到 {image_count:,} 张图片（{dir_count:,} 个文件夹）"
                )
                self.image_count_label.setStyleSheet("""
                    QLabel {
                        color: #10b981;
                        font-size: 15px;
                        font-weight: 600;
                        padding: 8px;
                    }
                """)

                # 自动填充总样本数
                self.total_patches_spin.setValue(min(image_count, 10000))

                # 检查输出目录是否已选择
                output_dir = self.dataset_output_edit.text().strip()
                if output_dir:
                    # 启用生成数据集按钮
                    self.generate_btn.setEnabled(True)
                    self._set_generation_status("idle", f"已准备生成 {image_count:,} 个样本")
                else:
                    self.generate_btn.setEnabled(False)
                    self._set_generation_status("idle", "请选择输出目录")
            else:
                self.image_count_label.setText("未找到图片文件")
                self.image_count_label.setStyleSheet("""
                    QLabel {
                        color: #ef4444;
                        font-size: 15px;
                        padding: 8px;
                    }
                """)
                self.generate_btn.setEnabled(False)

        except Exception as e:
            self.image_count_label.setText(f"统计失败：{e}")
            self.image_count_label.setStyleSheet("""
                QLabel {
                    color: #ef4444;
                    font-size: 15px;
                    padding: 8px;
                }
            """)
            self.generate_btn.setEnabled(False)

    def _display_noise_boxes(self):
        """在源图像预览上叠加显示噪声计算选取的区域盒。"""
        if not self.source_image_path or not self.noise_params:
            return

        try:
            # 加载源图像
            img_data = np.fromfile(self.source_image_path, dtype=np.uint8)
            if img_data is None or len(img_data) == 0:
                return

            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img is None:
                return

            # 转换为 RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 获取区域盒坐标
            box_coords = self.noise_params.get('box_coords', [])

            if not box_coords:
                return

            # 获取图像尺寸
            img_h, img_w = rgb_img.shape[:2]

            # 在图像上绘制区域盒（绿色边框，2 像素粗）
            for box in box_coords:
                x1, y1 = box['x1'], box['y1']
                x2, y2 = box['x2'], box['y2']
                # 绘制矩形框
                cv2.rectangle(rgb_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 转换为 QImage 显示
            h, w = rgb_img.shape[:2]
            rgb_bytes = rgb_img.tobytes()
            q_img = QImage(rgb_bytes, w, h, w * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            # 缩放以适配标签
            label_rect = self.source_image_label.frameGeometry()
            avail_width = label_rect.width() - 60
            avail_height = label_rect.height() - 60

            if avail_width > 0 and avail_height > 0:
                scaled = pixmap.scaled(avail_width, avail_height,
                                       Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.source_image_label.setPixmap(scaled)
            else:
                self.source_image_label.setPixmap(pixmap)

            self.source_image_label.setText("")
            print(f"Displayed {len(box_coords)} noise estimation boxes")
        except Exception as e:
            print(f"Error displaying noise boxes: {e}")
            import traceback
            traceback.print_exc()

    def _display_source_preview(self, image_path):
        """显示源图像预览。"""
        try:
            # 使用 numpy + cv2 解码，避免中文路径问题
            img_data = np.fromfile(image_path, dtype=np.uint8)
            if img_data is None or len(img_data) == 0:
                self.source_image_label.setText("无法读取图像文件")
                return

            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img is None:
                self.source_image_label.setText("无法解码图像")
                return

            # img 已经是 BGR 格式，转换为 RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 归一化到 uint8（如果不是）
            if rgb_img.dtype == np.uint16:
                rgb_img = (rgb_img.astype(np.float64) / 65535.0 * 255).astype(np.uint8)
            elif rgb_img.dtype != np.uint8:
                rgb_img = ((rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min()) * 255).astype(np.uint8)

            h, w = rgb_img.shape[:2]
            rgb_bytes = rgb_img.tobytes()
            q_img = QImage(rgb_bytes, w, h, w * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            # 强制更新标签尺寸后再缩放
            self.source_image_label.repaint()
            QApplication.processEvents()

            # 使用 frameGeometry 获取实际可用空间
            label_rect = self.source_image_label.frameGeometry()
            avail_width = label_rect.width() - 60
            avail_height = label_rect.height() - 60

            print(f"Label frame: {label_rect.width()}x{label_rect.height()}")
            print(f"Image: {w}x{h}, Avail: {avail_width}x{avail_height}")

            if avail_width > 0 and avail_height > 0:
                scaled = pixmap.scaled(avail_width, avail_height,
                                       Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.source_image_label.setPixmap(scaled)
            else:
                self.source_image_label.setPixmap(pixmap)

            self.source_image_label.setText("")
            print("Preview set successfully")
        except Exception as e:
            print(f"Display error: {e}")
            import traceback
            traceback.print_exc()
            self.source_image_label.setText(f"预览失败：{str(e)}")

    # ========== 步骤 2：数据集生成方法 ==========
    def start_dataset_generation(self):
        """开始数据集生成。"""
        print("=== 开始数据集生成 ===")

        # 检查是否已导入原始数据集
        base_image_dir = self.dataset_path_edit.text().strip()
        print(f"原始数据集路径：{base_image_dir}")

        if not base_image_dir:
            self._set_generation_status("error", "请先导入原始数据集")
            QMessageBox.warning(self, "警告", "请先导入原始数据集（点击'浏览'选择包含图像的文件夹）")
            return

        # 检查总样本数
        total_patches = self.total_patches_spin.value()
        print(f"总样本数：{total_patches}")

        if total_patches <= 0:
            self._set_generation_status("error", "总样本数必须大于 0")
            QMessageBox.warning(self, "警告", "总样本数必须大于 0，请先导入原始数据集或手动设置样本数")
            return

        # 检查噪声参数是否已设置（允许为 0，但需要用户确认）
        poisson_lambda = self.poisson_lambda_spin.value()
        awgn_sigma = self.awgn_sigma_spin.value()
        print(f"噪声参数：Poisson λ={poisson_lambda}, AWGN σ={awgn_sigma}")

        if poisson_lambda == 0 and awgn_sigma == 0:
            reply = QMessageBox.question(
                self, "噪声参数为零",
                "当前噪声参数均为 0，生成的数据集将没有噪声。是否继续？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        # 从可编辑控件获取当前噪声参数
        current_params = self.get_current_noise_params()
        print(f"完整噪声参数：{current_params}")

        # 获取输出目录
        output_dir = self.dataset_output_edit.text().strip()
        if not output_dir:
            output_dir = os.path.join(os.path.dirname(__file__), 'generated_dataset')
            self.dataset_output_edit.setText(output_dir)
        print(f"输出目录：{output_dir}")

        patch_size = self.patch_size_spin.value()
        train_split = self.train_split_spin.value()
        test_split = self.test_split_spin.value()
        val_split = self.val_split_spin.value()

        # 验证分割比例
        if train_split + test_split + val_split != 100:
            self._set_generation_status("error", "数据集比例之和不等于 100%")
            QMessageBox.warning(self, "警告", "训练集 + 测试集 + 验证集比例必须等于 100%")
            return

        # 更新状态为生成中
        self._set_generation_status("processing", "正在生成数据集...")

        self.step2_progress.setVisible(True)
        self.step2_progress.setValue(0)
        self.generate_btn.setEnabled(False)
        self.dataset_log_text.append("开始生成数据集...")

        print("创建 DatasetGenerationThread...")
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
        print("启动线程...")
        self.thread.start()

    def update_step2_progress(self, value, message):
        """更新步骤 2 进度。"""
        self.step2_progress.setValue(value)
        self.status_label.setText(message)
        self.dataset_log_text.append(message)
        # 更新状态显示
        self._set_generation_status("processing", message)

    def _set_generation_status(self, status, message):
        """设置数据集生成状态。"""
        if status == "idle":
            self.generation_status_label.setText("状态：未开始")
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
            self.generation_detail_label.setText(message)
        elif status == "processing":
            self.generation_status_label.setText("状态：正在生成数据集...")
            self.generation_status_label.setStyleSheet("""
                QLabel {
                    color: #0284c7;
                    font-size: 15px;
                    font-weight: 600;
                    padding: 10px;
                    background-color: #e0f2fe;
                    border-radius: 6px;
                    border: 1px solid #7dd3fc;
                }
            """)
            self.generation_detail_label.setText(message)
        elif status == "success":
            self.generation_status_label.setText("状态：生成完成 ✓")
            self.generation_status_label.setStyleSheet("""
                QLabel {
                    color: #047857;
                    font-size: 15px;
                    font-weight: 600;
                    padding: 10px;
                    background-color: #d1fae5;
                    border-radius: 6px;
                    border: 1px solid #6ee7b7;
                }
            """)
            self.generation_detail_label.setText(message)
        elif status == "error":
            self.generation_status_label.setText("状态：生成失败 ✗")
            self.generation_status_label.setStyleSheet("""
                QLabel {
                    color: #dc2626;
                    font-size: 15px;
                    font-weight: 600;
                    padding: 10px;
                    background-color: #fee2e2;
                    border-radius: 6px;
                    border: 1px solid #fca5a5;
                }
            """)
            self.generation_detail_label.setText(message)

    def dataset_generation_finished(self, success, message):
        """数据集生成完成回调。"""
        self.step2_progress.setVisible(False)
        self.generate_btn.setEnabled(True)

        if success:
            self._set_generation_status("success", message)
            QMessageBox.information(self, "成功", message)
            self.dataset_log_text.append(f"✓ {message}")
            # 启用预览刷新按钮
            self.refresh_preview_btn.setEnabled(True)
            # 加载预览
            self.load_preview_pairs()
        else:
            self._set_generation_status("error", message)
            QMessageBox.critical(self, "错误", message)
            self.dataset_log_text.append(f"✗ {message}")

    def load_preview_pairs(self):
        """从原始数据集随机加载一张图片，显示干净版本和加噪版本的对比。"""
        # 获取原始数据集路径
        dataset_path = self.dataset_path_edit.text().strip()
        if not dataset_path:
            self.preview_info_label.setText("请先导入原始数据集")
            return

        # 获取当前噪声参数
        poisson_lambda = self.poisson_lambda_spin.value()
        awgn_sigma = self.awgn_sigma_spin.value()
        blur_sigma = self.blur_sigma_spin.value()

        # 遍历目录获取所有图片路径
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

        # 随机选择一张图片
        import random
        selected_image_path = random.choice(image_files)

        try:
            # 读取原始图像（干净版本）
            img_data = np.fromfile(selected_image_path, dtype=np.uint8)
            if img_data is None or len(img_data) == 0:
                self.preview_info_label.setText("无法读取图片")
                return

            clean_img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if clean_img is None:
                self.preview_info_label.setText("无法解码图片")
                return

            # 转换为 RGB
            clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)

            # 生成加噪版本
            noisy_img = self._add_noise_for_preview(clean_img, poisson_lambda, awgn_sigma, blur_sigma)

            # 显示对比
            self._display_preview_image(clean_img, self.preview_clean_label)
            self._display_preview_image(noisy_img, self.preview_noisy_label)

            # 更新信息
            filename = os.path.basename(selected_image_path)
            self.preview_info_label.setText(
                f"原始图片：{filename}\n"
                f"Poisson λ={poisson_lambda:.1f}, AWGN σ={awgn_sigma:.4f}, Blur σ={blur_sigma:.1f}"
            )

        except Exception as e:
            self.preview_info_label.setText(f"加载预览失败：{e}")

    def _add_noise_for_preview(self, clean_img, poisson_lambda, awgn_sigma, blur_sigma):
        """为预览生成加噪图像（使用当前控件的噪声参数）"""
        # 如果所有噪声参数都为 0，直接返回原始图像
        if poisson_lambda == 0 and awgn_sigma == 0 and blur_sigma == 0:
            return clean_img.copy()

        # 转换到 float [0, 1]
        img_float = clean_img.astype(np.float64) / 255.0

        # 1. Poisson 噪声
        if poisson_lambda > 0:
            scaled = img_float * poisson_lambda
            noisy = np.random.poisson(scaled).astype(np.float64) / poisson_lambda
        else:
            noisy = img_float.copy()

        # 2. AWGN
        if awgn_sigma > 0:
            noisy += np.random.normal(0, awgn_sigma, noisy.shape)

        # 3. Gaussian Blur
        if blur_sigma > 0:
            kernel_size = int(6 * blur_sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            noisy = cv2.GaussianBlur(noisy, (kernel_size, kernel_size), blur_sigma)

        # 裁剪到 [0, 1]
        noisy = np.clip(noisy, 0, 1)

        # 转换回 uint8
        return (noisy * 255).astype(np.uint8)

    def _display_preview_image(self, img, label):
        """显示预览图像（接受 numpy 数组）。"""
        try:
            if img is None:
                label.setText("无法加载")
                label.setPixmap(QPixmap())
                return

            h, w = img.shape[:2]
            rgb_bytes = img.tobytes()
            q_img = QImage(rgb_bytes, w, h, w * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            # 缩放以适配标签
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

        except Exception as e:
            label.setText("显示失败")

    def apply_medical_style(self):
        """应用 Medical Minimalism 风格"""
        # 主标题
        for label in self.findChildren(QLabel):
            if label.text().startswith("图片预处理"):
                label.setStyleSheet("""
                    font-size: 24px;
                    font-weight: 600;
                    color: #1e293b;
                    padding: 14px 18px;
                    border-left: 4px solid #0ea5e9;
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #f0f9ff, stop:1 transparent);
                    border-radius: 8px;
                """)

        # 控制面板
        for panel in self.findChildren(QFrame):
            if panel.objectName() in ["controlPanel", "step1ControlPanel", "step2ControlPanel"]:
                panel.setStyleSheet("""
                    QFrame#controlPanel, QFrame#step1ControlPanel, QFrame#step2ControlPanel {
                        background-color: #f8fafc;
                        border-radius: 12px;
                        border: 1px solid #e2e8f0;
                        padding: 20px;
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

        # 按钮样式
        for btn in self.findChildren(QPushButton):
            if "加载" in btn.text() or "选择" in btn.text():
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
            elif "开始" in btn.text() or btn.objectName() in ["primaryBtn", "step1PrimaryBtn", "step2PrimaryBtn"]:
                btn.setStyleSheet("""
                    QPushButton#primaryBtn, QPushButton#step1PrimaryBtn, QPushButton#step2PrimaryBtn {
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0ea5e9, stop:1 #0284c7);
                        color: white;
                        border: none;
                        padding: 16px 32px;
                        border-radius: 8px;
                        font-weight: 600;
                        font-size: 16px;
                    }
                    QPushButton#primaryBtn:hover, QPushButton#step1PrimaryBtn:hover, QPushButton#step2PrimaryBtn:hover {
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #0369a1);
                    }
                    QPushButton#primaryBtn:disabled, QPushButton#step1PrimaryBtn:disabled, QPushButton#step2PrimaryBtn:disabled {
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
