"""
主窗口 - 多页面架构
包含：DenseNet（含噪音提取、数据集生成、算法训练）、Noise2Void、降噪与超分辨率三个页面
Medical Minimalism 风格
"""

import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFrame,
                             QStackedWidget, QToolButton, QSpacerItem, QSizePolicy,
                             QTextEdit, QDialog, QVBoxLayout)
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor
from PyQt5.QtCore import Qt, QSize

# 导入页面模块
from preprocess_page import PreprocessPage
from noise2void_page import Noise2VoidPage
from denoise_app import DenoiseWidget


# 医疗极简主义设计令牌
class DesignTokens:
    """UI 设计令牌 - Medical Minimalism 风格"""

    # 主色调 - Sky Blue
    PRIMARY_500 = "#0ea5e9"
    PRIMARY_600 = "#0284c7"
    PRIMARY_700 = "#0369a1"

    # 中性色
    BACKGROUND = "#f8fafc"
    SURFACE = "#ffffff"
    BORDER = "#e2e8f0"
    TEXT_PRIMARY = "#1e293b"
    TEXT_SECONDARY = "#64748b"
    TEXT_MUTED = "#94a3b8"

    # 功能色
    SUCCESS = "#10b981"
    WARNING = "#f59e0b"
    ERROR = "#ef4444"

    # 侧边栏深色主题
    SIDEBAR_DARK = "#1e293b"
    SIDEBAR_DARKER = "#0f172a"
    SIDEBAR_BORDER = "#334155"

    # 间距
    SPACING_4 = 4
    SPACING_8 = 8
    SPACING_12 = 12
    SPACING_16 = 16
    SPACING_24 = 24

    # 圆角
    RADIUS_SMALL = 4
    RADIUS_MEDIUM = 8
    RADIUS_LARGE = 12


class NavigationButton(QToolButton):
    """侧边栏导航按钮 - Medical Minimalism 风格"""

    def __init__(self, text, icon=None, checkable=True, parent=None):
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(checkable)
        self.setChecked(False)
        self.setAutoExclusive(checkable)
        self.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(64)
        self.setStyleSheet(self._get_style(checkable))

    def _get_style(self, checkable=True):
        if not checkable:
            # Action button style (like help guide)
            return f"""
                QToolButton {{
                    background-color: {DesignTokens.PRIMARY_500};
                    border: 1px solid {DesignTokens.PRIMARY_500};
                    border-radius: {DesignTokens.RADIUS_MEDIUM};
                    color: white;
                    font-size: 17px;
                    font-weight: 500;
                    padding: 16px 18px;
                    margin: 4px 8px;
                }}
                QToolButton:hover {{
                    background-color: {DesignTokens.PRIMARY_600};
                    border-color: {DesignTokens.PRIMARY_600};
                }}
            """
        else:
            # Navigation button style
            return f"""
                QToolButton {{
                    background-color: {DesignTokens.SIDEBAR_BORDER};
                    border: 1px solid {DesignTokens.SIDEBAR_BORDER};
                    border-radius: {DesignTokens.RADIUS_MEDIUM};
                    color: {DesignTokens.TEXT_MUTED};
                    font-size: 17px;
                    font-weight: 500;
                    padding: 16px 18px;
                    margin: 4px 8px;
                }}
                QToolButton:hover {{
                    background-color: {DesignTokens.PRIMARY_700};
                    border-color: {DesignTokens.PRIMARY_600};
                    color: {DesignTokens.SURFACE};
                }}
                QToolButton:checked {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {DesignTokens.PRIMARY_500}, stop:1 {DesignTokens.PRIMARY_600});
                    color: white;
                    font-weight: 600;
                    border-color: transparent;
                }}
            """


class MainWindow(QMainWindow):
    """主窗口 - 多页面应用架构 - Medical Minimalism 风格"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """初始化用户界面。"""
        self.setWindowTitle(f'X 射线图像降噪与超分辨率重构系统 v4.0.25')
        # 窗口最小尺寸增加 0.5 倍（接近全屏展示）
        # 1600 * 1.5 = 2400, 1100 * 1.5 = 1650
        self.setMinimumSize(2400, 1650)
        self.resize(2560, 1800)

        # 设置全局调色板
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(DesignTokens.BACKGROUND))
        palette.setColor(QPalette.WindowText, QColor(DesignTokens.TEXT_PRIMARY))
        palette.setColor(QPalette.Base, QColor(DesignTokens.SURFACE))
        palette.setColor(QPalette.AlternateBase, QColor(DesignTokens.BACKGROUND))
        palette.setColor(QPalette.Text, QColor(DesignTokens.TEXT_PRIMARY))
        palette.setColor(QPalette.Button, QColor(DesignTokens.SURFACE))
        palette.setColor(QPalette.ButtonText, QColor(DesignTokens.TEXT_PRIMARY))
        self.setPalette(palette)

        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {DesignTokens.BACKGROUND};
            }}
            QWidget {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
                font-size: 15px;
                color: {DesignTokens.TEXT_PRIMARY};
            }}
        """)

        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 左侧导航栏
        self.sidebar = self._create_sidebar()
        main_layout.addWidget(self.sidebar)

        # 右侧内容区域
        content_area = self._create_content_area()
        main_layout.addWidget(content_area, 1)

        # 状态栏
        self.status_bar = self.statusBar()
        self.status_bar.setStyleSheet(f"""
            QStatusBar {{
                background-color: {DesignTokens.SURFACE};
                color: {DesignTokens.TEXT_SECONDARY};
                border-top: 1px solid {DesignTokens.BORDER};
                font-size: 12px;
            }}
        """)
        self.status_bar.showMessage('就绪 - 请选择功能模块')

    def _create_sidebar(self):
        """创建左侧导航栏 - 深色医疗风格"""
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setStyleSheet(f"""
            QFrame#sidebar {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {DesignTokens.SIDEBAR_DARK}, stop:1 {DesignTokens.SIDEBAR_DARKER});
                border-right: 1px solid {DesignTokens.SIDEBAR_BORDER};
                min-width: 180px;
                max-width: 180px;
            }}
        """)

        layout = QVBoxLayout(sidebar)
        layout.setSpacing(DesignTokens.SPACING_8)
        layout.setContentsMargins(12, 24, 12, 12)

        # 导航按钮 - 从顶部向上分布
        self.nav_buttons = []

        self.btn_preprocess = NavigationButton("DenseNet")
        self.btn_preprocess.clicked.connect(lambda: self._switch_page(0))
        layout.addWidget(self.btn_preprocess)
        self.nav_buttons.append(self.btn_preprocess)

        self.btn_n2v = NavigationButton("Noise2Void")
        self.btn_n2v.clicked.connect(lambda: self._switch_page(1))
        layout.addWidget(self.btn_n2v)
        self.nav_buttons.append(self.btn_n2v)

        self.btn_denoise = NavigationButton("降噪与超分")
        self.btn_denoise.clicked.connect(lambda: self._switch_page(2))
        layout.addWidget(self.btn_denoise)
        self.nav_buttons.append(self.btn_denoise)

        # 底部弹性空间，将按钮推向顶部
        layout.addStretch()

        # 使用指南按钮
        self.help_btn = NavigationButton("使用指南", checkable=False)
        self.help_btn.clicked.connect(self.show_help_guide)
        layout.addWidget(self.help_btn)

        # 默认选中第一个
        self.btn_preprocess.setChecked(True)

        return sidebar

    def _create_content_area(self):
        """创建内容区域 - Medical Minimalism 风格"""
        content = QFrame()
        content.setStyleSheet("""
            QFrame {
                background-color: transparent;
            }
        """)

        layout = QVBoxLayout(content)
        layout.setSpacing(DesignTokens.SPACING_16)
        layout.setContentsMargins(DesignTokens.SPACING_16, DesignTokens.SPACING_16,
                                   DesignTokens.SPACING_16, DesignTokens.SPACING_16)

        # 页面堆栈
        self.page_stack = QStackedWidget()
        self.page_stack.setStyleSheet(f"""
            QStackedWidget {{
                background-color: transparent;
            }}
        """)

        # 添加页面
        self.preprocess_page = PreprocessPage()
        self.preprocess_page.apply_medical_style()
        self.page_stack.addWidget(self.preprocess_page)

        # Noise2Void 页面
        self.n2v_page = Noise2VoidPage()
        self.n2v_page.apply_medical_style()
        self.page_stack.addWidget(self.n2v_page)

        # 降噪页面
        self.denoise_widget = DenoiseWidget()
        self.denoise_widget.apply_medical_style()
        self.page_stack.addWidget(self.denoise_widget)

        layout.addWidget(self.page_stack)

        return content

    def _switch_page(self, index):
        """切换页面。"""
        self.page_stack.setCurrentIndex(index)

        # 更新导航按钮状态
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)

        # 更新状态栏
        page_names = ['DenseNet（噪音提取、数据集生成、算法训练）', 'Noise2Void 自监督训练', '降噪与超分辨率']
        self.status_bar.showMessage(f'当前页面：{page_names[index]}')

    def show_help_guide(self):
        """显示使用指南对话框 - 非模态，不影响其他操作。"""
        dialog = HelpGuideDialog(self)
        dialog.setWindowModality(Qt.NonModal)
        dialog.show()


class HelpGuideDialog(QDialog):
    """使用指南对话框 - 显示 Markdown 格式的使用说明。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("使用指南")
        self.setMinimumSize(800, 600)
        self.resize(900, 700)
        self.init_ui()
        self.apply_medical_style()

    def init_ui(self):
        """初始化界面。"""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Markdown 内容
        markdown_text = """
# X 射线图像降噪与超分辨率重构系统 - 使用指南

## 系统简介
本系统提供 X 射线图像的降噪处理和超分辨率重构功能，采用深度学习算法，有效提升图像质量。

## 功能模块

### 1. DenseNet
**用途**: 从单张 X 光图像提取噪声参数，生成训练数据集。

**操作步骤**:
1. 点击"噪音提取"标签页
2. 点击"加载"按钮加载 X 光图像（显示文件名、形状、数据类型、位深度）
3. 选择提取方法（局部标准差法/均匀区域法）
4. 点击"开始提取噪声"
5. 等待提取完成，查看提取的噪声参数

**生成文件**:
- `noise_params.json` - 噪声参数配置文件
- 包含 Poisson λ、AWGN σ、Gaussian Blur σ 等参数

### 2. 数据集生成
**用途**: 使用噪声参数生成合成噪声图像配对数据集。

**操作步骤**:
1. 点击"数据集生成"标签页
2. 导入干净图像数据集（包含 train/clean 和 train/noisy 目录）
3. 配置数据集参数：
   - 总样本数：生成图像对的数量
   - 训练集/测试集/验证集比例
4. 点击"开始生成数据集"
5. 等待生成完成

**输出目录结构**:
```
output/
├── train/
│   ├── clean/    # 干净图像
│   └── noisy/    # 噪声图像
├── test/
│   ├── clean/
│   └── noisy/
└── validation/
    ├── clean/
    └── noisy/
```

### 3. 算法训练
**用途**: 训练深度learning 降噪/超分辨率模型。

**操作步骤**:
1. 选择数据集目录（选择预处理生成的数据集）
2. 选择模型输出目录
3. 配置训练参数：
   - 训练轮数 (Epochs): 推荐 50-100
   - 批次大小 (Batch Size): 推荐 16-32
   - 学习率：推荐 0.001
   - 块大小 (Patch Size): 推荐 64
   - 模型类型：降噪模型/超分辨率模型
4. 点击"开始训练"
5. 实时监控训练进度和 Loss 曲线
6. 训练完成后，点击"添加"按钮将模型集成到对应算法

**训练完成**:
- 模型自动保存到输出目录
- 可点击"添加"按钮集成到降噪或超分算法
- 在"降噪与超分"页面可使用训练的模型

### 4. Noise2Void 自监督训练
**用途**: 通过单张噪声图像训练降噪模型，无需成对的噪声/干净图像。

**原理**: 使用"盲点"网络架构，训练时预测每个像素的值时不使用该像素本身，从而实现无需干净图像的自监督去噪训练。

**操作步骤**:
1. 点击"选择噪声图像"加载单张 X 光噪声图像
2. 配置训练参数：
   - 训练轮数 (Epochs): 推荐 30-100
   - 批次大小 (Batch Size): 推荐 8-32
   - 块大小 (Patch Size): 推荐 64-128
   - 学习率：推荐 0.001
3. 选择模型输出目录
4. 点击"开始训练"
5. 训练完成后，可选择集成到降噪算法列表

**优势**:
- 无需成对的噪声/干净图像
- 单张图像即可训练
- 适用于 X 射线等难以获取干净图像的领域

**输出文件**:
- `noise2void_model.pth` - 训练好的模型权重
- `n2v_config.json` - 训练配置和元数据
- `model_ready.marker` - 模型集成标记文件

### 5. 降噪与超分
**用途**: 对 X 射线图像进行降噪和超分辨率处理。

**操作步骤**:
1. 点击"加载"按钮加载图像
2. **降噪处理**:
   - 选择降噪算法（包括自定义训练模型）
   - 调整强度和其他参数
   - 点击"执行降噪"
   - 查看降噪结果和质量指标
3. **超分辨率重构**:
   - 选择算法
   - 选择放大倍数 (1.5x - 4.0x)
   - 点击"执行超分辨率"
   - 查看超分辨率结果
4. 保存处理结果

**可用算法**:
- 降噪：Hybrid、BM3D、Anisotropic Diffusion、Iterative Reconstruction、NLM、Bilateral、Wavelet、Gaussian、Neural Network、Trained Neural Denoise (自定义训练模型)
- 超分辨率：双三次插值、兰索斯插值、保边增强、神经网络 (训练模型)

**算法管理**:
- 点击每个模块的"⚙ 管理"按钮
- 可启用/禁用算法
- 可修改算法显示名称
- 可通过目录加载自定义训练模型

**自定义模型加载**:
1. 点击"⚙ 管理"按钮打开算法编辑器
2. 在"自定义模型加载"区域点击"浏览"
3. 选择包含模型文件的目录
4. 系统自动检测模型文件：
   - 降噪模型：`denoiser.onnx`, `best_denoiser.pth`, `model.pt`
   - 超分模型：`sr_model.onnx`, `best_sr_model.pth`, `model_sr.pt`
5. 点击"加载模型到算法列表"
6. 模型自动复制到 `integrated_model/{type}/{timestamp}/` 目录
7. 算法下拉框刷新显示新模型（带时间戳）

**模型目录结构**:
```
integrated_model/
├── denoise/
│   ├── 20260407_153000/      # 时间戳命名的子目录
│   │   ├── denoiser.onnx
│   │   ├── best_denoiser.pth
│   │   └── model_ready.marker
│   └── 20260408_102000/
│       └── ...
└── super_resolution/
    ├── 20260407_160000/
    │   ├── sr_model.onnx
    │   └── model_ready.marker
    └── ...
```

## 快捷操作流程

### 完整训练流程
1. DenseNet → 噪音提取 → 生成 noise_params.json
2. DenseNet → 数据集生成 → 生成训练数据集
3. 算法训练 → 选择数据集 → 开始训练 → 添加模型
4. 降噪与超分 → 加载图像 → 降噪 → 超分 → 保存

### Noise2Void 自监督训练流程
1. Noise2Void → 加载单张噪声图像
2. 配置训练参数 → 开始训练
3. 训练完成 → 集成到降噪算法
4. 降噪与超分 → 选择 Noise2Void 模型 → 执行降噪

### 直接使用预训练模型
1. 降噪与超分 → 加载图像
2. 选择"Neural Network (Trained)"算法（如有）
3. 执行降噪/超分处理

## 技术指标
- 支持图像格式：PNG、JPG、BMP、TIFF、DICOM
- 支持位深度：8 位、16 位灰度图像
- 评估指标：PSNR、SSIM、MSE
"""

        # 文本显示区域
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setMarkdown(markdown_text)
        layout.addWidget(self.text_display)

        # 关闭按钮
        close_btn = QPushButton("关闭")
        close_btn.setObjectName("closeBtn")
        close_btn.setMinimumHeight(44)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def apply_medical_style(self):
        """应用 Medical Minimalism 风格。"""
        self.setStyleSheet("""
            QDialog {
                background-color: #f8fafc;
            }
            QTextEdit {
                font-family: 'Microsoft YaHei', sans-serif;
                font-size: 14px;
                line-height: 1.6;
                background-color: white;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 16px;
                color: #1e293b;
            }
            QTextEdit h1 {
                font-size: 24px;
                font-weight: 600;
                color: #0ea5e9;
                margin-bottom: 16px;
            }
            QTextEdit h2 {
                font-size: 20px;
                font-weight: 600;
                color: #0284c7;
                margin-top: 20px;
                margin-bottom: 12px;
            }
            QTextEdit h3 {
                font-size: 16px;
                font-weight: 600;
                color: #475569;
                margin-top: 16px;
                margin-bottom: 8px;
            }
            QTextEdit p {
                margin-bottom: 12px;
            }
            QTextEdit ul, QTextEdit ol {
                margin-bottom: 12px;
                padding-left: 24px;
            }
            QTextEdit li {
                margin-bottom: 4px;
            }
            QTextEdit code {
                background-color: #f1f5f9;
                padding: 2px 6px;
                border-radius: 4px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 13px;
            }
            QTextEdit pre {
                background-color: #1e293b;
                color: #e2e8f0;
                padding: 12px;
                border-radius: 8px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 13px;
                overflow-x: auto;
            }
            QPushButton#closeBtn {
                background-color: #0ea5e9;
                color: white;
                font-size: 15px;
                font-weight: 600;
                padding: 12px 24px;
                border-radius: 8px;
            }
            QPushButton#closeBtn:hover {
                background-color: #0284c7;
            }
        """)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 设置全局字体
    font = QFont("Microsoft YaHei", 9)
    font.setStyleHint(QFont.SansSerif)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
