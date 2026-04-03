"""
主窗口 - 多页面架构
包含：图片预处理、算法训练、降噪与超分辨率三个页面
Medical Minimalism 风格
"""

import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFrame,
                             QStackedWidget, QToolButton, QSpacerItem, QSizePolicy)
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor
from PyQt5.QtCore import Qt, QSize

# 导入页面模块
from preprocess_page import PreprocessPage
from training_page import TrainingPage
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

    def __init__(self, text, icon=None, parent=None):
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(True)
        self.setChecked(False)
        self.setAutoExclusive(True)
        self.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(64)
        self.setStyleSheet(self._get_style())

    def _get_style(self):
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
        self.setWindowTitle(f'X 射线图像降噪与超分辨率重构系统 v3.1.3')
        # 根据界面布局计算的最小尺寸（确保按钮不重叠）：
        # 宽度 = 侧边栏 (180px) + 内容左边距 (20px) + 左面板 (400px) + 间距 (20px) + 右面板 (500px) + 内容右边距 (20px) = 1140px
        # 高度 = 标题栏 (55px) + 标签页 (50px) + 内容区 (750px) + 状态栏 (50px) + 边距 (40px) = 945px
        # 增加缓冲空间防止压缩
        self.setMinimumSize(1400, 980)
        self.resize(1600, 1000)

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

        self.btn_preprocess = NavigationButton("图片预处理")
        self.btn_preprocess.clicked.connect(lambda: self._switch_page(0))
        layout.addWidget(self.btn_preprocess)
        self.nav_buttons.append(self.btn_preprocess)

        self.btn_training = NavigationButton("算法训练")
        self.btn_training.clicked.connect(lambda: self._switch_page(1))
        layout.addWidget(self.btn_training)
        self.nav_buttons.append(self.btn_training)

        self.btn_denoise = NavigationButton("降噪与超分")
        self.btn_denoise.clicked.connect(lambda: self._switch_page(2))
        layout.addWidget(self.btn_denoise)
        self.nav_buttons.append(self.btn_denoise)

        # 底部弹性空间，将按钮推向顶部
        layout.addStretch()

        # 版本信息
        version_label = QLabel("v3.1.2")
        version_label.setStyleSheet(f"""
            QLabel {{
                color: {DesignTokens.TEXT_MUTED};
                font-size: 11px;
                padding: 8px;
                text-align: center;
            }}
        """)
        version_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(version_label)

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

        self.training_page = TrainingPage()
        self.training_page.apply_medical_style()
        self.page_stack.addWidget(self.training_page)

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
        page_names = ['图片预处理', '算法训练', '降噪与超分辨率']
        self.status_bar.showMessage(f'当前页面：{page_names[index]}')


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 设置全局字体
    font = QFont("Microsoft YaHei", 9)
    font.setStyleHint(QFont.SansSerif)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
