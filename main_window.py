"""
主窗口 - 多页面架构
包含：图片预处理、算法训练、降噪与超分辨率三个页面
"""

import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFrame,
                             QStackedWidget, QToolButton, QSpacerItem, QSizePolicy)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt, QSize

# 导入页面模块
from preprocess_page import PreprocessPage
from training_page import TrainingPage
from denoise_app import DenoiseWidget


class NavigationButton(QToolButton):
    """侧边栏导航按钮。"""

    def __init__(self, text, icon=None, parent=None):
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(True)
        self.setChecked(False)
        self.setAutoExclusive(True)
        self.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.setIconSize(QSize(32, 32))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(80)
        self.setStyleSheet(self._get_style())

    def _get_style(self):
        return """
            QToolButton {
                background-color: transparent;
                border: none;
                border-radius: 8px;
                color: #94a3b8;
                font-size: 13px;
                font-weight: 500;
                padding: 10px;
                margin: 4px 6px;
            }
            QToolButton:hover {
                background-color: #334155;
                color: #ffffff;
            }
            QToolButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3b82f6, stop:1 #8b5cf6);
                color: white;
                font-weight: 600;
            }
        """


class MainWindow(QMainWindow):
    """主窗口 - 多页面应用架构。"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """初始化用户界面。"""
        self.setWindowTitle('X 射线图像降噪与超分辨率重构系统 v2.0')
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #f1f5f9;
            }
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
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: white;
                color: #64748b;
                border-top: 1px solid #e2e8f0;
                font-size: 12px;
            }
        """)
        self.status_bar.showMessage('就绪 - 请选择功能模块')

    def _create_sidebar(self):
        """创建左侧导航栏。"""
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setStyleSheet("""
            QFrame#sidebar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1e293b, stop:1 #0f172a);
                border-right: 1px solid #334155;
                min-width: 180px;
                max-width: 180px;
            }
        """)

        layout = QVBoxLayout(sidebar)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 20, 12, 12)

        # 导航按钮
        self.nav_buttons = []

        self.btn_preprocess = NavigationButton("📊 图片预处理")
        self.btn_preprocess.clicked.connect(lambda: self._switch_page(0))
        layout.addWidget(self.btn_preprocess)
        self.nav_buttons.append(self.btn_preprocess)

        self.btn_training = NavigationButton("🧠 算法训练")
        self.btn_training.clicked.connect(lambda: self._switch_page(1))
        layout.addWidget(self.btn_training)
        self.nav_buttons.append(self.btn_training)

        self.btn_denoise = NavigationButton("🔍 降噪与超分")
        self.btn_denoise.clicked.connect(lambda: self._switch_page(2))
        layout.addWidget(self.btn_denoise)
        self.nav_buttons.append(self.btn_denoise)

        layout.addStretch()

        # 版本信息
        version_label = QLabel("v2.0")
        version_label.setStyleSheet("""
            QLabel {
                color: #64748b;
                font-size: 11px;
                padding: 8px;
                text-align: center;
            }
        """)
        version_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(version_label)

        # 默认选中第一个
        self.btn_preprocess.setChecked(True)

        return sidebar

    def _create_content_area(self):
        """创建内容区域。"""
        content = QFrame()
        content.setStyleSheet("""
            QFrame {
                background-color: transparent;
            }
        """)

        layout = QVBoxLayout(content)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)

        # 页面堆栈
        self.page_stack = QStackedWidget()
        self.page_stack.setStyleSheet("""
            QStackedWidget {
                background-color: transparent;
            }
        """)

        # 添加页面
        self.preprocess_page = PreprocessPage()
        self.page_stack.addWidget(self.preprocess_page)

        self.training_page = TrainingPage()
        self.page_stack.addWidget(self.training_page)

        # 降噪页面
        self.denoise_widget = DenoiseWidget()
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
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
