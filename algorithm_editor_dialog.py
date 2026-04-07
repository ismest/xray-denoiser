"""
算法编辑对话框
支持降噪和超分辨率算法的启用/禁用、重命名管理
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QTableWidget, QTableWidgetItem, QLabel, QMessageBox,
                             QWidget, QHeaderView, QCheckBox, QLineEdit,
                             QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from algorithm_config import (get_algorithm_config, update_algorithm,
                              add_algorithm, delete_algorithm, reset_to_defaults,
                              DEFAULT_DENOISE_ALGORITHMS, DEFAULT_SR_ALGORITHMS)


class AlgorithmEditorDialog(QDialog):
    """算法编辑对话框 - 支持降噪和超分辨率。"""

    def __init__(self, parent=None, algo_type="denoise"):
        """
        初始化对话框。

        Args:
            parent: 父窗口
            algo_type: 算法类型 ("denoise" 或 "super_resolution")
        """
        super().__init__(parent)
        self.algo_type = algo_type
        self.tab_name = "降噪算法" if algo_type == "denoise" else "超分辨率算法"
        self.setWindowTitle(f"{self.tab_name} - 管理")
        self.setMinimumSize(600, 450)
        self.resize(700, 550)

        self.init_ui()
        self.apply_medical_style()

    def init_ui(self):
        """初始化界面。"""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # 说明标签
        info_label = QLabel('提示：勾选启用算法，双击名称可修改。修改后点击"保存"生效。')
        info_label.setStyleSheet("color: #64748b; font-size: 14px; padding: 8px; background-color: #f8fafc; border-radius: 6px;")
        layout.addWidget(info_label)

        # 算法表格
        self.table = QTableWidget()
        self.table.setObjectName("algoTable")
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["启用", "算法名称"])

        # 表格样式
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setColumnWidth(0, 60)

        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.DoubleClicked)  # 双击可编辑
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)

        layout.addWidget(self.table)

        # 加载算法数据
        self._load_algorithms()

        # 底部按钮
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.reset_btn = QPushButton("恢复默认")
        self.reset_btn.setObjectName("resetBtn")
        self.reset_btn.clicked.connect(self._reset_defaults)
        button_layout.addWidget(self.reset_btn)

        button_layout.addStretch()

        self.save_btn = QPushButton("保存")
        self.save_btn.setObjectName("saveBtn")
        self.save_btn.clicked.connect(self._save_changes)
        button_layout.addWidget(self.save_btn)

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setObjectName("cancelBtn")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

    def _load_algorithms(self):
        """加载算法列表到表格。"""
        config = get_algorithm_config(self.algo_type)

        # 获取默认名称用于对比
        if self.algo_type == "denoise":
            defaults = {a["key"]: a["name"] for a in DEFAULT_DENOISE_ALGORITHMS}
        else:
            defaults = {a["key"]: a["name"] for a in DEFAULT_SR_ALGORITHMS}

        self.table.setRowCount(len(config))

        for row, algo in enumerate(config):
            # 启用复选框
            check_item = QTableWidgetItem()
            check_item.setFlags(check_item.flags() & ~Qt.ItemIsEditable)
            check_item.setCheckState(Qt.Checked if algo.get("enabled", True) else Qt.Unchecked)
            self.table.setItem(row, 0, check_item)

            # 算法名称（可编辑）
            name_item = QTableWidgetItem(algo["name"])
            name_item.setFlags(name_item.flags() | Qt.ItemIsEditable)
            # 标记是否修改过
            is_default = algo["name"] == defaults.get(algo["key"], "")
            if not is_default:
                name_item.setBackground(Qt.lightGray)
            self.table.setItem(row, 1, name_item)

    def _reset_defaults(self):
        """恢复默认配置。"""
        reply = QMessageBox.question(
            self, "确认恢复默认",
            f"确定要恢复{self.tab_name}的默认配置吗？\n\n所有自定义名称和启用状态将被重置。",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            if reset_to_defaults(self.algo_type):
                self._load_algorithms()
                QMessageBox.information(self, "成功", "已恢复默认配置")
            else:
                QMessageBox.warning(self, "失败", "恢复默认配置失败")

    def _save_changes(self):
        """保存修改。"""
        config = get_algorithm_config(self.algo_type)

        success = True
        for row in range(self.table.rowCount()):
            check_item = self.table.item(row, 0)
            name_item = self.table.item(row, 1)

            if row < len(config):
                key = config[row]["key"]
                enabled = check_item.checkState() == Qt.Checked
                name = name_item.text()

                if not update_algorithm(self.algo_type, key, name=name, enabled=enabled):
                    success = False

        if success:
            QMessageBox.information(self, "成功", "算法配置已保存")
            self.accept()
        else:
            QMessageBox.warning(self, "失败", "保存配置时出错")

    def apply_medical_style(self):
        """应用 Medical Minimalism 风格。"""
        self.setStyleSheet("""
            QDialog {
                background-color: #f8fafc;
            }
            QTableWidget {
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                background-color: white;
                gridline-color: #e2e8f0;
                font-size: 14px;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QTableWidget::item:selected {
                background-color: #e0f2fe;
                color: #0369a1;
            }
            QTableWidget::item:focus {
                border: 2px solid #0ea5e9;
            }
            QHeaderView::section {
                background-color: #f1f5f9;
                color: #475569;
                font-weight: 600;
                padding: 10px;
                border: none;
                font-size: 14px;
            }
            QPushButton#saveBtn {
                background-color: #0ea5e9;
                color: white;
                font-size: 15px;
                font-weight: 600;
                padding: 10px 24px;
                border-radius: 8px;
                min-width: 80px;
            }
            QPushButton#saveBtn:hover {
                background-color: #0284c7;
            }
            QPushButton#cancelBtn {
                background-color: #f1f5f9;
                color: #475569;
                font-size: 15px;
                font-weight: 500;
                padding: 10px 24px;
                border-radius: 8px;
                border: 1px solid #cbd5e1;
                min-width: 80px;
            }
            QPushButton#cancelBtn:hover {
                background-color: #e2e8f0;
            }
            QPushButton#resetBtn {
                background-color: #fef3c7;
                color: #92400e;
                font-size: 15px;
                font-weight: 500;
                padding: 10px 24px;
                border-radius: 8px;
                border: 1px solid #fcd34d;
                min-width: 80px;
            }
            QPushButton#resetBtn:hover {
                background-color: #fde68a;
            }
        """)


class DenoiseAlgorithmEditor(QDialog):
    """降噪算法编辑对话框。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.editor = AlgorithmEditorDialog(parent, "denoise")
        self.setWindowTitle(self.editor.windowTitle())
        self.setMinimumSize(self.editor.minimumSize())
        self.resize(self.editor.size())

        layout = QVBoxLayout(self)
        layout.addWidget(self.editor)

        # 连接信号
        self.editor.accepted.connect(self.accept)
        self.editor.rejected.connect(self.reject)


class SRAlgorithmEditor(QDialog):
    """超分辨率算法编辑对话框。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.editor = AlgorithmEditorDialog(parent, "super_resolution")
        self.setWindowTitle(self.editor.windowTitle())
        self.setMinimumSize(self.editor.minimumSize())
        self.resize(self.editor.size())

        layout = QVBoxLayout(self)
        layout.addWidget(self.editor)

        # 连接信号
        self.editor.accepted.connect(self.accept)
        self.editor.rejected.connect(self.reject)
