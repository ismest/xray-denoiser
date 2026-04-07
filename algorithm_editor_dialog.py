"""
算法编辑对话框
支持降噪和超分辨率算法的启用/禁用、重命名管理
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QTableWidget, QTableWidgetItem, QLabel, QMessageBox,
                             QTabWidget, QWidget, QHeaderView, QCheckBox, QLineEdit,
                             QFormLayout, QGroupBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from algorithm_config import (get_algorithm_config, update_algorithm,
                              add_algorithm, delete_algorithm, reset_to_defaults,
                              DEFAULT_DENOISE_ALGORITHMS, DEFAULT_SR_ALGORITHMS)


class AlgorithmEditorDialog(QDialog):
    """算法编辑对话框。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("算法管理")
        self.setMinimumSize(700, 500)
        self.resize(800, 600)

        self.current_changes = {}  # 暂存当前修改

        self.init_ui()
        self.apply_medical_style()

    def init_ui(self):
        """初始化界面。"""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # 说明标签
        info_label = QLabel("提示：勾选启用算法，双击名称可修改显示名称。修改后点击保存生效。")
        info_label.setStyleSheet("color: #64748b; font-size: 14px; padding: 8px; background-color: #f8fafc; border-radius: 6px;")
        layout.addWidget(info_label)

        # 标签页
        self.tabs = QTabWidget()
        self.tabs.setObjectName("algorithmTabs")

        # 降噪算法标签页
        self.denoise_widget = self._create_algorithm_tab("denoise")
        self.tabs.addTab(self.denoise_widget, "降噪算法")

        # 超分辨率算法标签页
        self.sr_widget = self._create_algorithm_tab("super_resolution")
        self.tabs.addTab(self.sr_widget, "超分辨率算法")

        layout.addWidget(self.tabs)

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

    def _create_algorithm_tab(self, algo_type: str) -> QWidget:
        """创建算法标签页。"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # 算法表格
        self.table = QTableWidget()
        self.table.setObjectName(f"{algo_type}Table")
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["启用", "算法名称", "操作"])

        # 表格样式
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        self.table.setColumnWidth(0, 60)
        self.table.setColumnWidth(2, 80)

        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)

        layout.addWidget(self.table)

        # 加载算法数据
        self._load_algorithms(algo_type)

        # 存储类型引用
        self.table.algo_type = algo_type

        return widget

    def _load_algorithms(self, algo_type: str):
        """加载算法列表到表格。"""
        config = get_algorithm_config(algo_type)

        # 获取默认名称用于对比
        if algo_type == "denoise":
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

            # 算法名称
            name_item = QTableWidgetItem(algo["name"])
            name_item.setFlags(name_item.flags() | Qt.ItemIsEditable)
            # 标记是否修改过
            is_default = algo["name"] == defaults.get(algo["key"], "")
            if not is_default:
                name_item.setBackground(Qt.lightGray)
            self.table.setItem(row, 1, name_item)

            # 删除按钮
            delete_btn = QPushButton("删除")
            delete_btn.setObjectName("deleteRowBtn")
            delete_btn.setStyleSheet("""
                QPushButton#deleteRowBtn {
                    background-color: #fee2e2;
                    color: #dc2626;
                    border: 1px solid #fca5a5;
                    border-radius: 4px;
                    padding: 4px 12px;
                    font-size: 13px;
                }
                QPushButton#deleteRowBtn:hover {
                    background-color: #fecaca;
                }
            """)
            delete_btn.clicked.connect(lambda checked, r=row: self._delete_row(algo_type, r))
            self.table.setCellWidget(row, 2, delete_btn)

    def _delete_row(self, algo_type: str, row: int):
        """删除一行算法。"""
        key = self.table.item(row, 1).text()
        # 从配置中查找实际的 key
        config = get_algorithm_config(algo_type)
        if row < len(config):
            actual_key = config[row]["key"]

            reply = QMessageBox.question(
                self, "确认删除",
                f"确定要从列表中删除算法吗？\n\n这只会从显示列表中移除，不会删除实际算法代码。",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                if delete_algorithm(algo_type, actual_key):
                    self.table.removeRow(row)
                    self._load_algorithms(algo_type)  # 重新加载
                    QMessageBox.information(self, "成功", "算法已从列表中移除")
                else:
                    QMessageBox.warning(self, "失败", "删除算法失败")

    def _save_changes(self):
        """保存修改。"""
        algo_type = self.tabs.currentWidget().table.algo_type
        config = get_algorithm_config(algo_type)

        success = True
        for row in range(self.table.rowCount()):
            check_item = self.table.item(row, 0)
            name_item = self.table.item(row, 1)

            if row < len(config):
                key = config[row]["key"]
                enabled = check_item.checkState() == Qt.Checked
                name = name_item.text()

                if not update_algorithm(algo_type, key, name=name, enabled=enabled):
                    success = False

        if success:
            QMessageBox.information(self, "成功", "算法配置已保存")
            self.accept()
        else:
            QMessageBox.warning(self, "失败", "保存配置时出错")

    def _reset_defaults(self):
        """恢复默认配置。"""
        algo_type = self.tabs.currentWidget().table.algo_type
        tab_name = "降噪" if algo_type == "denoise" else "超分辨率"

        reply = QMessageBox.question(
            self, "确认恢复默认",
            f"确定要恢复{tab_name}算法的默认配置吗？\n\n所有自定义名称和启用状态将被重置。",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            if reset_to_defaults(algo_type):
                self._load_algorithms(algo_type)
                QMessageBox.information(self, "成功", "已恢复默认配置")
            else:
                QMessageBox.warning(self, "失败", "恢复默认配置失败")

    def apply_medical_style(self):
        """应用 Medical Minimalism 风格。"""
        self.setStyleSheet("""
            QDialog {
                background-color: #f8fafc;
            }
            QTabWidget::pane {
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e2e8f0;
                color: #64748b;
                padding: 10px 20px;
                margin-right: 4px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-size: 15px;
                font-weight: 500;
            }
            QTabBar::tab:selected {
                background-color: white;
                color: #0ea5e9;
                font-weight: 600;
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
