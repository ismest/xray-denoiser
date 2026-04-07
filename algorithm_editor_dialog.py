"""
算法编辑对话框
支持降噪和超分辨率算法的启用/禁用、重命名管理
支持自定义通过目录加载训练模型
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QTableWidget, QTableWidgetItem, QLabel, QMessageBox,
                             QWidget, QHeaderView, QCheckBox, QLineEdit,
                             QFrame, QFileDialog, QGroupBox, QFormLayout,
                             QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from algorithm_config import (get_algorithm_config, update_algorithm,
                              add_algorithm, delete_algorithm, reset_to_defaults,
                              DEFAULT_DENOISE_ALGORITHMS, DEFAULT_SR_ALGORITHMS)

import os
import json
import shutil


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

        # 自定义模型加载区域
        self.custom_model_group = QGroupBox("自定义模型加载")
        self.custom_model_group.setObjectName("customModelGroup")
        custom_layout = QVBoxLayout(self.custom_model_group)
        custom_layout.setSpacing(8)

        # 模型目录选择
        dir_layout = QHBoxLayout()
        dir_layout.setSpacing(8)
        dir_label = QLabel("模型目录:")
        dir_label.setStyleSheet("font-size: 14px; color: #475569;")
        dir_layout.addWidget(dir_label)

        self.model_dir_text = QLineEdit()
        self.model_dir_text.setPlaceholderText("选择包含模型文件的目录...")
        self.model_dir_text.setMinimumHeight(36)
        self.model_dir_text.setStyleSheet("""
            QLineEdit {
                padding: 8px 12px;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                font-size: 14px;
            }
        """)
        dir_layout.addWidget(self.model_dir_text, 1)

        self.browse_dir_btn = QPushButton("浏览")
        self.browse_dir_btn.setObjectName("browseDirBtn")
        self.browse_dir_btn.setMinimumHeight(36)
        self.browse_dir_btn.setMinimumWidth(70)
        self.browse_dir_btn.clicked.connect(self.browse_model_directory)
        dir_layout.addWidget(self.browse_dir_btn)

        custom_layout.addLayout(dir_layout)

        # 模型信息显示
        self.model_info_label = QLabel("未选择模型目录")
        self.model_info_label.setStyleSheet("color: #94a3b8; font-size: 13px; padding: 6px;")
        custom_layout.addWidget(self.model_info_label)

        # 加载按钮
        self.load_model_btn = QPushButton("加载模型到算法列表")
        self.load_model_btn.setObjectName("loadModelBtn")
        self.load_model_btn.setMinimumHeight(38)
        self.load_model_btn.clicked.connect(self.load_custom_model)
        self.load_model_btn.setEnabled(False)
        custom_layout.addWidget(self.load_model_btn)

        layout.addWidget(self.custom_model_group)

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

        # 加载算法数据
        self._load_algorithms()

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

    def browse_model_directory(self):
        """浏览选择模型目录。"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择模型目录", "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if dir_path:
            self.model_dir_text.setText(dir_path)
            self._check_model_directory(dir_path)

    def _check_model_directory(self, dir_path):
        """检查模型目录并显示信息。"""
        if not os.path.isdir(dir_path):
            self.model_info_label.setText("无效的目录")
            self.model_info_label.setStyleSheet("color: #ef4444; font-size: 13px; padding: 6px;")
            self.load_model_btn.setEnabled(False)
            return

        # 检查模型文件
        model_files = []
        if self.algo_type == "denoise":
            # 检查降噪模型文件
            if os.path.exists(os.path.join(dir_path, 'denoiser.onnx')):
                model_files.append("denoiser.onnx")
            if os.path.exists(os.path.join(dir_path, 'best_denoiser.pth')):
                model_files.append("best_denoiser.pth")
            if os.path.exists(os.path.join(dir_path, 'model.pt')):
                model_files.append("model.pt")
        else:
            # 检查超分辨率模型文件
            if os.path.exists(os.path.join(dir_path, 'sr_model.onnx')):
                model_files.append("sr_model.onnx")
            if os.path.exists(os.path.join(dir_path, 'best_sr_model.pth')):
                model_files.append("best_sr_model.pth")
            if os.path.exists(os.path.join(dir_path, 'model_sr.pt')):
                model_files.append("model_sr.pt")

        if model_files:
            self.model_info_label.setText(f"找到模型文件：{', '.join(model_files)}")
            self.model_info_label.setStyleSheet("color: #10b981; font-size: 13px; padding: 6px;")
            self.load_model_btn.setEnabled(True)
        else:
            self.model_info_label.setText("未找到有效的模型文件")
            self.model_info_label.setStyleSheet("color: #ef4444; font-size: 13px; padding: 6px;")
            self.load_model_btn.setEnabled(False)

    def load_custom_model(self):
        """加载自定义模型到算法列表。"""
        model_dir = self.model_dir_text.text()
        if not os.path.isdir(model_dir):
            QMessageBox.warning(self, "警告", "请先选择有效的模型目录")
            return

        # 生成算法 key
        timestamp = None
        try:
            timestamp = os.path.basename(model_dir)
        except:
            pass

        import datetime
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.algo_type == "denoise":
            algo_key = f"trained_neural_denoise_{timestamp_str}"
            algo_name = f"Trained Neural Denoise [{timestamp_str}]"
            target_subdir = "denoise"
        else:
            algo_key = f"trained_sr_{timestamp_str}"
            algo_name = f"Trained Super Resolution [{timestamp_str}]"
            target_subdir = "super_resolution"

        # 复制模型文件到 integrated_model 目录
        integrated_base = os.path.join(os.path.dirname(__file__), 'integrated_model', target_subdir)

        # 创建目标目录（带时间戳）
        target_dir = os.path.join(integrated_base, timestamp_str)
        os.makedirs(target_dir, exist_ok=True)

        # 复制所有模型文件
        try:
            for item in os.listdir(model_dir):
                src_path = os.path.join(model_dir, item)
                dst_path = os.path.join(target_dir, item)
                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)

            # 创建 marker 文件
            marker_path = os.path.join(target_dir, 'model_ready.marker')
            with open(marker_path, 'w', encoding='utf-8') as f:
                f.write(f"Integrated at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Source: {model_dir}\n")

            # 添加到算法配置
            if add_algorithm(self.algo_type, algo_key, algo_name, enabled=True):
                self.model_info_label.setText(f"模型已加载：{algo_name}")
                self.model_info_label.setStyleSheet("color: #10b981; font-size: 13px; padding: 6px;")
                self.load_model_btn.setEnabled(False)
                self.model_dir_text.clear()

                # 重新加载算法列表
                self._load_algorithms()

                QMessageBox.information(self, "成功", f"模型已加载到算法列表:\n{algo_name}")
            else:
                QMessageBox.warning(self, "失败", "添加到算法列表失败")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型失败：{str(e)}")

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
            QGroupBox#customModelGroup {
                font-weight: 600;
                font-size: 15px;
                color: #475569;
                margin-top: 12px;
                padding-top: 12px;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                background-color: #f1f5f9;
            }
            QGroupBox#customModelGroup::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
            QPushButton#browseDirBtn {
                background-color: #f1f5f9;
                color: #475569;
                font-size: 14px;
                font-weight: 500;
                padding: 8px 16px;
                border-radius: 6px;
                border: 1px solid #cbd5e1;
            }
            QPushButton#browseDirBtn:hover {
                background-color: #e2e8f0;
            }
            QPushButton#loadModelBtn {
                background-color: #0ea5e9;
                color: white;
                font-size: 14px;
                font-weight: 600;
                padding: 8px 16px;
                border-radius: 6px;
            }
            QPushButton#loadModelBtn:hover {
                background-color: #0284c7;
            }
            QPushButton#loadModelBtn:disabled {
                background-color: #cbd5e1;
                color: #94a3b8;
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
