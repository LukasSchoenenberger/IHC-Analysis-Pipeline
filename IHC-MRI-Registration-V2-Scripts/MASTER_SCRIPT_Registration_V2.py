import os
import sys

# IMPORTANT: Set matplotlib backend BEFORE any GUI imports to avoid ICE errors
import matplotlib
matplotlib.use('Qt5Agg')

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QTextEdit, QCheckBox, QDoubleSpinBox, QSpinBox,
    QGroupBox, QScrollArea, QListWidget, QLineEdit, QDialog, QMessageBox,
    QFileDialog, QSplitter, QSizePolicy, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QUrl
from PyQt5.QtGui import QPixmap, QFont, QColor, QPalette, QCursor, QDesktopServices, QIcon
import subprocess
import threading
import json
import shutil
from datetime import datetime
from PIL import Image


class AnnotationManagerDialog(QDialog):
    """Dialog for managing annotation mask names"""

    def __init__(self, parent, existing_annotations=None):
        super().__init__(parent)
        self.result = None
        self.annotations = existing_annotations[:] if existing_annotations else []

        self.setWindowTitle("Annotation Mask Manager")
        self.setMinimumSize(500, 600)
        self.resize(500, 600)
        self.setModal(True)

        self.setup_dialog()

    def setup_dialog(self):
        """Setup the dialog interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)

        # Title and description
        title_label = QLabel("Annotation Mask Manager")
        title_label.setFont(QFont('Segoe UI', 14, QFont.Bold))
        layout.addWidget(title_label)

        desc_label = QLabel(
            "Add annotation mask names for QuPath script generation.\n"
            "These names should match your QuPath annotation classifications."
        )
        desc_label.setFont(QFont('Segoe UI', 9))
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        layout.addSpacing(15)

        # Input section
        input_group = QGroupBox("Add New Annotation")
        input_layout = QVBoxLayout(input_group)

        input_layout.addWidget(QLabel("Annotation Name:"))

        entry_layout = QHBoxLayout()
        self.entry = QLineEdit()
        self.entry.setFont(QFont('Segoe UI', 10))
        self.entry.returnPressed.connect(self.add_annotation)
        entry_layout.addWidget(self.entry)

        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self.add_annotation)
        entry_layout.addWidget(add_btn)
        input_layout.addLayout(entry_layout)

        layout.addWidget(input_group)

        # Current annotations section
        list_group = QGroupBox("Current Annotations")
        list_layout = QVBoxLayout(list_group)

        self.listbox = QListWidget()
        self.listbox.setFont(QFont('Segoe UI', 10))
        list_layout.addWidget(self.listbox)

        # List management buttons
        list_btn_layout = QHBoxLayout()

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_annotation)
        list_btn_layout.addWidget(remove_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_all)
        list_btn_layout.addWidget(clear_btn)

        list_btn_layout.addStretch()

        load_sample_btn = QPushButton("Load Sample")
        load_sample_btn.clicked.connect(self.load_sample_annotations)
        list_btn_layout.addWidget(load_sample_btn)

        list_layout.addLayout(list_btn_layout)
        layout.addWidget(list_group)

        # Bottom buttons
        button_layout = QHBoxLayout()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        button_layout.addStretch()

        self.status_label = QLabel()
        self.status_label.setFont(QFont('Segoe UI', 9))
        self.status_label.setStyleSheet("color: gray;")
        button_layout.addWidget(self.status_label)

        ok_btn = QPushButton("Generate Script")
        ok_btn.clicked.connect(self.on_ok)
        button_layout.addWidget(ok_btn)

        layout.addLayout(button_layout)

        # Populate existing annotations
        self.refresh_listbox()
        self.update_status()
        self.entry.setFocus()

    def add_annotation(self):
        """Add a new annotation to the list"""
        name = self.entry.text().strip()

        if not name:
            QMessageBox.warning(self, "Empty Name", "Please enter an annotation name.")
            return

        if name in self.annotations:
            QMessageBox.warning(self, "Duplicate Name", f"'{name}' already exists in the list.")
            return

        if any(char in name for char in ['\n', '\r', '\t']):
            QMessageBox.critical(self, "Invalid Name", "Annotation names cannot contain newlines or tabs.")
            return

        self.annotations.append(name)
        self.refresh_listbox()
        self.entry.clear()
        self.entry.setFocus()
        self.update_status()

    def remove_annotation(self):
        """Remove selected annotation from the list"""
        current_row = self.listbox.currentRow()
        if current_row < 0:
            QMessageBox.information(self, "No Selection", "Please select an annotation to remove.")
            return

        self.annotations.pop(current_row)
        self.refresh_listbox()
        self.update_status()

        if self.annotations:
            new_index = min(current_row, len(self.annotations) - 1)
            self.listbox.setCurrentRow(new_index)

    def clear_all(self):
        """Clear all annotations"""
        if self.annotations:
            reply = QMessageBox.question(
                self, "Clear All",
                "Are you sure you want to remove all annotations?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.annotations.clear()
                self.refresh_listbox()
                self.update_status()

    def load_sample_annotations(self):
        """Load sample annotation names"""
        sample_annotations = ["WM", "WM_Lesion", "GM", "Surrounding_WM", "WML_Perilesion"]

        if self.annotations:
            reply = QMessageBox.question(
                self, "Load Sample",
                "Do you want to replace existing annotations (Yes) or add to them (No)?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            if reply == QMessageBox.Cancel:
                return
            elif reply == QMessageBox.Yes:
                self.annotations.clear()

        for annotation in sample_annotations:
            if annotation not in self.annotations:
                self.annotations.append(annotation)

        self.refresh_listbox()
        self.update_status()

    def refresh_listbox(self):
        """Refresh the listbox with current annotations"""
        self.listbox.clear()
        for annotation in self.annotations:
            self.listbox.addItem(annotation)

    def update_status(self):
        """Update status message"""
        count = len(self.annotations)
        if count == 0:
            self.status_label.setText("No annotations added")
        elif count == 1:
            self.status_label.setText("1 annotation")
        else:
            self.status_label.setText(f"{count} annotations")

    def on_ok(self):
        """Handle OK button click"""
        if not self.annotations:
            reply = QMessageBox.question(
                self, "No Annotations",
                "No annotations have been added. Continue anyway?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        self.result = self.annotations[:]
        self.accept()


class BatchBlockSelectorDialog(QDialog):
    """Dialog to select multiple directories from a parent folder for batch processing."""

    def __init__(self, parent, initial_parent_dir=""):
        super().__init__(parent)
        self.setWindowTitle("Select Directories for Batch Processing")
        self.setMinimumSize(500, 500)
        self.resize(500, 500)
        self.setModal(True)
        self.parent_dir = initial_parent_dir or ""
        self._dir_checkboxes = []
        self.setup_ui()
        if self.parent_dir and os.path.isdir(self.parent_dir):
            self._populate_list()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Parent folder row
        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("Parent folder:"))
        self.dir_edit = QLineEdit(self.parent_dir)
        self.dir_edit.setReadOnly(True)
        self.dir_edit.setFont(QFont('Segoe UI', 9))
        folder_row.addWidget(self.dir_edit, 1)
        btn_browse = QPushButton("Browse...")
        btn_browse.setFont(QFont('Segoe UI', 9))
        btn_browse.clicked.connect(self._browse)
        folder_row.addWidget(btn_browse)
        layout.addLayout(folder_row)

        # Scrollable checkbox list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.StyledPanel)
        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setSpacing(2)
        self._list_layout.addStretch()
        scroll.setWidget(self._list_widget)
        layout.addWidget(scroll, 1)

        # Select All / Deselect All
        sel_row = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_all.setFont(QFont('Segoe UI', 9))
        btn_all.clicked.connect(lambda: self._set_all(True))
        sel_row.addWidget(btn_all)
        btn_none = QPushButton("Deselect All")
        btn_none.setFont(QFont('Segoe UI', 9))
        btn_none.clicked.connect(lambda: self._set_all(False))
        sel_row.addWidget(btn_none)
        sel_row.addStretch()
        layout.addLayout(sel_row)

        # OK / Cancel
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setFont(QFont('Segoe UI', 10))
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)
        btn_ok = QPushButton("OK")
        btn_ok.setFont(QFont('Segoe UI', 10, QFont.Bold))
        btn_ok.clicked.connect(self._on_ok)
        btn_row.addWidget(btn_ok)
        layout.addLayout(btn_row)

    def _browse(self):
        chosen = QFileDialog.getExistingDirectory(self, "Select Parent Folder", self.parent_dir or "")
        if chosen:
            self.parent_dir = chosen
            self.dir_edit.setText(chosen)
            self._populate_list()

    def _populate_list(self):
        # Clear existing checkboxes
        for cb, _ in self._dir_checkboxes:
            cb.setParent(None)
        self._dir_checkboxes = []

        if not os.path.isdir(self.parent_dir):
            return

        subdirs = sorted(
            d for d in os.listdir(self.parent_dir)
            if os.path.isdir(os.path.join(self.parent_dir, d))
        )
        # Insert before the stretch
        stretch_idx = self._list_layout.count() - 1
        for name in subdirs:
            cb = QCheckBox(name)
            cb.setFont(QFont('Segoe UI', 10))
            cb.setChecked(False)
            self._list_layout.insertWidget(stretch_idx, cb)
            self._dir_checkboxes.append((cb, os.path.join(self.parent_dir, name)))
            stretch_idx += 1

    def _set_all(self, checked):
        for cb, _ in self._dir_checkboxes:
            cb.setChecked(checked)

    def _on_ok(self):
        if not self._dir_checkboxes:
            QMessageBox.warning(self, "No Directories", "Please select a parent folder first.")
            return
        self.accept()

    def selected_dirs(self):
        return [path for cb, path in self._dir_checkboxes if cb.isChecked()]


class SubdirListConfigDialog(QDialog):
    """Generic config dialog for a list of strings (subdirectory names), saved to a JSON file."""

    def __init__(self, parent, title, json_path, list_key="subdirs"):
        super().__init__(parent)
        self.json_path = json_path
        self.list_key = list_key
        self.setWindowTitle(title)
        self.setMinimumSize(400, 350)
        self.resize(400, 350)
        self.setModal(True)
        self._load()
        self._setup_ui(title)

    def _load(self):
        self.items = []
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    self.items = json.load(f).get(self.list_key, [])
            except Exception:
                pass

    def _setup_ui(self, title):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        group = QGroupBox(title)
        group_layout = QVBoxLayout(group)

        self._list = QListWidget()
        self._list.setFont(QFont('Segoe UI', 9))
        self._list.setMaximumHeight(160)
        for item in self.items:
            self._add_item(item)
        group_layout.addWidget(self._list)

        add_row = QHBoxLayout()
        self._input = QLineEdit()
        self._input.setFont(QFont('Segoe UI', 9))
        self._input.setPlaceholderText("New entry...")
        self._input.returnPressed.connect(self._on_add)
        add_row.addWidget(self._input)
        btn_add = QPushButton("Add")
        btn_add.setFont(QFont('Segoe UI', 9))
        btn_add.setFixedWidth(50)
        btn_add.clicked.connect(self._on_add)
        add_row.addWidget(btn_add)
        group_layout.addLayout(add_row)

        layout.addWidget(group)
        layout.addStretch()

        bottom = QHBoxLayout()
        bottom.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setFont(QFont('Segoe UI', 10))
        btn_cancel.clicked.connect(self.reject)
        bottom.addWidget(btn_cancel)
        btn_save = QPushButton("Save")
        btn_save.setFont(QFont('Segoe UI', 10, QFont.Bold))
        btn_save.clicked.connect(self._save)
        bottom.addWidget(btn_save)
        layout.addLayout(bottom)

    def _add_item(self, text):
        from PyQt5.QtWidgets import QListWidgetItem
        item = QListWidgetItem(self._list)
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(2, 0, 2, 0)
        row_layout.setSpacing(4)
        lbl = QLabel(text)
        lbl.setFont(QFont('Segoe UI', 9))
        row_layout.addWidget(lbl, 1)
        btn_x = QPushButton("\u00d7")
        btn_x.setFixedSize(20, 20)
        btn_x.setFont(QFont('Segoe UI', 10, QFont.Bold))
        btn_x.setStyleSheet("QPushButton { border: none; color: #999; } QPushButton:hover { color: #e74c3c; }")
        btn_x.clicked.connect(lambda: self._remove_item(item))
        row_layout.addWidget(btn_x)
        item.setSizeHint(row_widget.sizeHint())
        self._list.setItemWidget(item, row_widget)

    def _remove_item(self, item):
        row = self._list.row(item)
        if row >= 0:
            self._list.takeItem(row)

    def _on_add(self):
        text = self._input.text().strip()
        if text:
            self._add_item(text)
            self._input.clear()
            self._input.setFocus()

    def _collect(self):
        result = []
        for i in range(self._list.count()):
            w = self._list.itemWidget(self._list.item(i))
            if w:
                lbl = w.findChild(QLabel)
                if lbl:
                    result.append(lbl.text())
        return result

    def _save(self):
        # Preserve other keys in the existing JSON
        data = {}
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                pass
        data[self.list_key] = self._collect()
        try:
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save configuration:\n{e}")


class TifBatchConfigDialog(QDialog):
    """Config dialog for TIF/PNG Batch: stain subdirectory list + directory-type checkboxes."""

    def __init__(self, parent, json_path):
        super().__init__(parent)
        self.json_path = json_path
        self.setWindowTitle("TIF/PNG Batch – Configuration")
        self.setMinimumSize(400, 400)
        self.resize(400, 400)
        self.setModal(True)
        self._load()
        self._setup_ui()

    def _load(self):
        self.items = []
        self.convert_density_maps = True
        self.convert_annotation_masks = False
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.items = data.get("subdirs", [])
                self.convert_density_maps = data.get("convert_density_maps", True)
                self.convert_annotation_masks = data.get("convert_annotation_masks", False)
            except Exception:
                pass

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        group = QGroupBox("Stain Subdirectories")
        group_layout = QVBoxLayout(group)

        self._list = QListWidget()
        self._list.setFont(QFont('Segoe UI', 9))
        self._list.setMaximumHeight(160)
        for item in self.items:
            self._add_item(item)
        group_layout.addWidget(self._list)

        add_row = QHBoxLayout()
        self._input = QLineEdit()
        self._input.setFont(QFont('Segoe UI', 9))
        self._input.setPlaceholderText("New entry...")
        self._input.returnPressed.connect(self._on_add)
        add_row.addWidget(self._input)
        btn_add = QPushButton("Add")
        btn_add.setFont(QFont('Segoe UI', 9))
        btn_add.setFixedWidth(50)
        btn_add.clicked.connect(self._on_add)
        add_row.addWidget(btn_add)
        group_layout.addLayout(add_row)

        layout.addWidget(group)

        dir_group = QGroupBox("Convert files in")
        dir_layout = QVBoxLayout(dir_group)
        self._chk_density = QCheckBox("Density_Maps")
        self._chk_density.setFont(QFont('Segoe UI', 9))
        self._chk_density.setChecked(self.convert_density_maps)
        self._chk_annotations = QCheckBox("Annotation_Masks")
        self._chk_annotations.setFont(QFont('Segoe UI', 9))
        self._chk_annotations.setChecked(self.convert_annotation_masks)
        dir_layout.addWidget(self._chk_density)
        dir_layout.addWidget(self._chk_annotations)
        layout.addWidget(dir_group)

        layout.addStretch()

        bottom = QHBoxLayout()
        bottom.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setFont(QFont('Segoe UI', 10))
        btn_cancel.clicked.connect(self.reject)
        bottom.addWidget(btn_cancel)
        btn_save = QPushButton("Save")
        btn_save.setFont(QFont('Segoe UI', 10, QFont.Bold))
        btn_save.clicked.connect(self._save)
        bottom.addWidget(btn_save)
        layout.addLayout(bottom)

    def _add_item(self, text):
        from PyQt5.QtWidgets import QListWidgetItem
        item = QListWidgetItem(self._list)
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(2, 0, 2, 0)
        row_layout.setSpacing(4)
        lbl = QLabel(text)
        lbl.setFont(QFont('Segoe UI', 9))
        row_layout.addWidget(lbl, 1)
        btn_x = QPushButton("\u00d7")
        btn_x.setFixedSize(20, 20)
        btn_x.setFont(QFont('Segoe UI', 10, QFont.Bold))
        btn_x.setStyleSheet("QPushButton { border: none; color: #999; } QPushButton:hover { color: #e74c3c; }")
        btn_x.clicked.connect(lambda: self._remove_item(item))
        row_layout.addWidget(btn_x)
        item.setSizeHint(row_widget.sizeHint())
        self._list.setItemWidget(item, row_widget)

    def _remove_item(self, item):
        row = self._list.row(item)
        if row >= 0:
            self._list.takeItem(row)

    def _on_add(self):
        text = self._input.text().strip()
        if text:
            self._add_item(text)
            self._input.clear()
            self._input.setFocus()

    def _collect(self):
        result = []
        for i in range(self._list.count()):
            w = self._list.itemWidget(self._list.item(i))
            if w:
                lbl = w.findChild(QLabel)
                if lbl:
                    result.append(lbl.text())
        return result

    def _save(self):
        data = {}
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                pass
        data["subdirs"] = self._collect()
        data["convert_density_maps"] = self._chk_density.isChecked()
        data["convert_annotation_masks"] = self._chk_annotations.isChecked()
        try:
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save configuration:\n{e}")


class TransformerBatchConfigDialog(QDialog):
    """Config dialog for Transformer Batch: stain subdirectories + copy-to-Density_Maps option."""

    def __init__(self, parent, json_path):
        super().__init__(parent)
        self.json_path = json_path
        self.setWindowTitle("Transformer Batch – Configuration")
        self.setMinimumSize(480, 560)
        self.resize(520, 600)
        self.setModal(True)
        self._load()
        self._setup_ui()

    def _load(self):
        self.items = []
        self.transform_density_maps = True
        self.transform_annotation_masks = False
        self.copy_to_density_maps = False
        self.rename_by_stain_dir = False
        self.use_naming_pattern = False
        self.naming_dict = []
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.items = data.get("subdirs", [])
                    self.transform_density_maps = data.get("transform_density_maps", True)
                    self.transform_annotation_masks = data.get("transform_annotation_masks", False)
                    self.copy_to_density_maps = data.get("copy_to_density_maps", False)
                    self.rename_by_stain_dir = data.get("rename_by_stain_dir", False)
                    self.use_naming_pattern = data.get("use_naming_pattern", False)
                    self.naming_dict = data.get("naming_dict", [])
            except Exception:
                pass

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        src_group = QGroupBox("Transform files in")
        src_layout = QHBoxLayout(src_group)
        self._chk_transform_density = QCheckBox("Density_Maps")
        self._chk_transform_density.setFont(QFont('Segoe UI', 9))
        self._chk_transform_density.setChecked(self.transform_density_maps)
        self._chk_transform_annotations = QCheckBox("Annotation_Masks")
        self._chk_transform_annotations.setFont(QFont('Segoe UI', 9))
        self._chk_transform_annotations.setChecked(self.transform_annotation_masks)
        src_layout.addWidget(self._chk_transform_density)
        src_layout.addWidget(self._chk_transform_annotations)
        src_layout.addStretch()
        layout.addWidget(src_group)

        group = QGroupBox("Stain Subdirectories")
        group_layout = QVBoxLayout(group)

        self._list = QListWidget()
        self._list.setFont(QFont('Segoe UI', 9))
        self._list.setMaximumHeight(130)
        for item in self.items:
            self._add_item(item)
        group_layout.addWidget(self._list)

        add_row = QHBoxLayout()
        self._input = QLineEdit()
        self._input.setFont(QFont('Segoe UI', 9))
        self._input.setPlaceholderText("New subdirectory name...")
        self._input.returnPressed.connect(self._on_add)
        add_row.addWidget(self._input)
        btn_add = QPushButton("Add")
        btn_add.setFont(QFont('Segoe UI', 9))
        btn_add.setFixedWidth(50)
        btn_add.clicked.connect(self._on_add)
        add_row.addWidget(btn_add)
        group_layout.addLayout(add_row)

        layout.addWidget(group)

        self._copy_chk = QCheckBox("Copy results to block-level Density_Maps / Annotation_Masks")
        self._copy_chk.setFont(QFont('Segoe UI', 9))
        self._copy_chk.setChecked(self.copy_to_density_maps)
        self._copy_chk.setToolTip(
            "Density_Maps results → copied to <block>/Density_Maps/ (rename rules apply).\n"
            "Annotation_Masks results → copied to <block>/Annotation_Masks/ as\n"
            "  <original_name>_transformed[_<threshold>].nii.gz (no rename rules)."
        )
        layout.addWidget(self._copy_chk)

        rename_row = QHBoxLayout()
        self._rename_chk = QCheckBox("Change name based on stain directory")
        self._rename_chk.setFont(QFont('Segoe UI', 9))
        self._rename_chk.setChecked(self.rename_by_stain_dir)
        self._rename_chk.setToolTip(
            "When checked, copied files are renamed using the stain directory name\n"
            "and the stain number extracted from the filename.\n"
            "Example: density_stain1_*.nii.gz in Bielschowsky/ → Bielschowsky_stain1_transformed.nii.gz"
        )
        rename_row.addWidget(self._rename_chk)

        self._naming_pattern_chk = QCheckBox("Set naming pattern")
        self._naming_pattern_chk.setFont(QFont('Segoe UI', 9))
        self._naming_pattern_chk.setChecked(self.use_naming_pattern)
        self._naming_pattern_chk.setToolTip(
            "Define per-file rename rules in the table below.\n"
            "File Pattern: plain text = substring match (e.g. 'stain1');\n"
            "  wildcards also supported (e.g. '*stain1*'). Case-insensitive.\n"
            "Block number is always prepended.\n"
            "Fallback order: dict match → <block>_<new_name>_transformed\n"
            "  → <block>_<stain_dir>_stainX_transformed\n"
            "  → <block>_<stain_dir>_<original_stem>_transformed"
        )
        self._naming_pattern_chk.toggled.connect(self._on_naming_pattern_toggled)
        rename_row.addWidget(self._naming_pattern_chk)
        rename_row.addStretch()
        layout.addLayout(rename_row)

        # Naming patterns table
        self._pattern_group = QGroupBox("Naming Patterns")
        pattern_layout = QVBoxLayout(self._pattern_group)
        pattern_layout.setContentsMargins(6, 6, 6, 6)
        pattern_layout.setSpacing(4)

        self._naming_table = QTableWidget(0, 3)
        self._naming_table.setFont(QFont('Segoe UI', 9))
        self._naming_table.setHorizontalHeaderLabels(["Stain Directory", "File Pattern (e.g. stain1)", "New Name"])
        self._naming_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        self._naming_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._naming_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Interactive)
        self._naming_table.setColumnWidth(0, 120)
        self._naming_table.setColumnWidth(2, 120)
        self._naming_table.verticalHeader().setVisible(False)
        self._naming_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._naming_table.setMinimumHeight(120)
        for entry in self.naming_dict:
            self._add_pattern_row(entry.get("stain_dir", ""), entry.get("pattern", ""), entry.get("new_name", ""))
        pattern_layout.addWidget(self._naming_table)

        btn_row = QHBoxLayout()
        btn_add_row = QPushButton("Add Row")
        btn_add_row.setFont(QFont('Segoe UI', 9))
        btn_add_row.clicked.connect(self._on_add_pattern_row)
        btn_row.addWidget(btn_add_row)
        btn_remove_row = QPushButton("Remove Selected")
        btn_remove_row.setFont(QFont('Segoe UI', 9))
        btn_remove_row.clicked.connect(self._on_remove_pattern_row)
        btn_row.addWidget(btn_remove_row)
        btn_row.addStretch()
        pattern_layout.addLayout(btn_row)

        layout.addWidget(self._pattern_group)
        self._pattern_group.setEnabled(self.use_naming_pattern)

        bottom = QHBoxLayout()
        bottom.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setFont(QFont('Segoe UI', 10))
        btn_cancel.clicked.connect(self.reject)
        bottom.addWidget(btn_cancel)
        btn_save = QPushButton("Save")
        btn_save.setFont(QFont('Segoe UI', 10, QFont.Bold))
        btn_save.clicked.connect(self._save)
        bottom.addWidget(btn_save)
        layout.addLayout(bottom)

    def _add_item(self, text):
        from PyQt5.QtWidgets import QListWidgetItem
        item = QListWidgetItem(self._list)
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(2, 0, 2, 0)
        row_layout.setSpacing(4)
        lbl = QLabel(text)
        lbl.setFont(QFont('Segoe UI', 9))
        row_layout.addWidget(lbl, 1)
        btn_x = QPushButton("\u00d7")
        btn_x.setFixedSize(20, 20)
        btn_x.setFont(QFont('Segoe UI', 10, QFont.Bold))
        btn_x.setStyleSheet("QPushButton { border: none; color: #999; } QPushButton:hover { color: #e74c3c; }")
        btn_x.clicked.connect(lambda: self._remove_item(item))
        row_layout.addWidget(btn_x)
        item.setSizeHint(row_widget.sizeHint())
        self._list.setItemWidget(item, row_widget)

    def _remove_item(self, item):
        row = self._list.row(item)
        if row >= 0:
            self._list.takeItem(row)

    def _on_add(self):
        text = self._input.text().strip()
        if text:
            self._add_item(text)
            self._input.clear()
            self._input.setFocus()

    def _collect(self):
        result = []
        for i in range(self._list.count()):
            w = self._list.itemWidget(self._list.item(i))
            if w:
                lbl = w.findChild(QLabel)
                if lbl:
                    result.append(lbl.text())
        return result

    def _on_naming_pattern_toggled(self, checked):
        self._pattern_group.setEnabled(checked)

    def _add_pattern_row(self, stain_dir="", pattern="", new_name=""):
        row = self._naming_table.rowCount()
        self._naming_table.insertRow(row)
        self._naming_table.setItem(row, 0, QTableWidgetItem(stain_dir))
        self._naming_table.setItem(row, 1, QTableWidgetItem(pattern))
        self._naming_table.setItem(row, 2, QTableWidgetItem(new_name))

    def _on_add_pattern_row(self):
        self._add_pattern_row()
        self._naming_table.scrollToBottom()
        self._naming_table.editItem(self._naming_table.item(self._naming_table.rowCount() - 1, 0))

    def _on_remove_pattern_row(self):
        rows = sorted(set(idx.row() for idx in self._naming_table.selectedIndexes()), reverse=True)
        for row in rows:
            self._naming_table.removeRow(row)

    def _collect_naming_dict(self):
        result = []
        for row in range(self._naming_table.rowCount()):
            stain_dir = (self._naming_table.item(row, 0) or QTableWidgetItem("")).text().strip()
            pattern   = (self._naming_table.item(row, 1) or QTableWidgetItem("")).text().strip()
            new_name  = (self._naming_table.item(row, 2) or QTableWidgetItem("")).text().strip()
            if stain_dir or pattern or new_name:
                result.append({"stain_dir": stain_dir, "pattern": pattern, "new_name": new_name})
        return result

    def _save(self):
        data = {}
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                pass
        data["subdirs"] = self._collect()
        data["transform_density_maps"] = self._chk_transform_density.isChecked()
        data["transform_annotation_masks"] = self._chk_transform_annotations.isChecked()
        data["copy_to_density_maps"] = self._copy_chk.isChecked()
        data["rename_by_stain_dir"] = self._rename_chk.isChecked()
        data["use_naming_pattern"] = self._naming_pattern_chk.isChecked()
        data["naming_dict"] = self._collect_naming_dict()
        try:
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save configuration:\n{e}")


class SplitBatchConfigDialog(QDialog):
    """Config dialog for Split Annotation Batch: mask filenames + copy-to-Annotation_Masks option."""

    def __init__(self, parent, json_path):
        super().__init__(parent)
        self.json_path = json_path
        self.setWindowTitle("Split Annotation Batch – Configuration")
        self.setMinimumSize(400, 400)
        self.resize(400, 400)
        self.setModal(True)
        self._load()
        self._setup_ui()

    def _load(self):
        self.items = []
        self.copy_to_annotation_masks = False
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.items = data.get("filenames", [])
                    self.copy_to_annotation_masks = data.get("copy_to_annotation_masks", False)
            except Exception:
                pass

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        group = QGroupBox("Mask Filenames")
        group_layout = QVBoxLayout(group)

        self._list = QListWidget()
        self._list.setFont(QFont('Segoe UI', 9))
        self._list.setMaximumHeight(160)
        for item in self.items:
            self._add_item(item)
        group_layout.addWidget(self._list)

        add_row = QHBoxLayout()
        self._input = QLineEdit()
        self._input.setFont(QFont('Segoe UI', 9))
        self._input.setPlaceholderText("New filename (e.g. T1_whole_lesion_mask.nii.gz)...")
        self._input.returnPressed.connect(self._on_add)
        add_row.addWidget(self._input)
        btn_add = QPushButton("Add")
        btn_add.setFont(QFont('Segoe UI', 9))
        btn_add.setFixedWidth(50)
        btn_add.clicked.connect(self._on_add)
        add_row.addWidget(btn_add)
        group_layout.addLayout(add_row)

        layout.addWidget(group)

        self._copy_chk = QCheckBox("Copy results to Annotation_Masks/")
        self._copy_chk.setFont(QFont('Segoe UI', 9))
        self._copy_chk.setChecked(self.copy_to_annotation_masks)
        self._copy_chk.setToolTip(
            "When checked, newly created split masks in Split_Annotation_Masks/ are\n"
            "also copied directly into Annotation_Masks/ for use in downstream steps."
        )
        layout.addWidget(self._copy_chk)
        layout.addStretch()

        bottom = QHBoxLayout()
        bottom.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setFont(QFont('Segoe UI', 10))
        btn_cancel.clicked.connect(self.reject)
        bottom.addWidget(btn_cancel)
        btn_save = QPushButton("Save")
        btn_save.setFont(QFont('Segoe UI', 10, QFont.Bold))
        btn_save.clicked.connect(self._save)
        bottom.addWidget(btn_save)
        layout.addLayout(bottom)

    def _add_item(self, text):
        from PyQt5.QtWidgets import QListWidgetItem
        item = QListWidgetItem(self._list)
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(2, 0, 2, 0)
        row_layout.setSpacing(4)
        lbl = QLabel(text)
        lbl.setFont(QFont('Segoe UI', 9))
        row_layout.addWidget(lbl, 1)
        btn_x = QPushButton("\u00d7")
        btn_x.setFixedSize(20, 20)
        btn_x.setFont(QFont('Segoe UI', 10, QFont.Bold))
        btn_x.setStyleSheet("QPushButton { border: none; color: #999; } QPushButton:hover { color: #e74c3c; }")
        btn_x.clicked.connect(lambda: self._remove_item(item))
        row_layout.addWidget(btn_x)
        item.setSizeHint(row_widget.sizeHint())
        self._list.setItemWidget(item, row_widget)

    def _remove_item(self, item):
        row = self._list.row(item)
        if row >= 0:
            self._list.takeItem(row)

    def _on_add(self):
        text = self._input.text().strip()
        if text:
            self._add_item(text)
            self._input.clear()
            self._input.setFocus()

    def _collect(self):
        result = []
        for i in range(self._list.count()):
            w = self._list.itemWidget(self._list.item(i))
            if w:
                lbl = w.findChild(QLabel)
                if lbl:
                    result.append(lbl.text())
        return result

    def _save(self):
        data = {}
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                pass
        data["filenames"] = self._collect()
        data["copy_to_annotation_masks"] = self._copy_chk.isChecked()
        try:
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save configuration:\n{e}")


class DocumentationDialog(QDialog):
    """Scrollable documentation dialog for the registration pipeline."""

    TEXT = (
        "EXPECTED FOLDER STRUCTURE\n"
        "══════════════════════════════════════════════════════════════\n\n"
        "  <project_dir>/\n"
        "    block_<name>/\n"
        "      <stain>/                              ← working directory\n"
        "        Density_Maps/                       ← input density maps\n"
        "        Annotation_Masks/                   ← annotation masks (.nii.gz)\n"
        "        Match_slice_results/\n"
        "          *matching_info.json               ← slice/orientation info\n"
        "        Linear_registration_results/\n"
        "          transform_manual.pkl              ← linear transform\n"
        "        Non-linear_registration_results/\n"
        "          nonlinear_transform.pkl           ← nonlinear params\n"
        "          deformation_field.pkl             ← deformation field\n"
        "        Transformation_results/             ← output (auto-created)\n"
        "          Binarized_Masks/\n"
        "      Annotation_Masks/\n"
        "        T1_whole_lesion_mask.nii.gz         ← block-level masks\n"
        "      MRI/\n"
        "        qT1_map.nii.gz\n\n"
        "──────────────────────────────────────────────────────────────\n"
        "WORKING DIRECTORY\n"
        "══════════════════════════════════════════════════════════════\n\n"
        "The 'Choose Working Directory' button sets the directory that\n"
        "single-block scripts run in (their cwd). This should be the\n"
        "stain subdirectory, e.g. block_01/Bielschowsky/.\n"
        "Batch operations use their own directory selection dialog and\n"
        "are not affected by the working directory setting.\n\n"
        "──────────────────────────────────────────────────────────────\n"
        "REGISTRATION PIPELINE\n"
        "══════════════════════════════════════════════════════════════\n\n"
        "1. Slice Matching\n"
        "   Find the best matching MRI slice for the IHC section.\n\n"
        "2. Linear Registration\n"
        "   Apply a 2D affine registration between the IHC overview\n"
        "   and the selected MRI slice.\n\n"
        "3. Non-Linear Registration\n"
        "   Apply a deformable (thin-plate spline) registration on top\n"
        "   of the linear result for finer alignment.\n\n"
        "4. Transformer\n"
        "   Apply the computed transforms to the density maps in\n"
        "   Density_Maps/. Output goes to Transformation_results/.\n"
        "   Options: Binarize, Intensity Threshold.\n\n"
        "   Transformer (Batch)\n"
        "   Runs the Transformer over multiple blocks × stain subdirs.\n"
        "   Configure specifies which stain subdirs to process.\n"
        "   See folder structure above for required layout.\n\n"
        "──────────────────────────────────────────────────────────────\n"
        "UTILITY TOOLS\n"
        "══════════════════════════════════════════════════════════════\n\n"
        "Generate Download Script\n"
        "   Create a QuPath Groovy script to download annotation masks.\n"
        "   Set working directory to stain, that annotations were made on.\n\n"
        "Convert TIF/PNG to NIfTI\n"
        "   Convert density map images (.tif, .tiff, .png) in the selected\n"
        "   Annotation_Masks folder to NIfTI format (.nii.gz) using the\n"
        "   slice and orientation from matching_info.json.\n\n"
        "   Convert TIF/PNG to NIfTI (Batch)\n"
        "   Same conversion, run over multiple blocks × stain subdirs.\n"
        "   Configure specifies which stain subdirs to process.\n"
        "   See folder structure above for required layout.\n\n"
        "Binarize Masks\n"
        "   Merge and binarize multiple (non-binarized) annotation masks with\n"
        "   conflict resolution. For voxels shared between masks the one with\n"
        "   the highest value wins (kept unchanged); all others are set to 0.\n"
        "   Non-shared voxels are binarized by an intensity threshold.\n"
        "   Two modes:\n"
        "     Combine/Binarize – applies the threshold directly and saves.\n"
        "     Combine/Threshold Evaluation – shows an interactive preview\n"
        "       overlaid on the reference MRI with adjustable threshold,\n"
        "       mask opacity and contrast controls before saving.\n"
        "   Defaults (set by working directory): masks from\n"
        "   Transformation_results/, MRI from ../MRI/, output to\n"
        "   Transformation_results/Binarized_Masks/.\n\n"
        "Split Annotation Masks\n"
        "   Split large annotation masks into patch-sized NIfTI regions.\n"
        "   Patch size is set with the 'Patch Size' spinbox below.\n\n"
        "   Split Annotation Masks (Batch)\n"
        "   Splits configured mask files across multiple blocks.\n"
        "   Configure specifies which filename(s) to look for and split.\n"
        "   See folder structure above for required layout.\n"
    )

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Documentation")
        self.setMinimumSize(600, 600)
        self.resize(650, 700)
        self.setModal(True)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFont(QFont('Consolas', 9))
        text_edit.setPlainText(self.TEXT)
        layout.addWidget(text_edit, 1)

        btn_close = QPushButton("Close")
        btn_close.setFont(QFont('Segoe UI', 10))
        btn_close.clicked.connect(self.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)


class CreateFolderStructureDialog(QDialog):
    """Dialog for creating the project folder structure interactively."""

    DEFAULT_SUBDIRS = ["MRI", "Density_Maps", "Annotation_Masks"]
    STAIN_PIPELINE_SUBDIRS = [
        "Density_Maps", "Annotation_Masks", "Match_slice_results",
        "Linear_registration_results", "Non-linear_registration_results",
        "Transformation_results",
    ]

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Create Folder Structure")
        self.setModal(True)
        # Step 1: choose parent directory
        start_dir = getattr(parent, 'working_dir', '')
        self.parent_dir = QFileDialog.getExistingDirectory(
            parent, "Select Project Parent Directory", start_dir)
        if not self.parent_dir:
            # User cancelled — close immediately via a deferred reject
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, self.reject)
            return
        self.setMinimumSize(480, 520)
        self.resize(480, 560)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Parent directory display
        dir_row = QHBoxLayout()
        dir_label = QLabel(f"Parent: {self.parent_dir}")
        dir_label.setFont(QFont('Segoe UI', 9))
        dir_label.setWordWrap(True)
        self._dir_label = dir_label
        dir_row.addWidget(dir_label, 1)
        btn_change = QPushButton("Change")
        btn_change.setFont(QFont('Segoe UI', 9))
        btn_change.setFixedWidth(60)
        btn_change.setAutoDefault(False)
        btn_change.clicked.connect(self._change_dir)
        dir_row.addWidget(btn_change)
        layout.addLayout(dir_row)

        # Block names group
        block_group = QGroupBox("Block Names (without 'block_' prefix)")
        block_group.setFont(QFont('Segoe UI', 9))
        block_layout = QVBoxLayout(block_group)
        self._block_list = QListWidget()
        self._block_list.setFont(QFont('Segoe UI', 9))
        self._block_list.setMaximumHeight(120)
        block_layout.addWidget(self._block_list)
        add_block_row = QHBoxLayout()
        self._block_input = QLineEdit()
        self._block_input.setFont(QFont('Segoe UI', 9))
        self._block_input.setPlaceholderText("e.g. 01, 02, frontal ...")
        self._block_input.returnPressed.connect(self._on_add_block)
        add_block_row.addWidget(self._block_input)
        btn_add_block = QPushButton("Add Block")
        btn_add_block.setFont(QFont('Segoe UI', 9))
        btn_add_block.setFixedWidth(70)
        btn_add_block.setAutoDefault(False)
        btn_add_block.clicked.connect(self._on_add_block)
        add_block_row.addWidget(btn_add_block)
        block_layout.addLayout(add_block_row)
        layout.addWidget(block_group)

        # Per-block subdirectories group
        subdir_group = QGroupBox("Per-Block Subdirectories")
        subdir_group.setFont(QFont('Segoe UI', 9))
        subdir_layout = QVBoxLayout(subdir_group)
        self._subdir_list = QListWidget()
        self._subdir_list.setFont(QFont('Segoe UI', 9))
        self._subdir_list.setMaximumHeight(140)
        subdir_layout.addWidget(self._subdir_list)
        # Add default entries (non-removable)
        for d in self.DEFAULT_SUBDIRS:
            self._add_subdir_item(d, is_default=True)
        add_subdir_row = QHBoxLayout()
        self._subdir_input = QLineEdit()
        self._subdir_input.setFont(QFont('Segoe UI', 9))
        self._subdir_input.setPlaceholderText("Stain directory, e.g. Bielschowsky")
        self._subdir_input.returnPressed.connect(self._on_add_subdir)
        add_subdir_row.addWidget(self._subdir_input)
        btn_add_subdir = QPushButton("Add")
        btn_add_subdir.setFont(QFont('Segoe UI', 9))
        btn_add_subdir.setFixedWidth(50)
        btn_add_subdir.setAutoDefault(False)
        btn_add_subdir.clicked.connect(self._on_add_subdir)
        add_subdir_row.addWidget(btn_add_subdir)
        subdir_layout.addLayout(add_subdir_row)
        info = QLabel(
            "Default entries (MRI, Density_Maps, Annotation_Masks) are created\n"
            "directly under each block. Additional entries are stain directories\n"
            "with full pipeline subfolders.")
        info.setFont(QFont('Segoe UI', 8))
        info.setStyleSheet("color: #666;")
        subdir_layout.addWidget(info)
        layout.addWidget(subdir_group)

        layout.addStretch()

        # Bottom buttons
        bottom = QHBoxLayout()
        bottom.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setFont(QFont('Segoe UI', 10))
        btn_cancel.setAutoDefault(False)
        btn_cancel.clicked.connect(self.reject)
        bottom.addWidget(btn_cancel)
        btn_create = QPushButton("Create")
        btn_create.setFont(QFont('Segoe UI', 10, QFont.Bold))
        btn_create.setAutoDefault(False)
        btn_create.clicked.connect(self._create_folders)
        bottom.addWidget(btn_create)
        layout.addLayout(bottom)

    # -- Block list helpers --------------------------------------------------

    def _add_block_item(self, text):
        from PyQt5.QtWidgets import QListWidgetItem
        item = QListWidgetItem(self._block_list)
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(2, 0, 2, 0)
        row_layout.setSpacing(4)
        lbl = QLabel(text)
        lbl.setFont(QFont('Segoe UI', 9))
        row_layout.addWidget(lbl, 1)
        btn_x = QPushButton("\u00d7")
        btn_x.setFixedSize(20, 20)
        btn_x.setFont(QFont('Segoe UI', 10, QFont.Bold))
        btn_x.setStyleSheet(
            "QPushButton { border: none; color: #999; } "
            "QPushButton:hover { color: #e74c3c; }")
        btn_x.clicked.connect(lambda: self._remove_item(self._block_list, item))
        row_layout.addWidget(btn_x)
        item.setSizeHint(row_widget.sizeHint())
        self._block_list.setItemWidget(item, row_widget)

    def _on_add_block(self):
        text = self._block_input.text().strip()
        if not text:
            return
        if '/' in text or '\\' in text:
            QMessageBox.warning(self, "Invalid Name",
                                "Block name must not contain / or \\.")
            return
        # Check duplicates
        existing = self._collect_blocks()
        if text in existing:
            QMessageBox.warning(self, "Duplicate", f"Block '{text}' already exists.")
            return
        self._add_block_item(text)
        self._block_input.clear()
        self._block_input.setFocus()

    def _collect_blocks(self):
        result = []
        for i in range(self._block_list.count()):
            w = self._block_list.itemWidget(self._block_list.item(i))
            if w:
                lbl = w.findChild(QLabel)
                if lbl:
                    result.append(lbl.text())
        return result

    # -- Subdir list helpers -------------------------------------------------

    def _add_subdir_item(self, text, is_default=False):
        from PyQt5.QtWidgets import QListWidgetItem
        item = QListWidgetItem(self._subdir_list)
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(2, 0, 2, 0)
        row_layout.setSpacing(4)
        lbl = QLabel(text)
        lbl.setFont(QFont('Segoe UI', 9))
        row_layout.addWidget(lbl, 1)
        btn_x = QPushButton("\u00d7")
        btn_x.setFixedSize(20, 20)
        btn_x.setFont(QFont('Segoe UI', 10, QFont.Bold))
        if is_default:
            btn_x.setEnabled(False)
            btn_x.setStyleSheet(
                "QPushButton { border: none; color: #ccc; }")
        else:
            btn_x.setStyleSheet(
                "QPushButton { border: none; color: #999; } "
                "QPushButton:hover { color: #e74c3c; }")
            btn_x.clicked.connect(lambda: self._remove_item(self._subdir_list, item))
        row_layout.addWidget(btn_x)
        item.setSizeHint(row_widget.sizeHint())
        self._subdir_list.setItemWidget(item, row_widget)

    def _on_add_subdir(self):
        text = self._subdir_input.text().strip()
        if not text:
            return
        if '/' in text or '\\' in text:
            QMessageBox.warning(self, "Invalid Name",
                                "Directory name must not contain / or \\.")
            return
        existing = self._collect_subdirs()
        if text in existing:
            QMessageBox.warning(self, "Duplicate", f"'{text}' already exists.")
            return
        self._add_subdir_item(text, is_default=False)
        self._subdir_input.clear()
        self._subdir_input.setFocus()

    def _collect_subdirs(self):
        result = []
        for i in range(self._subdir_list.count()):
            w = self._subdir_list.itemWidget(self._subdir_list.item(i))
            if w:
                lbl = w.findChild(QLabel)
                if lbl:
                    result.append(lbl.text())
        return result

    # -- General helpers -----------------------------------------------------

    @staticmethod
    def _remove_item(list_widget, item):
        row = list_widget.row(item)
        if row >= 0:
            list_widget.takeItem(row)

    def _change_dir(self):
        new_dir = QFileDialog.getExistingDirectory(
            self, "Select Project Parent Directory", self.parent_dir)
        if new_dir:
            self.parent_dir = new_dir
            self._dir_label.setText(f"Parent: {self.parent_dir}")

    # -- Create folders ------------------------------------------------------

    def _create_folders(self):
        blocks = self._collect_blocks()
        if not blocks:
            QMessageBox.warning(self, "No Blocks",
                                "Please add at least one block name.")
            return

        subdirs = self._collect_subdirs()
        stain_dirs = [s for s in subdirs if s not in self.DEFAULT_SUBDIRS]

        try:
            for block_name in blocks:
                block_path = os.path.join(self.parent_dir, f"block_{block_name}")
                os.makedirs(block_path, exist_ok=True)
                # Default subdirs directly under block
                for d in self.DEFAULT_SUBDIRS:
                    os.makedirs(os.path.join(block_path, d), exist_ok=True)
                # Stain directories with pipeline subfolders
                for stain in stain_dirs:
                    stain_path = os.path.join(block_path, stain)
                    os.makedirs(stain_path, exist_ok=True)
                    for sub in self.STAIN_PIPELINE_SUBDIRS:
                        os.makedirs(os.path.join(stain_path, sub), exist_ok=True)

            QMessageBox.information(
                self, "Success",
                f"Created folder structure for {len(blocks)} block(s) in:\n"
                f"{self.parent_dir}")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error",
                                 f"Failed to create folders:\n{e}")


class ScriptRunner(QThread):
    """Thread for running scripts"""
    output_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool)

    def __init__(self, script_path, args=None, cwd=None):
        super().__init__()
        self.script_path = script_path
        self.args = args or []
        self.cwd = cwd
        self.process = None
        self.cancelled = False

    def run(self):
        cmd = [sys.executable, self.script_path] + self.args
        self.output_signal.emit(f"Running: {' '.join(cmd)}")
        if self.cwd:
            self.output_signal.emit(f"Working directory: {self.cwd}")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1,
                cwd=self.cwd
            )

            while True:
                if self.cancelled:
                    self.output_signal.emit("Operation cancelled by user")
                    self.finished_signal.emit(False)
                    return

                poll_result = self.process.poll()
                if poll_result is not None:
                    remaining_stdout, remaining_stderr = self.process.communicate()
                    if remaining_stdout:
                        for line in remaining_stdout.splitlines():
                            if line.strip():
                                self.output_signal.emit(line.strip())
                    if remaining_stderr:
                        for line in remaining_stderr.splitlines():
                            if line.strip():
                                self.error_signal.emit(f"stderr: {line.strip()}")
                    break

                line = self.process.stdout.readline()
                if line:
                    self.output_signal.emit(line.strip())

            self.finished_signal.emit(self.process.returncode == 0)

        except Exception as e:
            self.error_signal.emit(f"Error running script: {e}")
            self.finished_signal.emit(False)

    def cancel(self):
        self.cancelled = True
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


class RegistrationPipeline(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IHC-MRI Registration Pipeline")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # Variables
        self.binarize_masks = False
        self.apply_threshold = False
        self.threshold_value = 15.0
        self.patch_size = 100
        self.logs = []
        self.running_thread = None
        self.current_step = None
        self._script_queue = []
        self._queue_active = False
        self._queue_name = ""

        # Directory paths
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.working_dir = self.script_dir
        self.subscripts_dir = os.path.join(self.script_dir, "subscripts")
        self.configs_dir = os.path.join(self.script_dir, "configs")

        # Create subdirectories and migrate files from main dir if needed
        self.setup_directories()

        # Script paths (in subscripts/ subdirectory)
        self.scripts = {
            "slice_matcher": os.path.join(self.subscripts_dir, "Slice_Matching_Qt.py"),
            "slice_and_linear_registration": os.path.join(self.subscripts_dir, "Slice_and_Linear_Registration_Qt.py"),
            "linear_registration": os.path.join(self.subscripts_dir, "Linear_Registration_Qt.py"),
            "nonlinear_registration": os.path.join(self.subscripts_dir, "Non_Linear_Registration_Qt.py"),
            "transformer": os.path.join(self.subscripts_dir, "Transformation.py"),
            "tif_to_nifti": os.path.join(self.subscripts_dir, "TIFF-to-NIFTI-Conversion.py"),
            "mask_binarizer": os.path.join(self.subscripts_dir, "Mask_Binarizer.py"),
            "segmentation_splitter": os.path.join(self.subscripts_dir, "Segmentation-Splitting.py"),
            "download_masks_generator": os.path.join(self.subscripts_dir, "Download-Masks-Generator.py"),
        }

        # Ensure config JSON files exist with defaults (in configs/ subdirectory)
        self.ensure_transformer_batch_config_json()
        self.ensure_tif_batch_config_json()
        self.ensure_split_batch_config_json()

        # Colors
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'background': '#F8F9FA',
            'surface': '#FFFFFF',
            'text': '#2C3E50',
            'text_light': '#7F8C8D',
            'success': '#27AE60',
            'error': '#E74C3C'
        }

        self.setup_gui()
        self.check_scripts()

    def setup_gui(self):
        """Set up the main GUI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Content area with splitter
        content_splitter = QSplitter(Qt.Horizontal)

        # LEFT SIDE - Processing content
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 10, 0)
        self.create_processing_area(left_layout)
        content_splitter.addWidget(left_widget)

        # RIGHT SIDE - Information panel
        right_widget = QWidget()
        right_widget.setFixedWidth(450)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 0, 0, 0)
        self.create_info_panel(right_layout)
        content_splitter.addWidget(right_widget)

        content_splitter.setStretchFactor(0, 1)
        content_splitter.setStretchFactor(1, 0)

        main_layout.addWidget(content_splitter, 1)  # stretch factor 1 to fill available space

        # Log and status area
        self.create_log_area(main_layout)

    def create_info_panel(self, parent_layout):
        """Create information panel with logo at top"""
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        info_layout.setContentsMargins(5, 5, 5, 5)

        # Logo at the top
        image_path = os.path.join(self.script_dir, "configs", "logo.png")
        if os.path.exists(image_path):
            try:
                pixmap = QPixmap(image_path)
                scaled_pixmap = pixmap.scaled(400, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                logo_btn = QPushButton()
                logo_btn.setIcon(QIcon(scaled_pixmap))
                logo_btn.setIconSize(QSize(scaled_pixmap.width(), scaled_pixmap.height()))
                logo_btn.setFlat(True)
                logo_btn.setCursor(QCursor(Qt.PointingHandCursor))
                logo_btn.setStyleSheet("QPushButton { border: none; padding: 0; }")
                logo_btn.clicked.connect(self.open_website)
                info_layout.addWidget(logo_btn)
            except Exception as e:
                self.log(f"Failed to load logo.png: {e}", error=True)

        # Session info below logo
        session_group = QGroupBox("Session Info")
        session_layout = QVBoxLayout(session_group)

        self.session_label = QLabel()
        self.session_label.setFont(QFont('Segoe UI', 11))
        self.session_label.setStyleSheet("color: black;")
        self.update_session_info()
        session_layout.addWidget(self.session_label)

        info_layout.addWidget(session_group)

        # Choose Working Directory button
        choose_dir_btn = QPushButton("Choose Working Directory")
        choose_dir_btn.setFont(QFont('Segoe UI', 11))
        choose_dir_btn.setMinimumHeight(36)
        choose_dir_btn.clicked.connect(self.choose_working_directory)
        info_layout.addWidget(choose_dir_btn)

        # Documentation button
        doc_btn = QPushButton("Documentation")
        doc_btn.setFont(QFont('Segoe UI', 11))
        doc_btn.setMinimumHeight(36)
        doc_btn.clicked.connect(self.open_documentation)
        info_layout.addWidget(doc_btn)

        # Create Folder Structure button
        create_folders_btn = QPushButton("Create Folder Structure")
        create_folders_btn.setFont(QFont('Segoe UI', 11))
        create_folders_btn.setMinimumHeight(36)
        create_folders_btn.clicked.connect(self.open_create_folder_structure)
        info_layout.addWidget(create_folders_btn)

        info_layout.addStretch()
        parent_layout.addWidget(info_widget)

    def update_session_info(self):
        """Update session info label"""
        parent_dir = os.path.basename(os.path.dirname(self.working_dir))
        current_dir = os.path.basename(self.working_dir)
        display_path = f"{parent_dir}/{current_dir}" if parent_dir else current_dir

        session_info = f"Working Dir: {display_path}\nPipeline Version: 2.0"
        self.session_label.setText(session_info)

    def choose_working_directory(self):
        """Open dialog to choose working directory"""
        new_dir = QFileDialog.getExistingDirectory(
            self,
            "Choose Working Directory",
            self.working_dir,
            QFileDialog.ShowDirsOnly
        )

        if new_dir:
            self.working_dir = new_dir
            self.update_session_info()
            self.log(f"Working directory changed to: {new_dir}")

    def open_website(self):
        """Open the project website"""
        url = "https://dbe.unibas.ch/en/research/imaging-modelling-diagnosis/translational-imaging-in-neurology-think-basel-group/"
        self.log(f"Opening: {url}")

        try:
            QDesktopServices.openUrl(QUrl(url))
            self.log("Browser opened successfully!")
        except Exception as e:
            self.log(f"Error opening browser: {e}", error=True)
            QMessageBox.information(
                self, "Website URL",
                f"Please copy this URL to your browser:\n\n{url}"
            )

    def create_processing_area(self, parent_layout):
        """Create the main processing area"""
        container = QWidget()
        tab_layout = QVBoxLayout(container)
        tab_layout.setContentsMargins(0, 10, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Individual pipeline steps (including processing options)
        steps_group = QGroupBox("Individual Pipeline Steps")
        steps_layout = QVBoxLayout(steps_group)

        # Standalone buttons are wider than those sharing a row with Configure.
        # Add right padding so the centered text aligns with the narrower buttons.
        _solo_btn_pad = "padding-right: 86px;"  # 80px Configure + ~6px spacing

        # Steps 1 & 2 side-by-side with the combined button on the right
        row_12 = QHBoxLayout()
        row_12.setSpacing(6)

        left_col = QVBoxLayout()
        left_col.setSpacing(6)
        left_col.setContentsMargins(0, 0, 0, 0)
        for text, command in [
            ("1. Slice Matcher", self.run_slice_matcher),
            ("2. Linear Registration", self.run_linear_registration),
        ]:
            btn = QPushButton(text)
            btn.setFont(QFont('Segoe UI', 11))
            btn.setMinimumHeight(36)
            btn.clicked.connect(command)
            left_col.addWidget(btn)

        left_widget = QWidget()
        left_widget.setLayout(left_col)
        left_widget.setContentsMargins(0, 0, 0, 0)
        row_12.addWidget(left_widget, stretch=1)

        btn_combined = QPushButton("Slice Matcher\n+ Linear Registration")
        btn_combined.setFont(QFont('Segoe UI', 11))
        btn_combined.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Ignored)
        btn_combined.clicked.connect(self.run_slice_and_linear_registration)
        row_12.addWidget(btn_combined, stretch=1)

        steps_layout.addLayout(row_12)

        # Steps 3 & 4
        for text, command in [
            ("3. Non-Linear Registration", self.run_nonlinear_registration),
            ("4. Transformer", self.run_transformer),
        ]:
            btn = QPushButton(text)
            btn.setFont(QFont('Segoe UI', 11))
            btn.setMinimumHeight(36)
            btn.setStyleSheet(_solo_btn_pad)
            btn.clicked.connect(command)
            steps_layout.addWidget(btn)

        # Processing options directly under Transformer button (no title/subtitle)
        options_widget = QWidget()
        options_layout = QVBoxLayout(options_widget)
        options_layout.setContentsMargins(10, 10, 10, 0)

        # Binarize option
        self.binarize_check = QCheckBox("Binarize images after transformation")
        self.binarize_check.setFont(QFont('Segoe UI', 10))
        self.binarize_check.stateChanged.connect(lambda s: setattr(self, 'binarize_masks', s == Qt.Checked))
        options_layout.addWidget(self.binarize_check)

        # Threshold options
        threshold_layout = QHBoxLayout()

        self.threshold_check = QCheckBox("Apply intensity threshold")
        self.threshold_check.setFont(QFont('Segoe UI', 10))
        self.threshold_check.stateChanged.connect(lambda s: setattr(self, 'apply_threshold', s == Qt.Checked))
        threshold_layout.addWidget(self.threshold_check)

        threshold_layout.addWidget(QLabel("Value:"))

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0, 1000)
        self.threshold_spin.setValue(15.0)
        self.threshold_spin.setFixedWidth(80)
        self.threshold_spin.valueChanged.connect(lambda v: setattr(self, 'threshold_value', v))
        threshold_layout.addWidget(self.threshold_spin)

        threshold_layout.addStretch()
        options_layout.addLayout(threshold_layout)

        steps_layout.addWidget(options_widget)

        # Run Transformer (Batch) row
        transformer_batch_row = QHBoxLayout()
        btn_transformer_batch = QPushButton("Transformer (Batch)")
        btn_transformer_batch.setFont(QFont('Segoe UI', 11))
        btn_transformer_batch.setMinimumHeight(36)
        btn_transformer_batch.clicked.connect(self.run_transformer_batch)
        transformer_batch_row.addWidget(btn_transformer_batch)
        btn_configure_transformer_batch = QPushButton("Configure")
        btn_configure_transformer_batch.setFont(QFont('Segoe UI', 9))
        btn_configure_transformer_batch.setMinimumHeight(36)
        btn_configure_transformer_batch.setFixedWidth(80)
        btn_configure_transformer_batch.clicked.connect(self.configure_transformer_batch)
        transformer_batch_row.addWidget(btn_configure_transformer_batch)
        steps_layout.addLayout(transformer_batch_row)

        scroll_layout.addWidget(steps_group)

        # Utility tools section
        utility_group = QGroupBox("Utility Tools")
        utility_layout = QVBoxLayout(utility_group)

        # Generate Download Script button
        btn_download = QPushButton("Generate Download Script")
        btn_download.setFont(QFont('Segoe UI', 11))
        btn_download.setMinimumHeight(36)
        btn_download.setStyleSheet(_solo_btn_pad)
        btn_download.clicked.connect(self.run_download_script_generator)
        utility_layout.addWidget(btn_download)

        # Convert TIF to NIfTI button
        btn_tif = QPushButton("Convert TIF/PNG to NIfTI")
        btn_tif.setFont(QFont('Segoe UI', 11))
        btn_tif.setMinimumHeight(36)
        btn_tif.setStyleSheet(_solo_btn_pad)
        btn_tif.clicked.connect(self.run_tif_to_nifti)
        utility_layout.addWidget(btn_tif)

        # Convert TIF/PNG to NIfTI (Batch) row
        tif_batch_row = QHBoxLayout()
        btn_tif_batch = QPushButton("Convert TIF/PNG to NIfTI (Batch)")
        btn_tif_batch.setFont(QFont('Segoe UI', 11))
        btn_tif_batch.setMinimumHeight(36)
        btn_tif_batch.clicked.connect(self.run_tif_to_nifti_batch)
        tif_batch_row.addWidget(btn_tif_batch)
        btn_configure_tif_batch = QPushButton("Configure")
        btn_configure_tif_batch.setFont(QFont('Segoe UI', 9))
        btn_configure_tif_batch.setMinimumHeight(36)
        btn_configure_tif_batch.setFixedWidth(80)
        btn_configure_tif_batch.clicked.connect(self.configure_tif_batch)
        tif_batch_row.addWidget(btn_configure_tif_batch)
        utility_layout.addLayout(tif_batch_row)

        # Combine/Binarize Masks button
        btn_threshold = QPushButton("Combine/Binarize Masks")
        btn_threshold.setFont(QFont('Segoe UI', 11))
        btn_threshold.setMinimumHeight(36)
        btn_threshold.setStyleSheet(_solo_btn_pad)
        btn_threshold.clicked.connect(self.run_mask_binarizer)
        utility_layout.addWidget(btn_threshold)

        # Split Annotation Masks button
        btn_split = QPushButton("Split Annotation Masks")
        btn_split.setFont(QFont('Segoe UI', 11))
        btn_split.setMinimumHeight(36)
        btn_split.setStyleSheet(_solo_btn_pad)
        btn_split.clicked.connect(self.run_segmentation_splitter)
        utility_layout.addWidget(btn_split)

        # Patch size setting (below Split Annotation Masks)
        patchsize_layout = QHBoxLayout()
        patchsize_layout.addSpacing(20)
        patchsize_label = QLabel("Patch Size:")
        patchsize_label.setFont(QFont('Segoe UI', 10))
        patchsize_layout.addWidget(patchsize_label)

        self.patchsize_spin = QSpinBox()
        self.patchsize_spin.setRange(10, 5000)
        self.patchsize_spin.setValue(100)
        self.patchsize_spin.setSuffix(" px")
        self.patchsize_spin.setFixedWidth(100)
        self.patchsize_spin.valueChanged.connect(lambda v: setattr(self, 'patch_size', v))
        patchsize_layout.addWidget(self.patchsize_spin)
        patchsize_layout.addStretch()
        utility_layout.addLayout(patchsize_layout)

        # Split Annotation Masks (Batch) row
        split_batch_row = QHBoxLayout()
        btn_split_batch = QPushButton("Split Annotation Masks (Batch)")
        btn_split_batch.setFont(QFont('Segoe UI', 11))
        btn_split_batch.setMinimumHeight(36)
        btn_split_batch.clicked.connect(self.run_segmentation_splitter_batch)
        split_batch_row.addWidget(btn_split_batch)
        btn_configure_split_batch = QPushButton("Configure")
        btn_configure_split_batch.setFont(QFont('Segoe UI', 9))
        btn_configure_split_batch.setMinimumHeight(36)
        btn_configure_split_batch.setFixedWidth(80)
        btn_configure_split_batch.clicked.connect(self.configure_split_batch)
        split_batch_row.addWidget(btn_configure_split_batch)
        utility_layout.addLayout(split_batch_row)

        scroll_layout.addWidget(utility_group)
        scroll_layout.addStretch()

        scroll_area.setWidget(scroll_content)
        tab_layout.addWidget(scroll_area)

        parent_layout.addWidget(container)

    def create_log_area(self, parent_layout):
        """Create log area"""
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 10, 0, 0)
        bottom_layout.setSpacing(4)

        # Log header with pop-out button
        log_header = QHBoxLayout()
        log_label = QLabel("Processing Log")
        log_label.setFont(QFont('Segoe UI', 10, QFont.Bold))
        log_header.addWidget(log_label)
        log_header.addStretch()
        btn_popout = QPushButton("Pop Out")
        btn_popout.setFont(QFont('Segoe UI', 8))
        btn_popout.setFixedHeight(22)
        btn_popout.setAutoDefault(False)
        btn_popout.clicked.connect(self._popout_log)
        log_header.addWidget(btn_popout)
        bottom_layout.addLayout(log_header)

        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont('Consolas', 9))
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #3C3C3C;
                color: #FFFFFF;
                border: 1px solid #505050;
                border-radius: 4px;
            }
        """)
        self.log_text.setMaximumHeight(120)
        bottom_layout.addWidget(self.log_text)

        # Status bar with cancel button
        status_layout = QHBoxLayout()
        status_layout.setSpacing(10)

        status_label_title = QLabel("Status:")
        status_label_title.setFont(QFont('Segoe UI', 10, QFont.Bold))
        status_layout.addWidget(status_label_title)

        self.status_label = QLabel("Ready - Select processing steps to begin")
        self.status_label.setFont(QFont('Segoe UI', 10))
        status_layout.addWidget(self.status_label, 1)

        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setFont(QFont('Segoe UI', 11, QFont.Bold))
        self.cancel_btn.setMinimumHeight(32)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_operation)
        self.cancel_btn.setFixedWidth(110)
        status_layout.addWidget(self.cancel_btn)

        bottom_layout.addLayout(status_layout)

        parent_layout.addWidget(bottom_widget)

    def setup_directories(self):
        """Create subscripts/ and configs/ subdirectories, migrate files from main dir."""
        os.makedirs(self.subscripts_dir, exist_ok=True)
        os.makedirs(self.configs_dir, exist_ok=True)

        # Scripts that belong in subscripts/
        subscript_files = [
            "Slice_Matching_Qt.py", "Linear_Registration_Qt.py",
            "Non_Linear_Registration_Qt.py", "Transformation.py",
            "TIFF-to-NIFTI-Conversion.py", "Mask_Binarizer.py",
            "Segmentation-Splitting.py",
            "Download-Masks-Generator.py"
        ]

        # Config files that belong in configs/
        config_files = [
            "transformer_batch_config.json", "tif_batch_config.json",
            "split_batch_config.json", "annotation_list.json"
        ]

        # Migrate subscripts from main dir if missing in subdirectory
        for filename in subscript_files:
            dest = os.path.join(self.subscripts_dir, filename)
            if not os.path.exists(dest):
                src = os.path.join(self.script_dir, filename)
                if os.path.exists(src):
                    shutil.move(src, dest)

        # Migrate config files from main dir if missing in subdirectory
        for filename in config_files:
            dest = os.path.join(self.configs_dir, filename)
            if not os.path.exists(dest):
                src = os.path.join(self.script_dir, filename)
                if os.path.exists(src):
                    shutil.move(src, dest)

    def ensure_transformer_batch_config_json(self):
        """Create transformer_batch_config.json with defaults if it doesn't exist."""
        json_path = os.path.join(self.configs_dir, "transformer_batch_config.json")
        if not os.path.exists(json_path):
            defaults = {"subdirs": ["Bielschowsky"], "last_parent_dir": "", "transform_density_maps": True, "transform_annotation_masks": False, "copy_to_density_maps": False, "rename_by_stain_dir": False, "use_naming_pattern": False, "naming_dict": []}
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(defaults, f, indent=4, ensure_ascii=False)

    def ensure_tif_batch_config_json(self):
        """Create tif_batch_config.json with defaults if it doesn't exist."""
        json_path = os.path.join(self.configs_dir, "tif_batch_config.json")
        if not os.path.exists(json_path):
            defaults = {
                "subdirs": ["Bielschowsky"],
                "last_parent_dir": "",
                "convert_density_maps": True,
                "convert_annotation_masks": False
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(defaults, f, indent=4, ensure_ascii=False)

    def ensure_split_batch_config_json(self):
        """Create split_batch_config.json with defaults if it doesn't exist."""
        json_path = os.path.join(self.configs_dir, "split_batch_config.json")
        if not os.path.exists(json_path):
            defaults = {"filenames": ["T1_whole_lesion_mask.nii.gz"], "last_parent_dir": "", "copy_to_annotation_masks": False}
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(defaults, f, indent=4, ensure_ascii=False)

    def check_scripts(self):
        """Check if all required scripts exist"""
        missing_scripts = []
        for name, path in self.scripts.items():
            if not os.path.exists(path):
                missing_scripts.append(f"{name}: {path}")

        if missing_scripts:
            error_message = "The following scripts were not found:\n\n" + "\n".join(missing_scripts)
            self.log(error_message, error=True)

    def _popout_log(self):
        """Open the processing log in a separate resizable window."""
        if hasattr(self, '_log_popout') and self._log_popout is not None and self._log_popout.isVisible():
            self._log_popout.raise_()
            self._log_popout.activateWindow()
            return
        win = QDialog(self)
        win.setWindowTitle("Processing Log")
        win.resize(700, 400)
        layout = QVBoxLayout(win)
        layout.setContentsMargins(6, 6, 6, 6)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFont(QFont('Consolas', 9))
        text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #3C3C3C;
                color: #FFFFFF;
                border: 1px solid #505050;
                border-radius: 4px;
            }
        """)
        text_edit.setHtml(self.log_text.toHtml())
        layout.addWidget(text_edit)
        win.log_text = text_edit
        win.setModal(False)
        win.show()
        self._log_popout = win

    def log(self, message, error=False):
        """Add message to log"""
        timestamp = datetime.now().strftime('%H:%M:%S')

        if error:
            color = self.colors['error']
            prefix = "ERROR: "
        elif "successfully" in message.lower() or "completed" in message.lower():
            color = self.colors['success']
            prefix = ""
        elif "progress:" in message.lower() or "processing" in message.lower() or "running" in message.lower():
            color = '#3498DB'
            prefix = ""
        else:
            color = '#ECF0F1'
            prefix = ""

        line = f'<span style="color:{color};">[{timestamp}] {prefix}{message}</span>'
        self.log_text.append(line)
        if hasattr(self, '_log_popout') and self._log_popout is not None and self._log_popout.isVisible():
            self._log_popout.log_text.append(line)
        self.logs.append(message)
        print(message)

    def update_status(self, message):
        """Update status message"""
        self.status_label.setText(message)

    def enable_buttons(self, enabled=True):
        """Enable or disable cancel button"""
        self.cancel_btn.setEnabled(not enabled)

    def cancel_operation(self):
        """Cancel the current operation"""
        self.log("Cancel requested by user...")

        if self.running_thread and self.running_thread.isRunning():
            self.running_thread.cancel()

        self.update_status("Cancelling operation...")

    def run_script(self, script_path, args=None):
        """Run a Python script in a thread"""
        self.enable_buttons(False)

        self.running_thread = ScriptRunner(script_path, args, cwd=self.working_dir)
        self.running_thread.output_signal.connect(lambda msg: self.log(msg))
        self.running_thread.error_signal.connect(lambda msg: self.log(msg, error=True))
        self.running_thread.finished_signal.connect(self.on_script_finished)
        self.running_thread.start()

    def on_script_finished(self, success):
        """Handle script completion"""
        self.enable_buttons(True)

        # Wait for the QThread to fully finish before releasing the reference,
        # otherwise Qt crashes when destroying a still-running thread
        if self.running_thread is not None:
            self.running_thread.wait()
            self.running_thread = None

        if success:
            self.log("Operation completed successfully")
            if self._queue_active:
                self._run_next_in_queue()
                return
            self.update_status("Operation completed successfully")
        else:
            self.update_status("Operation failed or cancelled")
            self.log("Operation failed or cancelled", error=True)
            if self._queue_active:
                self._script_queue = []
                self._queue_active = False
                self.log(f"{self._queue_name} aborted due to script failure.", error=True)

    def load_existing_annotations(self):
        """Load existing annotations from annotation_list.json (falls back to .txt)."""
        json_file = os.path.join(self.configs_dir, "annotation_list.json")
        txt_file = os.path.join(self.configs_dir, "annotation_list.txt")
        annotations = []

        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                annotations = data.get("annotations", [])
                self.log(f"Loaded {len(annotations)} existing annotations from annotation_list.json")
            except Exception as e:
                self.log(f"Error reading annotation_list.json: {e}", error=True)
        elif os.path.exists(txt_file):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    annotations = [line.strip() for line in f.readlines() if line.strip()]
                self.log(f"Loaded {len(annotations)} existing annotations from annotation_list.txt")
            except Exception as e:
                self.log(f"Error reading annotation_list.txt: {e}", error=True)

        return annotations

    def save_annotation_list(self, annotations):
        """Save annotation list to annotation_list.json in configs dir."""
        json_file = os.path.join(self.configs_dir, "annotation_list.json")

        try:
            data = {"annotations": annotations}
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

            self.log(f"Saved {len(annotations)} annotations to annotation_list.json")
            return True

        except Exception as e:
            self.log(f"Error saving annotation_list.json: {e}", error=True)
            QMessageBox.critical(self, "Save Error", f"Could not save annotation list:\n{e}")
            return False

    def run_download_script_generator(self):
        """Run the download script generator"""
        self.current_step = "download_script_generator"
        self.update_status("Managing annotation list...")
        self.log("Starting download script generator...")

        existing_annotations = self.load_existing_annotations()

        dialog = AnnotationManagerDialog(self, existing_annotations)
        if dialog.exec_() != QDialog.Accepted:
            self.log("Annotation management cancelled by user")
            self.update_status("Ready - Select processing steps to begin")
            return

        annotations = dialog.result

        if not annotations:
            self.log("No annotations provided - creating empty annotation_list.txt")

        if not self.save_annotation_list(annotations):
            self.update_status("Failed to save annotation list")
            return

        self.update_status("Running Download Masks Generator...")
        self.log("Launching Download-Masks-Generator.py...")
        config_path = os.path.join(self.configs_dir, "annotation_list.json")
        self.run_script(self.scripts["download_masks_generator"],
                        args=[config_path])

    def run_slice_matcher(self):
        """Run the slice matcher step"""
        self.current_step = "slice_matcher"
        self.update_status("Running slice matcher...")
        self.log("Starting slice matcher step...")
        self.run_script(self.scripts["slice_matcher"])

    def run_slice_and_linear_registration(self):
        """Run the combined slice matcher + linear registration step"""
        self.current_step = "slice_and_linear_registration"
        self.update_status("Running slice matcher + linear registration...")
        self.log("Starting combined slice matcher / linear registration step...")
        self.run_script(self.scripts["slice_and_linear_registration"])

    def run_linear_registration(self):
        """Run the linear registration step"""
        self.current_step = "linear_registration"
        self.update_status("Running linear registration...")
        self.log("Starting linear registration step...")
        self.run_script(self.scripts["linear_registration"])

    def run_nonlinear_registration(self):
        """Run the non-linear registration step"""
        self.current_step = "nonlinear_registration"
        self.update_status("Running non-linear registration...")
        self.log("Starting non-linear registration step...")
        self.run_script(self.scripts["nonlinear_registration"])


    def run_transformer(self):
        """Run the transformer step"""
        self.current_step = "transformer"
        self.update_status("Running transformer...")
        self.log("Starting transformer step...")

        args = []

        if self.binarize_masks:
            args.append("--binarize")

        if self.apply_threshold:
            args.append("--threshold")
            args.append(str(self.threshold_value))
            self.log(f"Using intensity threshold: {self.threshold_value}")

        self.run_script(self.scripts["transformer"], args)

    def run_tif_to_nifti(self):
        """Run the TIF to NIfTI converter"""
        self.current_step = "tif_to_nifti"
        self.update_status("Running TIF to NIfTI converter...")
        self.log("Starting TIF to NIfTI annotation mask conversion...")
        self.run_script(self.scripts["tif_to_nifti"])

    def run_mask_binarizer(self):
        """Run the Mask Binarizer tool"""
        self.current_step = "mask_binarizer"
        self.update_status("Running Mask Binarizer...")
        self.log("Starting Mask Binarizer...")
        self.run_script(self.scripts["mask_binarizer"],
                        args=["--working-dir", self.working_dir])

    def run_segmentation_splitter(self):
        """Run the segmentation splitter utility"""
        self.current_step = "segmentation_splitter"
        self.log("Starting segmentation splitter utility...")
        self.log(f"Using patch size: {self.patch_size} pixels")
        self.log("Opening file selection dialog for segmentation splitting...")
        self.log("Results will be saved to: Splitted_annotation_masks")
        self.run_script(self.scripts["segmentation_splitter"], ["--patchsize", str(self.patch_size)])

    # ── Batch: Configure / Run ────────────────────────────────────────────────

    def configure_transformer_batch(self):
        """Open config dialog for Transformer Batch settings."""
        json_path = os.path.join(self.configs_dir, "transformer_batch_config.json")
        dialog = TransformerBatchConfigDialog(self, json_path)
        if dialog.exec_() == QDialog.Accepted:
            self.log(f"Transformer batch config saved to {json_path}")

    def configure_tif_batch(self):
        """Open config dialog for TIF/PNG Batch subdirectories and directory types."""
        json_path = os.path.join(self.configs_dir, "tif_batch_config.json")
        dialog = TifBatchConfigDialog(self, json_path)
        if dialog.exec_() == QDialog.Accepted:
            self.log(f"TIF/PNG batch config saved to {json_path}")

    def configure_split_batch(self):
        """Open config dialog for Split Annotation Batch settings."""
        json_path = os.path.join(self.configs_dir, "split_batch_config.json")
        dialog = SplitBatchConfigDialog(self, json_path)
        if dialog.exec_() == QDialog.Accepted:
            self.log(f"Split annotation batch config saved to {json_path}")

    def open_documentation(self):
        """Open the documentation dialog."""
        dialog = DocumentationDialog(self)
        dialog.exec_()

    def open_create_folder_structure(self):
        """Open the Create Folder Structure dialog."""
        dialog = CreateFolderStructureDialog(self)
        dialog.exec_()

    def _load_batch_config(self, filename):
        """Load a batch config JSON; return {} on any error."""
        path = os.path.join(self.configs_dir, filename)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_batch_parent_dir(self, filename, parent_dir):
        """Persist last_parent_dir into a batch config JSON."""
        path = os.path.join(self.configs_dir, filename)
        data = self._load_batch_config(filename)
        data["last_parent_dir"] = parent_dir
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception:
            pass

    def _show_block_selector(self, config_filename):
        """
        Show BatchBlockSelectorDialog using last_parent_dir from config.
        Returns (dialog, selected_dirs) if accepted, or (None, []) if cancelled.
        """
        config = self._load_batch_config(config_filename)
        dialog = BatchBlockSelectorDialog(self, config.get("last_parent_dir", ""))
        if dialog.exec_() != QDialog.Accepted:
            return None, []
        selected = dialog.selected_dirs()
        if not selected:
            self.log("No directories selected.", error=True)
            return None, []
        self._save_batch_parent_dir(config_filename, dialog.parent_dir)
        return dialog, selected

    def run_transformer_batch(self):
        """Run the Transformer over configured source directories in multiple block/stain dirs."""
        if self._queue_active:
            self.log("A batch/run-all operation is already in progress.", error=True)
            return

        config = self._load_batch_config("transformer_batch_config.json")
        subdirs = config.get("subdirs", [])
        if not subdirs:
            self.log("Transformer Batch: no stain subdirs configured. Use Configure.", error=True)
            return

        transform_density    = config.get("transform_density_maps", True)
        transform_annotation = config.get("transform_annotation_masks", False)
        if not transform_density and not transform_annotation:
            self.log("Transformer Batch: neither Density_Maps nor Annotation_Masks selected. Use Configure.", error=True)
            return

        copy_results         = config.get("copy_to_density_maps", False)
        rename_by_stain_dir  = config.get("rename_by_stain_dir", False)
        use_naming_pattern   = config.get("use_naming_pattern", False)
        naming_dict          = config.get("naming_dict", [])

        _, block_dirs = self._show_block_selector("transformer_batch_config.json")
        if not block_dirs:
            return

        import time as _time

        # Closure factories --------------------------------------------------
        def _make_snapshot_func(src_dir, container):
            """Record {path: mtime} for all .nii.gz files in src_dir into container (dict)."""
            def _snap():
                import glob as _glob2
                container.clear()
                for f in _glob2.glob(os.path.join(src_dir, "*.nii.gz")):
                    container[f] = os.path.getmtime(f)
            return _snap

        def _is_new(f, snapshot):
            """True if file is absent from snapshot or was overwritten since snapshot."""
            return f not in snapshot or os.path.getmtime(f) > snapshot[f]

        def _make_density_copy_func(src, dest, pfx, stain_name, log_fn, snapshot,
                                    rename, use_pattern, nd):
            """Copy Density_Maps results (new/updated since snapshot) to block/Density_Maps."""
            def _copy():
                import shutil, re, fnmatch
                import glob as _glob2
                os.makedirs(dest, exist_ok=True)
                new_files = [f for f in _glob2.glob(os.path.join(src, "*.nii.gz"))
                             if _is_new(f, snapshot)]
                def _match(fname_, pat_):
                    if not pat_:
                        return False
                    if any(c in pat_ for c in ('*', '?', '[')):
                        return fnmatch.fnmatch(fname_.lower(), pat_.lower())
                    return pat_.lower() in fname_.lower()
                for f in new_files:
                    fname = os.path.basename(f)
                    dest_name = None
                    if use_pattern and nd:
                        for entry in nd:
                            if (entry.get("stain_dir", "").lower() == stain_name.lower() and
                                    _match(fname, entry.get("pattern", ""))):
                                new_nm = entry.get("new_name", "").strip()
                                if new_nm:
                                    dest_name = f"{pfx}{new_nm}_transformed.nii.gz"
                                break
                    if dest_name is None and (rename or use_pattern):
                        m = re.search(r'stain\d+', fname, re.IGNORECASE)
                        if m:
                            dest_name = f"{pfx}{stain_name}_{m.group(0)}_transformed.nii.gz"
                    if dest_name is None and use_pattern:
                        stem = fname
                        for ext in ('.nii.gz', '.nii'):
                            if stem.endswith(ext):
                                stem = stem[:-len(ext)]
                                break
                        if stem.endswith('_transformed'):
                            stem = stem[:-len('_transformed')]
                        dest_name = f"{pfx}{stain_name}_{stem}_transformed.nii.gz"
                    if dest_name is None:
                        dest_name = pfx + fname
                    shutil.copy2(f, os.path.join(dest, dest_name))
                log_fn(f"Copied {len(new_files)} new file(s) to {dest}")
            return _copy

        def _make_annotation_copy_func(src, dest, log_fn, snapshot, threshold_used, threshold_val):
            """Copy Annotation_Masks results (new/updated since snapshot) to block/Annotation_Masks."""
            def _copy():
                import shutil
                import glob as _glob2
                os.makedirs(dest, exist_ok=True)
                new_files = [f for f in _glob2.glob(os.path.join(src, "*.nii.gz"))
                             if _is_new(f, snapshot)]
                for f in new_files:
                    fname = os.path.basename(f)
                    stem = fname[:-len(".nii.gz")] if fname.endswith(".nii.gz") else fname
                    if not stem.endswith("_transformed"):
                        stem = stem + "_transformed"
                    dest_name = f"{stem}_{threshold_val}.nii.gz" if threshold_used else f"{stem}.nii.gz"
                    shutil.copy2(f, os.path.join(dest, dest_name))
                log_fn(f"Copied {len(new_files)} annotation mask(s) to {dest}")
            return _copy
        # --------------------------------------------------------------------

        queue = []
        for block in block_dirs:
            block_name = os.path.basename(block)
            block_number = block_name[len("block_"):] if block_name.startswith("block_") else block_name
            for subdir in subdirs:
                stain_dir = os.path.join(block, subdir)
                linear  = os.path.join(stain_dir, "Linear_registration_results", "transform_manual.pkl")
                nonlin  = os.path.join(stain_dir, "Non-linear_registration_results", "nonlinear_transform.pkl")
                deform  = os.path.join(stain_dir, "Non-linear_registration_results", "deformation_field.pkl")
                output  = os.path.join(stain_dir, "Transformation_results")

                missing_transforms = [os.path.basename(p) for p in [linear, nonlin, deform]
                                      if not os.path.exists(p)]
                if missing_transforms:
                    self.log(f"Skipping {block_name}/{subdir}: missing {missing_transforms}", error=True)
                    continue

                os.makedirs(output, exist_ok=True)
                base_args = ["--linear", linear, "--nonlinear", nonlin,
                             "--deformation", deform, "--output", output]
                if self.binarize_masks:
                    base_args.append("--binarize")
                if self.apply_threshold:
                    base_args += ["--threshold", str(self.threshold_value)]

                if transform_density:
                    masks_dir = os.path.join(stain_dir, "Density_Maps")
                    if not os.path.isdir(masks_dir):
                        self.log(f"Skipping {block_name}/{subdir}/Density_Maps: directory not found", error=True)
                    else:
                        snapshot = {}
                        queue.append({
                            "script": self.scripts["transformer"],
                            "args": ["--masks", masks_dir] + base_args,
                            "cwd": stain_dir,
                            "label": f"{block_name}/{subdir}/Density_Maps",
                            "pre_func": _make_snapshot_func(output, snapshot)
                        })
                        if copy_results:
                            dest_dir = os.path.join(block, "Density_Maps")
                            queue.append({
                                "type": "func",
                                "func": _make_density_copy_func(
                                    output, dest_dir, f"{block_number}_", subdir,
                                    self.log, snapshot, rename_by_stain_dir,
                                    use_naming_pattern, naming_dict),
                                "label": f"{block_name}/{subdir} → copy to Density_Maps"
                            })

                if transform_annotation:
                    masks_dir = os.path.join(stain_dir, "Annotation_Masks")
                    if not os.path.isdir(masks_dir):
                        self.log(f"Skipping {block_name}/{subdir}/Annotation_Masks: directory not found", error=True)
                    else:
                        snapshot = {}
                        queue.append({
                            "script": self.scripts["transformer"],
                            "args": ["--masks", masks_dir] + base_args,
                            "cwd": stain_dir,
                            "label": f"{block_name}/{subdir}/Annotation_Masks",
                            "pre_func": _make_snapshot_func(output, snapshot)
                        })
                        if copy_results:
                            dest_dir = os.path.join(block, "Annotation_Masks")
                            queue.append({
                                "type": "func",
                                "func": _make_annotation_copy_func(
                                    output, dest_dir, self.log, snapshot,
                                    self.apply_threshold, self.threshold_value),
                                "label": f"{block_name}/{subdir} → copy to Annotation_Masks"
                            })

        if not queue:
            self.log("Transformer Batch: no valid directories found to process.", error=True)
            return
        self._start_script_queue(queue, "Transformer Batch")

    def run_tif_to_nifti_batch(self):
        """Run TIF/PNG→NIfTI conversion over selected directory types in multiple block/stain dirs."""
        if self._queue_active:
            self.log("A batch/run-all operation is already in progress.", error=True)
            return

        config = self._load_batch_config("tif_batch_config.json")
        subdirs = config.get("subdirs", [])
        if not subdirs:
            self.log("TIF/PNG Batch: no stain subdirs configured. Use Configure.", error=True)
            return

        convert_density = config.get("convert_density_maps", True)
        convert_annotations = config.get("convert_annotation_masks", False)
        if not convert_density and not convert_annotations:
            self.log("TIF/PNG Batch: neither Density_Maps nor Annotation_Masks selected. Use Configure.", error=True)
            return

        _, block_dirs = self._show_block_selector("tif_batch_config.json")
        if not block_dirs:
            return

        import glob as _glob
        queue = []
        for block in block_dirs:
            block_name = os.path.basename(block)
            for subdir in subdirs:
                stain_dir = os.path.join(block, subdir)
                match_results = os.path.join(stain_dir, "Match_slice_results")
                matching_jsons = _glob.glob(os.path.join(match_results, "*matching_info.json"))
                if not matching_jsons:
                    self.log(f"Skipping {block_name}/{subdir}: no matching_info.json found", error=True)
                    continue

                if convert_density:
                    density_maps = os.path.join(stain_dir, "Density_Maps")
                    if os.path.isdir(density_maps):
                        queue.append({
                            "script": self.scripts["tif_to_nifti"],
                            "args": ["--annotations_dir", density_maps],
                            "cwd": stain_dir,
                            "label": f"{block_name}/{subdir}/Density_Maps"
                        })
                    else:
                        self.log(f"Skipping {block_name}/{subdir}: Density_Maps not found", error=True)

                if convert_annotations:
                    annotation_masks = os.path.join(stain_dir, "Annotation_Masks")
                    if os.path.isdir(annotation_masks):
                        queue.append({
                            "script": self.scripts["tif_to_nifti"],
                            "args": ["--annotations_dir", annotation_masks],
                            "cwd": stain_dir,
                            "label": f"{block_name}/{subdir}/Annotation_Masks"
                        })
                    else:
                        self.log(f"Skipping {block_name}/{subdir}: Annotation_Masks not found", error=True)

        if not queue:
            self.log("TIF/PNG Batch: no valid directories found to process.", error=True)
            return
        self._start_script_queue(queue, "TIF/PNG Batch")

    def run_segmentation_splitter_batch(self):
        """Run Split Annotation Masks over Annotation_Masks in multiple block directories."""
        if self._queue_active:
            self.log("A batch/run-all operation is already in progress.", error=True)
            return

        config = self._load_batch_config("split_batch_config.json")
        filenames = config.get("filenames", [])
        if not filenames:
            self.log("Split Batch: no mask filenames configured. Use Configure.", error=True)
            return

        copy_to_annotation_masks = config.get("copy_to_annotation_masks", False)

        _, block_dirs = self._show_block_selector("split_batch_config.json")
        if not block_dirs:
            return

        queue = []
        for block in block_dirs:
            block_name = os.path.basename(block)
            ann_masks_dir = os.path.join(block, "Annotation_Masks")
            if not os.path.isdir(ann_masks_dir):
                self.log(f"Skipping {block_name}: Annotation_Masks/ not found", error=True)
                continue

            files_to_process = [
                os.path.join(ann_masks_dir, fname) for fname in filenames
                if os.path.isfile(os.path.join(ann_masks_dir, fname))
            ]
            missing = [f for f in filenames
                       if not os.path.isfile(os.path.join(ann_masks_dir, f))]
            if missing:
                self.log(f"{block_name}: skipping missing files: {missing}", error=True)

            if not files_to_process:
                continue

            split_output_dir = os.path.join(ann_masks_dir, "Split_Annotation_Masks")

            queue.append({
                "script": self.scripts["segmentation_splitter"],
                "args": ["--files", ",".join(files_to_process),
                         "--patchsize", str(self.patch_size)],
                "cwd": ann_masks_dir,
                "label": f"{block_name} ({len(files_to_process)} file(s))"
            })

            if copy_to_annotation_masks:
                # Build expected output filenames from input filenames + patchsize
                from pathlib import Path as _Path
                expected_names = [
                    f"{_Path(f).stem}_kmeans_split_patchsize_{self.patch_size}.nii.gz"
                    for f in files_to_process
                ]
                def _make_split_copy_func(src, dest, log_fn, names):
                    def _copy():
                        import shutil
                        copied = 0
                        for name in names:
                            src_file = os.path.join(src, name)
                            if os.path.isfile(src_file):
                                shutil.copy2(src_file, os.path.join(dest, name))
                                copied += 1
                            else:
                                log_fn(f"Warning: expected split output not found: {name}")
                        log_fn(f"Copied {copied} split mask(s) to {dest}")
                    return _copy
                queue.append({
                    "type": "func",
                    "func": _make_split_copy_func(split_output_dir, ann_masks_dir, self.log, expected_names),
                    "label": f"{block_name} → copy to Annotation_Masks"
                })

        if not queue:
            self.log("Split Batch: no valid files found to process.", error=True)
            return
        self._start_script_queue(queue, "Split Batch")

    def _start_script_queue(self, items, queue_name="Batch"):
        """Start a sequential queue of script items: [{script, args, cwd, label}, ...]."""
        self._script_queue = list(items)
        self._queue_active = True
        self._queue_name = queue_name
        self.log(f"{queue_name}: starting {len(items)} script(s)...")
        self._run_next_in_queue()

    def _run_next_in_queue(self):
        """Pop and run the next item from the script queue."""
        if not self._script_queue:
            self._queue_active = False
            self.log(f"{self._queue_name} completed successfully.")
            self.update_status(f"{self._queue_name} completed successfully")
            self.enable_buttons(True)
            return

        item = self._script_queue.pop(0)
        self.current_step = item.get("label", "")
        self.log(f"{self._queue_name}: {item['label']}...")
        self.update_status(f"{self._queue_name}: {item['label']}...")
        self.enable_buttons(False)

        if item.get("type") == "func":
            # Run a Python callable synchronously then advance the queue
            try:
                item["func"]()
            except Exception as e:
                self.log(f"Error in {item['label']}: {e}", error=True)
                self._queue_active = False
                self.enable_buttons(True)
                self.update_status(f"{self._queue_name} failed: {item['label']}")
                return
            self._run_next_in_queue()
            return

        if "pre_func" in item:
            try:
                item["pre_func"]()
            except Exception as e:
                self.log(f"Pre-run hook failed for {item['label']}: {e}", error=True)

        cwd = item.get("cwd", self.working_dir)
        self.running_thread = ScriptRunner(item["script"], item.get("args", []), cwd=cwd)
        self.running_thread.output_signal.connect(lambda msg: self.log(msg))
        self.running_thread.error_signal.connect(lambda msg: self.log(msg, error=True))
        self.running_thread.finished_signal.connect(self.on_script_finished)
        self.running_thread.start()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = RegistrationPipeline()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
