#!/usr/bin/env python3
"""
Manual Linear Registration Script - PyQt5 Version
===================================================
Interactive GUI for manually applying translation, rotation, and scaling
to align histology images with MRI reference slices.

Author: Registration Pipeline
Date: 2024
"""

import os
import sys
import json
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import pickle
from datetime import datetime
from scipy.ndimage import affine_transform

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QFileDialog, QMessageBox,
    QGroupBox, QCheckBox, QDoubleSpinBox, QFrame, QSizePolicy,
    QDialog, QSpinBox, QLineEdit
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

# PyQtGraph for contrast curve dialog
import pyqtgraph as pg
pg.setConfigOptions(antialias=False)


class ContrastCurveDialog(QDialog):
    """Dialog for curve-based contrast adjustment with histogram."""

    contrast_changed = pyqtSignal(object)  # Emits the lookup table

    def __init__(self, image_data, parent=None, initial_points=None):
        super().__init__(parent)
        self.setWindowTitle("MRI Contrast Adjustment")
        self.setGeometry(200, 200, 700, 550)

        self.image_data = image_data
        self.original_data = image_data.copy()

        # Control points: list of (x, y) where x is intensity, y is output (0-1)
        if initial_points is not None:
            self.control_points = list(initial_points)
        else:
            mid_x = image_data.max() / 2
            self.control_points = [(0, 0), (mid_x, 0.5), (image_data.max(), 1)]

        self.selected_point_idx = 1  # Middle point selected by default
        self.dragging = False
        self.drag_point_idx = None

        self.setup_ui()
        self.update_histogram()
        self.update_curve()

    def setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)

        # PyQtGraph widget for histogram and curve
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('bottom', 'Image Intensity')
        self.plot_widget.setLabel('left', 'Output Level')
        self.plot_widget.setYRange(0, 1.1)
        self.plot_widget.setMouseEnabled(x=False, y=False)

        # Histogram bars
        self.histogram_item = pg.BarGraphItem(x=[], height=[], width=1, brush='b')
        self.plot_widget.addItem(self.histogram_item)

        # Curve line
        self.curve_line = pg.PlotDataItem(pen=pg.mkPen('r', width=2))
        self.plot_widget.addItem(self.curve_line)

        # Control points scatter
        self.points_scatter = pg.ScatterPlotItem(
            size=20, pen=pg.mkPen('k', width=2), brush=pg.mkBrush('y'),
            symbol='o'
        )
        self.plot_widget.addItem(self.points_scatter)

        # Connect mouse events
        self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_moved)
        self.plot_widget.installEventFilter(self)

        layout.addWidget(self.plot_widget)

        # Control point editing section
        control_frame = QFrame()
        control_frame.setFrameStyle(QFrame.StyledPanel)
        control_layout = QHBoxLayout(control_frame)

        control_layout.addWidget(QLabel("Selected point:"))

        control_layout.addWidget(QLabel("Id:"))
        self.id_spin = QSpinBox()
        self.id_spin.setMinimum(1)
        self.id_spin.setMaximum(len(self.control_points))
        self.id_spin.setValue(2)
        self.id_spin.valueChanged.connect(self.on_id_changed)
        control_layout.addWidget(self.id_spin)

        control_layout.addWidget(QLabel("x:"))
        self.x_input = QLineEdit()
        self.x_input.setMaximumWidth(100)
        self.x_input.editingFinished.connect(self.on_xy_edited)
        control_layout.addWidget(self.x_input)

        control_layout.addWidget(QLabel("y:"))
        self.y_input = QLineEdit()
        self.y_input.setMaximumWidth(100)
        self.y_input.editingFinished.connect(self.on_xy_edited)
        control_layout.addWidget(self.y_input)

        control_layout.addStretch()

        # Add/Remove buttons
        self.btn_add = QPushButton("+")
        self.btn_add.setMaximumWidth(40)
        self.btn_add.clicked.connect(self.add_point)
        control_layout.addWidget(self.btn_add)

        self.btn_remove = QPushButton("-")
        self.btn_remove.setMaximumWidth(40)
        self.btn_remove.clicked.connect(self.remove_point)
        control_layout.addWidget(self.btn_remove)

        layout.addWidget(control_frame)

        # Buttons row
        btn_layout = QHBoxLayout()

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset_curve)
        btn_layout.addWidget(self.btn_reset)

        btn_layout.addStretch()

        self.btn_apply = QPushButton("Apply")
        self.btn_apply.clicked.connect(self.apply_and_close)
        btn_layout.addWidget(self.btn_apply)

        layout.addLayout(btn_layout)

        self.update_point_display()

    def update_histogram(self):
        """Update histogram display."""
        data_flat = self.original_data.flatten()
        data_flat = data_flat[data_flat > 0]

        if len(data_flat) == 0:
            return

        num_bins = 100
        hist, bin_edges = np.histogram(data_flat, bins=num_bins)
        hist_normalized = hist / hist.max() * 0.3

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]

        self.histogram_item.setOpts(x=bin_centers, height=hist_normalized, width=bin_width)
        self.plot_widget.setXRange(0, self.original_data.max() * 1.05)

    def update_curve(self):
        """Update the curve display based on control points."""
        if len(self.control_points) < 2:
            return

        self.control_points.sort(key=lambda p: p[0])

        x_pts = [p[0] for p in self.control_points]
        y_pts = [p[1] for p in self.control_points]
        self.points_scatter.setData(x_pts, y_pts)

        x_curve = np.linspace(0, self.original_data.max(), 256)
        y_curve = np.interp(x_curve, x_pts, y_pts)
        y_curve = np.clip(y_curve, 0, 1)
        self.curve_line.setData(x_curve, y_curve)

    def update_point_display(self):
        """Update the selected point display."""
        if 0 <= self.selected_point_idx < len(self.control_points):
            pt = self.control_points[self.selected_point_idx]
            self.id_spin.blockSignals(True)
            self.id_spin.setMaximum(len(self.control_points))
            self.id_spin.setValue(self.selected_point_idx + 1)
            self.id_spin.blockSignals(False)
            self.x_input.setText(f"{pt[0]:.1f}")
            self.y_input.setText(f"{pt[1]:.3f}")

    def on_id_changed(self, value):
        """Handle ID spinner change."""
        self.selected_point_idx = value - 1
        self.update_point_display()

    def on_xy_edited(self):
        """Handle x/y input editing."""
        try:
            x = float(self.x_input.text())
            y = float(self.y_input.text())
            y = max(0, min(1, y))
            x = max(0, x)

            if 0 <= self.selected_point_idx < len(self.control_points):
                self.control_points[self.selected_point_idx] = (x, y)
                self.update_curve()
                self.apply_contrast()
        except ValueError:
            pass

    def eventFilter(self, obj, event):
        """Event filter to handle mouse click on plot widget."""
        if obj == self.plot_widget:
            if event.type() == event.MouseButtonPress and event.button() == Qt.LeftButton:
                pos = event.pos()
                scene_pos = self.plot_widget.mapToScene(pos)
                if self.plot_widget.sceneBoundingRect().contains(scene_pos):
                    mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(scene_pos)
                    x, y = mouse_point.x(), mouse_point.y()

                    if self.dragging:
                        self.dragging = False
                        self.drag_point_idx = None
                        return True

                    for i, (px, py) in enumerate(self.control_points):
                        x_range = self.plot_widget.viewRange()[0]
                        y_range = self.plot_widget.viewRange()[1]
                        dx = (x - px) / (x_range[1] - x_range[0]) if x_range[1] != x_range[0] else 0
                        dy = (y - py) / (y_range[1] - y_range[0]) if y_range[1] != y_range[0] else 0
                        dist = np.sqrt(dx**2 + dy**2)

                        if dist < 0.08:
                            self.dragging = True
                            self.drag_point_idx = i
                            self.selected_point_idx = i
                            self.update_point_display()
                            return True

        return super().eventFilter(obj, event)

    def on_mouse_moved(self, pos):
        """Handle mouse movement for dragging."""
        if self.dragging and self.drag_point_idx is not None and self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()

            x = max(0, min(self.original_data.max(), x))
            y = max(0, min(1, y))

            if self.drag_point_idx == 0:
                x = 0
            elif self.drag_point_idx == len(self.control_points) - 1:
                x = self.original_data.max()

            self.control_points[self.drag_point_idx] = (x, y)
            self.selected_point_idx = self.drag_point_idx
            self.update_curve()
            self.update_point_display()
            self.apply_contrast()

    def reset_curve(self):
        """Reset to linear curve with 3 points."""
        mid_x = self.original_data.max() / 2
        self.control_points = [(0, 0), (mid_x, 0.5), (self.original_data.max(), 1)]
        self.selected_point_idx = 1
        self.id_spin.setMaximum(3)
        self.update_curve()
        self.update_point_display()
        self.apply_contrast()

    def add_point(self):
        """Add a new control point."""
        if len(self.control_points) >= 2:
            mid_x = self.original_data.max() / 2
            x_pts = [p[0] for p in self.control_points]
            y_pts = [p[1] for p in self.control_points]
            mid_y = np.interp(mid_x, x_pts, y_pts)

            self.control_points.append((mid_x, mid_y))
            self.control_points.sort(key=lambda p: p[0])
            self.selected_point_idx = self.control_points.index((mid_x, mid_y))
            self.id_spin.setMaximum(len(self.control_points))

            self.update_curve()
            self.update_point_display()

    def remove_point(self):
        """Remove selected control point."""
        if len(self.control_points) > 2:
            if 0 < self.selected_point_idx < len(self.control_points) - 1:
                del self.control_points[self.selected_point_idx]
                self.selected_point_idx = min(self.selected_point_idx, len(self.control_points) - 1)
                self.id_spin.setMaximum(len(self.control_points))
                self.update_curve()
                self.update_point_display()
                self.apply_contrast()

    def apply_contrast(self):
        """Apply the contrast curve and emit signal."""
        x_pts = [p[0] for p in self.control_points]
        y_pts = [p[1] for p in self.control_points]

        def apply_curve(data):
            result = np.interp(data, x_pts, y_pts)
            return result

        self.contrast_changed.emit(apply_curve)

    def apply_and_close(self):
        """Apply the contrast curve and close the dialog."""
        self.apply_contrast()
        self.close()

    def get_control_points(self):
        """Return the current control points for saving."""
        return list(self.control_points)


class InteractiveImageLabel(QLabel):
    """Custom QLabel that supports mouse interaction for image manipulation.

    - Left-click drag: Pan/translate the IHC image
    - Right-click drag: Rotate the IHC image
    - Mouse wheel: Scale the IHC image
    """

    # Signals to communicate transform changes to parent
    translate_changed = pyqtSignal(float, float)  # dx, dy
    rotate_changed = pyqtSignal(float)  # delta_degrees
    scale_changed = pyqtSignal(float)  # scale_factor

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        # Mouse interaction state
        self._dragging = False
        self._drag_button = None
        self._last_pos = None

        # Sensitivity settings
        self._translate_sensitivity = 1.0  # pixels per mouse pixel
        self._rotate_sensitivity = 0.5  # degrees per mouse pixel
        self._scale_sensitivity = 0.002  # scale factor per mouse pixel or wheel delta

    def mousePressEvent(self, event):
        """Handle mouse button press."""
        if event.button() in (Qt.LeftButton, Qt.RightButton):
            self._dragging = True
            self._drag_button = event.button()
            self._last_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse button release."""
        if event.button() == self._drag_button:
            self._dragging = False
            self._drag_button = None
            self._last_pos = None
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse movement for dragging."""
        if self._dragging and self._last_pos is not None:
            delta = event.pos() - self._last_pos
            dx = delta.x()
            dy = delta.y()

            if self._drag_button == Qt.LeftButton:
                # Left drag: translate/pan
                self.translate_changed.emit(
                    dx * self._translate_sensitivity,
                    dy * self._translate_sensitivity
                )
            elif self._drag_button == Qt.RightButton:
                # Right drag: rotate (use horizontal movement)
                self.rotate_changed.emit(dx * self._rotate_sensitivity)

            self._last_pos = event.pos()
        super().mouseMoveEvent(event)

    def wheelEvent(self, event):
        """Handle mouse wheel for scaling."""
        # Get wheel delta (positive = scroll up = zoom in)
        delta = event.angleDelta().y()

        # Convert to scale factor change
        scale_delta = delta * self._scale_sensitivity * 0.1
        self.scale_changed.emit(scale_delta)

        event.accept()


class ManualLinearRegistration(QMainWindow):
    """Interactive GUI for manual 2D affine registration using PyQt5."""

    def __init__(self):
        super().__init__()

        # File paths
        self.ihc_path = None
        self.mri_path = None

        # Image data
        self.ihc_nifti = None
        self.mri_nifti = None
        self.ihc_slice = None
        self.mri_slice = None
        self.original_ihc_slice = None

        # Normalized slices for display
        self.mri_slice_norm = None
        self.ihc_slice_norm = None

        # Slice information
        self.slice_dim = None
        self.slice_idx = None

        # Transform parameters
        self.translation_x = 0.0
        self.translation_y = 0.0
        self.rotation_deg = 0.0
        self.scale_x = 1.0
        self.scale_y = 1.0

        # Output directory
        self.output_dir = "Linear_registration_results"

        # Contrast adjustment
        self.contrast_dialog = None
        self.contrast_control_points = None
        self.contrast_func = None

        # Opacity for overlay
        self.opacity = 0.5

        # Uniform scale flag
        self.uniform_scale = True

    def find_input_files(self, start_dir=None):
        """Automatically detect input files from standard directory structure."""
        if start_dir is None:
            start_dir = os.getcwd()

        ihc_path = None
        mri_path = None

        # Look for Match_slice_results folder
        match_dir = os.path.join(start_dir, "Match_slice_results")
        if os.path.exists(match_dir):
            for f in os.listdir(match_dir):
                if f.startswith('._'):
                    continue
                if f.endswith("_in_block.nii.gz"):
                    ihc_path = os.path.join(match_dir, f)
                    break

        # Look for MRI file
        for f in os.listdir(start_dir):
            if f.startswith('._'):
                continue
            if f.endswith(('.nii', '.nii.gz')):
                if 'FLASH' in f.upper():
                    mri_path = os.path.join(start_dir, f)
                    break

        if mri_path is None:
            for f in os.listdir(start_dir):
                if f.startswith('._'):
                    continue
                if f.endswith(('.nii', '.nii.gz')) and 'result' not in f.lower():
                    mri_path = os.path.join(start_dir, f)
                    break

        return ihc_path, mri_path

    def load_matching_info(self, start_dir=None):
        """Load matching info from JSON file including MRI path and contrast settings."""
        if start_dir is None:
            start_dir = os.getcwd()

        match_dir = os.path.join(start_dir, "Match_slice_results")
        if not os.path.exists(match_dir):
            print(f"Match_slice_results directory not found at: {match_dir}")
            return None

        # Find matching info JSON
        for f in os.listdir(match_dir):
            if f.startswith('._'):
                continue
            if f.endswith("_matching_info.json"):
                json_path = os.path.join(match_dir, f)
                print(f"Found matching info JSON: {json_path}")
                try:
                    with open(json_path, 'r') as jf:
                        data = json.load(jf)
                        print(f"JSON keys: {list(data.keys())}")
                        return data
                except Exception as e:
                    print(f"Warning: Could not load matching info: {e}")
                    import traceback
                    traceback.print_exc()
                    return None

        print("No matching info JSON file found")
        return None

    def get_mri_path_from_json(self, matching_info):
        """Extract MRI file path from matching info."""
        if matching_info and 'mri_file' in matching_info:
            mri_path = matching_info['mri_file']
            print(f"MRI path from JSON: {mri_path}")
            if os.path.exists(mri_path):
                return mri_path
            # Stored path is from a different OS/machine — try relative to working dir
            relative_candidate = os.path.join(os.getcwd(), os.path.basename(mri_path))
            if os.path.exists(relative_candidate):
                print(f"Resolved MRI path relative to working directory: {relative_candidate}")
                return relative_candidate
            print(f"Warning: MRI file from JSON does not exist: {mri_path}")
        return None

    def get_contrast_settings_from_json(self, matching_info):
        """Extract contrast curve settings from matching info."""
        if matching_info and 'contrast_curve' in matching_info:
            print("Found contrast_curve in JSON")
            if 'control_points' in matching_info['contrast_curve']:
                points = matching_info['contrast_curve']['control_points']
                print(f"Loaded {len(points)} control points: {points}")
                return [(float(p[0]), float(p[1])) for p in points]
            else:
                print("No 'control_points' key in contrast_curve")
        else:
            print("No 'contrast_curve' key in JSON - contrast was not adjusted during slice matching")
        return None

    def load_images(self, ihc_path, mri_path):
        """Load NIfTI images and extract relevant slices."""
        print(f"Loading IHC: {ihc_path}")
        print(f"Loading MRI: {mri_path}")

        self.ihc_path = ihc_path
        self.mri_path = mri_path

        self.ihc_nifti = nib.load(ihc_path)
        self.mri_nifti = nib.load(mri_path)

        ihc_data = self.ihc_nifti.get_fdata()
        mri_data = self.mri_nifti.get_fdata()

        self.slice_dim, self.slice_idx = self._find_nonempty_slice(ihc_data)

        if self.slice_dim is None:
            raise ValueError("Could not find non-empty slice in IHC image")

        print(f"Found IHC data on dimension {self.slice_dim}, slice {self.slice_idx}")

        if self.slice_dim == 0:
            self.ihc_slice = ihc_data[self.slice_idx, :, :].T
            self.mri_slice = mri_data[self.slice_idx, :, :].T
        elif self.slice_dim == 1:
            self.ihc_slice = ihc_data[:, self.slice_idx, :].T
            self.mri_slice = mri_data[:, self.slice_idx, :].T
        else:
            self.ihc_slice = ihc_data[:, :, self.slice_idx].T
            self.mri_slice = mri_data[:, :, self.slice_idx].T

        print(f"IHC slice shape: {self.ihc_slice.shape}")
        print(f"MRI slice shape: {self.mri_slice.shape}")

        # Check if shapes match
        if self.ihc_slice.shape != self.mri_slice.shape:
            raise ValueError(
                f"IHC and MR Image have different shapes!\n"
                f"IHC: {self.ihc_slice.shape}, MRI: {self.mri_slice.shape}\n"
                f"Please ensure both images have matching dimensions."
            )

        self.original_ihc_slice = self.ihc_slice.copy()

        self.mri_slice_norm = self._normalize(self.mri_slice)
        self.ihc_slice_norm = self._normalize(self.ihc_slice)

    def _find_nonempty_slice(self, data):
        """Find the dimension and index of the slice with most non-zero pixels."""
        best_slice = None
        best_dim = None
        max_nonzero = 0

        for dim in range(3):
            for idx in range(data.shape[dim]):
                if dim == 0:
                    slice_data = data[idx, :, :]
                elif dim == 1:
                    slice_data = data[:, idx, :]
                else:
                    slice_data = data[:, :, idx]

                nonzero_count = np.count_nonzero(slice_data)

                if nonzero_count > max_nonzero:
                    max_nonzero = nonzero_count
                    best_slice = idx
                    best_dim = dim

        if max_nonzero > 0:
            print(f"  Found slice with {max_nonzero} non-zero pixels")
            return best_dim, best_slice

        return None, None

    def _normalize(self, img):
        """Normalize image to 0-1 range."""
        img = img.astype(np.float64)
        if img.max() > img.min():
            return (img - img.min()) / (img.max() - img.min())
        return img

    def get_transform_matrix(self):
        """Build 2D affine transformation matrix from current parameters."""
        h, w = self.original_ihc_slice.shape
        cx, cy = w / 2, h / 2

        theta = np.radians(self.rotation_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        T1 = np.array([
            [1, 0, -cx],
            [0, 1, -cy],
            [0, 0, 1]
        ])

        S = np.array([
            [self.scale_x, 0, 0],
            [0, self.scale_y, 0],
            [0, 0, 1]
        ])

        R = np.array([
            [cos_t, -sin_t, 0],
            [sin_t, cos_t, 0],
            [0, 0, 1]
        ])

        T2 = np.array([
            [1, 0, cx + self.translation_x],
            [0, 1, cy + self.translation_y],
            [0, 0, 1]
        ])

        M = T2 @ R @ S @ T1
        return M

    def apply_transform(self):
        """Apply current transformation to IHC slice."""
        M = self.get_transform_matrix()
        M_inv = np.linalg.inv(M)

        matrix = M_inv[:2, :2]
        offset = M_inv[:2, 2]

        transformed = affine_transform(
            self.original_ihc_slice,
            matrix,
            offset=offset,
            order=1,
            mode='constant',
            cval=0
        )

        self.ihc_slice = transformed
        self.ihc_slice_norm = self._normalize(transformed)

        return transformed

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Manual Linear Registration (PyQt5)")
        self.setMinimumSize(1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Left panel - Controls
        control_panel = QWidget()
        control_panel.setMaximumWidth(420)
        control_layout = QVBoxLayout(control_panel)

        # Transform controls group
        transform_group = QGroupBox("Transform Controls")
        transform_layout = QVBoxLayout(transform_group)

        # Translation X
        self.create_slider_control(transform_layout, "Translation X:",
                                   -200, 200, 0, 1, self.on_trans_x_changed)

        # Translation Y
        self.create_slider_control(transform_layout, "Translation Y:",
                                   -200, 200, 0, 1, self.on_trans_y_changed, attr='trans_y')

        # Rotation
        self.create_slider_control(transform_layout, "Rotation (degrees):",
                                   -180, 180, 0, 0.5, self.on_rotation_changed, attr='rotation')

        # Scale X
        self.create_scale_slider(transform_layout, "Scale X:",
                                 0.5, 2.0, 1.0, 0.01, self.on_scale_x_changed, attr='scale_x')

        # Scale Y
        self.create_scale_slider(transform_layout, "Scale Y:",
                                 0.5, 2.0, 1.0, 0.01, self.on_scale_y_changed, attr='scale_y')

        # Uniform scale checkbox
        self.uniform_check = QCheckBox("Uniform Scale")
        self.uniform_check.setChecked(True)
        self.uniform_check.stateChanged.connect(self.on_uniform_changed)
        transform_layout.addWidget(self.uniform_check)

        control_layout.addWidget(transform_group)

        # Display controls group
        display_group = QGroupBox("Display Controls")
        display_layout = QVBoxLayout(display_group)

        # Opacity slider
        display_layout.addWidget(QLabel("Overlay Opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        display_layout.addWidget(self.opacity_slider)

        # Change Contrast button
        self.contrast_button = QPushButton("Change Contrast")
        self.contrast_button.setStyleSheet("background-color: #e0e0ff;")
        self.contrast_button.clicked.connect(self.show_contrast_dialog)
        display_layout.addWidget(self.contrast_button)

        control_layout.addWidget(display_group)

        # Action buttons
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout(action_group)

        reset_btn = QPushButton("Reset Transform")
        reset_btn.clicked.connect(self.reset_transform)
        action_layout.addWidget(reset_btn)

        apply_btn = QPushButton("Apply && Save")
        apply_btn.setStyleSheet("background-color: #90EE90; font-weight: bold;")
        apply_btn.clicked.connect(self.apply_and_save)
        action_layout.addWidget(apply_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.cancel)
        action_layout.addWidget(cancel_btn)

        control_layout.addWidget(action_group)
        control_layout.addStretch()

        main_layout.addWidget(control_panel)

        # Right panel - Visualization
        viz_panel = QWidget()
        viz_layout = QVBoxLayout(viz_panel)

        # Use interactive image label for mouse-based manipulation
        self.image_label = InteractiveImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("border: 1px solid gray;")

        # Connect mouse interaction signals
        self.image_label.translate_changed.connect(self.on_mouse_translate)
        self.image_label.rotate_changed.connect(self.on_mouse_rotate)
        self.image_label.scale_changed.connect(self.on_mouse_scale)

        viz_layout.addWidget(self.image_label)

        # Info label
        self.info_label = QLabel("Transform: Tx=0.0, Ty=0.0, R=0.0, Sx=1.00, Sy=1.00")
        self.info_label.setAlignment(Qt.AlignCenter)
        viz_layout.addWidget(self.info_label)

        # Mouse controls help text
        help_label = QLabel("Mouse: Left-drag=Pan | Right-drag=Rotate | Scroll wheel=Scale")
        help_label.setAlignment(Qt.AlignCenter)
        help_label.setStyleSheet("color: #666666; font-size: 11px;")
        viz_layout.addWidget(help_label)

        main_layout.addWidget(viz_panel, stretch=1)

    def create_slider_control(self, parent_layout, label, min_val, max_val, default, step, callback, attr='trans_x'):
        """Create a slider with +/- nudge buttons and textbox for value display and direct input."""
        parent_layout.addWidget(QLabel(label))

        row = QHBoxLayout()

        # Nudge step size (small increment for fine adjustment)
        nudge_step = 0.5  # 0.5 pixel nudge for translation

        # Minus button for small decrements
        minus_btn = QPushButton("-")
        minus_btn.setFixedWidth(30)
        minus_btn.setToolTip(f"Decrease by {nudge_step}")
        row.addWidget(minus_btn)

        # Slider - wider with minimum width
        slider = QSlider(Qt.Horizontal)
        slider.setMinimumWidth(180)
        # Use integer steps internally, multiply by 10 for 0.1 precision
        slider.setRange(int(min_val * 10), int(max_val * 10))
        slider.setValue(int(default * 10))
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(int((max_val - min_val) * 10 / 10))
        row.addWidget(slider, stretch=1)

        # Plus button for small increments
        plus_btn = QPushButton("+")
        plus_btn.setFixedWidth(30)
        plus_btn.setToolTip(f"Increase by {nudge_step}")
        row.addWidget(plus_btn)

        # Textbox for value display and direct input
        textbox = QLineEdit()
        textbox.setMaximumWidth(70)
        textbox.setText(f"{default:.1f}")
        textbox.setAlignment(Qt.AlignCenter)
        row.addWidget(textbox)

        # Store references
        setattr(self, f'{attr}_slider', slider)
        setattr(self, f'{attr}_textbox', textbox)

        # Connect slider to update textbox and trigger callback
        def on_slider_change(value):
            actual_value = value / 10.0
            textbox.blockSignals(True)
            textbox.setText(f"{actual_value:.1f}")
            textbox.blockSignals(False)
            callback(actual_value)

        slider.valueChanged.connect(on_slider_change)

        # Connect textbox to update slider
        def on_textbox_change():
            try:
                value = float(textbox.text())
                value = max(min_val, min(max_val, value))
                slider.blockSignals(True)
                slider.setValue(int(value * 10))
                slider.blockSignals(False)
                textbox.setText(f"{value:.1f}")
                callback(value)
            except ValueError:
                pass

        textbox.editingFinished.connect(on_textbox_change)

        # Connect nudge buttons
        def nudge_minus():
            current = slider.value() / 10.0
            new_val = max(min_val, current - nudge_step)
            slider.setValue(int(new_val * 10))

        def nudge_plus():
            current = slider.value() / 10.0
            new_val = min(max_val, current + nudge_step)
            slider.setValue(int(new_val * 10))

        minus_btn.clicked.connect(nudge_minus)
        plus_btn.clicked.connect(nudge_plus)

        parent_layout.addLayout(row)

    def create_scale_slider(self, parent_layout, label, min_val, max_val, default, step, callback, attr='scale_x'):
        """Create a scale slider with +/- nudge buttons and textbox for value display and direct input."""
        parent_layout.addWidget(QLabel(label))

        row = QHBoxLayout()

        # Nudge step size (small increment for fine adjustment)
        nudge_step = 0.01  # 1% scale nudge

        # Minus button for small decrements
        minus_btn = QPushButton("-")
        minus_btn.setFixedWidth(30)
        minus_btn.setToolTip(f"Decrease by {nudge_step}")
        row.addWidget(minus_btn)

        # Slider - wider with minimum width, use 100x multiplier for 0.01 precision
        slider = QSlider(Qt.Horizontal)
        slider.setMinimumWidth(180)
        slider.setRange(int(min_val * 100), int(max_val * 100))
        slider.setValue(int(default * 100))
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(int((max_val - min_val) * 100 / 10))
        row.addWidget(slider, stretch=1)

        # Plus button for small increments
        plus_btn = QPushButton("+")
        plus_btn.setFixedWidth(30)
        plus_btn.setToolTip(f"Increase by {nudge_step}")
        row.addWidget(plus_btn)

        # Textbox for value display and direct input
        textbox = QLineEdit()
        textbox.setMaximumWidth(70)
        textbox.setText(f"{default:.2f}")
        textbox.setAlignment(Qt.AlignCenter)
        row.addWidget(textbox)

        # Store references
        setattr(self, f'{attr}_slider', slider)
        setattr(self, f'{attr}_textbox', textbox)

        # Connect slider to update textbox and trigger callback
        def on_slider_change(value):
            actual_value = value / 100.0
            textbox.blockSignals(True)
            textbox.setText(f"{actual_value:.2f}")
            textbox.blockSignals(False)
            callback(actual_value)

        slider.valueChanged.connect(on_slider_change)

        # Connect textbox to update slider
        def on_textbox_change():
            try:
                value = float(textbox.text())
                value = max(min_val, min(max_val, value))
                slider.blockSignals(True)
                slider.setValue(int(value * 100))
                slider.blockSignals(False)
                textbox.setText(f"{value:.2f}")
                callback(value)
            except ValueError:
                pass

        textbox.editingFinished.connect(on_textbox_change)

        # Connect nudge buttons
        def nudge_minus():
            current = slider.value() / 100.0
            new_val = max(min_val, current - nudge_step)
            slider.setValue(int(new_val * 100))

        def nudge_plus():
            current = slider.value() / 100.0
            new_val = min(max_val, current + nudge_step)
            slider.setValue(int(new_val * 100))

        minus_btn.clicked.connect(nudge_minus)
        plus_btn.clicked.connect(nudge_plus)

        parent_layout.addLayout(row)

    def on_trans_x_changed(self, value):
        self.translation_x = value
        self.update_transform()

    def on_trans_y_changed(self, value):
        self.translation_y = value
        self.update_transform()

    def on_rotation_changed(self, value):
        self.rotation_deg = value
        self.update_transform()

    def on_scale_x_changed(self, value):
        self.scale_x = value
        if self.uniform_scale:
            self.scale_y = value
            self.scale_y_slider.blockSignals(True)
            self.scale_y_slider.setValue(int(value * 100))
            self.scale_y_slider.blockSignals(False)
            self.scale_y_textbox.setText(f"{value:.2f}")
        self.update_transform()

    def on_scale_y_changed(self, value):
        self.scale_y = value
        if self.uniform_scale:
            self.scale_x = value
            self.scale_x_slider.blockSignals(True)
            self.scale_x_slider.setValue(int(value * 100))
            self.scale_x_slider.blockSignals(False)
            self.scale_x_textbox.setText(f"{value:.2f}")
        self.update_transform()

    def on_uniform_changed(self, state):
        self.uniform_scale = (state == Qt.Checked)

    def on_opacity_changed(self, value):
        self.opacity = value / 100.0
        self.update_display()

    def on_mouse_translate(self, dx, dy):
        """Handle mouse-based translation (left-click drag)."""
        # Compute scale factor between displayed image and actual image pixels
        pixmap = self.image_label.pixmap()
        if pixmap and pixmap.width() > 0 and self.original_ihc_slice is not None:
            image_h, image_w = self.original_ihc_slice.shape
            screen_to_image = image_w / pixmap.width()
        else:
            screen_to_image = 1.0

        # Update translation values (swap dx/dy to match screen coordinates,
        # scale by display ratio so panning tracks the cursor exactly)
        new_x = self.translation_x + dy * screen_to_image
        new_y = self.translation_y + dx * screen_to_image

        # Clamp to slider range
        new_x = max(-200, min(200, new_x))
        new_y = max(-200, min(200, new_y))

        # Update sliders (which will trigger the transform update)
        self.trans_x_slider.setValue(int(new_x * 10))
        self.trans_y_slider.setValue(int(new_y * 10))

    def on_mouse_rotate(self, delta_deg):
        """Handle mouse-based rotation (right-click drag)."""
        new_rot = self.rotation_deg + delta_deg

        # Clamp to slider range
        new_rot = max(-180, min(180, new_rot))

        # Update slider (which will trigger the transform update)
        self.rotation_slider.setValue(int(new_rot * 10))

    def on_mouse_scale(self, scale_delta):
        """Handle mouse-based scaling (mouse wheel or middle-click drag)."""
        # Apply scale change
        new_scale_x = self.scale_x + scale_delta
        new_scale_y = self.scale_y + scale_delta

        # Clamp to slider range
        new_scale_x = max(0.5, min(2.0, new_scale_x))
        new_scale_y = max(0.5, min(2.0, new_scale_y))

        if self.uniform_scale:
            # Update both scales together
            self.scale_x_slider.setValue(int(new_scale_x * 100))
            # scale_y will be updated automatically due to uniform_scale
        else:
            # Update both independently
            self.scale_x_slider.setValue(int(new_scale_x * 100))
            self.scale_y_slider.setValue(int(new_scale_y * 100))

    def show_contrast_dialog(self):
        """Show the contrast adjustment dialog for MRI."""
        if self.mri_slice is None:
            QMessageBox.warning(self, "No Image", "Load MRI image first.")
            return

        # Create dialog with current MRI slice data
        self.contrast_dialog = ContrastCurveDialog(
            self.mri_slice, self,
            initial_points=self.contrast_control_points
        )
        self.contrast_dialog.contrast_changed.connect(self.on_contrast_changed)
        self.contrast_dialog.show()

    def on_contrast_changed(self, contrast_func):
        """Handle contrast curve change from dialog."""
        self.contrast_func = contrast_func
        if self.contrast_dialog:
            self.contrast_control_points = self.contrast_dialog.get_control_points()
        self.update_display()

    def update_transform(self):
        """Update the transform and display."""
        self.apply_transform()
        self.update_display()

    def update_display(self):
        """Update the visualization display."""
        if self.mri_slice_norm is None or self.ihc_slice_norm is None:
            return

        # Apply contrast to MRI if available
        if self.contrast_func is not None:
            mri_display = self.contrast_func(self.mri_slice)
            mri_display = self._normalize(mri_display)
        else:
            mri_display = self.mri_slice_norm

        # Create RGB overlay image
        h, w = mri_display.shape
        overlay = np.zeros((h, w, 3), dtype=np.float32)

        # MRI in grayscale (all channels)
        overlay[:, :, 0] = mri_display * (1 - self.opacity * self.ihc_slice_norm)
        overlay[:, :, 1] = mri_display * (1 - self.opacity * self.ihc_slice_norm)
        overlay[:, :, 2] = mri_display * (1 - self.opacity * self.ihc_slice_norm)

        # IHC overlay in magenta
        overlay[:, :, 0] += self.ihc_slice_norm * self.opacity
        overlay[:, :, 2] += self.ihc_slice_norm * self.opacity * 0.5

        # Clip and convert to uint8
        overlay = np.clip(overlay, 0, 1)
        overlay_uint8 = (overlay * 255).astype(np.uint8)

        # Ensure contiguous array
        overlay_uint8 = np.ascontiguousarray(overlay_uint8)

        # Convert to QImage and QPixmap
        qimage = QImage(overlay_uint8.tobytes(), w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        # Scale to fit label
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

        # Update info label
        self.info_label.setText(
            f"Transform: Tx={self.translation_x:.1f}, Ty={self.translation_y:.1f}, "
            f"R={self.rotation_deg:.1f}°, Sx={self.scale_x:.2f}, Sy={self.scale_y:.2f}"
        )

    def reset_transform(self):
        """Reset all transform parameters."""
        # Reset translation sliders (value * 10)
        self.trans_x_slider.setValue(0)
        self.trans_x_textbox.setText("0.0")
        self.trans_y_slider.setValue(0)
        self.trans_y_textbox.setText("0.0")

        # Reset rotation slider (value * 10)
        self.rotation_slider.setValue(0)
        self.rotation_textbox.setText("0.0")

        # Reset scale sliders (value * 100)
        self.scale_x_slider.setValue(100)
        self.scale_x_textbox.setText("1.00")
        self.scale_y_slider.setValue(100)
        self.scale_y_textbox.setText("1.00")

    def apply_and_save(self):
        """Apply transformation and save results."""
        os.makedirs(self.output_dir, exist_ok=True)

        self.apply_transform()
        self._save_transformed_nifti()
        self._save_transform_files()
        self._save_visualization()

        print("\nRegistration complete!")
        print(f"Results saved to: {self.output_dir}/")

        QMessageBox.information(
            self,
            "Success",
            f"Registration saved successfully!\n\nOutput: {self.output_dir}/"
        )

        self.close()

    def _save_transformed_nifti(self):
        """Save transformed IHC as NIfTI file."""
        output_data = np.zeros(self.ihc_nifti.shape, dtype=self.ihc_nifti.get_data_dtype())

        if self.slice_dim == 0:
            output_data[self.slice_idx, :, :] = self.ihc_slice.T
        elif self.slice_dim == 1:
            output_data[:, self.slice_idx, :] = self.ihc_slice.T
        else:
            output_data[:, :, self.slice_idx] = self.ihc_slice.T

        output_nifti = nib.Nifti1Image(output_data, self.ihc_nifti.affine, self.ihc_nifti.header)

        output_path = os.path.join(self.output_dir, "ihc_to_mri_affine.nii.gz")
        nib.save(output_nifti, output_path)
        print(f"Saved transformed image: {output_path}")

    def _save_transform_files(self):
        """Save transform in SimpleITK format and pickle metadata."""
        M = self.get_transform_matrix()

        a11 = M[0, 0]
        a12 = M[0, 1]
        a21 = M[1, 0]
        a22 = M[1, 1]
        b1 = M[0, 2]
        b2 = M[1, 2]

        sitk_transform = sitk.AffineTransform(2)
        sitk_transform.SetMatrix([a11, a21, a12, a22])
        sitk_transform.SetTranslation([b1, b2])
        sitk_transform.SetCenter([0.0, 0.0])

        tfm_path = os.path.join(self.output_dir, "transform_manual.tfm")
        sitk.WriteTransform(sitk_transform, tfm_path)
        print(f"Saved transform file: {tfm_path}")

        pkl_data = {
            'dimension': 2,
            'parameters': [a11, a21, a12, a22, b1, b2],
            'fixed_parameters': [0.0, 0.0],
            'slice_info': {
                'dimension': self.slice_dim,
                'slice_index': self.slice_idx
            },
            'transform_params': {
                'translation_x': self.translation_x,
                'translation_y': self.translation_y,
                'rotation_deg': self.rotation_deg,
                'scale_x': self.scale_x,
                'scale_y': self.scale_y
            },
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'method': 'manual'
        }

        # Add contrast settings if available
        if self.contrast_control_points is not None:
            pkl_data['contrast_curve'] = {
                'control_points': [[float(x), float(y)] for x, y in self.contrast_control_points]
            }

        pkl_path = os.path.join(self.output_dir, "transform_manual.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(pkl_data, f)
        print(f"Saved transform metadata: {pkl_path}")

    def _save_visualization(self):
        """Save registration result visualization."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Apply contrast to MRI display if available
        if self.contrast_func is not None:
            mri_display = self.contrast_func(self.mri_slice)
            mri_display = self._normalize(mri_display)
        else:
            mri_display = self.mri_slice_norm

        axes[0].imshow(mri_display, cmap='gray')
        axes[0].set_title('MRI Reference')
        axes[0].axis('off')

        axes[1].imshow(self.ihc_slice_norm, cmap='gray')
        axes[1].set_title('Registered IHC')
        axes[1].axis('off')

        overlay = np.zeros((*mri_display.shape, 3))
        overlay[:, :, 0] = self.ihc_slice_norm
        overlay[:, :, 1] = mri_display
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay (R=IHC, G=MRI)')
        axes[2].axis('off')

        plt.tight_layout()

        viz_path = os.path.join(self.output_dir, "manual_registration_result.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization: {viz_path}")

    def cancel(self):
        """Cancel registration."""
        reply = QMessageBox.question(
            self, "Cancel",
            "Are you sure you want to cancel?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.close()
            sys.exit(1)

    def resizeEvent(self, event):
        """Handle window resize."""
        super().resizeEvent(event)
        self.update_display()


def main():
    """Entry point."""
    print("=" * 60)
    print("Manual Linear Registration (PyQt5)")
    print("=" * 60)

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = ManualLinearRegistration()

    # Load matching info from JSON first
    matching_info = window.load_matching_info()

    # Auto-detect IHC file
    ihc_path, _ = window.find_input_files()

    # Get MRI path from JSON (preferred) or auto-detect
    mri_path = None
    if matching_info:
        mri_path = window.get_mri_path_from_json(matching_info)

    # If not found, ask user
    if ihc_path is None or mri_path is None:
        print("\nCould not auto-detect input files. Please select manually.")

        if ihc_path is None:
            ihc_path, _ = QFileDialog.getOpenFileName(
                None,
                "Select IHC NIfTI file (*_in_block.nii.gz)",
                os.getcwd(),
                "NIfTI files (*.nii *.nii.gz)"
            )
            if not ihc_path:
                print("No IHC file selected. Exiting.")
                sys.exit(1)

        if mri_path is None:
            mri_default_dir = os.path.join(os.path.dirname(os.getcwd()), "MRI")
            if not os.path.isdir(mri_default_dir):
                mri_default_dir = os.path.dirname(os.getcwd())
            mri_path, _ = QFileDialog.getOpenFileName(
                None,
                "Select MRI Reference NIfTI file",
                mri_default_dir,
                "NIfTI files (*.nii *.nii.gz)"
            )
            if not mri_path:
                print("No MRI file selected. Exiting.")
                sys.exit(1)

    # Load contrast settings from JSON
    contrast_points = window.get_contrast_settings_from_json(matching_info)
    if contrast_points is not None:
        print(f"Loaded contrast settings from matching info JSON")
        window.contrast_control_points = contrast_points
        # Create contrast function from loaded points
        x_pts = [p[0] for p in contrast_points]
        y_pts = [p[1] for p in contrast_points]
        window.contrast_func = lambda data, xp=x_pts, yp=y_pts: np.interp(data, xp, yp)

    # Load images
    try:
        window.load_images(ihc_path, mri_path)
    except Exception as e:
        print(f"Error loading images: {e}")
        QMessageBox.critical(None, "Error", f"Error loading images: {e}")
        sys.exit(1)

    # Initialize UI and show
    window.init_ui()
    window.update_display()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
