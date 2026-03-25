#!/usr/bin/env python3
"""
Combined Slice Matching + Linear Registration
=============================================
Two-panel GUI:
  Left  — IHC/histology image + orientation selector
  Right — large registration overlay (greyscale or checkerboard)
           with two columns of controls below:
             Left col:  Reset Transform button
             Right col: slice slider, opacity, contrast, info, Confirm, Cancel

Saved outputs
  Match_slice_results/         (relative to histology file)
    *_in_block.nii.gz
    *_matching_info.json
    *_visualization.png
  Linear_registration_results/ (relative to working directory)
    ihc_to_mri_affine.nii.gz
    transform_manual.tfm
    transform_manual.pkl
    manual_registration_result.png
"""

import os
import math
import sys
import json
import time
import pickle
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from datetime import datetime
from skimage import io as sk_io
from skimage.transform import resize as sk_resize
from skimage.color import rgb2gray
from scipy.ndimage import affine_transform as scipy_affine_transform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QRadioButton, QButtonGroup, QSlider, QPushButton,
    QFileDialog, QMessageBox, QGroupBox, QSizePolicy, QDialog,
    QFrame, QSpinBox, QLineEdit
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor

import pyqtgraph as pg
pg.setConfigOptions(antialias=False)

# ---------------------------------------------------------------------------
# Contrast curve dialog
# ---------------------------------------------------------------------------

class ContrastCurveDialog(QDialog):
    contrast_changed = pyqtSignal(object)

    def __init__(self, image_data, parent=None, initial_points=None):
        super().__init__(parent)
        self.setWindowTitle("MRI Contrast Adjustment")
        self.setGeometry(200, 200, 700, 550)
        self.image_data = image_data
        self.original_data = image_data.copy()
        mid_x = image_data.max() / 2
        self.control_points = (list(initial_points) if initial_points is not None
                               else [(0, 0), (mid_x, 0.5), (image_data.max(), 1)])
        self.selected_point_idx = 1
        self.dragging = False
        self.drag_point_idx = None
        self._setup_ui()
        self._update_histogram()
        self._update_curve()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('bottom', 'Image Intensity')
        self.plot_widget.setLabel('left', 'Output Level')
        self.plot_widget.setYRange(0, 1.1)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.histogram_item = pg.BarGraphItem(x=[], height=[], width=1, brush='b')
        self.plot_widget.addItem(self.histogram_item)
        self.curve_line = pg.PlotDataItem(pen=pg.mkPen('r', width=2))
        self.plot_widget.addItem(self.curve_line)
        self.points_scatter = pg.ScatterPlotItem(
            size=20, pen=pg.mkPen('k', width=2), brush=pg.mkBrush('y'), symbol='o')
        self.plot_widget.addItem(self.points_scatter)
        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.plot_widget.installEventFilter(self)
        layout.addWidget(self.plot_widget)

        cf = QFrame()
        cf.setFrameStyle(QFrame.StyledPanel)
        cl = QHBoxLayout(cf)
        cl.addWidget(QLabel("Selected point:"))
        cl.addWidget(QLabel("Id:"))
        self.id_spin = QSpinBox()
        self.id_spin.setMinimum(1); self.id_spin.setMaximum(3); self.id_spin.setValue(2)
        self.id_spin.valueChanged.connect(self._on_id_changed)
        cl.addWidget(self.id_spin)
        cl.addWidget(QLabel("x:")); self.x_input = QLineEdit(); self.x_input.setMaximumWidth(100)
        self.x_input.editingFinished.connect(self._on_xy_edited); cl.addWidget(self.x_input)
        cl.addWidget(QLabel("y:")); self.y_input = QLineEdit(); self.y_input.setMaximumWidth(100)
        self.y_input.editingFinished.connect(self._on_xy_edited); cl.addWidget(self.y_input)
        cl.addStretch()
        for txt, fn in [("+", self._add_point), ("-", self._remove_point)]:
            b = QPushButton(txt); b.setMaximumWidth(40); b.clicked.connect(fn); cl.addWidget(b)
        layout.addWidget(cf)

        br = QHBoxLayout()
        rst = QPushButton("Reset"); rst.clicked.connect(self._reset_curve); br.addWidget(rst)
        br.addStretch()
        apl = QPushButton("Apply"); apl.clicked.connect(self.apply_and_close); br.addWidget(apl)
        layout.addLayout(br)
        self._update_point_display()

    def _update_histogram(self):
        d = self.original_data.flatten(); d = d[d > 0]
        if len(d) == 0: return
        hist, edges = np.histogram(d, bins=100)
        centers = (edges[:-1] + edges[1:]) / 2
        self.histogram_item.setOpts(x=centers, height=hist / hist.max() * 0.3,
                                    width=edges[1] - edges[0])
        self.plot_widget.setXRange(0, self.original_data.max() * 1.05)

    def _update_curve(self):
        if len(self.control_points) < 2: return
        self.control_points.sort(key=lambda p: p[0])
        xs, ys = zip(*self.control_points)
        self.points_scatter.setData(list(xs), list(ys))
        xc = np.linspace(0, self.original_data.max(), 256)
        self.curve_line.setData(xc, np.clip(np.interp(xc, xs, ys), 0, 1))

    def _update_point_display(self):
        if 0 <= self.selected_point_idx < len(self.control_points):
            pt = self.control_points[self.selected_point_idx]
            self.id_spin.blockSignals(True)
            self.id_spin.setMaximum(len(self.control_points))
            self.id_spin.setValue(self.selected_point_idx + 1)
            self.id_spin.blockSignals(False)
            self.x_input.setText(f"{pt[0]:.1f}"); self.y_input.setText(f"{pt[1]:.3f}")

    def _on_id_changed(self, v):
        self.selected_point_idx = v - 1; self._update_point_display()

    def _on_xy_edited(self):
        try:
            x = max(0, float(self.x_input.text()))
            y = max(0, min(1, float(self.y_input.text())))
            if 0 <= self.selected_point_idx < len(self.control_points):
                self.control_points[self.selected_point_idx] = (x, y)
                self._update_curve(); self._emit()
        except ValueError: pass

    def eventFilter(self, obj, event):
        if obj == self.plot_widget and event.type() == event.MouseButtonPress \
                and event.button() == Qt.LeftButton:
            sp = self.plot_widget.mapToScene(event.pos())
            if self.plot_widget.sceneBoundingRect().contains(sp):
                mp = self.plot_widget.plotItem.vb.mapSceneToView(sp)
                x, y = mp.x(), mp.y()
                if self.dragging:
                    self.dragging = False; self.drag_point_idx = None; return True
                xr, yr = self.plot_widget.viewRange()
                for i, (px, py) in enumerate(self.control_points):
                    dx = (x-px)/(xr[1]-xr[0]) if xr[1]!=xr[0] else 0
                    dy = (y-py)/(yr[1]-yr[0]) if yr[1]!=yr[0] else 0
                    if (dx**2+dy**2)**0.5 < 0.08:
                        self.dragging = True; self.drag_point_idx = i
                        self.selected_point_idx = i; self._update_point_display(); return True
        return super().eventFilter(obj, event)

    def _on_mouse_moved(self, pos):
        if self.dragging and self.drag_point_idx is not None \
                and self.plot_widget.sceneBoundingRect().contains(pos):
            mp = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            x = max(0, min(self.original_data.max(), mp.x()))
            y = max(0, min(1, mp.y()))
            n = len(self.control_points)
            if self.drag_point_idx == 0: x = 0
            elif self.drag_point_idx == n-1: x = self.original_data.max()
            self.control_points[self.drag_point_idx] = (x, y)
            self.selected_point_idx = self.drag_point_idx
            self._update_curve(); self._update_point_display(); self._emit()

    def _reset_curve(self):
        m = self.original_data.max()
        self.control_points = [(0, 0), (m/2, 0.5), (m, 1)]
        self.selected_point_idx = 1; self.id_spin.setMaximum(3)
        self._update_curve(); self._update_point_display(); self._emit()

    def _add_point(self):
        m = self.original_data.max() / 2
        xs, ys = zip(*self.control_points)
        self.control_points.append((m, float(np.interp(m, xs, ys))))
        self.control_points.sort(key=lambda p: p[0])
        self.selected_point_idx = next(i for i,(x,_) in enumerate(self.control_points) if x==m)
        self.id_spin.setMaximum(len(self.control_points))
        self._update_curve(); self._update_point_display()

    def _remove_point(self):
        if len(self.control_points) > 2 \
                and 0 < self.selected_point_idx < len(self.control_points)-1:
            del self.control_points[self.selected_point_idx]
            self.selected_point_idx = min(self.selected_point_idx, len(self.control_points)-1)
            self.id_spin.setMaximum(len(self.control_points))
            self._update_curve(); self._update_point_display(); self._emit()

    def _emit(self):
        xs, ys = zip(*self.control_points)
        self.contrast_changed.emit(lambda d, xp=xs, yp=ys: np.interp(d, xp, yp))

    def apply_and_close(self):
        self._emit(); self.close()

    def get_control_points(self):
        return list(self.control_points)


# ---------------------------------------------------------------------------
# Slice matching core logic
# ---------------------------------------------------------------------------

class HistologyMatcher:
    def __init__(self):
        self.mri_path = None
        self.histology_path = None
        self.mri_nifti = None
        self.mri_data = None
        self.histology_img = None
        self.optimal_plane = None

    def normalize(self, img):
        lo, hi = img.min(), img.max()
        return (img - lo) / (hi - lo) if hi > lo else img * 0.0

    def find_optimal_plane(self, mri_data):
        avgs = []
        for plane in range(3):
            n = mri_data.shape[plane]
            total = sum(
                np.count_nonzero(mri_data[i] if plane == 0
                                 else mri_data[:, i, :] if plane == 1
                                 else mri_data[:, :, i])
                for i in range(n))
            avgs.append(total / n if n else 0)
        opt = int(np.argmax(avgs))
        print(f"Optimal plane: {['Sagittal','Coronal','Axial'][opt]}")
        return opt

    def extract_slice(self, mri_data, plane, idx):
        if plane == 0: return mri_data[idx, :, :]
        if plane == 1: return mri_data[:, idx, :]
        return mri_data[:, :, idx]

    def get_num_slices(self, plane):
        return self.mri_data.shape[plane]

    def generate_all_orientations(self, image):
        r = np.rot90(image, k=-1)
        return [
            (image.copy(),              "Original"),
            (np.fliplr(image),          "Flipped horizontally"),
            (np.flipud(image),          "Flipped vertically"),
            (np.flipud(np.fliplr(image)), "Flipped H+V"),
            (r.copy(),                  "Rotated 90 CW"),
            (np.fliplr(r),              "Rot 90 CW + Flip H"),
            (np.flipud(r),              "Rot 90 CW + Flip V"),
            (np.flipud(np.fliplr(r)),   "Rot 90 CW + Flip H+V"),
        ]

    def create_histology_nifti(self, mri_nifti, hist_slice, plane, idx):
        vol = np.zeros(mri_nifti.shape)
        if plane == 0:   vol[idx, :, :] = hist_slice
        elif plane == 1: vol[:, idx, :] = hist_slice
        else:            vol[:, :, idx] = hist_slice
        return nib.Nifti1Image(vol, mri_nifti.affine, mri_nifti.header)

    def save_matching_info(self, slice_idx, orientation_desc, orientation_idx,
                           output_path, contrast_curve=None):
        info = {
            "mri_file":        self.mri_path,
            "histology_file":  self.histology_path,
            "anatomical_plane": ['Sagittal','Coronal','Axial'][int(self.optimal_plane)],
            "plane_index":     int(self.optimal_plane),
            "selected_slice":  {"index": int(slice_idx)},
            "orientation":     {"description": orientation_desc,
                                "index": int(orientation_idx)},
            "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if contrast_curve:
            info["contrast_curve"] = {
                "control_points": [[float(x), float(y)] for x, y in contrast_curve]}

        class _E(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, np.integer): return int(o)
                if isinstance(o, np.floating): return float(o)
                if isinstance(o, np.ndarray): return o.tolist()
                return super().default(o)

        with open(output_path, 'w') as f:
            json.dump(info, f, indent=4, cls=_E)


# ---------------------------------------------------------------------------
# Interactive image label
# ---------------------------------------------------------------------------

class InteractiveImageLabel(QLabel):
    # translate: image-space delta (ix, iy)
    translate_changed    = pyqtSignal(float, float)
    # rotate: delta degrees + pivot in image coords
    rotate_around_changed = pyqtSignal(float, float, float)
    # scale: fractional factor delta (e.g. 0.02 → ×1.02)
    scale_changed        = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self._image_size     = None   # (h, w) — for coordinate mapping

        # left-drag translate
        self._translating    = False
        self._last_pos       = None

        # right-drag rotate
        self._rotating       = False
        self._rotate_pivot_screen = None   # (sx, sy)
        self._rotate_pivot_img    = None   # (ix, iy)
        self._rotate_last_angle   = None   # radians

    # ---- coordinate helpers -----------------------------------------------

    def _img_scale_offset(self):
        if self._image_size is None or self.pixmap() is None:
            return 1.0, 0.0, 0.0
        ih, iw = self._image_size
        lw, lh = self.width(), self.height()
        scale = min(lw / iw, lh / ih)
        ox = (lw - iw * scale) / 2
        oy = (lh - ih * scale) / 2
        return scale, ox, oy

    def _screen_to_img(self, sx, sy):
        s, ox, oy = self._img_scale_offset()
        return (sx - ox) / s, (sy - oy) / s

    # ---- mouse events -----------------------------------------------------

    def mousePressEvent(self, event):
        pos = event.pos()
        if event.button() == Qt.LeftButton:
            self._translating = True
            self._last_pos = pos
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.RightButton:
            ix, iy = self._screen_to_img(pos.x(), pos.y())
            self._rotating = True
            self._rotate_pivot_screen = (pos.x(), pos.y())
            self._rotate_pivot_img    = (ix, iy)
            self._rotate_last_angle   = 0.0
            self.setCursor(Qt.SizeAllCursor)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._translating = False
            self._last_pos = None
            self.setCursor(Qt.ArrowCursor)
        elif event.button() == Qt.RightButton:
            self._rotating = False
            self._rotate_pivot_screen = None
            self._rotate_pivot_img    = None
            self._rotate_last_angle   = None
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        pos = event.pos()
        if self._translating and self._last_pos is not None:
            d = pos - self._last_pos
            s, _, _ = self._img_scale_offset()
            self.translate_changed.emit(d.x() / s, d.y() / s)
            self._last_pos = pos
            return
        if self._rotating and self._rotate_pivot_screen is not None:
            px, py = self._rotate_pivot_screen
            cur_angle = math.atan2(pos.y() - py, pos.x() - px)
            delta = math.degrees(cur_angle - self._rotate_last_angle)
            if abs(delta) < 90:
                self.rotate_around_changed.emit(
                    delta, self._rotate_pivot_img[0], self._rotate_pivot_img[1])
            self._rotate_last_angle = cur_angle
        super().mouseMoveEvent(event)

    def wheelEvent(self, event):
        self.scale_changed.emit(event.angleDelta().y() * 0.0002)
        event.accept()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class CombinedRegistrationWindow(QMainWindow):

    def __init__(self, matcher):
        super().__init__()
        self.matcher = matcher

        # Slice-matching state
        self.current_orientation_idx = 0
        self.current_slice_idx = 0
        self.resized_orientations = []

        # Contrast
        self.contrast_dialog = None
        self.contrast_control_points = None
        self.contrast_func = None

        # Registration state
        self.original_ihc_slice = None
        self.ihc_slice = None
        self.mri_reg_slice = None
        self.opacity = 0.0
        self.checkerboard_mode = False
        self.checkerboard_size = 32

        # Transform parameters
        self.trans_x = 0.0
        self.trans_y = 0.0
        self.rotation_deg = 0.0
        self.scale_val = 1.0

        self.output_dir_lin = "Linear_registration_results"
        self._init_ui()

    # -----------------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------------

    def _init_ui(self):
        self.setWindowTitle("Slice Matching + Linear Registration")
        self.setMinimumSize(1400, 900)

        root = QWidget()
        self.setCentralWidget(root)
        main_vbox = QVBoxLayout(root)
        main_vbox.setSpacing(6)

        # ── TOP ROW: both image panels side by side ───────────────────────
        img_row = QHBoxLayout()
        img_row.setSpacing(6)

        self.ihc_label = QLabel()
        self.ihc_label.setAlignment(Qt.AlignCenter)
        self.ihc_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ihc_label.setStyleSheet("border: 1px solid gray;")
        img_row.addWidget(self.ihc_label, stretch=1)

        self.reg_label = InteractiveImageLabel()
        self.reg_label.setAlignment(Qt.AlignCenter)
        self.reg_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.reg_label.setStyleSheet("border: 1px solid gray;")
        self.reg_label.translate_changed.connect(self._on_mouse_translate)
        self.reg_label.rotate_around_changed.connect(self._on_mouse_rotate_around)
        self.reg_label.scale_changed.connect(self._on_mouse_scale)
        img_row.addWidget(self.reg_label, stretch=2)

        main_vbox.addLayout(img_row, stretch=1)

        # ── BOTTOM ROW: three columns, top-aligned ────────────────────────
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(10)

        # --- Left column: Orientation ---
        og = QGroupBox("Orientation")
        ol = QVBoxLayout(og)
        self.orientation_button_group = QButtonGroup()
        self.orientation_buttons = []
        for i, lbl in enumerate([
            "Original", "Flipped horizontally", "Flipped vertically", "Flipped H+V",
            "Rotated 90 CW", "Rot 90 CW + Flip H", "Rot 90 CW + Flip V",
            "Rot 90 CW + Flip H+V",
        ]):
            rb = QRadioButton(lbl)
            self.orientation_button_group.addButton(rb, i)
            self.orientation_buttons.append(rb)
            ol.addWidget(rb)
            if i == 0: rb.setChecked(True)
        self.orientation_button_group.buttonClicked.connect(self._on_orientation_changed)
        ctrl_row.addWidget(og, 1, Qt.AlignTop)

        # --- Middle column: Transform Controls ---
        tg = QGroupBox("Transform Controls")
        tg.setToolTip(
            "Left drag: translate  |  Right drag: rotate around click point\n"
            "Scroll wheel: scale")
        tl = QVBoxLayout(tg)

        self._create_compact_control(tl, "Translation X:",
                                     -200, 200, 0.0, 0.5, self._on_slider_transform, 'trans_x', '.1f')
        self._create_compact_control(tl, "Translation Y:",
                                     -200, 200, 0.0, 0.5, self._on_slider_transform, 'trans_y', '.1f')
        self._create_compact_control(tl, "Rotation:",
                                     -180, 180, 0.0, 0.5, self._on_slider_transform, 'rotation_deg', '.1f')
        self._create_compact_control(tl, "Scale:",
                                     0.10, 5.0, 1.0, 0.01, self._on_slider_transform, 'scale_val', '.2f')

        reset_btn = QPushButton("Reset Transform")
        reset_btn.clicked.connect(self._reset_transform)
        tl.addWidget(reset_btn)
        ctrl_row.addWidget(tg, 1, Qt.AlignTop)

        # --- Right column: Slice / Overlay / Actions ---
        rc = QWidget()
        rcl = QVBoxLayout(rc)
        rcl.setContentsMargins(0, 0, 0, 0)
        rcl.setSpacing(8)

        sg = QGroupBox("Slice Selection")
        sl = QVBoxLayout(sg)
        self.slice_info_label = QLabel("Slice: 0 / 0")
        self.slice_info_label.setAlignment(Qt.AlignCenter)
        sl.addWidget(self.slice_info_label)
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0); self.slice_slider.setMaximum(100)
        self.slice_slider.setValue(50)
        self.slice_slider.valueChanged.connect(self._on_slice_changed)
        sl.addWidget(self.slice_slider)
        rcl.addWidget(sg)

        opa_group = QGroupBox("Overlay")
        opa_l = QVBoxLayout(opa_group)

        self.chk_mode = QPushButton("Mode: Alpha Blend")
        self.chk_mode.setCheckable(True)
        self.chk_mode.setChecked(False)
        self.chk_mode.setToolTip(
            "Alpha blend: smooth greyscale mix.\n"
            "Checkerboard: alternating blocks of MRI / IHC —\n"
            "structure edges snap together when aligned.")
        self.chk_mode.toggled.connect(self._on_mode_toggled)
        opa_l.addWidget(self.chk_mode)

        opa_row = QHBoxLayout()
        self.overlay_slider_label = QLabel("Opacity:")
        opa_row.addWidget(self.overlay_slider_label)
        self.overlay_slider = QSlider(Qt.Horizontal)
        self.overlay_slider.setMinimum(0)
        self.overlay_slider.setMaximum(100)
        self.overlay_slider.setValue(0)
        self.overlay_slider.valueChanged.connect(self._on_overlay_slider_changed)
        opa_row.addWidget(self.overlay_slider)
        opa_l.addLayout(opa_row)

        rcl.addWidget(opa_group)

        self.contrast_btn = QPushButton("Change Contrast")
        self.contrast_btn.setStyleSheet("background-color: #e0e0ff;")
        self.contrast_btn.clicked.connect(self._show_contrast_dialog)
        rcl.addWidget(self.contrast_btn)

        rcl.addStretch()

        confirm_btn = QPushButton("Confirm && Save")
        confirm_btn.setMinimumHeight(48)
        confirm_btn.setStyleSheet("font-size: 13px; font-weight: bold;"
                                  "background-color: #90EE90;")
        confirm_btn.clicked.connect(self._on_confirm)
        rcl.addWidget(confirm_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setMinimumHeight(32)
        cancel_btn.clicked.connect(self._on_cancel)
        rcl.addWidget(cancel_btn)

        ctrl_row.addWidget(rc, 1, Qt.AlignTop)

        main_vbox.addLayout(ctrl_row)

    # -----------------------------------------------------------------------
    # UI helpers
    # -----------------------------------------------------------------------

    def _create_compact_control(self, parent_layout, label, min_val, max_val,
                                default, step, callback, attr, fmt='.1f'):
        """Create a compact [-] [textbox] [+] control row."""
        parent_layout.addWidget(QLabel(label))
        row = QHBoxLayout()

        minus_btn = QPushButton("-")
        minus_btn.setFixedWidth(30)
        minus_btn.setToolTip(f"Decrease by {step}")
        row.addWidget(minus_btn)

        textbox = QLineEdit()
        textbox.setMaximumWidth(70)
        textbox.setText(format(default, fmt))
        textbox.setAlignment(Qt.AlignCenter)
        row.addWidget(textbox, stretch=1)

        plus_btn = QPushButton("+")
        plus_btn.setFixedWidth(30)
        plus_btn.setToolTip(f"Increase by {step}")
        row.addWidget(plus_btn)

        setattr(self, f'{attr}_textbox', textbox)

        def nudge(delta):
            current = getattr(self, attr)
            new_val = max(min_val, min(max_val, current + delta))
            setattr(self, attr, new_val)
            textbox.blockSignals(True)
            textbox.setText(format(new_val, fmt))
            textbox.blockSignals(False)
            callback()

        minus_btn.clicked.connect(lambda: nudge(-step))
        plus_btn.clicked.connect(lambda: nudge(step))

        def on_textbox_change():
            try:
                value = float(textbox.text())
                value = max(min_val, min(max_val, value))
                setattr(self, attr, value)
                textbox.setText(format(value, fmt))
                callback()
            except ValueError:
                pass

        textbox.editingFinished.connect(on_textbox_change)
        parent_layout.addLayout(row)

    def _on_slider_transform(self):
        """Called when any transform control changes via the UI."""
        self._update_transform()

    def _sync_controls(self):
        """Update textbox values to match current transform params."""
        for attr, fmt in [('trans_x', '.1f'), ('trans_y', '.1f'),
                          ('rotation_deg', '.1f'), ('scale_val', '.2f')]:
            textbox = getattr(self, f'{attr}_textbox', None)
            if textbox:
                textbox.blockSignals(True)
                textbox.setText(format(getattr(self, attr), fmt))
                textbox.blockSignals(False)

    # -----------------------------------------------------------------------
    # Transform matrix
    # -----------------------------------------------------------------------

    def _get_transform_matrix(self):
        """Build 2D affine matrix from current parameters (input → output)."""
        h, w = self.original_ihc_slice.shape
        cx, cy = w / 2, h / 2

        theta = np.radians(self.rotation_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Center → scale → rotate → translate back + offset
        T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
        S  = np.array([[self.scale_val, 0, 0],
                        [0, self.scale_val, 0],
                        [0, 0, 1]])
        R  = np.array([[cos_t, -sin_t, 0],
                        [sin_t,  cos_t, 0],
                        [0,      0,     1]])
        T2 = np.array([[1, 0, cx + self.trans_x],
                        [0, 1, cy + self.trans_y],
                        [0, 0, 1]])
        return T2 @ R @ S @ T1

    # -----------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------

    def load_data(self):
        plane = self.matcher.optimal_plane
        if plane == 0:   target = (self.matcher.mri_data.shape[1], self.matcher.mri_data.shape[2])
        elif plane == 1: target = (self.matcher.mri_data.shape[0], self.matcher.mri_data.shape[2])
        else:            target = (self.matcher.mri_data.shape[0], self.matcher.mri_data.shape[1])

        hist_resized = sk_resize(self.matcher.histology_img, target, anti_aliasing=True)
        raw_orientations = self.matcher.generate_all_orientations(hist_resized)
        self.resized_orientations = [
            (sk_resize(img, target, anti_aliasing=True) if img.shape != target else img, desc)
            for img, desc in raw_orientations
        ]

        n = self.matcher.get_num_slices(plane)
        self.slice_slider.setMaximum(n - 1)
        self.current_slice_idx = n // 2
        self.slice_slider.setValue(self.current_slice_idx)

        self._try_load_contrast()
        self._refresh_ihc_base()
        self._update_histology_display()
        self._update_registration_display()

    def _try_load_contrast(self):
        match_dir = os.path.join(os.path.dirname(self.matcher.histology_path),
                                 "Match_slice_results")
        if not os.path.isdir(match_dir): return
        for f in os.listdir(match_dir):
            if f.startswith('._') or not f.endswith("_matching_info.json"): continue
            try:
                with open(os.path.join(match_dir, f)) as jf:
                    data = json.load(jf)
                if 'contrast_curve' in data and 'control_points' in data['contrast_curve']:
                    pts = [(float(p[0]), float(p[1]))
                           for p in data['contrast_curve']['control_points']]
                    self.contrast_control_points = pts
                    xs, ys = zip(*pts)
                    self.contrast_func = lambda d, xp=xs, yp=ys: np.interp(d, xp, yp)
            except Exception: pass
            break

    # -----------------------------------------------------------------------
    # Display
    # -----------------------------------------------------------------------

    @staticmethod
    def _norm(img):
        lo, hi = float(img.min()), float(img.max())
        return (img.astype(np.float64) - lo) / (hi - lo) if hi > lo else np.zeros_like(img, float)

    @staticmethod
    def _to_qpixmap(img_norm, size=None):
        u8 = (np.clip(img_norm, 0, 1) * 255).astype(np.uint8)
        u8 = np.ascontiguousarray(u8)
        h, w = u8.shape
        px = QPixmap.fromImage(QImage(u8.tobytes(), w, h, w, QImage.Format_Grayscale8))
        return px.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation) if size else px

    def _mri_raw(self):
        return self.matcher.extract_slice(
            self.matcher.mri_data, self.matcher.optimal_plane, self.current_slice_idx)

    def _mri_display(self):
        raw = self._mri_raw()
        return self._norm(self.contrast_func(raw) if self.contrast_func else raw)

    def _update_histology_display(self):
        if not self.resized_orientations: return
        img, _ = self.resized_orientations[self.current_orientation_idx]
        self.ihc_label.setPixmap(self._to_qpixmap(self._norm(img), self.ihc_label.size()))

    def _update_registration_display(self):
        if self.ihc_slice is None or self.mri_reg_slice is None: return

        self.reg_label._image_size = self.ihc_slice.shape  # (h, w) for coord mapping

        mri_n = self._mri_display()
        ihc_n = self._norm(self.ihc_slice)

        if self.checkerboard_mode:
            bs = self.checkerboard_size
            h, w = mri_n.shape
            rows = np.arange(h) // bs
            cols = np.arange(w) // bs
            mask = (rows[:, None] + cols[None, :]) % 2 == 0
            blend = np.where(mask, mri_n, ihc_n)
        else:
            blend = np.clip(self.opacity * ihc_n + (1.0 - self.opacity) * mri_n, 0, 1)

        px = self._to_qpixmap(blend)
        self.reg_label.setPixmap(
            px.scaled(self.reg_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        plane_names = ["Sagittal", "Coronal", "Axial"]
        n = self.matcher.get_num_slices(self.matcher.optimal_plane)
        self.slice_info_label.setText(
            f"Slice: {self.current_slice_idx} / {n-1}"
            f" ({plane_names[self.matcher.optimal_plane]})")

    # -----------------------------------------------------------------------
    # IHC / transform
    # -----------------------------------------------------------------------

    def _refresh_ihc_base(self):
        if not self.resized_orientations: return
        img, _ = self.resized_orientations[self.current_orientation_idx]
        self.original_ihc_slice = img.copy()
        self.mri_reg_slice = self._mri_raw().copy()
        self.trans_x = 0.0
        self.trans_y = 0.0
        self.rotation_deg = 0.0
        self.scale_val = 1.0
        self._sync_controls()
        self._apply_transform()

    def _apply_transform(self):
        if self.original_ihc_slice is None:
            return
        M = self._get_transform_matrix()
        M_inv = np.linalg.inv(M)
        h, w = self.original_ihc_slice.shape
        self.ihc_slice = scipy_affine_transform(
            self.original_ihc_slice, M_inv[:2, :2], offset=M_inv[:2, 2],
            output_shape=(h, w), order=1, mode='constant', cval=0)

    def _update_transform(self):
        self._apply_transform()
        self._update_registration_display()

    # -----------------------------------------------------------------------
    # Event handlers
    # -----------------------------------------------------------------------

    def _on_orientation_changed(self, btn):
        self.current_orientation_idx = self.orientation_button_group.id(btn)
        self._update_histology_display()
        self._refresh_ihc_base()
        self._update_registration_display()

    def _on_slice_changed(self, value):
        self.current_slice_idx = value
        self.mri_reg_slice = self._mri_raw().copy()
        self._update_registration_display()

    def _on_overlay_slider_changed(self, value):
        if self.checkerboard_mode:
            self.checkerboard_size = value
        else:
            self.opacity = value / 100.0
        self._update_registration_display()

    def _on_mode_toggled(self, checked):
        self.checkerboard_mode = checked
        self.chk_mode.setText("Mode: Checkerboard" if checked else "Mode: Alpha Blend")
        if checked:
            self.overlay_slider_label.setText("Square size:")
            self.overlay_slider.setMinimum(4)
            self.overlay_slider.setMaximum(64)
            self.overlay_slider.setValue(self.checkerboard_size)
        else:
            self.overlay_slider_label.setText("Opacity:")
            self.overlay_slider.setMinimum(0)
            self.overlay_slider.setMaximum(100)
            self.overlay_slider.setValue(int(self.opacity * 100))
        self._update_registration_display()

    def _on_mouse_translate(self, dx_img, dy_img):
        if self.original_ihc_slice is None: return
        # Swap: screen Y (dy) → trans_x (matrix row axis = vertical),
        #        screen X (dx) → trans_y (matrix col axis = horizontal)
        self.trans_x += dy_img
        self.trans_y += dx_img
        self._sync_controls()
        self._update_transform()

    def _on_mouse_rotate_around(self, delta_deg, pivot_ix, pivot_iy):
        if self.original_ihc_slice is None: return
        # Negate: the matrix rotation direction is inverted in scipy [row,col] space
        visual_angle = delta_deg * 0.1
        h, w = self.original_ihc_slice.shape
        # Image center in screen coords (accounting for axis swap in matrix)
        center_sx = h / 2 + self.trans_y   # screen X = cols
        center_sy = w / 2 + self.trans_x   # screen Y = rows
        # Rotate center around pivot in screen space (CW positive)
        t = np.radians(visual_angle)
        c, s = np.cos(t), np.sin(t)
        dx = center_sx - pivot_ix
        dy = center_sy - pivot_iy
        new_sx = pivot_ix + c * dx - s * dy
        new_sy = pivot_iy + s * dx + c * dy
        self.trans_y = new_sx - h / 2
        self.trans_x = new_sy - w / 2
        self.rotation_deg -= visual_angle
        self._sync_controls()
        self._update_transform()

    def _on_mouse_scale(self, factor):
        if self.original_ihc_slice is None: return
        self.scale_val *= (1.0 + factor)
        self.scale_val = max(0.1, min(5.0, self.scale_val))
        self._sync_controls()
        self._update_transform()

    def _show_contrast_dialog(self):
        self.contrast_dialog = ContrastCurveDialog(
            self._mri_raw(), self, initial_points=self.contrast_control_points)
        self.contrast_dialog.contrast_changed.connect(self._on_contrast_changed)
        self.contrast_dialog.show()

    def _on_contrast_changed(self, fn):
        self.contrast_func = fn
        if self.contrast_dialog:
            self.contrast_control_points = self.contrast_dialog.get_control_points()
        self._update_registration_display()

    def _reset_transform(self):
        self.trans_x = 0.0
        self.trans_y = 0.0
        self.rotation_deg = 0.0
        self.scale_val = 1.0
        self._sync_controls()
        self._update_transform()

    # -----------------------------------------------------------------------
    # Confirm & Save
    # -----------------------------------------------------------------------

    def _on_confirm(self):
        try:
            oi = self.current_orientation_idx
            si = self.current_slice_idx
            plane = self.matcher.optimal_plane
            oriented_img, orient_desc = self.resized_orientations[oi]

            # --- Slice matching results ---
            match_dir = os.path.join(os.path.dirname(self.matcher.histology_path),
                                     "Match_slice_results")
            os.makedirs(match_dir, exist_ok=True)

            mri_sl = self.matcher.extract_slice(self.matcher.mri_data, plane, si)
            nz = mri_sl[mri_sl > 0]
            mri_lo, mri_hi = (np.percentile(nz, 1), np.percentile(nz, 99)) if len(nz) else (0, 1)

            hist_norm = self.matcher.normalize(oriented_img) * (mri_hi - mri_lo) + mri_lo
            nib.save(self.matcher.create_histology_nifti(
                         self.matcher.mri_nifti, hist_norm, plane, si),
                     os.path.join(match_dir, f"{self._stem()}_in_block.nii.gz"))

            self.matcher.save_matching_info(
                si, orient_desc, oi,
                os.path.join(match_dir, f"{self._stem()}_matching_info.json"),
                contrast_curve=self.contrast_control_points)

            self._save_slice_vis(oriented_img, mri_sl, si, orient_desc, match_dir)

            # --- Linear registration results ---
            lin_dir = self.output_dir_lin
            os.makedirs(lin_dir, exist_ok=True)

            self._apply_transform()
            reg_norm = self.matcher.normalize(self.ihc_slice) * (mri_hi - mri_lo) + mri_lo
            nib.save(self.matcher.create_histology_nifti(
                         self.matcher.mri_nifti, reg_norm, plane, si),
                     os.path.join(lin_dir, "ihc_to_mri_affine.nii.gz"))

            self._save_transform_files(lin_dir, plane, si)
            self._save_lin_vis(lin_dir)

            QMessageBox.information(
                self, "Saved",
                f"Slice matching → {match_dir}/\n\nLinear registration → {lin_dir}/")
            self.close()

        except Exception as exc:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Save failed:\n{exc}")

    def _stem(self):
        s = os.path.splitext(os.path.basename(self.matcher.histology_path))[0]
        return os.path.splitext(s)[0] if s.lower().endswith('.nii') else s

    def _save_slice_vis(self, hist_img, mri_sl, idx, desc, out_dir):
        vis = self.contrast_func(mri_sl) if self.contrast_func else mri_sl
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(hist_img, cmap='gray'); axes[0].set_title(f"Histology\n({desc})"); axes[0].axis('off')
        axes[1].imshow(vis, cmap='gray')
        axes[1].set_title(f"MRI Slice {idx}\n({['Sagittal','Coronal','Axial'][self.matcher.optimal_plane]})")
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{self._stem()}_visualization.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _save_transform_files(self, out_dir, slice_dim, slice_idx):
        # Internal matrix operates in (row, col) space.
        # Linear_Registration_Qt.py and Transformation.py expect (col, row) = transposed space.
        # Convert via axis-swap: M_save = P @ M_internal @ P where P swaps axes.
        M_internal = self._get_transform_matrix()
        P = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        M = P @ M_internal @ P

        a11, a12, b1 = float(M[0, 0]), float(M[0, 1]), float(M[0, 2])
        a21, a22, b2 = float(M[1, 0]), float(M[1, 1]), float(M[1, 2])

        t = sitk.AffineTransform(2)
        t.SetMatrix([a11, a21, a12, a22]); t.SetTranslation([b1, b2]); t.SetCenter([0., 0.])
        sitk.WriteTransform(t, os.path.join(out_dir, "transform_manual.tfm"))

        pkl = {
            'dimension': 2,
            'parameters': [a11, a21, a12, a22, b1, b2],
            'fixed_parameters': [0.0, 0.0],
            'slice_info': {'dimension': slice_dim, 'slice_index': slice_idx},
            'transform_params': {
                'translation_x': self.trans_x,
                'translation_y': self.trans_y,
                'rotation_deg': -self.rotation_deg,
                'scale_x': self.scale_val,
                'scale_y': self.scale_val,
            },
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'method': 'manual',
        }
        if self.contrast_control_points:
            pkl['contrast_curve'] = {'control_points': [[float(x), float(y)]
                                     for x, y in self.contrast_control_points]}
        with open(os.path.join(out_dir, "transform_manual.pkl"), 'wb') as f:
            pickle.dump(pkl, f)

    def _save_lin_vis(self, out_dir):
        raw = self._mri_raw()
        mri_d = self._norm(self.contrast_func(raw) if self.contrast_func else raw)
        ihc_d = self._norm(self.ihc_slice)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(mri_d, cmap='gray'); axes[0].set_title('MRI Reference'); axes[0].axis('off')
        axes[1].imshow(ihc_d, cmap='gray'); axes[1].set_title('Registered IHC'); axes[1].axis('off')
        ov = np.zeros((*mri_d.shape, 3)); ov[:,:,0] = ihc_d; ov[:,:,1] = mri_d
        axes[2].imshow(ov); axes[2].set_title('Overlay (R=IHC, G=MRI)'); axes[2].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "manual_registration_result.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _on_cancel(self):
        if QMessageBox.question(self, "Cancel", "Discard changes and close?",
                                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self.close()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_histology_display()
        self._update_registration_display()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _select_files():
    mri_default = os.path.join(os.path.dirname(os.getcwd()), "MRI")
    if not os.path.isdir(mri_default):
        mri_default = os.path.dirname(os.getcwd())
    mri_path, _ = QFileDialog.getOpenFileName(
        None, "Select MRI NIfTI file", mri_default,
        "NIfTI files (*.nii *.nii.gz);;All files (*.*)")
    if not mri_path: return None, None
    hist_path, _ = QFileDialog.getOpenFileName(
        None, "Select Histology image file", os.getcwd(),
        "Image files (*.jpg *.jpeg *.tif *.tiff *.png);;All files (*.*)",
        options=QFileDialog.DontUseNativeDialog)
    if not hist_path: return None, None
    return mri_path, hist_path


def main():
    print("===== Slice Matching + Linear Registration =====")
    app = QApplication.instance() or QApplication(sys.argv)

    mri_path, hist_path = _select_files()
    if not mri_path or not hist_path:
        print("File selection cancelled."); return

    matcher = HistologyMatcher()
    matcher.mri_path = mri_path; matcher.histology_path = hist_path
    print(f"Loading MRI:       {mri_path}")
    matcher.mri_nifti = nib.load(mri_path); matcher.mri_data = matcher.mri_nifti.get_fdata()
    print(f"  shape: {matcher.mri_data.shape}")
    print(f"Loading histology: {hist_path}")
    img = sk_io.imread(hist_path)
    if img.ndim > 2: img = rgb2gray(img)
    matcher.histology_img = img
    print("Finding optimal plane…")
    matcher.optimal_plane = matcher.find_optimal_plane(matcher.mri_data)

    window = CombinedRegistrationWindow(matcher)
    window.load_data()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
