#!/usr/bin/env python3
"""
Non-Linear Registration Tool - PyQt5/PyQtGraph Version
=======================================================
GPU-accelerated landmark-based registration with smooth panning and zooming.

"""

import os
import sys
import json
import subprocess
import numpy as np
import nibabel as nib
from scipy import interpolate
import pickle
from datetime import datetime
from skimage import io as sk_io
from skimage.transform import warp as sk_warp, AffineTransform as sk_AffineTransform, resize as sk_resize

# PyQt5 imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QLabel, QFileDialog, QMessageBox,
    QGroupBox, QRadioButton, QButtonGroup, QLineEdit, QSplitter,
    QStatusBar, QToolBar, QAction, QSpinBox, QDoubleSpinBox,
    QCheckBox, QFrame, QTextEdit, QDialog, QScrollArea, QSlider
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer, QPointF, QRectF
from PyQt5.QtGui import QKeySequence, QFont, QColor, QPen

# PyQtGraph for fast image display
import pyqtgraph as pg

# Configure PyQtGraph for better performance
pg.setConfigOptions(antialias=False, useOpenGL=True)


class ImageViewWidget(pg.GraphicsLayoutWidget):
    """Custom image view with landmark overlay support."""

    clicked = pyqtSignal(float, float, int)  # x, y, button (1=left, 2=right)
    mouse_moved = pyqtSignal(float, float)  # x, y for ruler tracking
    wheel_rotated = pyqtSignal(float)  # delta for ruler rotation
    left_drag_started = pyqtSignal(float, float)  # x, y
    left_drag_moved = pyqtSignal(float, float)    # x, y
    left_drag_ended = pyqtSignal(float, float)    # x, y

    def __init__(self, title="Image", parent=None):
        super().__init__(parent)

        self.title = title
        self._panning = False
        self._pan_start = None
        self._right_dragging = False
        self._right_drag_start = None
        self._wheel_zoom_enabled = True  # Can be disabled during ruler positioning
        self._drag_mode_enabled = False  # When True, left-click starts a drag instead of emitting clicked
        self._left_dragging = False

        # Create plot area
        self.plot = self.addPlot(title=title)
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)  # Match image coordinate system
        self.plot.hideAxis('left')
        self.plot.hideAxis('bottom')

        # Disable default mouse interactions - we'll handle them ourselves
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setMenuEnabled(False)

        # Image item
        self.image_item = pg.ImageItem()
        self.plot.addItem(self.image_item)

        # Colour overlay item (for overview PNG overlay)
        self.overlay_item = pg.ImageItem()
        self.overlay_item.setZValue(1)  # above grayscale image_item
        self.plot.addItem(self.overlay_item)
        self.overlay_item.hide()

        # Pixel-border grid overlay (shown together with the overview overlay)
        _grid_pen = pg.mkPen(color=(180, 180, 180, 180), width=1, cosmetic=True)
        self._grid_h = pg.PlotDataItem(pen=_grid_pen)
        self._grid_v = pg.PlotDataItem(pen=_grid_pen)
        self._grid_h.setZValue(2)
        self._grid_v.setZValue(2)
        self.plot.addItem(self._grid_h)
        self.plot.addItem(self._grid_v)
        self._grid_h.hide()
        self._grid_v.hide()
        self._grid_overlay_active = False   # True when overlay checkbox is on
        self._grid_ncols = 0               # low-res image width (NIfTI pixels)
        # Show grid only when each NIfTI pixel covers ≥ this many screen pixels
        self._GRID_THRESHOLD = 20
        self.plot.vb.sigRangeChanged.connect(self._update_grid_visibility)

        # Landmark scatter plot
        self.landmark_scatter = pg.ScatterPlotItem(
            size=12, pen=pg.mkPen('y', width=2), brush=pg.mkBrush('r'),
            symbol='o'
        )
        self.plot.addItem(self.landmark_scatter)
        self.landmark_scatter.setZValue(2)

        # Text items for landmark labels
        self.label_items = []

        # Highlight scatter for drag operations
        self.highlight_scatter = pg.ScatterPlotItem(
            size=18, pen=pg.mkPen('g', width=3), brush=pg.mkBrush(0, 255, 0, 80),
            symbol='o'
        )
        self.plot.addItem(self.highlight_scatter)
        self.highlight_scatter.setZValue(2)

        # Line preview for line mode
        self.line_preview = pg.PlotDataItem(pen=pg.mkPen('r', width=2, style=Qt.DashLine))
        self.plot.addItem(self.line_preview)
        self.line_preview.setZValue(2)
        self.line_preview.hide()

        # Line points markers
        self.line_points_scatter = pg.ScatterPlotItem(
            size=8, pen=pg.mkPen('r', width=1), brush=pg.mkBrush('r'),
            symbol='o'
        )
        self.plot.addItem(self.line_points_scatter)
        self.line_points_scatter.setZValue(2)

        # Ruler items
        self.ruler_line = pg.PlotDataItem(pen=pg.mkPen('r', width=3))
        self.ruler_text = pg.TextItem(color='r', anchor=(0.5, 1))
        self.ruler_center = pg.ScatterPlotItem(
            size=8, pen=pg.mkPen('r', width=2), brush=pg.mkBrush('r'),
            symbol='o'
        )
        self.plot.addItem(self.ruler_line)
        self.plot.addItem(self.ruler_text)
        self.plot.addItem(self.ruler_center)
        self.ruler_line.setZValue(2)
        self.ruler_text.setZValue(2)
        self.ruler_center.setZValue(2)
        self.ruler_line.hide()
        self.ruler_text.hide()
        self.ruler_center.hide()

        # Line overlay items for Move Line mode
        self.line_overlay_items = []       # current landmark connections (red dashed)
        self.line_overlay_bg_items = []    # original control point polylines (dim background)

        # Store image data
        self.image_data = None

        # Enable mouse tracking
        self.setMouseTracking(True)

    def set_drag_mode(self, enabled):
        """Enable or disable left-button drag mode."""
        self._drag_mode_enabled = enabled
        if not enabled:
            self._left_dragging = False

    def set_overlay_image(self, rgba_data, display_shape=None):
        """Display a colour overlay from an (H, W, 4) RGBA uint8 array.

        display_shape: (H, W) in image coordinates at which the overlay should
        be displayed.  Pass this when the RGBA data is at a higher resolution
        than the underlying NIfTI slice so that the overlay is stretched back to
        the correct spatial extent.  If None the image is displayed 1:1.
        """
        # pyqtgraph ImageItem expects (W, H, channels) — transpose row/col axes
        self.overlay_item.setImage(rgba_data.transpose(1, 0, 2))
        if display_shape is not None:
            dh, dw = display_shape
            self.overlay_item.setRect(QRectF(0, 0, dw, dh))
            self._set_pixel_grid(dh, dw)
        else:
            dh, dw = rgba_data.shape[0], rgba_data.shape[1]
            self.overlay_item.setRect(QRectF(0, 0, dw, dh))
            self._set_pixel_grid(dh, dw)

    def _set_pixel_grid(self, nrows, ncols):
        """Build horizontal and vertical grid lines at every low-res pixel boundary."""
        self._grid_ncols = ncols
        # Horizontal lines: one per row boundary (y = 0, 1, …, nrows)
        n_h = nrows + 1
        xs_h = np.empty(n_h * 3)
        ys_h = np.empty(n_h * 3)
        xs_h[0::3] = 0
        xs_h[1::3] = ncols
        xs_h[2::3] = np.nan
        ys_h[0::3] = np.arange(n_h)
        ys_h[1::3] = np.arange(n_h)
        ys_h[2::3] = np.nan
        self._grid_h.setData(xs_h, ys_h)

        # Vertical lines: one per column boundary (x = 0, 1, …, ncols)
        n_v = ncols + 1
        xs_v = np.empty(n_v * 3)
        ys_v = np.empty(n_v * 3)
        xs_v[0::3] = np.arange(n_v)
        xs_v[1::3] = np.arange(n_v)
        xs_v[2::3] = np.nan
        ys_v[0::3] = 0
        ys_v[1::3] = nrows
        ys_v[2::3] = np.nan
        self._grid_v.setData(xs_v, ys_v)

    def _update_grid_visibility(self):
        """Show the pixel grid only when sufficiently zoomed in."""
        if not self._grid_overlay_active or self._grid_ncols == 0:
            self._grid_h.hide()
            self._grid_v.hide()
            return
        [[x0, x1], _] = self.plot.vb.viewRange()
        visible_nifti_pixels = max(x1 - x0, 1)
        screen_pixels_per_nifti = self.width() / visible_nifti_pixels
        if screen_pixels_per_nifti >= self._GRID_THRESHOLD:
            self._grid_h.show()
            self._grid_v.show()
        else:
            self._grid_h.hide()
            self._grid_v.hide()

    def show_overlay(self, visible):
        """Show or hide the colour overlay and pixel-border grid."""
        self._grid_overlay_active = visible
        if visible:
            self.overlay_item.show()
        else:
            self.overlay_item.hide()
        self._update_grid_visibility()

    def set_image(self, data):
        """Set the image data."""
        self.image_data = data
        if data is not None:
            # Normalize to 0-255 for display
            normalized = self.normalize_image(data)
            self.image_item.setImage(normalized.T)  # Transpose for correct orientation
            self.plot.autoRange()

    def normalize_image(self, img):
        """Normalize image to 0-255 range."""
        img_min, img_max = np.min(img), np.max(img)
        if img_max > img_min:
            return ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        return np.zeros_like(img, dtype=np.uint8)

    def get_image_coords(self, pos):
        """Convert scene position to image coordinates."""
        mouse_point = self.plot.vb.mapSceneToView(pos)
        return mouse_point.x(), mouse_point.y()

    def is_in_image(self, x, y):
        """Check if coordinates are within image bounds."""
        if self.image_data is None:
            return False
        h, w = self.image_data.shape
        return 0 <= x < w and 0 <= y < h

    def mousePressEvent(self, event):
        """Handle mouse press events."""
        pos = event.pos()
        x, y = self.get_image_coords(pos)

        if event.button() == Qt.RightButton:
            if self.is_in_image(x, y):
                # Emit click for modes that use right-click (e.g., ruler placement)
                self.clicked.emit(x, y, 2)
            # Always start right-drag for zooming
            self._right_dragging = True
            self._right_drag_start = pos

        elif event.button() == Qt.LeftButton:
            if self.is_in_image(x, y):
                if self._drag_mode_enabled:
                    self._left_dragging = True
                    self.left_drag_started.emit(x, y)
                else:
                    self.clicked.emit(x, y, 1)  # Left click

        elif event.button() == Qt.MiddleButton:
            # Middle button starts panning
            self._panning = True
            self._pan_start = pos

    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        if event.button() == Qt.RightButton:
            self._right_dragging = False
            self._right_drag_start = None
        elif event.button() == Qt.LeftButton:
            if self._left_dragging:
                self._left_dragging = False
                pos = event.pos()
                x, y = self.get_image_coords(pos)
                self.left_drag_ended.emit(x, y)
        elif event.button() == Qt.MiddleButton:
            self._panning = False
            self._pan_start = None

    def mouseMoveEvent(self, event):
        """Handle mouse move events."""
        pos = event.pos()

        if self._left_dragging:
            x, y = self.get_image_coords(pos)
            self.left_drag_moved.emit(x, y)
            return

        if self._right_dragging and self._right_drag_start is not None:
            # Right-drag zooming (vertical movement)
            delta_y = pos.y() - self._right_drag_start.y()
            self._right_drag_start = pos

            # Zoom based on vertical drag (drag up = zoom in, drag down = zoom out)
            if abs(delta_y) > 1:
                # Calculate zoom factor
                factor = 1.0 + delta_y * 0.005  # Smaller multiplier for smoother zoom
                factor = max(0.5, min(2.0, factor))  # Clamp to reasonable range

                # Get center of view for zoom
                vb = self.plot.vb
                view_range = vb.viewRange()
                center_x = (view_range[0][0] + view_range[0][1]) / 2
                center_y = (view_range[1][0] + view_range[1][1]) / 2

                # Apply zoom centered on view center
                vb.scaleBy((factor, factor), center=(center_x, center_y))

        elif self._panning and self._pan_start is not None:
            # Middle-button panning
            delta = pos - self._pan_start
            self._pan_start = pos

            # Get current view range
            vb = self.plot.vb
            view_range = vb.viewRange()

            # Calculate pan amount in data coordinates
            view_width = view_range[0][1] - view_range[0][0]
            view_height = view_range[1][1] - view_range[1][0]

            widget_size = self.size()
            dx = -delta.x() * view_width / widget_size.width()
            dy = -delta.y() * view_height / widget_size.height()

            # Apply pan
            vb.translateBy(x=dx, y=dy)
        else:
            # Emit mouse position for ruler tracking
            x, y = self.get_image_coords(pos)
            if self.is_in_image(x, y):
                self.mouse_moved.emit(x, y)

    def set_wheel_zoom_enabled(self, enabled):
        """Enable or disable wheel zooming (used during ruler positioning)."""
        self._wheel_zoom_enabled = enabled

    def wheelEvent(self, event):
        """Handle mouse wheel events."""
        # Get scroll delta
        delta = event.angleDelta().y()

        # Check if we should emit for ruler rotation
        if event.modifiers() == Qt.NoModifier:
            # Emit wheel event for potential ruler rotation
            self.wheel_rotated.emit(delta)

            # Only do zooming if wheel zoom is enabled
            if self._wheel_zoom_enabled:
                pos = event.pos()
                x, y = self.get_image_coords(pos)

                # Zoom factor
                if delta > 0:
                    factor = 1.1
                else:
                    factor = 0.9

                # Zoom centered on mouse position
                self.plot.vb.scaleBy((1/factor, 1/factor), center=(x, y))

    def update_landmarks(self, landmarks, show_labels=True, force_all_labels=False):
        """Update landmark display."""
        # Clear old labels
        for item in self.label_items:
            self.plot.removeItem(item)
        self.label_items = []

        if not landmarks:
            self.landmark_scatter.setData([], [])
            return

        # Update scatter plot
        x_coords = [p[0] for p in landmarks]
        y_coords = [p[1] for p in landmarks]
        self.landmark_scatter.setData(x_coords, y_coords)

        # Add labels
        if show_labels:
            num_landmarks = len(landmarks)
            # Show all labels when forced (e.g. delete spinbox focused), else use density thinning
            if force_all_labels:
                label_indices = set(range(num_landmarks))
            elif num_landmarks > 50:
                label_indices = set(range(0, num_landmarks, 5))
                label_indices.add(num_landmarks - 1)
            elif num_landmarks > 20:
                label_indices = set(range(0, num_landmarks, 2))
                label_indices.add(num_landmarks - 1)
            else:
                label_indices = set(range(num_landmarks))

            for i, (x, y) in enumerate(landmarks):
                if i in label_indices:
                    text = pg.TextItem(str(i + 1), color='y', anchor=(0.5, 1))
                    text.setPos(x, y - 5)
                    self.plot.addItem(text)
                    self.label_items.append(text)

    def highlight_landmark(self, x, y):
        """Highlight a landmark at the given position."""
        self.highlight_scatter.setData([x], [y])

    def clear_highlight(self):
        """Clear landmark highlight."""
        self.highlight_scatter.setData([], [])

    def update_line_overlays(self, lines, original_polylines=None):
        """Show polylines connecting landmarks for each drawn line.

        Args:
            lines: list of lists of (x, y) tuples - current landmark positions per line.
            original_polylines: list of lists of (x, y) tuples - original control point
                polylines (shown as dim background so user sees full extent).
        """
        self.clear_line_overlays()

        # Draw original polylines first (behind) - dim, so user can see full extent
        if original_polylines:
            for points in original_polylines:
                if len(points) < 2:
                    continue
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                item = pg.PlotDataItem(
                    x_coords, y_coords,
                    pen=pg.mkPen(QColor(255, 165, 0, 100), width=2, style=Qt.DotLine)
                )
                self.plot.addItem(item)
                self.line_overlay_bg_items.append(item)

        # Draw current landmark connections on top
        for points in lines:
            if len(points) < 2:
                continue
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            item = pg.PlotDataItem(
                x_coords, y_coords,
                pen=pg.mkPen('r', width=2, style=Qt.DashLine)
            )
            self.plot.addItem(item)
            self.line_overlay_items.append(item)

    def clear_line_overlays(self):
        """Remove all line overlay items."""
        for item in self.line_overlay_items:
            self.plot.removeItem(item)
        self.line_overlay_items = []
        for item in self.line_overlay_bg_items:
            self.plot.removeItem(item)
        self.line_overlay_bg_items = []

    def update_line_preview(self, points, current_pos=None):
        """Update line preview during drawing."""
        if not points:
            self.line_preview.hide()
            self.line_points_scatter.setData([], [])
            return

        all_points = list(points)
        if current_pos:
            all_points.append(current_pos)

        # Draw line
        if len(all_points) >= 2:
            x_coords = [p[0] for p in all_points]
            y_coords = [p[1] for p in all_points]
            self.line_preview.setData(x_coords, y_coords)
            self.line_preview.show()
        else:
            self.line_preview.hide()

        # Draw points
        x_pts = [p[0] for p in points]
        y_pts = [p[1] for p in points]
        self.line_points_scatter.setData(x_pts, y_pts)

    def clear_line_preview(self):
        """Clear line preview."""
        self.line_preview.hide()
        self.line_preview.setData([], [])
        self.line_points_scatter.setData([], [])

    def update_ruler(self, position, angle, length, visible=True, color='r'):
        """Update ruler display."""
        if not visible or position is None:
            self.ruler_line.hide()
            self.ruler_text.hide()
            self.ruler_center.hide()
            return

        x, y = position
        angle_rad = np.radians(angle)
        dx = length * np.cos(angle_rad) / 2
        dy = length * np.sin(angle_rad) / 2

        x1, y1 = x - dx, y - dy
        x2, y2 = x + dx, y + dy

        pen = pg.mkPen(color, width=3)
        self.ruler_line.setPen(pen)
        self.ruler_line.setData([x1, x2], [y1, y2])
        self.ruler_line.show()

        self.ruler_text.setText(f'{length:.1f} px, {angle:.1f}°')
        self.ruler_text.setPos(x, y - 15)
        self.ruler_text.setColor(color)
        self.ruler_text.show()

        self.ruler_center.setData([x], [y])
        self.ruler_center.show()


class ManualDialog(QDialog):
    """Dialog showing the manual/help text."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual - Non-Linear Registration")
        self.setGeometry(200, 200, 600, 500)

        layout = QVBoxLayout(self)

        # Scroll area for text
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        # Text widget
        text_widget = QTextEdit()
        text_widget.setReadOnly(True)

        # Load manual text
        manual_text = self.load_manual_text()
        text_widget.setPlainText(manual_text)

        scroll.setWidget(text_widget)
        layout.addWidget(scroll)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

    def load_manual_text(self):
        """Load manual text from file or return default."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        manual_path = os.path.join(script_dir, "manual_nonlinear_registration.txt")

        if os.path.exists(manual_path):
            try:
                with open(manual_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                return f"Error loading manual: {e}"

        # Default manual text
        return """NON-LINEAR REGISTRATION V2 - MANUAL
====================================

MODES:
------
1. LANDMARK MODE (default) [Key: 1]
   - Left-click on IHC image to place a landmark
   - Then left-click on corresponding point on MRI image
   - Landmarks are numbered and connected

2. LINE MODE [Key: 2]
   - Left-click on IHC to start drawing a line
   - Continue clicking to add points
   - Press ENTER to finish the IHC line
   - Repeat for MRI image
   - Landmarks are auto-generated along the lines

3. RULER MODE [Key: 3]
   - Move mouse to position the ruler
   - Scroll wheel to rotate (0.5 deg per tick)
   - Left-click to place/fixate the ruler
   - First on IHC (red), then on MRI (blue)
   - Press 'R' to toggle ruler visibility

4. MOVE LANDMARK MODE [Key: 4]
   - Click and drag any placed landmark to move it
   - Works on both IHC and MRI images independently
   - Release to confirm new position

5. MOVE LINE MODE [Key: 5]
   - Only works on landmarks that were created via Line Mode
   - Two sub-modes (selectable via radio buttons):
     a) FREE MOVE: Drag any line landmark freely
     b) MOVE ENDPOINT: Drag the first or last landmark of
        a line along the original polyline path
   - After moving, press ENTER to redistribute all points
     on the line evenly (same spacing as original)

NAVIGATION:
-----------
- RIGHT-CLICK + DRAG: Zoom in/out (drag up/down)
- MIDDLE-CLICK + DRAG: Pan the view
- SCROLL WHEEL: Zoom in/out (disabled while positioning ruler)

KEYBOARD SHORTCUTS:
-------------------
- 1-5: Switch modes (Landmark/Line/Ruler/Move Lm/Move Line)
- 6: Focus delete landmark input
- BACKSPACE: Remove last landmark (or last line point)
- R: Toggle ruler visibility
- C: Toggle connection lines
- T: Toggle landmark labels
- ENTER: Finish line / Redistribute (Move Line) / Save
- ESCAPE: Cancel line drawing / Cancel drag
- TAB: Toggle active image (IHC/MRI)

WORKFLOW:
---------
1. Place at least 3 landmark pairs (more = better)
2. Use Move Landmark/Move Line to refine positions
3. Click "Preview" to see the result
4. Adjust landmarks if needed
5. Click "Save Registration" to save the result

OUTPUT:
-------
Results are saved to: Non-linear_registration_results/
- ihc_to_mri_nonlinear.nii.gz (registered image)
- Visualization PNG
"""


class ContrastCurveDialog(QDialog):
    """Dialog for curve-based contrast adjustment with histogram."""

    contrast_changed = pyqtSignal(object)  # Emits the lookup table

    def __init__(self, image_data, parent=None, initial_control_points=None):
        super().__init__(parent)
        self.setWindowTitle("Curve-Based Image Contrast Adjustment")
        self.setGeometry(200, 200, 700, 550)

        self.image_data = image_data
        self.original_data = image_data.copy()

        # Control points: list of (x, y) where x is intensity, y is output (0-1)
        if initial_control_points is not None and len(initial_control_points) >= 2:
            # Use provided control points (from matching_info.json)
            self.control_points = list(initial_control_points)
            print(f"Initialized contrast dialog with {len(self.control_points)} imported control points")
        else:
            # Default: Fixed 3 points: start, middle, end
            mid_x = image_data.max() / 2
            self.control_points = [(0, 0), (mid_x, 0.5), (image_data.max(), 1)]
        self.selected_point_idx = 1  # Middle point selected by default
        self.dragging = False
        self.drag_point_idx = None  # Which point is being dragged

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
        self.plot_widget.setLabel('left', 'Index into Color Map')
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

        # Connect mouse events - use scene for movement tracking
        self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_moved)

        # Install event filter to catch mouse press/release on plot widget
        self.plot_widget.installEventFilter(self)

        layout.addWidget(self.plot_widget)

        # Control point editing section
        control_frame = QFrame()
        control_frame.setFrameStyle(QFrame.StyledPanel)
        control_layout = QHBoxLayout(control_frame)

        control_layout.addWidget(QLabel("Selected control point:"))

        control_layout.addWidget(QLabel("Id:"))
        self.id_spin = QSpinBox()
        self.id_spin.setMinimum(1)
        self.id_spin.setMaximum(3)
        self.id_spin.setValue(2)  # Middle point selected by default
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
        # Calculate histogram
        data_flat = self.original_data.flatten()
        data_flat = data_flat[data_flat > 0]  # Ignore zero values

        if len(data_flat) == 0:
            return

        # Create histogram bins
        num_bins = 100
        hist, bin_edges = np.histogram(data_flat, bins=num_bins)

        # Normalize histogram to fit in plot
        hist_normalized = hist / hist.max() * 0.3  # Scale to 30% of plot height

        # Calculate bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]

        self.histogram_item.setOpts(x=bin_centers, height=hist_normalized, width=bin_width)

        # Set x range based on data
        self.plot_widget.setXRange(0, self.original_data.max() * 1.05)

    def update_curve(self):
        """Update the curve display based on control points."""
        if len(self.control_points) < 2:
            return

        # Sort control points by x
        self.control_points.sort(key=lambda p: p[0])

        # Update scatter plot
        x_pts = [p[0] for p in self.control_points]
        y_pts = [p[1] for p in self.control_points]
        self.points_scatter.setData(x_pts, y_pts)

        # Linear interpolation between points
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
            y = max(0, min(1, y))  # Clamp y to 0-1
            x = max(0, x)  # x must be positive

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

                    # If already dragging, release on click
                    if self.dragging:
                        self.dragging = False
                        self.drag_point_idx = None
                        return True

                    # Check if clicking near existing point to start dragging
                    for i, (px, py) in enumerate(self.control_points):
                        # Calculate distance in normalized coordinates
                        x_range = self.plot_widget.viewRange()[0]
                        y_range = self.plot_widget.viewRange()[1]
                        dx = (x - px) / (x_range[1] - x_range[0]) if x_range[1] != x_range[0] else 0
                        dy = (y - py) / (y_range[1] - y_range[0]) if y_range[1] != y_range[0] else 0
                        dist = np.sqrt(dx**2 + dy**2)

                        if dist < 0.08:  # Click near point - start dragging
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

            # Clamp values
            x = max(0, min(self.original_data.max(), x))
            y = max(0, min(1, y))

            # Don't allow moving first/last point's x position
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
        # Add point in the middle of the curve
        if len(self.control_points) >= 2:
            mid_x = self.original_data.max() / 2
            # Find y value on current curve at mid_x
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
            # Don't remove first or last point
            if 0 < self.selected_point_idx < len(self.control_points) - 1:
                del self.control_points[self.selected_point_idx]
                self.selected_point_idx = min(self.selected_point_idx, len(self.control_points) - 1)
                self.id_spin.setMaximum(len(self.control_points))
                self.update_curve()
                self.update_point_display()
                self.apply_contrast()

    def apply_contrast(self):
        """Apply the contrast curve and emit signal."""
        # Get control points for linear interpolation
        x_pts = [p[0] for p in self.control_points]
        y_pts = [p[1] for p in self.control_points]

        def apply_curve(data):
            """Apply the linear contrast curve to data."""
            result = np.interp(data, x_pts, y_pts)
            return (result * 255).astype(np.uint8)

        self.contrast_changed.emit(apply_curve)

    def apply_and_close(self):
        """Apply the contrast curve and close the dialog."""
        self.apply_contrast()
        self.close()

    def get_contrast_function(self):
        """Return the current contrast function."""
        x_pts = [p[0] for p in self.control_points]
        y_pts = [p[1] for p in self.control_points]

        def apply_curve(data):
            result = np.interp(data, x_pts, y_pts)
            return (result * 255).astype(np.uint8)

        return apply_curve


class OverlayPreviewDialog(QDialog):
    """Preview dialog showing two grayscale images overlayed with an opacity slider."""

    def __init__(self, warped, reference, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Registration Preview - Overlay")
        self.setMinimumSize(700, 700)
        self.resize(800, 800)
        self._parent = parent  # NonLinearRegistrationQt instance

        # Normalize images to 0-255 uint8
        self.warped = self._normalize(warped)
        self.reference = self._normalize(reference)

        layout = QVBoxLayout(self)

        # PyQtGraph view for the overlay
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.view = self.graphics_widget.addViewBox(row=0, col=0)
        self.view.setAspectLocked(True)
        self.view.invertY(True)

        # Bottom image (MRI reference)
        self.ref_item = pg.ImageItem()
        self.ref_item.setImage(self.reference.T)
        self.view.addItem(self.ref_item)

        # Top image (warped IHC) - opacity controlled by slider
        self.warped_item = pg.ImageItem()
        self.warped_item.setImage(self.warped.T)
        self.warped_item.setOpacity(0.5)
        self.view.addItem(self.warped_item)

        layout.addWidget(self.graphics_widget)

        # Slider row
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("MRI"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(10)
        self.slider.valueChanged.connect(self._on_slider_changed)
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(QLabel("IHC"))
        self.pct_label = QLabel("50%")
        self.pct_label.setFixedWidth(40)
        slider_layout.addWidget(self.pct_label)

        self.toggle_btn = QPushButton("Toggle")
        self.toggle_btn.setFixedWidth(60)
        self._toggle_state = False  # False = showing blend, True = showing IHC
        self.toggle_btn.clicked.connect(self._on_toggle)
        slider_layout.addWidget(self.toggle_btn)

        layout.addLayout(slider_layout)

        # Bottom row: Live Update checkbox + Close button
        bottom_layout = QHBoxLayout()
        self.live_update_cb = QCheckBox("Live Update")
        self.live_update_cb.setToolTip("Recompute preview whenever landmarks change")
        self.live_update_cb.toggled.connect(self._on_live_update_toggled)
        bottom_layout.addWidget(self.live_update_cb)
        bottom_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        bottom_layout.addWidget(close_btn)
        layout.addLayout(bottom_layout)

        # Status label for live update feedback
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self.status_label)

    def _on_live_update_toggled(self, checked):
        """Connect/disconnect live update to parent's landmarks_changed signal."""
        if self._parent is not None and hasattr(self._parent, 'landmarks_changed'):
            if checked:
                self._parent.landmarks_changed.connect(self._on_landmarks_changed)
                self.status_label.setText("Live update enabled")
                self._on_landmarks_changed()
            else:
                self._parent.landmarks_changed.disconnect(self._on_landmarks_changed)
                self.status_label.setText("")

    def _on_landmarks_changed(self):
        """Recompute registration and update preview images."""
        if self._parent is None:
            return
        result = self._parent.compute_warped_image()
        if result is None:
            self.status_label.setText("Cannot compute (need 3+ matched pairs)")
            return
        _, warped, mri_img = result
        # Apply contrast if available
        if self._parent.contrast_func is not None:
            mri_img = self._parent.contrast_func(mri_img)
        self.warped = self._normalize(warped)
        self.reference = self._normalize(mri_img)
        self.warped_item.setImage(self.warped.T)
        self.ref_item.setImage(self.reference.T)
        self.status_label.setText("Updated")

    def closeEvent(self, event):
        """Disconnect signal on close to avoid dangling connections."""
        if self.live_update_cb.isChecked() and self._parent is not None:
            try:
                self._parent.landmarks_changed.disconnect(self._on_landmarks_changed)
            except TypeError:
                pass
        super().closeEvent(event)

    def _normalize(self, img):
        """Normalize image to 0-255 float for display."""
        img = img.astype(np.float64)
        vmin, vmax = img.min(), img.max()
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin) * 255.0
        return img

    def _on_slider_changed(self, value):
        """Slider: 0 = fully MRI, 100 = fully IHC."""
        opacity = value / 100.0
        self.warped_item.setOpacity(opacity)
        self.ref_item.setOpacity(1.0 - opacity)
        self.pct_label.setText(f"{value}%")

    def _on_toggle(self):
        """Toggle between 0% (MRI only) and 100% (IHC only)."""
        self._toggle_state = not self._toggle_state
        self.slider.setValue(100 if self._toggle_state else 0)


class _MriOverlayWorker(QThread):
    """Compute the MRI overview overlay (linear + TPS) in a background thread.

    All heavy numpy/scipy work happens in run(); no Qt UI calls are made there.
    """
    computed = pyqtSignal(str)   # emits save_path on success
    failed   = pyqtSignal(str)   # emits error message on failure
    status   = pyqtSignal(str)   # progress text for the status bar

    def __init__(self, png_path, M, landmarks_ihc, landmarks_mri,
                 slice_shape, orientation_desc, output_path, parent=None):
        super().__init__(parent)
        self._png_path = png_path
        self._M        = M
        self._lm_ihc   = list(landmarks_ihc)
        self._lm_mri   = list(landmarks_mri)
        self._shape    = slice_shape        # (H, W) at 1×
        self._orient   = orientation_desc   # str or None
        self._out      = output_path

    # ------------------------------------------------------------------
    def run(self):
        try:
            import numpy as np
            from scipy import interpolate
            from skimage import io as sk_io
            from skimage.transform import (resize as sk_resize,
                                           warp as sk_warp,
                                           AffineTransform as sk_AffineTransform)
            import matplotlib
            import matplotlib.pyplot as plt

            # --- load PNG ---
            self.status.emit("Loading overview PNG…")
            img = sk_io.imread(self._png_path)
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
            img = img.astype(np.uint8)

            while max(img.shape[:2]) > 2000:
                h, w = img.shape[0] // 2, img.shape[1] // 2
                img = sk_resize(img, (h, w), anti_aliasing=True,
                                preserve_range=True).astype(np.uint8)

            if self._orient:
                img = self._apply_orientation(img, self._orient)

            # --- affine at 4× ---
            self.status.emit("Applying linear transform…")
            H, W = self._shape
            img_resized = sk_resize(img, (H * 4, W * 4),
                                    anti_aliasing=True,
                                    preserve_range=True).astype(np.uint8)
            if self._M is not None:
                S     = np.diag([4., 4., 1.])
                S_inv = np.diag([.25, .25, 1.])
                M_4x  = S @ self._M @ S_inv
                tform = sk_AffineTransform(matrix=M_4x)
                channels = []
                for c in range(3):
                    ch = sk_warp(img_resized[:, :, c].astype(np.float32),
                                 tform.inverse, preserve_range=True,
                                 cval=255., order=1)
                    channels.append(ch)
                img_affine = np.clip(np.stack(channels, axis=-1), 0, 255).astype(np.uint8)
            else:
                img_affine = img_resized

            # --- inverse TPS at 4× ---
            if len(self._lm_ihc) >= 3:
                self.status.emit("Applying nonlinear transform (TPS) — please wait…")
                src_pts = np.array(self._lm_ihc) * 4.
                dst_pts = np.array(self._lm_mri) * 4.

                tps_inv_x = interpolate.Rbf(dst_pts[:, 0], dst_pts[:, 1], src_pts[:, 0],
                                             function='thin_plate', smooth=0.)
                tps_inv_y = interpolate.Rbf(dst_pts[:, 0], dst_pts[:, 1], src_pts[:, 1],
                                             function='thin_plate', smooth=0.)

                height, width = img_affine.shape[:2]
                grid_y, grid_x = np.mgrid[0:height, 0:width]
                ihc_x = tps_inv_x(grid_x, grid_y)
                ihc_y = tps_inv_y(grid_x, grid_y)

                rows = np.arange(height)
                cols = np.arange(width)
                channels = []
                for c in range(img_affine.shape[2]):
                    interp = interpolate.RegularGridInterpolator(
                        (rows, cols), img_affine[:, :, c].astype(float),
                        method='linear', bounds_error=False, fill_value=255.)
                    query = np.stack([ihc_y.ravel(), ihc_x.ravel()], axis=-1)
                    ch = interp(query).reshape(height, width)
                    channels.append(ch)
                img_tps = np.clip(np.stack(channels, axis=-1), 0, 255).astype(np.uint8)
            else:
                img_tps = img_affine

            # --- build RGBA and save ---
            self.status.emit("Saving overlay…")
            h, w = img_tps.shape[:2]
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = img_tps
            rgba[:, :, 3]  = np.where(np.all(img_tps > 240, axis=-1), 0, 255).astype(np.uint8)

            orig_backend = matplotlib.get_backend()
            matplotlib.use('Agg')
            plt.switch_backend('Agg')
            plt.imsave(self._out, rgba)
            matplotlib.use(orig_backend)
            plt.switch_backend(orig_backend)

            self.computed.emit(self._out)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.failed.emit(str(e))

    # ------------------------------------------------------------------
    @staticmethod
    def _apply_orientation(image, orientation_desc):
        """Mirror of NonLinearRegistrationQt._apply_orientation for use in thread."""
        img  = image.copy()
        desc = orientation_desc.lower()
        is_rotated_90 = any(x in desc for x in ["rotated 90", "rot 90 cw", "90° clockwise"])
        has_flip_h    = any(x in desc for x in ["flipped horizontally", "flip h", "flipped h"])
        has_flip_v    = any(x in desc for x in ["flipped vertically",   "flip v", "flipped v"])
        if "h+v" in desc or "horizontally and vertically" in desc or "both" in desc:
            has_flip_h = has_flip_v = True
        import numpy as np
        if is_rotated_90:
            img = np.rot90(img, k=-1)
        if has_flip_h:
            img = np.fliplr(img)
        if has_flip_v:
            img = np.flipud(img)
        return img


class NonLinearRegistrationQt(QMainWindow):
    """Main window for non-linear registration."""

    landmarks_changed = pyqtSignal()  # Emitted when landmarks are added/moved/deleted

    def __init__(self):
        super().__init__()

        # File paths
        self.ihc_path = None
        self.mri_path = None
        self.registered_ihc_path = None

        # Image data
        self.ihc_nii = None
        self.mri_nii = None
        self.registered_ihc_data = None
        self.mri_data = None

        # Slice info
        self.slice_dimension = None
        self.slice_index = None

        # Landmarks
        self.landmarks_ihc = []
        self.landmarks_mri = []
        self.active_image = 'ihc'

        # Mode states
        self.landmark_mode = True
        self.line_mode = False
        self.ruler_mode = False

        # Line drawing state
        self.line_drawing_ihc = False
        self.line_drawing_mri = False
        self.line_points_ihc = []
        self.line_points_mri = []
        self.landmark_spacing = 8.0

        # Line storage: tracks which landmarks came from lines
        # Each entry: {'ihc_control_points': [...], 'mri_control_points': [...], 'landmark_indices': [...]}
        self.drawn_lines = []

        # Move modes
        self.move_landmark_mode = False
        self.move_line_mode = False
        self.move_line_sub_mode = 'free'  # 'free' or 'endpoint'

        # Drag state
        self._dragging = False
        self._drag_index = None       # index into landmarks_ihc/mri
        self._drag_side = None        # 'ihc' or 'mri'
        self._drag_line_idx = None    # index into drawn_lines (for move line mode)
        self._drag_original_pos = None  # original position before drag (for cancel)
        self._modified_line_idx = None  # tracks which line was last modified (for Enter redistribution)

        # Ruler state
        self.ruler_visible = False
        self.ruler_position_ihc = None
        self.ruler_position_mri = None
        self.ruler_angle_ihc = 0.0
        self.ruler_angle_mri = 0.0
        self.ruler_length = 100
        self.ruler_fixated_ihc = False
        self.ruler_fixated_mri = False

        # Display options
        self.show_labels = True
        self.show_connections = True

        # Output directory
        self.output_dir = None

        # Contrast adjustment - MRI
        self.mri_slice_original = None  # Original MRI slice data
        self.contrast_dialog = None
        self.contrast_control_points = None  # Loaded from matching_info.json
        self.contrast_func = None  # Interpolation function from control points
        self.matching_info = None  # Cached matching info from JSON

        # Contrast adjustment - IHC
        self.ihc_slice_original = None
        self.ihc_contrast_dialog = None
        self.ihc_contrast_control_points = None
        self.ihc_contrast_func = None

        # Overview overlay
        self.overview_png_path = None  # source PNG selected by user
        self._mri_overlay_worker = None  # QThread kept alive during background compute

        # Setup UI
        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("Non-Linear Registration (PyQt5)")
        self.setMinimumSize(900, 600)
        self.resize(1400, 900)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # Splitter for image views
        splitter = QSplitter(Qt.Horizontal)

        # IHC view
        ihc_container = QWidget()
        ihc_layout = QVBoxLayout(ihc_container)
        ihc_layout.setContentsMargins(0, 0, 0, 0)

        self.ihc_view = ImageViewWidget("IHC (Click here first)")
        self.ihc_view.clicked.connect(self.on_ihc_clicked)
        self.ihc_view.mouse_moved.connect(self.on_ihc_mouse_moved)
        self.ihc_view.wheel_rotated.connect(self.on_ihc_wheel)
        self.ihc_view.left_drag_started.connect(lambda x, y: self.on_drag_started(x, y, 'ihc'))
        self.ihc_view.left_drag_moved.connect(lambda x, y: self.on_drag_moved(x, y, 'ihc'))
        self.ihc_view.left_drag_ended.connect(lambda x, y: self.on_drag_ended(x, y, 'ihc'))
        ihc_layout.addWidget(self.ihc_view)

        # Buttons below IHC view
        ihc_btn_layout = QHBoxLayout()
        ihc_btn_layout.addStretch()
        self.btn_change_contrast_ihc = QPushButton("Change Contrast")
        self.btn_change_contrast_ihc.setStyleSheet("background-color: #f0f0f0;")
        self.btn_change_contrast_ihc.clicked.connect(self.show_ihc_contrast_dialog)
        ihc_btn_layout.addWidget(self.btn_change_contrast_ihc)
        self.chk_overlay_overview = QCheckBox("Overlay")
        self.chk_overlay_overview.setToolTip(
            "Overlay the original colour PNG over the IHC view.\n"
            "First use: opens a file dialog to choose the PNG; result is saved to\n"
            "Non-linear_registration_results/overview_overlay.png.\n"
            "Subsequent uses: reloads the saved file."
        )
        self.chk_overlay_overview.toggled.connect(self.on_overview_overlay_toggled)
        ihc_btn_layout.addWidget(self.chk_overlay_overview)
        btn_del_overlay_ihc = QPushButton("\u00d7")
        btn_del_overlay_ihc.setFixedSize(22, 22)
        btn_del_overlay_ihc.setFont(QFont(btn_del_overlay_ihc.font().family(), 14))
        btn_del_overlay_ihc.setToolTip("Delete cached IHC overlay")
        btn_del_overlay_ihc.setContentsMargins(0, 0, 0, 0)
        btn_del_overlay_ihc.setStyleSheet(
            "QPushButton { border: none; color: #999; padding: 0px; margin-left: -4px; } "
            "QPushButton:hover { color: #e74c3c; }")
        btn_del_overlay_ihc.clicked.connect(self._delete_ihc_overlay)
        ihc_btn_layout.addWidget(btn_del_overlay_ihc)
        ihc_btn_layout.addStretch()
        ihc_layout.addLayout(ihc_btn_layout)

        # MRI view
        mri_container = QWidget()
        mri_layout = QVBoxLayout(mri_container)
        mri_layout.setContentsMargins(0, 0, 0, 0)

        self.mri_view = ImageViewWidget("MRI Reference")
        self.mri_view.clicked.connect(self.on_mri_clicked)
        self.mri_view.mouse_moved.connect(self.on_mri_mouse_moved)
        self.mri_view.wheel_rotated.connect(self.on_mri_wheel)
        self.mri_view.left_drag_started.connect(lambda x, y: self.on_drag_started(x, y, 'mri'))
        self.mri_view.left_drag_moved.connect(lambda x, y: self.on_drag_moved(x, y, 'mri'))
        self.mri_view.left_drag_ended.connect(lambda x, y: self.on_drag_ended(x, y, 'mri'))
        mri_layout.addWidget(self.mri_view)

        # Change Contrast button at bottom center of MRI view
        mri_btn_layout = QHBoxLayout()
        mri_btn_layout.addStretch()
        self.btn_change_contrast = QPushButton("Change Contrast")
        self.btn_change_contrast.setStyleSheet("background-color: #f0f0f0;")
        self.btn_change_contrast.clicked.connect(self.show_contrast_dialog)
        mri_btn_layout.addWidget(self.btn_change_contrast)
        self.chk_overlay_overview_mri = QCheckBox("Overlay")
        self.chk_overlay_overview_mri.setToolTip(
            "Overlay the original colour PNG over the MRI view.\n"
            "Applies linear + nonlinear (TPS) transforms to the PNG.\n"
            "First use: opens a file dialog to choose the PNG; result is saved to\n"
            "Non-linear_registration_results/overview_overlay_mri.png.\n"
            "Subsequent uses: reloads the saved file."
        )
        self.chk_overlay_overview_mri.toggled.connect(self.on_mri_overview_overlay_toggled)
        mri_btn_layout.addWidget(self.chk_overlay_overview_mri)
        btn_del_overlay_mri = QPushButton("\u00d7")
        btn_del_overlay_mri.setFixedSize(22, 22)
        btn_del_overlay_mri.setFont(QFont(btn_del_overlay_mri.font().family(), 14))
        btn_del_overlay_mri.setToolTip("Delete cached MRI overlay")
        btn_del_overlay_mri.setContentsMargins(0, 0, 0, 0)
        btn_del_overlay_mri.setStyleSheet(
            "QPushButton { border: none; color: #999; padding: 0px; margin-left: -4px; } "
            "QPushButton:hover { color: #e74c3c; }")
        btn_del_overlay_mri.clicked.connect(self._delete_mri_overlay)
        mri_btn_layout.addWidget(btn_del_overlay_mri)
        mri_btn_layout.addStretch()
        mri_layout.addLayout(mri_btn_layout)

        splitter.addWidget(ihc_container)
        splitter.addWidget(mri_container)
        splitter.setSizes([700, 700])

        main_layout.addWidget(splitter, stretch=1)

        # Control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.update_status("Ready - Load images to begin")

        # Keyboard shortcuts
        self.setup_shortcuts()

    def create_control_panel(self):
        """Create the control panel with buttons."""
        panel = QWidget()
        panel.setMaximumHeight(80)
        panel.setStyleSheet("font-size: 11px;")
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        # Manual button (outside mode group)
        self.btn_manual = QPushButton("Manual")
        self.btn_manual.setStyleSheet("background-color: #e0e0ff;")
        self.btn_manual.clicked.connect(self.show_manual)
        layout.addWidget(self.btn_manual)

        # Mode selection group
        mode_group = QGroupBox("Mode")
        mode_layout = QHBoxLayout(mode_group)
        mode_layout.setContentsMargins(4, 2, 4, 2)
        mode_layout.setSpacing(3)

        self.btn_landmark_mode = QPushButton("Landmark")
        self.btn_landmark_mode.setCheckable(True)
        self.btn_landmark_mode.setChecked(True)
        self.btn_landmark_mode.clicked.connect(lambda: self.set_mode('landmark'))

        self.btn_line_mode = QPushButton("Line")
        self.btn_line_mode.setCheckable(True)
        self.btn_line_mode.clicked.connect(lambda: self.set_mode('line'))

        self.btn_ruler_mode = QPushButton("Ruler")
        self.btn_ruler_mode.setCheckable(True)
        self.btn_ruler_mode.clicked.connect(lambda: self.set_mode('ruler'))

        self.btn_move_landmark_mode = QPushButton("Move Lm")
        self.btn_move_landmark_mode.setCheckable(True)
        self.btn_move_landmark_mode.clicked.connect(lambda: self.set_mode('move_landmark'))

        self.btn_move_line_mode = QPushButton("Move Line")
        self.btn_move_line_mode.setCheckable(True)
        self.btn_move_line_mode.clicked.connect(lambda: self.set_mode('move_line'))

        mode_layout.addWidget(self.btn_landmark_mode)
        mode_layout.addWidget(self.btn_line_mode)
        mode_layout.addWidget(self.btn_ruler_mode)
        mode_layout.addWidget(self.btn_move_landmark_mode)
        mode_layout.addWidget(self.btn_move_line_mode)

        layout.addWidget(mode_group)

        # Move Line sub-mode panel (hidden by default)
        self.move_line_submode_frame = QFrame()
        self.move_line_submode_frame.setFrameStyle(QFrame.StyledPanel)
        submode_layout = QHBoxLayout(self.move_line_submode_frame)
        submode_layout.setContentsMargins(4, 2, 4, 2)
        submode_layout.addWidget(QLabel("Sub-mode:"))
        self.radio_free_move = QRadioButton("Free Move")
        self.radio_free_move.setChecked(True)
        self.radio_free_move.toggled.connect(self._on_move_line_submode_changed)
        self.radio_endpoint_move = QRadioButton("Move Endpoint")
        self.radio_endpoint_move.toggled.connect(self._on_move_line_submode_changed)
        submode_layout.addWidget(self.radio_free_move)
        submode_layout.addWidget(self.radio_endpoint_move)
        self.move_line_submode_frame.setVisible(False)
        layout.addWidget(self.move_line_submode_frame)

        # Active image selection
        active_group = QGroupBox("Active Image")
        active_layout = QHBoxLayout(active_group)
        active_layout.setContentsMargins(4, 2, 4, 2)
        active_layout.setSpacing(3)

        self.radio_ihc = QRadioButton("IHC")
        self.radio_ihc.setChecked(True)
        self.radio_ihc.toggled.connect(self.on_active_image_changed)

        self.radio_mri = QRadioButton("MRI")
        self.radio_mri.toggled.connect(self.on_active_image_changed)

        active_layout.addWidget(self.radio_ihc)
        active_layout.addWidget(self.radio_mri)

        layout.addWidget(active_group)

        # Landmark management
        landmark_group = QGroupBox("Landmarks")
        landmark_layout = QHBoxLayout(landmark_group)
        landmark_layout.setContentsMargins(4, 2, 4, 2)
        landmark_layout.setSpacing(3)

        self.btn_save_landmarks = QPushButton("Save")
        self.btn_save_landmarks.clicked.connect(self.save_landmarks)

        self.btn_load_landmarks = QPushButton("Load")
        self.btn_load_landmarks.clicked.connect(self.load_landmarks)

        self.landmark_input = QSpinBox()
        self.landmark_input.setMinimum(1)
        self.landmark_input.setMaximum(9999)
        self.landmark_input.setPrefix("Delete #")
        self.landmark_input.installEventFilter(self)

        self.btn_delete = QPushButton("Delete")
        self.btn_delete.clicked.connect(self.delete_specific_landmark)

        self.btn_reset = QPushButton("Reset All")
        self.btn_reset.setStyleSheet("background-color: #ffcccc;")
        self.btn_reset.clicked.connect(self.reset_landmarks)

        landmark_layout.addWidget(self.landmark_input)
        landmark_layout.addWidget(self.btn_delete)
        landmark_layout.addWidget(self.btn_save_landmarks)
        landmark_layout.addWidget(self.btn_load_landmarks)
        landmark_layout.addWidget(self.btn_reset)

        layout.addWidget(landmark_group)

        # Registration buttons
        reg_group = QGroupBox("Registration")
        reg_layout = QHBoxLayout(reg_group)
        reg_layout.setContentsMargins(4, 2, 4, 2)
        reg_layout.setSpacing(3)

        self.btn_preview = QPushButton("Preview")
        self.btn_preview.setStyleSheet("background-color: #ccffcc;")
        self.btn_preview.clicked.connect(self.preview_registration)

        self.btn_save_reg = QPushButton("Save Registration")
        self.btn_save_reg.setStyleSheet("background-color: #99ff99;")
        self.btn_save_reg.clicked.connect(self.save_registration)

        reg_layout.addWidget(self.btn_preview)
        reg_layout.addWidget(self.btn_save_reg)

        layout.addWidget(reg_group)

        # Landmark counter
        self.landmark_label = QLabel("Landmarks: 0 pairs")
        self.landmark_label.setStyleSheet("font-weight: bold; color: blue;")
        layout.addWidget(self.landmark_label)

        layout.addStretch()

        return panel

    def setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Backspace - context-sensitive (line point or landmark)
        backspace = QAction(self)
        backspace.setShortcut(QKeySequence(Qt.Key_Backspace))
        backspace.triggered.connect(self.on_backspace)
        self.addAction(backspace)

        # R - toggle ruler
        r_key = QAction(self)
        r_key.setShortcut(QKeySequence('R'))
        r_key.triggered.connect(self.toggle_ruler_visibility)
        self.addAction(r_key)

        # C - toggle connections
        c_key = QAction(self)
        c_key.setShortcut(QKeySequence('C'))
        c_key.triggered.connect(self.toggle_connections)
        self.addAction(c_key)

        # T - toggle labels
        t_key = QAction(self)
        t_key.setShortcut(QKeySequence('T'))
        t_key.triggered.connect(self.toggle_labels)
        self.addAction(t_key)

        # Enter - finish line or save (or delete landmark if spinbox focused)
        enter_key = QAction(self)
        enter_key.setShortcut(QKeySequence(Qt.Key_Return))
        enter_key.triggered.connect(self.on_enter_pressed)
        self.addAction(enter_key)

        # Escape - cancel or close
        esc_key = QAction(self)
        esc_key.setShortcut(QKeySequence(Qt.Key_Escape))
        esc_key.triggered.connect(self.on_escape_pressed)
        self.addAction(esc_key)

        # 1 - Landmark mode
        key_1 = QAction(self)
        key_1.setShortcut(QKeySequence('1'))
        key_1.triggered.connect(lambda: self.set_mode('landmark'))
        self.addAction(key_1)

        # 2 - Line mode
        key_2 = QAction(self)
        key_2.setShortcut(QKeySequence('2'))
        key_2.triggered.connect(lambda: self.set_mode('line'))
        self.addAction(key_2)

        # 3 - Ruler mode
        key_3 = QAction(self)
        key_3.setShortcut(QKeySequence('3'))
        key_3.triggered.connect(lambda: self.set_mode('ruler'))
        self.addAction(key_3)

        # 4 - Move Landmark mode
        key_4 = QAction(self)
        key_4.setShortcut(QKeySequence('4'))
        key_4.triggered.connect(lambda: self.set_mode('move_landmark'))
        self.addAction(key_4)

        # 5 - Move Line mode
        key_5 = QAction(self)
        key_5.setShortcut(QKeySequence('5'))
        key_5.triggered.connect(lambda: self.set_mode('move_line'))
        self.addAction(key_5)

        # 6 - Focus delete landmark textbox
        key_6 = QAction(self)
        key_6.setShortcut(QKeySequence('6'))
        key_6.triggered.connect(self.focus_delete_input)
        self.addAction(key_6)

        # Tab - Toggle between IHC and MRI
        tab_key = QAction(self)
        tab_key.setShortcut(QKeySequence(Qt.Key_Tab))
        tab_key.triggered.connect(self.toggle_active_image)
        self.addAction(tab_key)

    def show_manual(self):
        """Show the manual dialog."""
        dialog = ManualDialog(self)
        dialog.exec_()

    def show_contrast_dialog(self):
        """Show the contrast adjustment dialog for MRI."""
        if self.mri_slice_original is None:
            QMessageBox.warning(self, "No Image", "Load MRI image first.")
            return

        # Create dialog with current MRI slice data and any loaded control points
        self.contrast_dialog = ContrastCurveDialog(
            self.mri_slice_original, self,
            initial_control_points=self.contrast_control_points
        )
        self.contrast_dialog.contrast_changed.connect(self.on_contrast_changed)
        self.contrast_dialog.show()

    def on_contrast_changed(self, contrast_func):
        """Handle contrast curve change from dialog."""
        if self.mri_slice_original is None:
            return

        # Store the contrast function for use in previews/saving
        self.contrast_func = contrast_func

        # Also store control points from the dialog for potential re-use
        if self.contrast_dialog is not None:
            self.contrast_control_points = list(self.contrast_dialog.control_points)

        # Apply contrast function to original data
        adjusted = contrast_func(self.mri_slice_original)

        # Update MRI view with adjusted image
        self.mri_view.image_item.setImage(adjusted.T)

    def show_ihc_contrast_dialog(self):
        """Show the contrast adjustment dialog for IHC."""
        if self.ihc_slice_original is None:
            QMessageBox.warning(self, "No Image", "Load IHC image first.")
            return

        self.ihc_contrast_dialog = ContrastCurveDialog(
            self.ihc_slice_original, self,
            initial_control_points=self.ihc_contrast_control_points
        )
        self.ihc_contrast_dialog.contrast_changed.connect(self.on_ihc_contrast_changed)
        self.ihc_contrast_dialog.show()

    def on_ihc_contrast_changed(self, contrast_func):
        """Handle contrast curve change for IHC."""
        if self.ihc_slice_original is None:
            return

        self.ihc_contrast_func = contrast_func

        # Store control points so the curve persists when reopening
        if self.ihc_contrast_dialog is not None:
            self.ihc_contrast_control_points = list(self.ihc_contrast_dialog.control_points)

        # Apply contrast function to original data
        adjusted = contrast_func(self.ihc_slice_original)

        # Update IHC view with adjusted image
        self.ihc_view.image_item.setImage(adjusted.T)

    def set_mode(self, mode):
        """Set the current interaction mode."""
        self.landmark_mode = (mode == 'landmark')
        self.line_mode = (mode == 'line')
        self.ruler_mode = (mode == 'ruler')
        self.move_landmark_mode = (mode == 'move_landmark')
        self.move_line_mode = (mode == 'move_line')

        self.btn_landmark_mode.setChecked(mode == 'landmark')
        self.btn_line_mode.setChecked(mode == 'line')
        self.btn_ruler_mode.setChecked(mode == 'ruler')
        self.btn_move_landmark_mode.setChecked(mode == 'move_landmark')
        self.btn_move_line_mode.setChecked(mode == 'move_line')

        # Show/hide move line sub-mode panel
        self.move_line_submode_frame.setVisible(mode == 'move_line')

        # Clear line drawing state if switching away from line mode
        if not self.line_mode:
            self.line_drawing_ihc = False
            self.line_drawing_mri = False
            self.line_points_ihc = []
            self.line_points_mri = []
            self.ihc_view.clear_line_preview()
            self.mri_view.clear_line_preview()

        # Enable drag mode for move modes
        drag_enabled = self.move_landmark_mode or self.move_line_mode
        self.ihc_view.set_drag_mode(drag_enabled)
        self.mri_view.set_drag_mode(drag_enabled)

        # Clear any drag state
        self._dragging = False
        self._drag_index = None
        self._drag_side = None
        self._drag_line_idx = None
        self._drag_original_pos = None
        self.ihc_view.clear_highlight()
        self.mri_view.clear_highlight()

        # Show/hide line overlays
        self._update_line_overlays()

        # Initialize ruler mode
        if self.ruler_mode:
            self.ruler_visible = True
            self.ruler_fixated_ihc = False
            self.ruler_fixated_mri = False
            # Disable wheel zoom while positioning rulers
            self.ihc_view.set_wheel_zoom_enabled(False)
            self.mri_view.set_wheel_zoom_enabled(False)
            self.update_status("Ruler mode: Move mouse to position, scroll to rotate, left-click to place")
        else:
            # Re-enable wheel zoom when leaving ruler mode
            self.ihc_view.set_wheel_zoom_enabled(True)
            self.mri_view.set_wheel_zoom_enabled(True)
            if mode == 'move_landmark':
                self.update_status("Move Landmark: Click and drag a landmark to move it")
            elif mode == 'move_line':
                self.update_status(f"Move Line ({self.move_line_sub_mode}): Click and drag a line landmark")
            else:
                self.update_status(f"Mode: {mode.capitalize()}")

    def _update_line_overlays(self):
        """Update line overlays on both views from drawn_lines data."""
        if not self.move_line_mode or not self.drawn_lines:
            self.ihc_view.clear_line_overlays()
            self.mri_view.clear_line_overlays()
            return
        ihc_lines = []
        mri_lines = []
        ihc_originals = []
        mri_originals = []
        for line in self.drawn_lines:
            ihc_pts = [self.landmarks_ihc[i] for i in line['landmark_indices']
                       if i < len(self.landmarks_ihc)]
            mri_pts = [self.landmarks_mri[i] for i in line['landmark_indices']
                       if i < len(self.landmarks_mri)]
            ihc_lines.append(ihc_pts)
            mri_lines.append(mri_pts)
            ihc_originals.append(line['ihc_control_points'])
            mri_originals.append(line['mri_control_points'])
        self.ihc_view.update_line_overlays(ihc_lines, ihc_originals)
        self.mri_view.update_line_overlays(mri_lines, mri_originals)

    def _on_move_line_submode_changed(self):
        """Handle move line sub-mode radio button change."""
        if self.radio_free_move.isChecked():
            self.move_line_sub_mode = 'free'
        else:
            self.move_line_sub_mode = 'endpoint'
        if self.move_line_mode:
            self.update_status(f"Move Line ({self.move_line_sub_mode}): Click and drag a line landmark")

    def on_active_image_changed(self):
        """Handle active image radio button change."""
        self.active_image = 'ihc' if self.radio_ihc.isChecked() else 'mri'

    def toggle_active_image(self):
        """Toggle between IHC and MRI active image."""
        if self.radio_ihc.isChecked():
            self.radio_mri.setChecked(True)
        else:
            self.radio_ihc.setChecked(True)

    def eventFilter(self, obj, event):
        """Refresh landmark labels when the delete spinbox gains or loses focus."""
        if obj is self.landmark_input and event.type() in (event.FocusIn, event.FocusOut):
            self.update_landmarks_display()
        return super().eventFilter(obj, event)

    def focus_delete_input(self):
        """Focus the delete landmark spinbox."""
        self.landmark_input.setFocus()
        self.landmark_input.selectAll()

    def on_ihc_clicked(self, x, y, button):
        """Handle click on IHC image."""
        # Left click
        if button == 1:
            if self.ruler_mode:
                # Place/fixate ruler with left click
                self.ruler_fixated_ihc = not self.ruler_fixated_ihc
                if self.ruler_fixated_ihc:
                    self.ruler_position_ihc = (x, y)
                    self.update_ruler_display()
                    # Re-enable wheel zoom on IHC since ruler is placed
                    self.ihc_view.set_wheel_zoom_enabled(True)
                    self.update_status("IHC ruler placed. Now position ruler on MRI.")
                else:
                    # Disable wheel zoom again since ruler is being repositioned
                    self.ihc_view.set_wheel_zoom_enabled(False)
                    self.update_status("IHC ruler released. Move mouse to reposition.")
                return

        if self.line_mode:
            self.handle_line_click_ihc(x, y)
            return

        if self.landmark_mode and self.active_image == 'ihc':
            if len(self.landmarks_ihc) > len(self.landmarks_mri):
                self.update_status("Place corresponding landmark on MRI first")
                return

            self.landmarks_ihc.append((x, y))
            self.active_image = 'mri'
            self.radio_mri.setChecked(True)

            self.update_landmarks_display()
            self.update_status(f"Landmark {len(self.landmarks_ihc)} placed on IHC. Now click MRI.")

    def on_mri_clicked(self, x, y, button):
        """Handle click on MRI image."""
        # Left click
        if button == 1:
            if self.ruler_mode:
                # Place/fixate ruler with left click (only if IHC ruler is already placed)
                if self.ruler_fixated_ihc:
                    self.ruler_fixated_mri = not self.ruler_fixated_mri
                    if self.ruler_fixated_mri:
                        self.ruler_position_mri = (x, y)
                        self.update_ruler_display()
                        # Re-enable wheel zoom on MRI since ruler is placed
                        self.mri_view.set_wheel_zoom_enabled(True)
                        self.update_status("MRI ruler placed.")
                    else:
                        # Disable wheel zoom again since ruler is being repositioned
                        self.mri_view.set_wheel_zoom_enabled(False)
                        self.update_status("MRI ruler released. Move mouse to reposition.")
                else:
                    self.update_status("Place IHC ruler first.")
                return

        if self.line_mode:
            self.handle_line_click_mri(x, y)
            return

        if self.landmark_mode and self.active_image == 'mri':
            if len(self.landmarks_ihc) <= len(self.landmarks_mri):
                self.update_status("Place landmark on IHC first")
                return

            self.landmarks_mri.append((x, y))
            self.active_image = 'ihc'
            self.radio_ihc.setChecked(True)

            self.update_landmarks_display()
            self.landmark_label.setText(f"Landmarks: {len(self.landmarks_mri)} pairs")
            self.update_status(f"Landmark pair {len(self.landmarks_mri)} complete.")

    def on_ihc_mouse_moved(self, x, y):
        """Handle mouse movement on IHC for ruler positioning."""
        if self.ruler_mode and not self.ruler_fixated_ihc:
            self.ruler_position_ihc = (x, y)
            self.update_ruler_display()

    def on_mri_mouse_moved(self, x, y):
        """Handle mouse movement on MRI for ruler positioning."""
        if self.ruler_mode and self.ruler_fixated_ihc and not self.ruler_fixated_mri:
            self.ruler_position_mri = (x, y)
            self.update_ruler_display()

    def on_ihc_wheel(self, delta):
        """Handle wheel rotation on IHC for ruler rotation."""
        if self.ruler_mode and not self.ruler_fixated_ihc:
            # Rotate ruler by 0.5 degrees per tick
            rotation = 0.5 if delta > 0 else -0.5
            self.ruler_angle_ihc = (self.ruler_angle_ihc + rotation) % 360
            self.update_ruler_display()

    def on_mri_wheel(self, delta):
        """Handle wheel rotation on MRI for ruler rotation."""
        if self.ruler_mode and self.ruler_fixated_ihc and not self.ruler_fixated_mri:
            # Rotate ruler by 0.5 degrees per tick
            rotation = 0.5 if delta > 0 else -0.5
            self.ruler_angle_mri = (self.ruler_angle_mri + rotation) % 360
            self.update_ruler_display()

    def update_ruler_display(self):
        """Update ruler display on both images."""
        if self.ruler_visible:
            self.ihc_view.update_ruler(
                self.ruler_position_ihc, self.ruler_angle_ihc,
                self.ruler_length, True, 'r'
            )
            if self.ruler_fixated_ihc:
                self.mri_view.update_ruler(
                    self.ruler_position_mri, self.ruler_angle_mri,
                    self.ruler_length, True, 'b'
                )
            else:
                self.mri_view.update_ruler(None, 0, 0, False)
        else:
            self.ihc_view.update_ruler(None, 0, 0, False)
            self.mri_view.update_ruler(None, 0, 0, False)

    def handle_line_click_ihc(self, x, y):
        """Handle line mode click on IHC."""
        if not self.line_drawing_ihc and not self.line_drawing_mri:
            self.line_drawing_ihc = True
            self.line_points_ihc = [(x, y)]
            self.ihc_view.update_line_preview(self.line_points_ihc)
            self.update_status("IHC line started. Click to add points, press Enter to finish.")
        elif self.line_drawing_ihc:
            self.line_points_ihc.append((x, y))
            self.ihc_view.update_line_preview(self.line_points_ihc)
            self.update_status(f"IHC line: {len(self.line_points_ihc)} points")

    def handle_line_click_mri(self, x, y):
        """Handle line mode click on MRI."""
        if self.line_drawing_mri:
            self.line_points_mri.append((x, y))
            self.mri_view.update_line_preview(self.line_points_mri)
            self.update_status(f"MRI line: {len(self.line_points_mri)} points")

    # ---- Drag handlers for Move Landmark / Move Line ----

    def _find_nearest_landmark(self, x, y, side, threshold=15.0):
        """Find nearest landmark to (x, y) within threshold. Returns index or None."""
        landmarks = self.landmarks_ihc if side == 'ihc' else self.landmarks_mri
        if not landmarks:
            return None
        best_idx = None
        best_dist = threshold
        for i, (lx, ly) in enumerate(landmarks):
            dist = np.sqrt((lx - x) ** 2 + (ly - y) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    def _find_line_for_landmark(self, landmark_idx):
        """Find which drawn_line contains this landmark index. Returns line index or None."""
        for i, line in enumerate(self.drawn_lines):
            if landmark_idx in line['landmark_indices']:
                return i
        return None

    def _is_line_endpoint(self, landmark_idx, line_idx):
        """Check if landmark is the first or last in a line."""
        indices = self.drawn_lines[line_idx]['landmark_indices']
        return landmark_idx == indices[0] or landmark_idx == indices[-1]

    def on_drag_started(self, x, y, side):
        """Handle start of a left-button drag on either image."""
        if self.move_landmark_mode:
            idx = self._find_nearest_landmark(x, y, side)
            if idx is not None:
                self._dragging = True
                self._drag_index = idx
                self._drag_side = side
                landmarks = self.landmarks_ihc if side == 'ihc' else self.landmarks_mri
                self._drag_original_pos = landmarks[idx]
                view = self.ihc_view if side == 'ihc' else self.mri_view
                view.highlight_landmark(x, y)
                self.update_status(f"Moving landmark {idx + 1}")

        elif self.move_line_mode:
            idx = self._find_nearest_landmark(x, y, side)
            if idx is None:
                return
            line_idx = self._find_line_for_landmark(idx)
            if line_idx is None:
                self.update_status("That landmark doesn't belong to a line")
                return

            if self.move_line_sub_mode == 'endpoint':
                if not self._is_line_endpoint(idx, line_idx):
                    self.update_status("Endpoint mode: select first or last landmark of a line")
                    return

            self._dragging = True
            self._drag_index = idx
            self._drag_side = side
            self._drag_line_idx = line_idx
            landmarks = self.landmarks_ihc if side == 'ihc' else self.landmarks_mri
            self._drag_original_pos = landmarks[idx]
            view = self.ihc_view if side == 'ihc' else self.mri_view
            view.highlight_landmark(x, y)
            self.update_status(f"Moving line landmark {idx + 1} ({self.move_line_sub_mode} mode)")

    def on_drag_moved(self, x, y, side):
        """Handle drag movement."""
        if not self._dragging or side != self._drag_side:
            return

        if self.move_landmark_mode:
            # Move landmark freely
            if self._drag_side == 'ihc':
                self.landmarks_ihc[self._drag_index] = (x, y)
            else:
                self.landmarks_mri[self._drag_index] = (x, y)
            view = self.ihc_view if side == 'ihc' else self.mri_view
            view.highlight_landmark(x, y)
            self.update_landmarks_display()

        elif self.move_line_mode:
            if self.move_line_sub_mode == 'free':
                # Free move: update position directly
                if self._drag_side == 'ihc':
                    self.landmarks_ihc[self._drag_index] = (x, y)
                else:
                    self.landmarks_mri[self._drag_index] = (x, y)
                view = self.ihc_view if side == 'ihc' else self.mri_view
                view.highlight_landmark(x, y)
                self.update_landmarks_display()

            elif self.move_line_sub_mode == 'endpoint':
                # Endpoint mode: project onto original polyline
                line = self.drawn_lines[self._drag_line_idx]
                control_pts = line['ihc_control_points'] if self._drag_side == 'ihc' else line['mri_control_points']
                proj_x, proj_y = self._project_onto_polyline(x, y, control_pts)
                if self._drag_side == 'ihc':
                    self.landmarks_ihc[self._drag_index] = (proj_x, proj_y)
                else:
                    self.landmarks_mri[self._drag_index] = (proj_x, proj_y)
                view = self.ihc_view if side == 'ihc' else self.mri_view
                view.highlight_landmark(proj_x, proj_y)
                self.update_landmarks_display()

    def on_drag_ended(self, x, y, side):
        """Handle end of drag."""
        if not self._dragging or side != self._drag_side:
            return

        # Final position update
        self.on_drag_moved(x, y, side)

        if self.move_line_mode:
            self._modified_line_idx = self._drag_line_idx

        # Clear drag state
        self._dragging = False
        self._drag_index = None
        self._drag_side = None
        self._drag_line_idx = None
        self._drag_original_pos = None
        self.ihc_view.clear_highlight()
        self.mri_view.clear_highlight()

        if self.move_landmark_mode:
            self.update_status("Landmark moved. Drag another or switch mode.")
        elif self.move_line_mode:
            self.update_status("Line landmark moved. Press Enter to redistribute points evenly.")

    def _project_onto_polyline(self, px, py, control_points):
        """Project point (px, py) onto the nearest position on a polyline."""
        best_x, best_y = control_points[0]
        best_dist_sq = (px - best_x) ** 2 + (py - best_y) ** 2

        for i in range(len(control_points) - 1):
            x1, y1 = control_points[i]
            x2, y2 = control_points[i + 1]
            dx, dy = x2 - x1, y2 - y1
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-12:
                continue
            t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            dist_sq = (px - proj_x) ** 2 + (py - proj_y) ** 2
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_x, best_y = proj_x, proj_y

        return best_x, best_y

    def _redistribute_line_landmarks(self, line_idx):
        """Redistribute landmarks evenly along a line after move.

        For free mode: uses current landmark positions as the new polyline.
        For endpoint mode: uses original control points but re-interpolates
        between the (possibly moved) endpoint positions.
        """
        line = self.drawn_lines[line_idx]
        indices = line['landmark_indices']
        num_landmarks = len(indices)
        if num_landmarks < 2:
            return

        def cumulative_distances(points):
            distances = [0]
            for i in range(1, len(points)):
                dx = points[i][0] - points[i - 1][0]
                dy = points[i][1] - points[i - 1][1]
                dist = np.sqrt(dx ** 2 + dy ** 2)
                distances.append(distances[-1] + dist)
            return distances, distances[-1]

        def interpolate_along(points, distances, total, num_pts):
            """Interpolate num_pts evenly spaced points along a polyline."""
            result = []
            for i in range(num_pts):
                t = i / (num_pts - 1) if num_pts > 1 else 0.5
                target_dist = t * total
                for j in range(len(distances) - 1):
                    if distances[j] <= target_dist <= distances[j + 1]:
                        seg_len = distances[j + 1] - distances[j]
                        seg_t = (target_dist - distances[j]) / seg_len if seg_len > 0 else 0
                        lx = points[j][0] + seg_t * (points[j + 1][0] - points[j][0])
                        ly = points[j][1] + seg_t * (points[j + 1][1] - points[j][1])
                        result.append((lx, ly))
                        break
                else:
                    # Fallback: use last point
                    result.append(points[-1])
            return result

        if self.move_line_sub_mode == 'free':
            # Use current landmark positions as new polyline for BOTH sides
            ihc_points = [self.landmarks_ihc[i] for i in indices]
            mri_points = [self.landmarks_mri[i] for i in indices]

            ihc_dists, ihc_total = cumulative_distances(ihc_points)
            mri_dists, mri_total = cumulative_distances(mri_points)

            new_ihc = interpolate_along(ihc_points, ihc_dists, ihc_total, num_landmarks)
            new_mri = interpolate_along(mri_points, mri_dists, mri_total, num_landmarks)

            # Update control points to match new positions
            line['ihc_control_points'] = list(new_ihc)
            line['mri_control_points'] = list(new_mri)

        elif self.move_line_sub_mode == 'endpoint':
            # Use original control points but trim/extend to current endpoint positions
            # The endpoint may have been moved along the polyline, so we need to find
            # the parametric range and redistribute within it
            for side_key, lm_list in [('ihc', self.landmarks_ihc), ('mri', self.landmarks_mri)]:
                cp_key = f'{side_key}_control_points'
                control_pts = line[cp_key]
                cp_dists, cp_total = cumulative_distances(control_pts)

                # Find parametric positions of current first and last landmarks on original polyline
                first_pos = lm_list[indices[0]]
                last_pos = lm_list[indices[-1]]
                t_first = self._parametric_position_on_polyline(first_pos[0], first_pos[1], control_pts, cp_dists, cp_total)
                t_last = self._parametric_position_on_polyline(last_pos[0], last_pos[1], control_pts, cp_dists, cp_total)

                # Ensure t_first < t_last
                if t_first > t_last:
                    t_first, t_last = t_last, t_first

                # Generate evenly spaced points between t_first and t_last
                new_points = []
                for i in range(num_landmarks):
                    t = t_first + (t_last - t_first) * (i / (num_landmarks - 1) if num_landmarks > 1 else 0.5)
                    target_dist = t * cp_total
                    for j in range(len(cp_dists) - 1):
                        if cp_dists[j] <= target_dist <= cp_dists[j + 1]:
                            seg_len = cp_dists[j + 1] - cp_dists[j]
                            seg_t = (target_dist - cp_dists[j]) / seg_len if seg_len > 0 else 0
                            lx = control_pts[j][0] + seg_t * (control_pts[j + 1][0] - control_pts[j][0])
                            ly = control_pts[j][1] + seg_t * (control_pts[j + 1][1] - control_pts[j][1])
                            new_points.append((lx, ly))
                            break
                    else:
                        new_points.append(control_pts[-1])

                for k, idx in enumerate(indices):
                    lm_list[idx] = new_points[k]

            # Skip the per-index assignment below since we did it in the loop
            self.update_landmarks_display()
            self.landmark_label.setText(f"Landmarks: {len(self.landmarks_ihc)} pairs")
            return

        # Assign new positions (for free mode)
        for k, idx in enumerate(indices):
            self.landmarks_ihc[idx] = new_ihc[k]
            self.landmarks_mri[idx] = new_mri[k]

        self.update_landmarks_display()
        self.landmark_label.setText(f"Landmarks: {len(self.landmarks_ihc)} pairs")

    def _parametric_position_on_polyline(self, px, py, control_points, cp_dists, cp_total):
        """Find the parametric position t (0-1) of the projection of (px,py) onto the polyline."""
        best_t = 0.0
        best_dist_sq = float('inf')

        for i in range(len(control_points) - 1):
            x1, y1 = control_points[i]
            x2, y2 = control_points[i + 1]
            dx, dy = x2 - x1, y2 - y1
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-12:
                continue
            seg_t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
            seg_t = max(0.0, min(1.0, seg_t))
            proj_x = x1 + seg_t * dx
            proj_y = y1 + seg_t * dy
            dist_sq = (px - proj_x) ** 2 + (py - proj_y) ** 2
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                # Parametric position along entire polyline
                seg_len = np.sqrt(seg_len_sq)
                arc_at_proj = cp_dists[i] + seg_t * seg_len
                best_t = arc_at_proj / cp_total if cp_total > 0 else 0

        return best_t

    def on_backspace(self):
        """Handle backspace key - context sensitive."""
        if self.line_mode:
            # In line mode, delete last point from active line
            if self.line_drawing_ihc and len(self.line_points_ihc) > 1:
                self.line_points_ihc.pop()
                self.ihc_view.update_line_preview(self.line_points_ihc)
                self.update_status(f"Removed point. IHC line: {len(self.line_points_ihc)} points")
                return
            elif self.line_drawing_mri and len(self.line_points_mri) > 1:
                self.line_points_mri.pop()
                self.mri_view.update_line_preview(self.line_points_mri)
                self.update_status(f"Removed point. MRI line: {len(self.line_points_mri)} points")
                return

        # Default: remove last landmark
        self.remove_last_landmark()

    def finish_line(self):
        """Finish current line drawing."""
        if self.line_drawing_ihc:
            if len(self.line_points_ihc) >= 2:
                self.line_drawing_ihc = False
                self.line_drawing_mri = True
                self.line_points_mri = []
                self.active_image = 'mri'
                self.radio_mri.setChecked(True)
                # Keep IHC line visible but stop updating
                self.update_status("IHC line complete. Now draw line on MRI.")
            else:
                self.update_status("Line needs at least 2 points")

        elif self.line_drawing_mri:
            if len(self.line_points_mri) >= 2:
                self.generate_landmarks_from_lines()
                self.line_drawing_mri = False
                self.ihc_view.clear_line_preview()
                self.mri_view.clear_line_preview()
            else:
                self.update_status("Line needs at least 2 points")

    def generate_landmarks_from_lines(self):
        """Generate evenly distributed landmarks from drawn lines."""
        if len(self.line_points_ihc) < 2 or len(self.line_points_mri) < 2:
            return

        def cumulative_distances(points):
            distances = [0]
            for i in range(1, len(points)):
                dx = points[i][0] - points[i-1][0]
                dy = points[i][1] - points[i-1][1]
                dist = np.sqrt(dx**2 + dy**2)
                distances.append(distances[-1] + dist)
            return distances, distances[-1]

        ihc_distances, ihc_total = cumulative_distances(self.line_points_ihc)
        mri_distances, mri_total = cumulative_distances(self.line_points_mri)

        avg_length = (ihc_total + mri_total) / 2.0
        num_landmarks = max(2, int(np.round(avg_length / self.landmark_spacing)) + 1)

        for i in range(num_landmarks):
            t = i / (num_landmarks - 1) if num_landmarks > 1 else 0.5

            # IHC point
            target_dist_ihc = t * ihc_total
            for j in range(len(ihc_distances) - 1):
                if ihc_distances[j] <= target_dist_ihc <= ihc_distances[j + 1]:
                    seg_t = (target_dist_ihc - ihc_distances[j]) / (ihc_distances[j + 1] - ihc_distances[j])
                    lm_ihc_x = self.line_points_ihc[j][0] + seg_t * (self.line_points_ihc[j + 1][0] - self.line_points_ihc[j][0])
                    lm_ihc_y = self.line_points_ihc[j][1] + seg_t * (self.line_points_ihc[j + 1][1] - self.line_points_ihc[j][1])
                    break

            # MRI point
            target_dist_mri = t * mri_total
            for j in range(len(mri_distances) - 1):
                if mri_distances[j] <= target_dist_mri <= mri_distances[j + 1]:
                    seg_t = (target_dist_mri - mri_distances[j]) / (mri_distances[j + 1] - mri_distances[j])
                    lm_mri_x = self.line_points_mri[j][0] + seg_t * (self.line_points_mri[j + 1][0] - self.line_points_mri[j][0])
                    lm_mri_y = self.line_points_mri[j][1] + seg_t * (self.line_points_mri[j + 1][1] - self.line_points_mri[j][1])
                    break

            self.landmarks_ihc.append((lm_ihc_x, lm_ihc_y))
            self.landmarks_mri.append((lm_mri_x, lm_mri_y))

        # Store line data for Move Line mode
        start_idx = len(self.landmarks_ihc) - num_landmarks
        line_entry = {
            'ihc_control_points': list(self.line_points_ihc),
            'mri_control_points': list(self.line_points_mri),
            'landmark_indices': list(range(start_idx, start_idx + num_landmarks))
        }
        self.drawn_lines.append(line_entry)

        self.line_points_ihc = []
        self.line_points_mri = []

        self.update_landmarks_display()
        self.landmark_label.setText(f"Landmarks: {len(self.landmarks_ihc)} pairs")
        self.update_status(f"Generated {num_landmarks} landmarks from lines. Ready for next line.")

        # Stay in line mode - switch active image back to IHC for next line
        self.active_image = 'ihc'
        self.radio_ihc.setChecked(True)

    def update_landmarks_display(self):
        """Update landmark display on both images."""
        force_all = self.landmark_input.hasFocus()
        self.ihc_view.update_landmarks(self.landmarks_ihc, self.show_labels, force_all_labels=force_all)
        self.mri_view.update_landmarks(self.landmarks_mri, self.show_labels, force_all_labels=force_all)
        # Keep line overlays in sync when in move line mode
        if self.move_line_mode:
            self._update_line_overlays()
        self.landmarks_changed.emit()

    def remove_last_landmark(self):
        """Remove the last landmark pair."""
        if len(self.landmarks_mri) == len(self.landmarks_ihc) and self.landmarks_ihc:
            deleted_idx = len(self.landmarks_ihc) - 1
            self.landmarks_ihc.pop()
            self.landmarks_mri.pop()
            self._update_drawn_lines_after_delete(deleted_idx)
            self.update_landmarks_display()
            self.landmark_label.setText(f"Landmarks: {len(self.landmarks_ihc)} pairs")
            self.update_status("Removed last landmark pair")
        elif len(self.landmarks_ihc) > len(self.landmarks_mri):
            self.landmarks_ihc.pop()
            self.update_landmarks_display()
            self.update_status("Removed last IHC landmark")

    def delete_specific_landmark(self):
        """Delete a specific landmark by number."""
        idx = self.landmark_input.value() - 1
        if 0 <= idx < len(self.landmarks_ihc):
            del self.landmarks_ihc[idx]
            if idx < len(self.landmarks_mri):
                del self.landmarks_mri[idx]
            self._update_drawn_lines_after_delete(idx)
            self.update_landmarks_display()
            self.landmark_label.setText(f"Landmarks: {min(len(self.landmarks_ihc), len(self.landmarks_mri))} pairs")
            self.update_status(f"Deleted landmark {idx + 1}")

    def _update_drawn_lines_after_delete(self, deleted_idx):
        """Update drawn_lines indices after a landmark is deleted."""
        lines_to_remove = []
        for i, line in enumerate(self.drawn_lines):
            # Remove deleted index from this line
            if deleted_idx in line['landmark_indices']:
                line['landmark_indices'].remove(deleted_idx)
            # Shift all indices above the deleted one
            line['landmark_indices'] = [
                idx - 1 if idx > deleted_idx else idx
                for idx in line['landmark_indices']
            ]
            # Mark degenerate lines for removal
            if len(line['landmark_indices']) < 2:
                lines_to_remove.append(i)
        # Remove degenerate lines in reverse order
        for i in reversed(lines_to_remove):
            del self.drawn_lines[i]

    def reset_landmarks(self):
        """Reset all landmarks."""
        reply = QMessageBox.question(self, "Reset Landmarks",
                                      "Are you sure you want to remove all landmarks?",
                                      QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.landmarks_ihc = []
            self.landmarks_mri = []
            self.drawn_lines = []
            self.update_landmarks_display()
            self.landmark_label.setText("Landmarks: 0 pairs")
            self.update_status("All landmarks reset")

    def toggle_ruler_visibility(self):
        """Toggle ruler visibility."""
        self.ruler_visible = not self.ruler_visible
        self.update_ruler_display()
        self.update_status(f"Ruler {'visible' if self.ruler_visible else 'hidden'}")

    def toggle_connections(self):
        """Toggle connection lines visibility."""
        self.show_connections = not self.show_connections
        self.update_status(f"Connections {'visible' if self.show_connections else 'hidden'}")

    def toggle_labels(self):
        """Toggle landmark labels visibility."""
        self.show_labels = not self.show_labels
        self.update_landmarks_display()
        self.update_status(f"Labels {'visible' if self.show_labels else 'hidden'}")

    def on_enter_pressed(self):
        """Handle Enter key press."""
        # If the delete landmark spinbox has focus, delete landmark instead
        if self.landmark_input.hasFocus():
            self.delete_specific_landmark()
            return

        if self.line_mode and (self.line_drawing_ihc or self.line_drawing_mri):
            self.finish_line()
        elif self.move_line_mode and self._modified_line_idx is not None:
            # Redistribute landmarks evenly along the modified line
            self._redistribute_line_landmarks(self._modified_line_idx)
            self.update_status(f"Redistributed landmarks on line. Points are evenly spaced again.")
            self._modified_line_idx = None
        else:
            self.save_registration()

    def on_escape_pressed(self):
        """Handle Escape key press."""
        if self.line_mode and (self.line_drawing_ihc or self.line_drawing_mri):
            self.line_drawing_ihc = False
            self.line_drawing_mri = False
            self.line_points_ihc = []
            self.line_points_mri = []
            self.ihc_view.clear_line_preview()
            self.mri_view.clear_line_preview()
            self.update_status("Line drawing cancelled")
        elif (self.move_landmark_mode or self.move_line_mode) and self._dragging:
            # Cancel current drag - restore original position
            if self._drag_original_pos is not None and self._drag_index is not None:
                if self._drag_side == 'ihc':
                    self.landmarks_ihc[self._drag_index] = self._drag_original_pos
                else:
                    self.landmarks_mri[self._drag_index] = self._drag_original_pos
                self.update_landmarks_display()
            self._dragging = False
            self._drag_index = None
            self._drag_side = None
            self._drag_line_idx = None
            self._drag_original_pos = None
            self.ihc_view.clear_highlight()
            self.mri_view.clear_highlight()
            self.update_status("Drag cancelled")
        else:
            # Unfocus any focused textbox/widget
            focused = QApplication.focusWidget()
            if focused is not None:
                focused.clearFocus()

    def update_status(self, message):
        """Update status bar message."""
        self.status_bar.showMessage(message)

    def get_output_dir(self):
        """Get or create output directory."""
        if self.output_dir is None:
            if self.registered_ihc_path:
                base_dir = os.path.dirname(self.registered_ihc_path)
                parent_dir = os.path.dirname(base_dir)
                self.output_dir = os.path.join(parent_dir, "Non-linear_registration_results")
            else:
                self.output_dir = os.path.join(os.getcwd(), "Non-linear_registration_results")

        os.makedirs(self.output_dir, exist_ok=True)
        return self.output_dir

    def save_landmarks(self):
        """Save landmarks to file."""
        if not self.landmarks_ihc:
            QMessageBox.warning(self, "No Landmarks", "No landmarks to save.")
            return

        # Default to output directory — create if it doesn't exist yet
        default_dir = self.get_output_dir()
        os.makedirs(default_dir, exist_ok=True)
        default_path = os.path.join(default_dir, "landmarks.pkl")

        path = self._open_save_dialog(
            "Save Landmarks", default_path,
            "Pickle files (*.pkl);;All files (*.*)"
        )

        if path:
            data = {
                'landmarks_ihc': self.landmarks_ihc,
                'landmarks_mri': self.landmarks_mri,
                'drawn_lines': self.drawn_lines,
                'slice_info': {
                    'dimension': self.slice_dimension,
                    'slice_index': self.slice_index
                }
            }
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            self.update_status(f"Saved {len(self.landmarks_ihc)} landmark pairs")

    def load_landmarks(self):
        """Load landmarks from file."""
        # Default to output directory
        default_dir = self.get_output_dir()

        path = self._open_file_dialog(
            "Load Landmarks", default_dir,
            "Pickle files (*.pkl);;All files (*.*)"
        )

        if path:
            with open(path, 'rb') as f:
                data = pickle.load(f)

            self.landmarks_ihc = data['landmarks_ihc']
            self.landmarks_mri = data['landmarks_mri']
            self.drawn_lines = data.get('drawn_lines', [])
            self.update_landmarks_display()
            self.landmark_label.setText(f"Landmarks: {len(self.landmarks_ihc)} pairs")
            self.update_status(f"Loaded {len(self.landmarks_ihc)} landmark pairs")

    def preview_registration(self):
        """Preview registration result."""
        self.perform_registration(preview_only=True)

    def save_registration(self):
        """Save registration result."""
        self.perform_registration(preview_only=False)

    # ------------------------------------------------------------------
    # Overview overlay helpers
    # ------------------------------------------------------------------

    def _apply_orientation(self, image, orientation_desc):
        """Apply the slice-matching orientation (rotation / flips) to an image array.

        Compatible with both orientation description formats:
          Long  (Slice-Matching.py):   "Rotated 90° clockwise + flipped horizontally"
          Short (Slice-MatchingV2.py): "Rot 90 CW + Flip H"
        Works for both 2-D (H, W) and 3-D (H, W, C) arrays.
        """
        img = image.copy()
        desc = orientation_desc.lower()

        is_rotated_90 = any(x in desc for x in ["rotated 90", "rot 90 cw", "90° clockwise"])
        has_flip_h    = any(x in desc for x in ["flipped horizontally", "flip h", "flipped h"])
        has_flip_v    = any(x in desc for x in ["flipped vertically",   "flip v", "flipped v"])

        if "h+v" in desc or "horizontally and vertically" in desc or "both" in desc:
            has_flip_h = has_flip_v = True

        if is_rotated_90:
            img = np.rot90(img, k=-1)   # k=-1 → clockwise
        if has_flip_h:
            img = np.fliplr(img)
        if has_flip_v:
            img = np.flipud(img)

        return img

    def _get_linear_reg_dir(self):
        """Return path to Linear_registration_results/ directory."""
        if self.registered_ihc_path:
            return os.path.dirname(self.registered_ihc_path)
        return os.path.join(os.getcwd(), "Linear_registration_results")

    def _load_linear_transform(self):
        """Load the affine 3×3 matrix from Linear_registration_results/transform_manual.pkl.

        Returns the matrix M (in (col=x, row=y) homogeneous space) or None.
        """
        pkl_path = os.path.join(self._get_linear_reg_dir(), "transform_manual.pkl")
        if not os.path.exists(pkl_path):
            print(f"Linear transform not found: {pkl_path}")
            return None
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        a11, a21, a12, a22, b1, b2 = data['parameters']
        M = np.array([[a11, a12, b1],
                      [a21, a22, b2],
                      [0.0, 0.0, 1.0]])
        print(f"Loaded linear transform from {pkl_path}")
        return M

    def _get_ihc_slice_shape(self):
        """Return (H, W) pixel shape of the IHC NIfTI slice."""
        s = self.ihc_nii.shape
        if self.slice_dimension == 0:
            return (s[1], s[2])
        elif self.slice_dimension == 1:
            return (s[0], s[2])
        else:
            return (s[0], s[1])

    def _get_mri_slice_shape(self):
        """Return (H, W) pixel shape of the MRI NIfTI slice."""
        s = self.mri_nii.shape
        if self.slice_dimension == 0:
            return (s[1], s[2])
        elif self.slice_dimension == 1:
            return (s[0], s[2])
        else:
            return (s[0], s[1])

    def _apply_affine_rgb(self, rgb_img, M):
        """Apply a 3×3 affine matrix M to an RGB uint8 image, channel by channel.

        M is in (col=x, row=y) homogeneous space (as produced by get_transform_matrix()).
        Returns a uint8 RGB array of the same shape as rgb_img.
        """
        tform = sk_AffineTransform(matrix=M)
        channels = []
        for c in range(rgb_img.shape[2]):
            ch = sk_warp(
                rgb_img[:, :, c].astype(np.float32),
                tform.inverse,
                preserve_range=True,
                cval=255.0,   # white fill for out-of-bounds (typical IHC background)
                order=1
            )
            channels.append(ch)
        return np.clip(np.stack(channels, axis=-1), 0, 255).astype(np.uint8)

    def _apply_tps_rgb(self, rgb_img):
        """Apply the current TPS warp to an RGB uint8 image, channel by channel.

        Uses the same landmarks and Rbf/griddata approach as compute_warped_image.
        Returns a uint8 RGB array of the same shape as rgb_img.
        """
        if len(self.landmarks_ihc) < 3:
            return rgb_img

        src_pts = np.array(self.landmarks_ihc)
        dst_pts = np.array(self.landmarks_mri)

        tps_x = interpolate.Rbf(src_pts[:, 0], src_pts[:, 1], dst_pts[:, 0],
                                 function='thin_plate', smooth=0.0)
        tps_y = interpolate.Rbf(src_pts[:, 0], src_pts[:, 1], dst_pts[:, 1],
                                 function='thin_plate', smooth=0.0)

        height, width = rgb_img.shape[:2]
        grid_y, grid_x = np.mgrid[0:height, 0:width]
        x_warped = tps_x(grid_x, grid_y)
        y_warped = tps_y(grid_x, grid_y)

        points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
        xi = np.vstack([x_warped.ravel(), y_warped.ravel()]).T

        channels = []
        for c in range(rgb_img.shape[2]):
            values = rgb_img[:, :, c].astype(np.float32).flatten()
            ch = interpolate.griddata(xi, values, points, method='linear', fill_value=255.0)
            ch = ch.reshape(height, width)
            if np.any(np.isnan(ch)):
                nan_mask = np.isnan(ch)
                nn = interpolate.griddata(xi, values, points, method='nearest')
                ch[nan_mask] = nn.reshape(height, width)[nan_mask]
            channels.append(ch)

        return np.clip(np.stack(channels, axis=-1), 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # File-dialog helpers — use zenity (GTK, separate process) so that
    # Qt's XCB/XWayland modal-dialog freeze is avoided.  Falls back to
    # QFileDialog with DontUseNativeDialog when zenity is not available.
    # ------------------------------------------------------------------

    @staticmethod
    def _zenity_available():
        try:
            subprocess.run(['zenity', '--version'], capture_output=True, check=True)
            return True
        except Exception:
            return False

    def _open_file_dialog(self, title, start_dir, file_filter="All files (*.*)"):
        """Return a selected file path, or '' if cancelled."""
        if self._zenity_available():
            # Use dummy filename so zenity opens *inside* the directory
            cmd = ['zenity', '--file-selection', f'--title={title}',
                   f'--filename={start_dir}/_']
            # parse file_filter groups into zenity --file-filter entries
            for part in file_filter.split(';;'):
                part = part.strip()
                if not part:
                    continue
                name = part[:part.find('(')].strip()
                inside = part[part.find('(')+1:part.find(')')]
                globs = ' '.join(inside.split())
                cmd += ['--file-filter', f'{name} | {globs}' if name else globs]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                return result.stdout.strip()
            except Exception:
                pass  # fall through to Qt dialog
        path, _ = QFileDialog.getOpenFileName(
            self, title, start_dir, file_filter,
            options=QFileDialog.DontUseNativeDialog
        )
        return path

    def _open_save_dialog(self, title, default_path, file_filter="All files (*.*)"):
        """Return a save path, or '' if cancelled."""
        if self._zenity_available():
            cmd = ['zenity', '--file-selection', '--save',
                   f'--title={title}', f'--filename={default_path}']
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                return result.stdout.strip()
            except Exception:
                pass
        path, _ = QFileDialog.getSaveFileName(
            self, title, default_path, file_filter,
            options=QFileDialog.DontUseNativeDialog
        )
        return path

    def _open_dir_dialog(self, title, start_dir):
        """Return a selected directory path, or '' if cancelled."""
        if self._zenity_available():
            cmd = ['zenity', '--file-selection', '--directory',
                   f'--title={title}', f'--filename={start_dir}/']
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                return result.stdout.strip()
            except Exception:
                pass
        chosen = QFileDialog.getExistingDirectory(
            self, title, start_dir,
            QFileDialog.DontUseNativeDialog | QFileDialog.ShowDirsOnly
        )
        return chosen

    def _select_reg_dir(self):
        """Open a directory dialog so the user can choose a registration root directory.

        Looks for:
          <dir>/Linear_registration_results/transform_manual.pkl
          <dir>/Non-linear_registration_results/landmarks.pkl

        Returns the chosen directory path, or None if cancelled.
        """
        start_dir = os.path.dirname(self._get_linear_reg_dir())
        chosen = self._open_dir_dialog("Select Registration Root Directory", start_dir)
        return chosen if chosen else None

    def _load_transforms_from_dir(self, base_dir):
        """Load linear transform and TPS landmarks from a registration root directory.

        Expects:
          <base_dir>/Linear_registration_results/transform_manual.pkl
          <base_dir>/Non-linear_registration_results/landmarks.pkl

        Returns (M, landmarks_ihc, landmarks_mri).
        M is None if not found; landmark lists are empty if not found.
        """
        # --- Linear transform ---
        lin_pkl = os.path.join(base_dir, "Linear_registration_results", "transform_manual.pkl")
        M = None
        if os.path.exists(lin_pkl):
            with open(lin_pkl, 'rb') as f:
                data = pickle.load(f)
            a11, a21, a12, a22, b1, b2 = data['parameters']
            M = np.array([[a11, a12, b1],
                          [a21, a22, b2],
                          [0.0, 0.0, 1.0]])
            print(f"Loaded linear transform from {lin_pkl}")
        else:
            print(f"Linear transform not found: {lin_pkl}")

        # --- TPS landmarks ---
        nlr_pkl = os.path.join(base_dir, "Non-linear_registration_results", "landmarks.pkl")
        landmarks_ihc, landmarks_mri = [], []
        if os.path.exists(nlr_pkl):
            with open(nlr_pkl, 'rb') as f:
                lm_data = pickle.load(f)
            landmarks_ihc = lm_data.get('landmarks_ihc', [])
            landmarks_mri = lm_data.get('landmarks_mri', [])
            print(f"Loaded {len(landmarks_ihc)} landmark pairs from {nlr_pkl}")
        else:
            print(f"Landmarks file not found: {nlr_pkl}")

        return M, landmarks_ihc, landmarks_mri

    def _compute_overview_overlay(self):
        """Open file dialog, apply linear transform to colour PNG, save result.

        Saves to Non-linear_registration_results/overview_overlay.png.
        Returns the saved path, or None on failure/cancel.
        """
        # Choose source PNG — start in stain dir (parent of output dir)
        png_path = self._open_file_dialog(
            "Select Overview PNG", os.path.dirname(self.get_output_dir()),
            "Image files (*.png *.tif *.tiff *.jpg *.jpeg);;All files (*.*)"
        )
        if not png_path:
            return None
        self.overview_png_path = png_path

        # Choose registration root directory (contains Linear_registration_results/)
        reg_dir = self._select_reg_dir()
        if reg_dir is None:
            return None

        self.update_status("Computing overview overlay — this may take a moment…")
        QApplication.processEvents()

        try:
            # Load transforms from the chosen directory
            M, _lm_ihc, _lm_mri = self._load_transforms_from_dir(reg_dir)

            # Load image, ensure RGB
            img = sk_io.imread(png_path)
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
            img = img.astype(np.uint8)

            # Pre-downsample: halve repeatedly until both sides are < 2000 px
            while max(img.shape[:2]) > 2000:
                new_h, new_w = img.shape[0] // 2, img.shape[1] // 2
                img = sk_resize(img, (new_h, new_w),
                                anti_aliasing=True, preserve_range=True).astype(np.uint8)
            print(f"PNG size after pre-downsampling: {img.shape[1]}×{img.shape[0]}")

            # Apply slice-matching orientation (rotation / flips) before registration.
            # This must match what TIFF-to-NIFTI-Conversion did when creating the IHC NIfTI.
            if self.matching_info and 'orientation' in self.matching_info:
                orientation_desc = self.matching_info['orientation']['description']
                img = self._apply_orientation(img, orientation_desc)
                print(f"Applied orientation: {orientation_desc}")
            else:
                print("Warning: no orientation info found in matching_info — skipping orientation step")

            # Resize to 4× IHC NIfTI slice dimensions for a sharper overlay.
            # The affine matrix is scaled accordingly; the display rect in
            # set_overlay_image maps the 4× image back to 1× spatial coords.
            # Only pass spatial dims — skimage preserves the channel axis automatically.
            slice_shape = self._get_ihc_slice_shape()  # (H, W)
            img_resized = sk_resize(
                img, (slice_shape[0] * 4, slice_shape[1] * 4),
                anti_aliasing=True, preserve_range=True
            ).astype(np.uint8)

            # Apply linear (affine) registration.
            # The stored transform was fitted at 1× (slice_shape) pixel coordinates.
            # Scale it to 4× space: M_4x = S @ M @ S⁻¹, where S = diag(4, 4, 1).
            if M is not None:
                S = np.diag([4.0, 4.0, 1.0])
                S_inv = np.diag([0.25, 0.25, 1.0])
                M_4x = S @ M @ S_inv
                img_affine = self._apply_affine_rgb(img_resized, M_4x)
            else:
                print("Warning: no linear transform found — skipping affine step")
                img_affine = img_resized

            # Build RGBA explicitly: mask out white background (all channels > 240) → alpha=0
            h, w = img_affine.shape[:2]
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = img_affine
            rgba[:, :, 3] = np.where(np.all(img_affine > 240, axis=-1), 0, 255).astype(np.uint8)

            # Save via matplotlib (reliable for RGBA PNG, already used elsewhere in this file)
            import matplotlib
            import matplotlib.pyplot as plt
            orig_backend = matplotlib.get_backend()
            matplotlib.use('Agg')
            plt.switch_backend('Agg')
            output_dir = self.get_output_dir()
            save_path = os.path.join(output_dir, "overview_overlay.png")
            plt.imsave(save_path, rgba)
            matplotlib.use(orig_backend)
            plt.switch_backend(orig_backend)
            print(f"Saved overview overlay to: {save_path}")
            self.update_status(f"Overview overlay saved: {save_path}")
            return save_path

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Overlay Error", f"Failed to compute overlay:\n{str(e)}")
            self.update_status("Overlay computation failed.")
            return None

    def on_overview_overlay_toggled(self, checked):
        """Handle Overlay Overview checkbox toggle."""
        if not checked:
            self.ihc_view.show_overlay(False)
            return

        if self.registered_ihc_data is None:
            QMessageBox.warning(self, "No Image Loaded", "Load images before enabling the overlay.")
            self.chk_overlay_overview.setChecked(False)
            return

        output_dir = self.get_output_dir()
        cached_path = os.path.join(output_dir, "overview_overlay.png")

        if not os.path.exists(cached_path):
            # First time — compute and save
            cached_path = self._compute_overview_overlay()
            if cached_path is None:
                self.chk_overlay_overview.setChecked(False)
                return

        # Load RGBA and display
        rgba = sk_io.imread(cached_path)
        if rgba.ndim == 3 and rgba.shape[2] == 3:
            # No alpha channel in file — generate one
            alpha = np.where(np.all(rgba > 240, axis=-1), 0, 255).astype(np.uint8)
            rgba = np.dstack([rgba, alpha])

        self.ihc_view.set_overlay_image(rgba, display_shape=self._get_ihc_slice_shape())
        self.ihc_view.show_overlay(True)

    def _compute_mri_overview_overlay(self):
        """Open dialogs to collect inputs, then launch background computation.

        The actual work runs in _MriOverlayWorker (QThread) so the UI stays
        responsive.  Slots _on_mri_overlay_computed / _on_mri_overlay_failed
        handle the result when the thread finishes.
        Returns True if the worker was started, False if the user cancelled.
        """
        # Choose source PNG — start in stain dir (parent of output dir)
        png_path = self._open_file_dialog(
            "Select Overview PNG", os.path.dirname(self.get_output_dir()),
            "Image files (*.png *.tif *.tiff *.jpg *.jpeg);;All files (*.*)"
        )
        if not png_path:
            return False
        self.overview_png_path = png_path

        # Choose registration root directory
        reg_dir = self._select_reg_dir()
        if reg_dir is None:
            return False

        # Load transforms (fast disk reads — stay on main thread)
        M, landmarks_ihc, landmarks_mri = self._load_transforms_from_dir(reg_dir)

        orientation_desc = None
        if self.matching_info and 'orientation' in self.matching_info:
            orientation_desc = self.matching_info['orientation']['description']

        output_dir = self.get_output_dir()
        save_path  = os.path.join(output_dir, "overview_overlay_mri.png")

        # Launch background thread
        self._mri_overlay_worker = _MriOverlayWorker(
            png_path, M, landmarks_ihc, landmarks_mri,
            self._get_mri_slice_shape(), orientation_desc, save_path,
            parent=self
        )
        self._mri_overlay_worker.computed.connect(self._on_mri_overlay_computed)
        self._mri_overlay_worker.failed.connect(self._on_mri_overlay_failed)
        self._mri_overlay_worker.status.connect(self.update_status)

        # Disable checkbox while computing so the user can't double-trigger
        self.chk_overlay_overview_mri.setEnabled(False)
        self.update_status("Computing MRI overview overlay in background…")
        self._mri_overlay_worker.start()
        return True

    def _on_mri_overlay_computed(self, save_path):
        """Called on the main thread when the background worker finishes."""
        self.chk_overlay_overview_mri.setEnabled(True)
        self.update_status(f"MRI overview overlay saved: {save_path}")

        rgba = sk_io.imread(save_path)
        if rgba.ndim == 3 and rgba.shape[2] == 3:
            alpha = np.where(np.all(rgba > 240, axis=-1), 0, 255).astype(np.uint8)
            rgba = np.dstack([rgba, alpha])

        self.mri_view.set_overlay_image(rgba, display_shape=self._get_mri_slice_shape())
        self.mri_view.show_overlay(True)

    def _on_mri_overlay_failed(self, msg):
        """Called on the main thread when the background worker raises an exception."""
        self.chk_overlay_overview_mri.setEnabled(True)
        self.chk_overlay_overview_mri.setChecked(False)
        QMessageBox.critical(self, "Overlay Error", f"Failed to compute MRI overlay:\n{msg}")
        self.update_status("MRI overlay computation failed.")

    def on_mri_overview_overlay_toggled(self, checked):
        """Handle MRI Overlay Overview checkbox toggle."""
        if not checked:
            self.mri_view.show_overlay(False)
            return

        if self.mri_data is None:
            QMessageBox.warning(self, "No Image Loaded", "Load images before enabling the overlay.")
            self.chk_overlay_overview_mri.setChecked(False)
            return

        output_dir = self.get_output_dir()
        cached_path = os.path.join(output_dir, "overview_overlay_mri.png")

        if os.path.exists(cached_path):
            # Cached result available — load and display immediately
            rgba = sk_io.imread(cached_path)
            if rgba.ndim == 3 and rgba.shape[2] == 3:
                alpha = np.where(np.all(rgba > 240, axis=-1), 0, 255).astype(np.uint8)
                rgba = np.dstack([rgba, alpha])
            self.mri_view.set_overlay_image(rgba, display_shape=self._get_mri_slice_shape())
            self.mri_view.show_overlay(True)
        else:
            # First time — launch background computation
            started = self._compute_mri_overview_overlay()
            if not started:
                self.chk_overlay_overview_mri.setChecked(False)

    def _delete_ihc_overlay(self):
        """Delete the cached IHC overlay PNG and hide the overlay."""
        self.chk_overlay_overview.setChecked(False)
        output_dir = self.get_output_dir()
        cached_path = os.path.join(output_dir, "overview_overlay.png")
        if os.path.exists(cached_path):
            os.remove(cached_path)
            self.update_status("Deleted IHC overlay cache.")
        else:
            self.update_status("No IHC overlay cache to delete.")

    def _delete_mri_overlay(self):
        """Delete the cached MRI overlay PNG and hide the overlay."""
        self.chk_overlay_overview_mri.setChecked(False)
        output_dir = self.get_output_dir()
        cached_path = os.path.join(output_dir, "overview_overlay_mri.png")
        if os.path.exists(cached_path):
            os.remove(cached_path)
            self.update_status("Deleted MRI overlay cache.")
        else:
            self.update_status("No MRI overlay cache to delete.")

    # ------------------------------------------------------------------

    def compute_warped_image(self):
        """Compute TPS warped image from current landmarks. Returns (ihc_img, warped, mri_img) or None."""
        if len(self.landmarks_ihc) < 3 or len(self.landmarks_mri) < 3:
            return None
        if len(self.landmarks_ihc) != len(self.landmarks_mri):
            return None

        try:
            # Get slice data
            if self.slice_dimension == 0:
                ihc_img = self.registered_ihc_data[self.slice_index, :, :]
                mri_img = self.mri_data[self.slice_index, :, :]
            elif self.slice_dimension == 1:
                ihc_img = self.registered_ihc_data[:, self.slice_index, :]
                mri_img = self.mri_data[:, self.slice_index, :]
            else:
                ihc_img = self.registered_ihc_data[:, :, self.slice_index]
                mri_img = self.mri_data[:, :, self.slice_index]

            src_pts = np.array(self.landmarks_ihc)
            dst_pts = np.array(self.landmarks_mri)

            tps_x = interpolate.Rbf(src_pts[:, 0], src_pts[:, 1], dst_pts[:, 0],
                                    function='thin_plate', smooth=0.0)
            tps_y = interpolate.Rbf(src_pts[:, 0], src_pts[:, 1], dst_pts[:, 1],
                                    function='thin_plate', smooth=0.0)

            height, width = ihc_img.shape
            grid_y, grid_x = np.mgrid[0:height, 0:width]
            x_warped = tps_x(grid_x, grid_y)
            y_warped = tps_y(grid_x, grid_y)

            points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
            xi = np.vstack([x_warped.ravel(), y_warped.ravel()]).T
            values = ihc_img.flatten()

            result = interpolate.griddata(xi, values, points, method='linear', fill_value=0)
            result = result.reshape(height, width)

            if np.any(np.isnan(result)):
                nan_mask = np.isnan(result)
                nn_interp = interpolate.griddata(xi, values, points, method='nearest')
                result[nan_mask] = nn_interp.reshape(height, width)[nan_mask]

            return ihc_img, result, mri_img
        except Exception:
            return None

    def perform_registration(self, preview_only=False):
        """Perform thin-plate spline registration."""
        if len(self.landmarks_ihc) < 3 or len(self.landmarks_mri) < 3:
            QMessageBox.warning(self, "Not Enough Landmarks",
                               "Please place at least 3 landmark pairs.")
            return

        if len(self.landmarks_ihc) != len(self.landmarks_mri):
            QMessageBox.warning(self, "Unmatched Landmarks",
                               "Equal number of landmarks required on both images.")
            return

        self.update_status("Performing registration...")
        QApplication.processEvents()

        try:
            result_data = self.compute_warped_image()
            if result_data is None:
                QMessageBox.critical(self, "Error", "Registration computation failed.")
                return

            ihc_img, result, mri_img = result_data

            # Show preview
            self.show_result_preview(ihc_img, result, mri_img)

            if not preview_only:
                # Compute transform params for saving
                src_pts = np.array(self.landmarks_ihc)
                dst_pts = np.array(self.landmarks_mri)
                smoothness = 0.0

                height, width = ihc_img.shape
                grid_y, grid_x = np.mgrid[0:height, 0:width]
                tps_x = interpolate.Rbf(src_pts[:, 0], src_pts[:, 1], dst_pts[:, 0],
                                        function='thin_plate', smooth=smoothness)
                tps_y = interpolate.Rbf(src_pts[:, 0], src_pts[:, 1], dst_pts[:, 1],
                                        function='thin_plate', smooth=smoothness)
                x_warped = tps_x(grid_x, grid_y)
                y_warped = tps_y(grid_x, grid_y)

                tps_params = {
                    'type': 'thin_plate_spline',
                    'src_pts': src_pts.tolist(),
                    'dst_pts': dst_pts.tolist(),
                    'smoothness': smoothness,
                    'slice_info': {
                        'dimension': self.slice_dimension,
                        'slice_index': self.slice_index
                    }
                }
                deformation_field = {
                    'x_orig': grid_x,
                    'y_orig': grid_y,
                    'x_warped': x_warped,
                    'y_warped': y_warped,
                    'shape': ihc_img.shape
                }
                self.save_result(result, tps_params, deformation_field)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Registration failed: {str(e)}")
            import traceback
            traceback.print_exc()

    def show_result_preview(self, original, warped, reference):
        """Show registration result as an overlay preview with opacity slider."""
        # Apply contrast settings to MRI reference if available
        if self.contrast_func is not None:
            reference = self.contrast_func(reference)

        self._preview_dialog = OverlayPreviewDialog(warped, reference, parent=self)
        self._preview_dialog.show()

    def save_result(self, result, tps_params=None, deformation_field=None):
        """Save registration result to file.

        Args:
            result: The warped image data (2D array)
            tps_params: TPS transform parameters dict (for Transformation.py compatibility)
            deformation_field: Deformation field dict (for Transformation.py compatibility)
        """
        # Create output volume
        output_data = np.zeros_like(self.registered_ihc_data)

        if self.slice_dimension == 0:
            output_data[self.slice_index, :, :] = result
        elif self.slice_dimension == 1:
            output_data[:, self.slice_index, :] = result
        else:
            output_data[:, :, self.slice_index] = result

        # Get output directory
        output_dir = self.get_output_dir()

        # Save NIfTI
        output_path = os.path.join(output_dir, "ihc_to_mri_nonlinear.nii.gz")
        new_nii = nib.Nifti1Image(output_data, self.ihc_nii.affine, self.ihc_nii.header)
        nib.save(new_nii, output_path)
        print(f"Saved registered image to: {output_path}")

        # Save transform parameters for Transformation.py compatibility
        if tps_params is not None:
            tps_path = os.path.join(output_dir, "nonlinear_transform.pkl")
            with open(tps_path, 'wb') as f:
                pickle.dump(tps_params, f)
            print(f"Saved non-linear transform to: {tps_path}")

        # Save deformation field for Transformation.py compatibility
        if deformation_field is not None:
            deformation_path = os.path.join(output_dir, "deformation_field.pkl")
            with open(deformation_path, 'wb') as f:
                pickle.dump(deformation_field, f)
            print(f"Saved deformation field to: {deformation_path}")

        # Save landmarks automatically
        landmarks_path = os.path.join(output_dir, "landmarks.pkl")
        landmarks_data = {
            'landmarks_ihc': self.landmarks_ihc,
            'landmarks_mri': self.landmarks_mri,
            'drawn_lines': self.drawn_lines,
            'slice_info': {
                'dimension': self.slice_dimension,
                'slice_index': self.slice_index
            }
        }
        with open(landmarks_path, 'wb') as f:
            pickle.dump(landmarks_data, f)
        print(f"Saved landmarks to: {landmarks_path}")

        # Save visualization
        self._save_visualization(output_dir)

        self.update_status(f"Saved to: {output_path}")
        QMessageBox.information(self, "Success",
                                f"Registration saved to:\n{output_dir}\n\n"
                                f"Files saved:\n"
                                f"- ihc_to_mri_nonlinear.nii.gz\n"
                                f"- nonlinear_transform.pkl\n"
                                f"- deformation_field.pkl\n"
                                f"- landmarks.pkl\n"
                                f"- nonlinear_registration_result.png\n\n"
                                f"Landmarks saved")

    def _save_visualization(self, output_dir):
        """Save registration visualization as PNG."""
        import matplotlib
        import matplotlib.pyplot as plt
        original_backend = matplotlib.get_backend()
        matplotlib.use('Agg')  # Non-interactive backend for saving
        plt.switch_backend('Agg')

        # Get slice data
        if self.slice_dimension == 0:
            ihc_img = self.registered_ihc_data[self.slice_index, :, :]
            mri_img = self.mri_data[self.slice_index, :, :]
        elif self.slice_dimension == 1:
            ihc_img = self.registered_ihc_data[:, self.slice_index, :]
            mri_img = self.mri_data[:, self.slice_index, :]
        else:
            ihc_img = self.registered_ihc_data[:, :, self.slice_index]
            mri_img = self.mri_data[:, :, self.slice_index]

        # Load the just-saved result
        result_path = os.path.join(output_dir, "ihc_to_mri_nonlinear.nii.gz")
        result_nii = nib.load(result_path)
        result_data = result_nii.get_fdata()

        if self.slice_dimension == 0:
            warped_img = result_data[self.slice_index, :, :]
        elif self.slice_dimension == 1:
            warped_img = result_data[:, self.slice_index, :]
        else:
            warped_img = result_data[:, :, self.slice_index]

        # Apply contrast settings to MRI if available
        if self.contrast_func is not None:
            mri_img = self.contrast_func(mri_img)

        def normalize(img):
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                return (img - img_min) / (img_max - img_min)
            return img

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].imshow(normalize(ihc_img), cmap='gray')
        axes[0, 0].set_title('Linear Registered IHC')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(normalize(mri_img), cmap='gray')
        axes[0, 1].set_title('MRI Reference')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(normalize(warped_img), cmap='gray')
        axes[1, 0].set_title('Non-linear Registered IHC')
        axes[1, 0].axis('off')

        overlay = np.zeros((warped_img.shape[0], warped_img.shape[1], 3))
        overlay[:, :, 0] = normalize(warped_img)
        overlay[:, :, 1] = normalize(mri_img)
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Overlay (Red=IHC, Green=MRI)')
        axes[1, 1].axis('off')

        # Add landmarks to visualization
        for i, (x, y) in enumerate(self.landmarks_ihc):
            axes[0, 0].plot(x, y, 'ro', markersize=4)

        for i, (x, y) in enumerate(self.landmarks_mri):
            axes[0, 1].plot(x, y, 'ro', markersize=4)
            axes[1, 1].plot(x, y, 'wo', markersize=3)

        plt.tight_layout()
        plt.suptitle('Non-linear Registration Result', fontsize=16, y=0.98)

        output_path = os.path.join(output_dir, "nonlinear_registration_result.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        # Restore original backend so interactive preview still works
        import matplotlib
        matplotlib.use(original_backend)
        plt.switch_backend(original_backend)
        print(f"Saved visualization to: {output_path}")

    def load_matching_info(self, start_dir=None):
        """Load matching info from JSON file including contrast settings.

        Compatible with both Slice-Matching.py (matplotlib) and Slice-MatchingV2.py (PyQt5) outputs.
        """
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

    def select_files(self):
        """Select input files."""
        # Try auto-detection from cwd first
        working_dir = os.getcwd()
        linear_reg_dir = os.path.join(working_dir, "Linear_registration_results")

        if os.path.isdir(linear_reg_dir):
            for f in os.listdir(linear_reg_dir):
                if f.startswith('._'):
                    continue
                if f.endswith('.nii.gz') and 'affine' in f.lower():
                    self.registered_ihc_path = os.path.join(linear_reg_dir, f)
                    break

        # If auto-detection failed, ask user to pick the sample working directory
        if not self.registered_ihc_path:
            working_dir = self._open_dir_dialog(
                "Select Sample Working Directory (containing Linear_registration_results/)",
                os.getcwd()
            )
            if not working_dir:
                return False

            # Retry auto-detection from chosen directory
            linear_reg_dir = os.path.join(working_dir, "Linear_registration_results")
            if os.path.isdir(linear_reg_dir):
                for f in os.listdir(linear_reg_dir):
                    if f.startswith('._'):
                        continue
                    if f.endswith('.nii.gz') and 'affine' in f.lower():
                        self.registered_ihc_path = os.path.join(linear_reg_dir, f)
                        break

        # If still not found, ask for the file directly
        if not self.registered_ihc_path:
            path = self._open_file_dialog(
                "Select Linearly Registered IHC",
                working_dir, "NIfTI files (*.nii *.nii.gz)"
            )
            if not path:
                return False
            self.registered_ihc_path = path

        # Find original IHC
        parent_dir = os.path.dirname(os.path.dirname(self.registered_ihc_path))
        match_dir = os.path.join(parent_dir, "Match_slice_results")

        if os.path.isdir(match_dir):
            for f in os.listdir(match_dir):
                if f.startswith('._'):
                    continue
                if f.endswith('_in_block.nii.gz'):
                    self.ihc_path = os.path.join(match_dir, f)
                    break

        if not self.ihc_path:
            path = self._open_file_dialog(
                "Select Original IHC",
                working_dir, "NIfTI files (*.nii *.nii.gz)"
            )
            if not path:
                return False
            self.ihc_path = path

        # Find MRI
        if os.path.isdir(parent_dir):
            for f in os.listdir(parent_dir):
                if f.startswith('._'):
                    continue
                if f.endswith(('.nii', '.nii.gz')) and 'FLASH' in f.upper():
                    self.mri_path = os.path.join(parent_dir, f)
                    break

        if not self.mri_path:
            mri_default_dir = os.path.join(os.path.dirname(working_dir), "MRI")
            if not os.path.isdir(mri_default_dir):
                mri_default_dir = os.path.dirname(working_dir)
            path = self._open_file_dialog(
                "Select MRI Reference",
                mri_default_dir, "NIfTI files (*.nii *.nii.gz)"
            )
            if not path:
                return False
            self.mri_path = path

        return True

    def load_data(self):
        """Load image data."""
        print(f"Loading registered IHC: {self.registered_ihc_path}")
        print(f"Loading IHC: {self.ihc_path}")
        print(f"Loading MRI: {self.mri_path}")

        # Load matching info and contrast settings
        self.matching_info = self.load_matching_info()
        if self.matching_info:
            contrast_points = self.get_contrast_settings_from_json(self.matching_info)
            if contrast_points is not None:
                print("Loaded contrast settings from matching info JSON")
                self.contrast_control_points = contrast_points
                # Create contrast function from loaded points
                x_pts = [p[0] for p in contrast_points]
                y_pts = [p[1] for p in contrast_points]
                self.contrast_func = lambda data, xp=x_pts, yp=y_pts: np.interp(data, xp, yp)

        # Load NIfTI files
        self.ihc_nii = nib.load(self.ihc_path)
        self.mri_nii = nib.load(self.mri_path)
        registered_nii = nib.load(self.registered_ihc_path)

        self.registered_ihc_data = registered_nii.get_fdata()
        self.mri_data = self.mri_nii.get_fdata()

        # Find non-empty slice
        self.slice_index, self.slice_dimension = self.find_non_empty_slice(self.registered_ihc_data)

        print(f"Using slice {self.slice_index} along dimension {self.slice_dimension}")

        # Get slice data
        if self.slice_dimension == 0:
            ihc_slice = self.registered_ihc_data[self.slice_index, :, :]
            mri_slice = self.mri_data[self.slice_index, :, :]
        elif self.slice_dimension == 1:
            ihc_slice = self.registered_ihc_data[:, self.slice_index, :]
            mri_slice = self.mri_data[:, self.slice_index, :]
        else:
            ihc_slice = self.registered_ihc_data[:, :, self.slice_index]
            mri_slice = self.mri_data[:, :, self.slice_index]

        # Store original slices for contrast adjustment
        self.mri_slice_original = mri_slice.copy()
        self.ihc_slice_original = ihc_slice.copy()

        # Apply loaded contrast settings if available
        if self.contrast_func is not None:
            mri_slice = self.contrast_func(mri_slice)
            print("Applied imported contrast settings to MRI display")

        # Set images
        self.ihc_view.set_image(ihc_slice)
        self.mri_view.set_image(mri_slice)

        self.update_status("Images loaded. Place landmarks to begin registration.")

    def find_non_empty_slice(self, data):
        """Find slice with most non-zero content."""
        for axis in range(3):
            dim_size = data.shape[axis]
            for idx in range(dim_size):
                if axis == 0:
                    slice_data = data[idx, :, :]
                elif axis == 1:
                    slice_data = data[:, idx, :]
                else:
                    slice_data = data[:, :, idx]

                if np.count_nonzero(slice_data) > 1000:
                    return idx, axis

        return data.shape[2] // 2, 2

    def run(self):
        """Main entry point."""
        if not self.select_files():
            return False

        self.load_data()
        self.show()
        return True


def main():
    """Entry point."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = NonLinearRegistrationQt()
    if window.run():
        sys.exit(app.exec_())
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
