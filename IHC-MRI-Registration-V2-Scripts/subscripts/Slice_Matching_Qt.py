import os
import sys
import numpy as np
import nibabel as nib
from skimage import io, transform, exposure, filters
from skimage.color import rgb2gray
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import time
import re
import json

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QRadioButton, QButtonGroup, QSlider, QPushButton,
    QFileDialog, QMessageBox, QGroupBox, QSizePolicy, QDialog,
    QFrame, QSpinBox, QLineEdit
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

# PyQtGraph for contrast curve dialog
import pyqtgraph as pg
pg.setConfigOptions(antialias=False)


class HistologyMatcher:
    """Core logic for histology-MRI matching (preserved from original)"""

    def __init__(self):
        self.mri_path = None
        self.histology_path = None
        self.mri_nifti = None
        self.mri_data = None
        self.histology_img = None
        self.histology_original = None
        self.optimal_plane = None
        self.all_orientations = None
        self.output_dir = None

    def normalize_image(self, img):
        """Normalize image to range [0, 1]"""
        img_min, img_max = np.min(img), np.max(img)
        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        return img

    def enhance_edges(self, img, sigma=1.0, edge_weight=0.3):
        """Apply edge enhancement to image"""
        smoothed = gaussian_filter(img, sigma=sigma)
        gradient_x = filters.sobel_h(smoothed)
        gradient_y = filters.sobel_v(smoothed)
        gradient = np.sqrt(gradient_x**2 + gradient_y**2)
        enhanced = self.normalize_image(img) * (1 - edge_weight) + self.normalize_image(gradient) * edge_weight
        return self.normalize_image(enhanced)

    def preprocess_image(self, img):
        """Apply all preprocessing steps to an image"""
        if img.dtype != np.float32 and img.dtype != np.float64:
            img = img.astype(np.float32)
        img_norm = self.normalize_image(img)
        img_eq = exposure.equalize_hist(img_norm)
        img_enhanced = self.enhance_edges(img_eq)
        return img_eq

    def generate_all_orientations(self, image):
        """Generate all 8 possible orientations of a 2D image"""
        orientations = []

        # Original
        orientations.append((image.copy(), "Original"))

        # Flipped horizontally
        flipped_h = np.fliplr(image)
        orientations.append((flipped_h, "Flipped horizontally"))

        # Flipped vertically
        flipped_v = np.flipud(image)
        orientations.append((flipped_v, "Flipped vertically"))

        # Flipped both horizontally and vertically
        flipped_hv = np.flipud(np.fliplr(image))
        orientations.append((flipped_hv, "Flipped H+V"))

        # Rotated 90 clockwise
        rotated_90 = np.rot90(image, k=-1)
        orientations.append((rotated_90, "Rotated 90 CW"))

        # Rotated 90 clockwise + flipped horizontally
        rotated_90_flipped_h = np.fliplr(rotated_90)
        orientations.append((rotated_90_flipped_h, "Rot 90 CW + Flip H"))

        # Rotated 90 clockwise + flipped vertically
        rotated_90_flipped_v = np.flipud(rotated_90)
        orientations.append((rotated_90_flipped_v, "Rot 90 CW + Flip V"))

        # Rotated 90 clockwise + flipped both ways
        rotated_90_flipped_hv = np.flipud(np.fliplr(rotated_90))
        orientations.append((rotated_90_flipped_hv, "Rot 90 CW + Flip H+V"))

        return orientations

    def find_optimal_plane(self, mri_data):
        """Find the anatomical plane with the highest average number of non-zero pixels per slice"""
        plane_names = {0: "Sagittal", 1: "Coronal", 2: "Axial"}
        avg_nonzero_per_plane = []

        for plane in range(3):
            if plane == 0:  # Sagittal
                dim_size = mri_data.shape[0]
            elif plane == 1:  # Coronal
                dim_size = mri_data.shape[1]
            else:  # Axial
                dim_size = mri_data.shape[2]

            total_nonzero = 0
            for slice_idx in range(dim_size):
                if plane == 0:  # Sagittal
                    slice_data = mri_data[slice_idx, :, :]
                elif plane == 1:  # Coronal
                    slice_data = mri_data[:, slice_idx, :]
                else:  # Axial
                    slice_data = mri_data[:, :, slice_idx]

                nonzero_count = np.count_nonzero(slice_data)
                total_nonzero += nonzero_count

            avg_nonzero = total_nonzero / dim_size if dim_size > 0 else 0
            avg_nonzero_per_plane.append(avg_nonzero)
            print(f"{plane_names[plane]} plane: {avg_nonzero:.2f} average non-zero pixels per slice")

        optimal_plane = np.argmax(avg_nonzero_per_plane)
        print(f"Optimal plane: {plane_names[optimal_plane]} with {avg_nonzero_per_plane[optimal_plane]:.2f} avg non-zero pixels")

        return optimal_plane

    def extract_slice(self, mri_data, plane, slice_idx):
        """Extract a specific slice from the MRI volume"""
        if plane == 0:  # Sagittal
            return mri_data[slice_idx, :, :]
        elif plane == 1:  # Coronal
            return mri_data[:, slice_idx, :]
        else:  # Axial
            return mri_data[:, :, slice_idx]

    def get_num_slices(self, plane):
        """Get the number of slices for a given plane"""
        if plane == 0:  # Sagittal
            return self.mri_data.shape[0]
        elif plane == 1:  # Coronal
            return self.mri_data.shape[1]
        else:  # Axial
            return self.mri_data.shape[2]

    def create_histology_nifti(self, mri_nifti, histology_oriented, plane, slice_idx):
        """
        Create a new standalone NIfTI file containing only the histology slice
        at the correct position and orientation in an otherwise empty volume
        """
        # Get the shape and affine from the original MRI
        mri_shape = mri_nifti.shape

        # Create an empty volume with the same dimensions as the MRI
        histology_volume = np.zeros(mri_shape)

        # Insert the histology image at the specified slice position only
        if plane == 0:  # Sagittal
            histology_volume[slice_idx, :, :] = histology_oriented
        elif plane == 1:  # Coronal
            histology_volume[:, slice_idx, :] = histology_oriented
        else:  # Axial
            histology_volume[:, :, slice_idx] = histology_oriented

        # Create a new NIfTI image with the histology data only
        # Using the same affine and header ensures spatial alignment with the MRI
        new_nifti = nib.Nifti1Image(histology_volume, mri_nifti.affine, mri_nifti.header)

        return new_nifti

    def save_matching_info(self, slice_idx, orientation_desc, orientation_idx, output_path, contrast_curve=None):
        """Save the slice matching and orientation information to a JSON file"""
        plane_names = ["Sagittal", "Coronal", "Axial"]

        info = {
            "mri_file": self.mri_path,
            "histology_file": self.histology_path,
            "anatomical_plane": plane_names[int(self.optimal_plane)],
            "plane_index": int(self.optimal_plane),
            "selected_slice": {
                "index": int(slice_idx)
            },
            "orientation": {
                "description": orientation_desc,
                "index": int(orientation_idx)
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Add contrast curve settings if available
        if contrast_curve is not None:
            info["contrast_curve"] = {
                "control_points": [[float(x), float(y)] for x, y in contrast_curve]
            }

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)

        with open(output_path, 'w') as f:
            json.dump(info, f, indent=4, cls=NumpyEncoder)

        print(f"Matching information saved to: {output_path}")

    def create_visualization(self, histology_oriented, mri_slice, slice_idx, orientation_desc, output_path):
        """Create a visualization showing the matched histology and MRI slice"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(histology_oriented, cmap='gray')
        axes[0].set_title(f"Histology\n({orientation_desc})", fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(mri_slice, cmap='gray')
        plane_names = ["Sagittal", "Coronal", "Axial"]
        axes[1].set_title(f"MRI Slice {slice_idx}\n({plane_names[self.optimal_plane]} plane)", fontsize=12)
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Visualization saved to: {output_path}")

    def block_name_from_path(self, path):
        """Extract block name from path using regex patterns"""
        patterns = [
            r'Block[_\s]*(\d+[_\s]*\d+[_\s]*\d+)',
            r'Block[_\s]*(\d+[_\s]*\d+)',
            r'Block[_\s]*(\d+)',
            r'([bB]\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, path)
            if match:
                return match.group(1).replace(' ', '_')

        dir_name = os.path.basename(os.path.dirname(path))
        if dir_name and dir_name != "." and dir_name != "":
            return dir_name

        return "block"


class ContrastCurveDialog(QDialog):
    """Dialog for curve-based contrast adjustment with histogram."""

    contrast_changed = pyqtSignal(object)  # Emits the lookup table

    def __init__(self, image_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MRI Contrast Adjustment")
        self.setGeometry(200, 200, 700, 550)

        self.image_data = image_data
        self.original_data = image_data.copy()

        # Control points: list of (x, y) where x is intensity, y is output (0-1)
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
        self.id_spin.setMaximum(3)
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


class SliceMatcherWindow(QMainWindow):
    """PyQt5 Main Window for Slice Matching"""

    def __init__(self, matcher):
        super().__init__()
        self.matcher = matcher
        self.current_orientation_idx = 0
        self.current_slice_idx = 0
        self.histology_resized = None
        self.resized_orientations = None

        # Contrast adjustment state
        self.contrast_dialog = None
        self.contrast_control_points = None  # Store control points for saving
        self.contrast_func = None  # Current contrast function

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Histology-MRI Slice Matching")
        self.setMinimumSize(1200, 700)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Left panel (Histology)
        left_panel = QVBoxLayout()

        # Histology image display
        self.histology_label = QLabel()
        self.histology_label.setAlignment(Qt.AlignCenter)
        self.histology_label.setMinimumSize(500, 500)
        self.histology_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.histology_label.setStyleSheet("border: 1px solid gray;")
        left_panel.addWidget(self.histology_label)

        # Orientation group
        orientation_group = QGroupBox("Orientation")
        orientation_layout = QVBoxLayout()

        self.orientation_button_group = QButtonGroup()
        self.orientation_buttons = []

        orientation_labels = [
            "Original",
            "Flipped horizontally",
            "Flipped vertically",
            "Flipped H+V",
            "Rotated 90 CW",
            "Rot 90 CW + Flip H",
            "Rot 90 CW + Flip V",
            "Rot 90 CW + Flip H+V"
        ]

        for i, label in enumerate(orientation_labels):
            radio = QRadioButton(label)
            self.orientation_button_group.addButton(radio, i)
            self.orientation_buttons.append(radio)
            orientation_layout.addWidget(radio)
            if i == 0:
                radio.setChecked(True)

        self.orientation_button_group.buttonClicked.connect(self.on_orientation_changed)
        orientation_group.setLayout(orientation_layout)
        left_panel.addWidget(orientation_group)

        main_layout.addLayout(left_panel)

        # Right panel (MRI)
        right_panel = QVBoxLayout()

        # MRI image display
        self.mri_label = QLabel()
        self.mri_label.setAlignment(Qt.AlignCenter)
        self.mri_label.setMinimumSize(500, 500)
        self.mri_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.mri_label.setStyleSheet("border: 1px solid gray;")
        right_panel.addWidget(self.mri_label)

        # Change Contrast button
        self.contrast_button = QPushButton("Change Contrast")
        self.contrast_button.setStyleSheet("background-color: #e0e0ff;")
        self.contrast_button.clicked.connect(self.show_contrast_dialog)
        right_panel.addWidget(self.contrast_button)

        # Slice selection group
        slice_group = QGroupBox("Slice Selection")
        slice_layout = QVBoxLayout()

        # Slice label
        self.slice_info_label = QLabel("Slice: 0 / 0")
        self.slice_info_label.setAlignment(Qt.AlignCenter)
        slice_layout.addWidget(self.slice_info_label)

        # Slice slider
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(100)
        self.slice_slider.setValue(50)
        self.slice_slider.valueChanged.connect(self.on_slice_changed)
        slice_layout.addWidget(self.slice_slider)

        slice_group.setLayout(slice_layout)
        right_panel.addWidget(slice_group)

        # Confirm button
        self.confirm_button = QPushButton("Confirm && Save")
        self.confirm_button.setMinimumHeight(50)
        self.confirm_button.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.confirm_button.clicked.connect(self.on_confirm)
        right_panel.addWidget(self.confirm_button)

        main_layout.addLayout(right_panel)

    def load_data(self):
        """Load and prepare data after file selection"""
        # Get target shape based on optimal plane
        if self.matcher.optimal_plane == 0:  # Sagittal
            target_shape = (self.matcher.mri_data.shape[1], self.matcher.mri_data.shape[2])
        elif self.matcher.optimal_plane == 1:  # Coronal
            target_shape = (self.matcher.mri_data.shape[0], self.matcher.mri_data.shape[2])
        else:  # Axial
            target_shape = (self.matcher.mri_data.shape[0], self.matcher.mri_data.shape[1])

        print(f"Resizing histology to match MRI slice dimensions: {target_shape}")

        # Resize histology
        self.histology_resized = transform.resize(
            self.matcher.histology_img, target_shape, anti_aliasing=True
        )

        # Generate all orientations
        self.matcher.all_orientations = self.matcher.generate_all_orientations(self.histology_resized)

        # Pre-resize all orientations to match MRI slice dimensions
        self.resized_orientations = []
        for oriented_img, desc in self.matcher.all_orientations:
            if oriented_img.shape != target_shape:
                resized_img = transform.resize(oriented_img, target_shape, anti_aliasing=True)
            else:
                resized_img = oriented_img
            self.resized_orientations.append((resized_img, desc))

        # Set up slider
        num_slices = self.matcher.get_num_slices(self.matcher.optimal_plane)
        self.slice_slider.setMaximum(num_slices - 1)
        self.current_slice_idx = num_slices // 2
        self.slice_slider.setValue(self.current_slice_idx)

        # Update displays
        self.update_histology_display()
        self.update_mri_display()

    def numpy_to_qpixmap(self, img, target_size=None):
        """Convert a numpy array to QPixmap for display"""
        # Normalize to 0-255
        img_normalized = self.matcher.normalize_image(img)
        img_uint8 = (img_normalized * 255).astype(np.uint8)

        # Ensure array is contiguous in memory
        img_uint8 = np.ascontiguousarray(img_uint8)

        # Create QImage
        height, width = img_uint8.shape
        bytes_per_line = width
        qimage = QImage(img_uint8.tobytes(), width, height, bytes_per_line, QImage.Format_Grayscale8)

        # Convert to QPixmap and scale if needed
        pixmap = QPixmap.fromImage(qimage)

        if target_size:
            pixmap = pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        return pixmap

    def update_histology_display(self):
        """Update the histology image display based on current orientation"""
        if self.resized_orientations is None:
            return

        oriented_img, desc = self.resized_orientations[self.current_orientation_idx]

        # Get label size for scaling
        label_size = self.histology_label.size()
        pixmap = self.numpy_to_qpixmap(oriented_img, label_size)
        self.histology_label.setPixmap(pixmap)

    def update_mri_display(self):
        """Update the MRI slice display based on current slice index"""
        if self.matcher.mri_data is None:
            return

        mri_slice = self.matcher.extract_slice(
            self.matcher.mri_data,
            self.matcher.optimal_plane,
            self.current_slice_idx
        )

        # Apply contrast adjustment if available
        if self.contrast_func is not None:
            mri_slice_display = self.contrast_func(mri_slice)
        else:
            mri_slice_display = mri_slice

        # Get label size for scaling
        label_size = self.mri_label.size()
        pixmap = self.numpy_to_qpixmap(mri_slice_display, label_size)
        self.mri_label.setPixmap(pixmap)

        # Update slice info label
        num_slices = self.matcher.get_num_slices(self.matcher.optimal_plane)
        plane_names = ["Sagittal", "Coronal", "Axial"]
        self.slice_info_label.setText(
            f"Slice: {self.current_slice_idx} / {num_slices - 1} ({plane_names[self.matcher.optimal_plane]})"
        )

    def on_orientation_changed(self, button):
        """Handle orientation radio button changes"""
        self.current_orientation_idx = self.orientation_button_group.id(button)
        self.update_histology_display()

    def on_slice_changed(self, value):
        """Handle slice slider changes"""
        self.current_slice_idx = value
        self.update_mri_display()

    def show_contrast_dialog(self):
        """Show the contrast adjustment dialog for MRI."""
        if self.matcher.mri_data is None:
            QMessageBox.warning(self, "No Image", "Load MRI image first.")
            return

        # Get current MRI slice for histogram calculation
        mri_slice = self.matcher.extract_slice(
            self.matcher.mri_data,
            self.matcher.optimal_plane,
            self.current_slice_idx
        )

        # Create dialog with current MRI slice data
        self.contrast_dialog = ContrastCurveDialog(mri_slice, self)
        self.contrast_dialog.contrast_changed.connect(self.on_contrast_changed)

        # Restore previous control points if available
        if self.contrast_control_points is not None:
            self.contrast_dialog.control_points = list(self.contrast_control_points)
            self.contrast_dialog.update_curve()
            self.contrast_dialog.update_point_display()

        self.contrast_dialog.show()

    def on_contrast_changed(self, contrast_func):
        """Handle contrast curve change from dialog."""
        self.contrast_func = contrast_func
        # Store control points for saving later
        if self.contrast_dialog:
            self.contrast_control_points = self.contrast_dialog.get_control_points()
        self.update_mri_display()

    def on_confirm(self):
        """Handle confirm button click - save outputs"""
        try:
            # Get current selections
            orientation_idx = self.current_orientation_idx
            slice_idx = self.current_slice_idx
            oriented_img, orientation_desc = self.resized_orientations[orientation_idx]

            # Create output directory
            base_dir = os.path.dirname(self.matcher.histology_path)
            self.matcher.output_dir = os.path.join(base_dir, "Match_slice_results")
            os.makedirs(self.matcher.output_dir, exist_ok=True)

            # Normalize histology to MRI intensity range
            mri_slice = self.matcher.extract_slice(
                self.matcher.mri_data,
                self.matcher.optimal_plane,
                slice_idx
            )

            # Get MRI intensity range (using non-zero values)
            mri_nonzero = mri_slice[mri_slice > 0]
            if len(mri_nonzero) > 0:
                mri_range = (np.percentile(mri_nonzero, 1), np.percentile(mri_nonzero, 99))
            else:
                mri_range = (0, 1)

            # Normalize histology
            histology_normalized = self.matcher.normalize_image(oriented_img)
            histology_normalized = histology_normalized * (mri_range[1] - mri_range[0]) + mri_range[0]

            # Create NIfTI
            histology_nifti = self.matcher.create_histology_nifti(
                self.matcher.mri_nifti,
                histology_normalized,
                self.matcher.optimal_plane,
                slice_idx
            )

            # Generate output filename
            histology_filename = os.path.splitext(os.path.basename(self.matcher.histology_path))[0]
            if histology_filename.lower().endswith('.nii'):
                histology_filename = os.path.splitext(histology_filename)[0]

            # Save NIfTI
            output_nifti_path = os.path.join(
                self.matcher.output_dir,
                f"{histology_filename}_in_block.nii.gz"
            )
            nib.save(histology_nifti, output_nifti_path)
            print(f"NIfTI saved to: {output_nifti_path}")

            # Save JSON matching info (including contrast curve if set)
            info_path = os.path.join(
                self.matcher.output_dir,
                f"{histology_filename}_matching_info.json"
            )
            self.matcher.save_matching_info(
                slice_idx, orientation_desc, orientation_idx, info_path,
                contrast_curve=self.contrast_control_points
            )

            # Save visualization (apply contrast if set)
            vis_path = os.path.join(
                self.matcher.output_dir,
                f"{histology_filename}_visualization.png"
            )
            mri_slice_vis = mri_slice.copy()
            if self.contrast_func is not None:
                mri_slice_vis = self.contrast_func(mri_slice_vis)
            self.matcher.create_visualization(
                oriented_img, mri_slice_vis, slice_idx, orientation_desc, vis_path
            )

            # Show success message
            QMessageBox.information(
                self,
                "Success",
                f"Files saved successfully!\n\n"
                f"NIfTI: {output_nifti_path}\n\n"
                f"Info: {info_path}\n\n"
                f"Visualization: {vis_path}"
            )

            self.close()

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while saving:\n{str(e)}"
            )

    def resizeEvent(self, event):
        """Handle window resize to update image displays"""
        super().resizeEvent(event)
        self.update_histology_display()
        self.update_mri_display()


def select_files():
    """Select MRI and histology files using QFileDialog"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Select MRI file - default to ../MRI if it exists
    mri_default_dir = os.path.join(os.path.dirname(os.getcwd()), "MRI")
    if not os.path.isdir(mri_default_dir):
        mri_default_dir = os.path.dirname(os.getcwd())
    mri_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select MRI NIfTI file",
        mri_default_dir,
        "NIfTI files (*.nii *.nii.gz);;All files (*.*)"
    )

    if not mri_path:
        print("No MRI file selected. Exiting.")
        return None, None

    # Select histology file — DontUseNativeDialog forces Qt's own picker so
    # initial_dir is respected (native GTK picker uses its own last-visited dir).
    initial_dir = os.getcwd()
    histology_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select Histology image file",
        initial_dir,
        "Image files (*.jpg *.jpeg *.tif *.tiff *.png);;All files (*.*)",
        options=QFileDialog.DontUseNativeDialog
    )

    if not histology_path:
        print("No histology file selected. Exiting.")
        return None, None

    return mri_path, histology_path


def main():
    """Main function to run the histology matcher"""
    print("\n===== MRI-Histology Slice Matching (PyQt5) =====")

    # Create application
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Select files
    mri_path, histology_path = select_files()
    if not mri_path or not histology_path:
        return

    # Create matcher and load data
    matcher = HistologyMatcher()
    matcher.mri_path = mri_path
    matcher.histology_path = histology_path

    print(f"\nLoading MRI: {mri_path}")
    matcher.mri_nifti = nib.load(mri_path)
    matcher.mri_data = matcher.mri_nifti.get_fdata()
    print(f"MRI dimensions: {matcher.mri_data.shape}")

    print(f"\nLoading histology: {histology_path}")
    matcher.histology_img = io.imread(histology_path)
    print(f"Original histology dimensions: {matcher.histology_img.shape}")

    # Store original for reference
    matcher.histology_original = matcher.histology_img.copy()

    # Convert to grayscale if RGB
    if len(matcher.histology_img.shape) > 2:
        matcher.histology_img = rgb2gray(matcher.histology_img)
        print(f"Converted histology to grayscale: {matcher.histology_img.shape}")

    # Find optimal plane
    print("\nFinding optimal anatomical plane...")
    matcher.optimal_plane = matcher.find_optimal_plane(matcher.mri_data)

    # Create and show window
    window = SliceMatcherWindow(matcher)
    window.load_data()
    window.show()

    # Run application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
