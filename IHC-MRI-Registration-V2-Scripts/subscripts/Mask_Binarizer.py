"""
Mask_Binarizer.py
-----------------
Binarize and merge multiple (non-binarized) annotation masks with conflict resolution.

Algorithm
---------
For every voxel position the following rules apply (evaluated per-mask):

1. **Shared voxels** (non-zero in ≥ 2 masks at the same position)
   → The mask with the *highest* value wins: its value is kept unchanged.
   → All other masks that had a non-zero value there are set to 0.

2. **Non-shared voxels** (non-zero in exactly 1 mask)
   → If the value is already 255: keep it.
   → Otherwise apply the threshold:
       value ≥ threshold  →  255
       value < threshold  →  0

Voxels that are 0 in all masks stay 0.

Two modes
---------
• Combine / Binarize            – runs the full pipeline and saves directly.
• Combine / Threshold Evaluation – resolves conflicts, then opens an interactive
  preview (mask overlay on MRI with threshold, opacity, contrast controls and
  manual annotation editing) before you click Confirm/Save.

Manual annotation editing (Threshold Evaluation only)
------------------------------------------------------
In the preview dialog you can manually refine annotation masks before saving:
  • Select a label from the "Selected label" dropdown.
  • Left-click (and drag) to draw on the selected label (sets voxels to 255).
  • Right-click (and drag) to erase from the selected label (sets voxels to 0).
  • Mouse wheel to zoom in/out centered on cursor.
  • Middle-button drag to pan the view.
  • Brush size is adjustable.
Drawn voxels are exclusive: drawing on a label automatically clears the same
voxels from all other labels.

After Confirm/Save the threshold and manual edits are applied to the full 3-D
volumes and each label is saved as a separate binarized NIfTI file.

Contrast auto-loading
---------------------
For Threshold Evaluation the MRI contrast curve is loaded from:

    <block_dir>/<stain_dir>/Match_slice_results/*_matching_info.json

where
    block_dir  = grandparent of the MRI NIfTI  (…/MRI/file.nii.gz → …/)
    stain_dir  = grandparent of the first annotation mask
                 (…/<stain>/Transformation_results/mask.nii.gz → <stain>)

The slice displayed is determined by find_optimal_slice on the annotation masks
(the annotation mask NIfTI already encodes the correct slice).
"""

import os
import sys
import json
import numpy as np
import nibabel as nib

from PyQt5.QtWidgets import (
    QApplication, QWidget, QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem,
    QGroupBox, QFileDialog, QSpinBox, QComboBox, QTextEdit,
    QMessageBox, QAbstractItemView, QSizePolicy, QSlider, QShortcut
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QKeySequence

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

# ContrastCurveDialog – optional, from the registration pipeline
try:
    from Non_Linear_Registration_Qt import ContrastCurveDialog
except ImportError:
    try:
        from subscripts.Non_Linear_Registration_Qt import ContrastCurveDialog
    except ImportError:
        ContrastCurveDialog = None

# Distinct colors for up to 10 masks
MASK_COLORS = plt.cm.tab10.colors  # list of (R, G, B) float tuples


# ─────────────────────────────────────────────────────────────────────────────
# NIfTI / slice helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_canonical(path):
    """Load NIfTI in its original orientation (no reorientation).

    The name is kept for backwards compatibility.  Reorienting to RAS canonical
    swaps array axes relative to the on-disk layout, so the saved output ends up
    with different dimensions than the original file.  Since all masks in this
    pipeline are already co-registered to the same reference space, reorientation
    is not needed and is explicitly avoided here.
    """
    return nib.load(path)


def find_optimal_slice(data):
    """Return (axis, slice_idx) of the slice with the most non-zero voxels."""
    best_axis, best_idx, best_count = 0, 0, 0
    for axis in range(3):
        for idx in range(data.shape[axis]):
            sl = (data[idx] if axis == 0
                  else (data[:, idx] if axis == 1 else data[:, :, idx]))
            n = np.count_nonzero(sl)
            if n > best_count:
                best_count, best_axis, best_idx = n, axis, idx
    return best_axis, best_idx


def extract_slice(data, axis, idx):
    if axis == 0:
        return data[idx, :, :]
    elif axis == 1:
        return data[:, idx, :]
    else:
        return data[:, :, idx]


# ─────────────────────────────────────────────────────────────────────────────
# Matching-info JSON – contrast settings only
# ─────────────────────────────────────────────────────────────────────────────

def _find_matching_info_json(match_slice_dir):
    """Return the first *_matching_info.json found in match_slice_dir, or None."""
    if not os.path.isdir(match_slice_dir):
        return None
    for f in sorted(os.listdir(match_slice_dir)):
        if not f.startswith('._') and f.endswith('_matching_info.json'):
            return os.path.join(match_slice_dir, f)
    return None


def load_contrast_from_matching_info(mri_path, mask_paths, log_fn=print):
    """Load contrast control points from the matching-info JSON.

    Path: <grandparent of mri_path> / <stain_dir of first mask> / Match_slice_results/

    Returns list of (x, y) tuples, or None if not found.
    """
    if not mri_path or not mask_paths:
        return None

    mri_dir    = os.path.dirname(os.path.abspath(mri_path))
    block_dir  = os.path.dirname(mri_dir)
    first_mask = mask_paths[0]
    mask_dir   = os.path.dirname(os.path.abspath(first_mask))
    stain_name = os.path.basename(os.path.dirname(mask_dir))

    match_dir = os.path.join(block_dir, stain_name, "Match_slice_results")
    log_fn(f"  Looking for contrast settings in: {match_dir}")

    json_path = _find_matching_info_json(match_dir)
    if json_path is None:
        log_fn("  No matching-info JSON found — MRI will use linear normalization.")
        return None

    log_fn(f"  Found: {os.path.basename(json_path)}")
    try:
        with open(json_path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        if 'contrast_curve' in data and 'control_points' in data['contrast_curve']:
            pts = [(float(p[0]), float(p[1]))
                   for p in data['contrast_curve']['control_points']]
            log_fn(f"  Loaded {len(pts)} contrast control points")
            return pts
        else:
            log_fn("  No contrast_curve in JSON.")
    except Exception as e:
        log_fn(f"  Warning: could not parse matching-info JSON: {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Core algorithm
# ─────────────────────────────────────────────────────────────────────────────

def load_and_resolve(mask_paths, log_fn=print):
    """Load masks and resolve conflicts (shared voxels) without applying threshold.

    Returns
    -------
    resolved : list of np.float32 arrays
        • shared winner  → 255
        • shared loser   → 0
        • non-shared     → original intensity (kept for threshold evaluation)
    niis     : list of nibabel canonical images
    names    : list of base filenames (without .nii / .nii.gz)
    """
    if len(mask_paths) < 1:
        raise ValueError("Please select at least 1 mask file.")

    log_fn(f"Loading {len(mask_paths)} masks …")
    niis, arrays = [], []
    for p in mask_paths:
        nii  = load_canonical(p)
        data = nii.get_fdata().astype(np.float32)
        niis.append(nii)
        arrays.append(data)
        log_fn(f"  Loaded: {os.path.basename(p)}  shape={data.shape}  "
               f"max={data.max():.0f}  nonzero={np.count_nonzero(data)}")

    ref_shape = arrays[0].shape
    for i, a in enumerate(arrays[1:], 1):
        if a.shape != ref_shape:
            raise ValueError(
                f"Shape mismatch: '{os.path.basename(mask_paths[0])}' has shape {ref_shape} "
                f"but '{os.path.basename(mask_paths[i])}' has shape {a.shape}."
            )

    stacked       = np.stack(arrays, axis=0)
    nonzero_count = np.sum(stacked > 0, axis=0)
    shared        = nonzero_count >= 2

    log_fn(f"\nVoxel overlap analysis:")
    log_fn(f"  Shared voxels  (≥2 masks): {np.sum(shared):,}")
    log_fn(f"  Non-shared voxels (1 mask): {np.sum(nonzero_count == 1):,}")

    resolved = [a.copy() for a in arrays]
    if np.any(shared):
        winner_idx = np.argmax(stacked, axis=0)
        wins = []
        for i in range(len(arrays)):
            win_here = shared & (winner_idx == i)
            # Winner keeps its original value unchanged.
            for j in range(len(arrays)):
                if j != i:
                    resolved[j][win_here] = 0.0
            wins.append(int(np.sum(win_here)))
        log_fn("  Conflict resolution – winner (highest value) kept unchanged, rest → 0")
        for i, p in enumerate(mask_paths):
            log_fn(f"    {os.path.basename(p)}: won {wins[i]:,} shared voxels")

    names = []
    for p in mask_paths:
        bn = os.path.basename(p)
        names.append(bn[:-7] if bn.endswith('.nii.gz') else
                     bn[:-4]  if bn.endswith('.nii')    else bn)

    return resolved, niis, names


def apply_threshold_to_resolved(resolved_arrays, threshold, log_fn=print):
    """Apply threshold; return final 0/255 output arrays."""
    outputs = []
    for i, arr in enumerate(resolved_arrays):
        out = np.zeros_like(arr)
        out[arr >= 255.0] = 255.0
        needs = (arr > 0) & (arr < 255.0)
        above = needs & (arr >= threshold)
        out[above] = 255.0
        log_fn(f"  Mask {i+1}: {int(np.sum(out > 0)):,} voxels kept, "
               f"{int(np.sum(needs & (arr < threshold))):,} zeroed (threshold={threshold})")
        outputs.append(out)
    return outputs


def binarize_masks(mask_paths, threshold, log_fn=print):
    """All-in-one: load → resolve → threshold → return (nibabel_img, name) pairs."""
    resolved, niis, names = load_and_resolve(mask_paths, log_fn)
    log_fn(f"\nApplying threshold = {threshold} …")
    outputs = apply_threshold_to_resolved(resolved, threshold, log_fn)
    # Create output images without copying the header so that nibabel builds a
    # fresh, consistent header from the data shape and affine.  Copying the
    # original header after as_closest_canonical() reorientation can leave
    # stale dim/qform/sform fields that cause dimension-mismatch errors in
    # external viewers such as ITK-SNAP.
    return [(nib.Nifti1Image(out.astype(np.uint8), nii.affine), name)
            for out, nii, name in zip(outputs, niis, names)]


# ─────────────────────────────────────────────────────────────────────────────
# Overlay rendering
# ─────────────────────────────────────────────────────────────────────────────

def make_overlay_rgba(resolved_slices, threshold, opacity_frac):
    """Build RGBA image compositing all mask slices.

    • value == 255 or value ≥ threshold  → full mask color at opacity_frac
    • 0 < value < threshold              → dimmed color at 30% opacity (shows cut region)
    • value == 0                         → transparent
    """
    H, W = resolved_slices[0].shape
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    for i, sl in enumerate(resolved_slices):
        c         = MASK_COLORS[i % len(MASK_COLORS)]
        fully_on  = sl >= 255.0
        above_thr = (sl > 0) & (sl < 255.0) & (sl >= threshold)
        below_thr = (sl > 0) & (sl < threshold)

        shown = fully_on | above_thr
        rgba[shown, :3] = c[:3]
        rgba[shown,  3] = opacity_frac

        rgba[below_thr, :3] = [x * 0.55 for x in c[:3]]
        rgba[below_thr,  3] = opacity_frac * 0.30
    return rgba


# ─────────────────────────────────────────────────────────────────────────────
# Worker threads
# ─────────────────────────────────────────────────────────────────────────────

class CombineBinarizeWorker(QThread):
    log_signal  = pyqtSignal(str)
    done_signal = pyqtSignal(bool, str)

    def __init__(self, mask_paths, threshold, output_dir):
        super().__init__()
        self.mask_paths = mask_paths
        self.threshold  = threshold
        self.output_dir = output_dir

    def run(self):
        try:
            results = binarize_masks(self.mask_paths, self.threshold, self.log_signal.emit)
            self.log_signal.emit(f"\nSaving to: {self.output_dir}")
            os.makedirs(self.output_dir, exist_ok=True)
            for out_nii, name in results:
                path = os.path.join(self.output_dir, f"{name}_binarized.nii.gz")
                nib.save(out_nii, path)
                self.log_signal.emit(f"  Saved: {os.path.basename(path)}")
            self.log_signal.emit(f"\nDone. {len(results)} mask(s) saved.")
            self.done_signal.emit(True, "Processing complete.")
        except Exception as e:
            import traceback
            self.log_signal.emit(f"\nERROR: {e}\n{traceback.format_exc()}")
            self.done_signal.emit(False, str(e))


class CombineForPreviewWorker(QThread):
    """Loads + resolves conflicts, extracts 2-D preview slices, emits data for the dialog."""
    log_signal   = pyqtSignal(str)
    result_ready = pyqtSignal(object)
    error_signal = pyqtSignal(str)

    def __init__(self, mask_paths, mri_path, output_dir):
        super().__init__()
        self.mask_paths = mask_paths
        self.mri_path   = mri_path
        self.output_dir = output_dir

    def run(self):
        try:
            resolved_3d, niis, names = load_and_resolve(self.mask_paths, self.log_signal.emit)

            # Determine the slice to display from the annotation masks themselves
            union = np.zeros_like(resolved_3d[0])
            for arr in resolved_3d:
                np.maximum(union, arr, out=union)
            opt_axis, opt_idx = find_optimal_slice(union)
            self.log_signal.emit(
                f"\nPreview slice: axis={opt_axis}, index={opt_idx}")

            # Extract 2-D mask slices for the preview.
            # extract_slice returns numpy *views* so in-place edits in the dialog
            # propagate back to resolved_3d automatically.
            resolved_slices_2d = [extract_slice(arr, opt_axis, opt_idx)
                                   for arr in resolved_3d]

            # Load raw MRI slice (raw intensities – contrast applied in dialog)
            mri_slice_raw = None
            if self.mri_path and os.path.exists(self.mri_path):
                self.log_signal.emit(
                    f"Loading MRI: {os.path.basename(self.mri_path)}")
                mri_nii  = load_canonical(self.mri_path)
                mri_data = mri_nii.get_fdata().astype(np.float32)
                self.log_signal.emit(f"  MRI shape: {mri_data.shape}")
                if mri_data.shape[opt_axis] > opt_idx:
                    mri_slice_raw = extract_slice(mri_data, opt_axis, opt_idx).copy()
                    self.log_signal.emit(
                        f"  Extracted MRI slice shape: {mri_slice_raw.shape}, "
                        f"range: [{mri_slice_raw.min():.1f}, {mri_slice_raw.max():.1f}]")
                else:
                    self.log_signal.emit(
                        "  Warning: MRI axis too short for mask slice index.")
            else:
                self.log_signal.emit("No MRI selected — preview will show masks only.")

            # Load contrast settings from matching-info JSON
            contrast_points = load_contrast_from_matching_info(
                self.mri_path, self.mask_paths, self.log_signal.emit)

            self.result_ready.emit({
                'resolved_slices_2d': resolved_slices_2d,
                'resolved_3d':        resolved_3d,
                'niis':               niis,
                'names':              names,
                'mri_slice_raw':      mri_slice_raw,
                'contrast_points':    contrast_points,
                'output_dir':         self.output_dir,
                'opt_axis':           opt_axis,
                'opt_idx':            opt_idx,
            })
        except Exception as e:
            import traceback
            self.log_signal.emit(f"\nERROR: {e}\n{traceback.format_exc()}")
            self.error_signal.emit(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Interactive threshold preview dialog
# ─────────────────────────────────────────────────────────────────────────────

class ThresholdPreviewDialog(QDialog):
    """Interactive preview: MRI background + colored mask overlay + manual editing.

    Controls
    --------
    Threshold slider  – live threshold on non-shared voxels.
    Opacity slider    – mask overlay transparency.
    Selected label    – choose which label to draw on.
    Brush size        – radius of the draw/erase brush (pixels).

    Mouse interactions
    ------------------
    Left-click / drag  – draw on the selected label (sets voxels to 255).
                         Clears the same voxels from all other labels.
    Right-click / drag – erase from the selected label (sets voxels to 0).
    Mouse wheel        – zoom in/out centered on cursor.
    Middle-button drag – pan the view.

    Buttons
    -------
    Change Contrast  – opens ContrastCurveDialog for MRI display adjustment.
    Confirm/Save     – applies current threshold + manual edits to 3-D volumes
                       and saves one binarized NIfTI per label.
    """

    def __init__(self, resolved_slices_2d, resolved_3d, niis, names,
                 mri_slice_raw, contrast_points, output_dir,
                 opt_axis=2, opt_idx=0,
                 initial_threshold=30, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Threshold Evaluation Preview")
        self.setMinimumSize(950, 820)

        self.resolved_slices_2d = resolved_slices_2d  # 2-D views into resolved_3d
        self.resolved_3d        = resolved_3d
        self.niis               = niis
        self.names              = names
        self.mri_slice_raw      = mri_slice_raw
        self.output_dir         = output_dir
        self.opt_axis           = opt_axis
        self.opt_idx            = opt_idx

        # Drawing state
        self.selected_label  = 0
        self.brush_size      = 3
        self._painting       = False
        self._erase_mode     = False
        self._pan_start      = None   # (display_x, display_y) at pan start
        self._pan_xlim       = None
        self._pan_ylim       = None
        self._overlay_visible = True

        # Build contrast function from JSON points (raw intensity → uint8 display)
        self.contrast_func   = None
        self.contrast_points = None
        if contrast_points:
            self._set_contrast_from_points(contrast_points)

        self._build_ui(initial_threshold)
        self._init_figure()

    # ── Contrast helpers ─────────────────────────────────────────────────────

    def _set_contrast_from_points(self, points):
        xp = [p[0] for p in points]
        yp = [p[1] for p in points]
        self.contrast_func   = lambda d, _xp=xp, _yp=yp: (
            np.interp(d, _xp, _yp) * 255).astype(np.uint8)
        self.contrast_points = list(points)

    def _apply_contrast(self, raw_data):
        if self.contrast_func is not None:
            return self.contrast_func(raw_data)
        mn, mx = raw_data.min(), raw_data.max()
        if mx > mn:
            return ((raw_data - mn) / (mx - mn) * 255).astype(np.uint8)
        return np.zeros_like(raw_data, dtype=np.uint8)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self, initial_threshold):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        title = QLabel("Threshold Evaluation Preview")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Canvas
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas, stretch=1)

        # ── Drawing tools ─────────────────────────────────────────────────────
        draw_group  = QGroupBox("Drawing Tools")
        draw_layout = QHBoxLayout(draw_group)

        draw_layout.addWidget(QLabel("Selected label:"))
        self.label_combo = QComboBox()
        self.label_combo.setMinimumWidth(200)
        for i, name in enumerate(self.names):
            self.label_combo.addItem(name)
            r, g, b = [int(x * 255) for x in MASK_COLORS[i % len(MASK_COLORS)][:3]]
            self.label_combo.setItemData(i, QColor(r, g, b), Qt.BackgroundRole)
            # Choose black or white text for readability
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            self.label_combo.setItemData(
                i, QColor(0, 0, 0) if lum > 128 else QColor(255, 255, 255),
                Qt.ForegroundRole)
        self.label_combo.currentIndexChanged.connect(self._on_label_changed)
        draw_layout.addWidget(self.label_combo)

        draw_layout.addSpacing(16)
        draw_layout.addWidget(QLabel("Brush size:"))
        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(1, 50)
        self.brush_spin.setValue(self.brush_size)
        self.brush_spin.setToolTip("Brush radius in pixels")
        self.brush_spin.valueChanged.connect(lambda v: setattr(self, 'brush_size', v))
        draw_layout.addWidget(self.brush_spin)

        draw_layout.addSpacing(16)
        hint = QLabel("LMB: draw  |  RMB: erase  |  Wheel: zoom  |  MMB drag: pan  |  S: toggle overlay")
        hint.setStyleSheet("color: gray;")
        draw_layout.addWidget(hint)
        draw_layout.addStretch()
        layout.addWidget(draw_group)

        # ── Sliders ───────────────────────────────────────────────────────────
        ctrl_group  = QGroupBox("Controls")
        ctrl_layout = QVBoxLayout(ctrl_group)

        # Threshold
        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel("Threshold:"))
        self.thr_slider = QSlider(Qt.Horizontal)
        self.thr_slider.setRange(0, 255)
        self.thr_slider.setValue(initial_threshold)
        self.thr_slider.setTickInterval(10)
        self.thr_slider.setTickPosition(QSlider.TicksBelow)
        self.thr_slider.valueChanged.connect(self._on_threshold_changed)
        thr_row.addWidget(self.thr_slider, stretch=1)
        self.thr_label = QLabel(str(initial_threshold))
        self.thr_label.setMinimumWidth(30)
        thr_row.addWidget(self.thr_label)
        ctrl_layout.addLayout(thr_row)

        # Opacity
        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("Mask opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(70)
        self.opacity_slider.setTickInterval(10)
        self.opacity_slider.setTickPosition(QSlider.TicksBelow)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        op_row.addWidget(self.opacity_slider, stretch=1)
        self.op_label = QLabel("70%")
        self.op_label.setMinimumWidth(35)
        op_row.addWidget(self.op_label)
        ctrl_layout.addLayout(op_row)

        layout.addWidget(ctrl_group)

        # ── Bottom buttons ────────────────────────────────────────────────────
        btn_row = QHBoxLayout()

        no_mri    = self.mri_slice_raw is None
        no_dialog = ContrastCurveDialog is None
        self.contrast_btn = QPushButton("Change Contrast")
        self.contrast_btn.setEnabled(not no_mri and not no_dialog)
        if no_mri:
            self.contrast_btn.setToolTip("No MRI loaded.")
        elif no_dialog:
            self.contrast_btn.setToolTip("ContrastCurveDialog not available.")
        self.contrast_btn.clicked.connect(self._open_contrast_dialog)
        btn_row.addWidget(self.contrast_btn)

        btn_row.addStretch()

        confirm_btn = QPushButton("Confirm/Save")
        confirm_btn.setFont(QFont("Arial", 10, QFont.Bold))
        confirm_btn.setMinimumHeight(36)
        confirm_btn.clicked.connect(self._on_confirm)
        btn_row.addWidget(confirm_btn)

        layout.addLayout(btn_row)

        # Window-level shortcut so it fires regardless of which widget has focus
        toggle_sc = QShortcut(QKeySequence('S'), self)
        toggle_sc.setContext(Qt.WindowShortcut)
        toggle_sc.activated.connect(self._toggle_overlay)

    # ── Figure ────────────────────────────────────────────────────────────────

    def _init_figure(self):
        self.ax = self.figure.add_subplot(111)
        self.ax.set_axis_off()

        # MRI background
        if self.mri_slice_raw is not None:
            disp = self._apply_contrast(self.mri_slice_raw)
            self.bg_im = self.ax.imshow(
                disp, cmap='gray', interpolation='none', aspect='auto')
        else:
            h, w = self.resolved_slices_2d[0].shape
            self.bg_im = self.ax.imshow(
                np.zeros((h, w)), cmap='gray', interpolation='none', aspect='auto')
            self.ax.text(0.5, 0.5, "No MRI loaded",
                         transform=self.ax.transAxes,
                         ha='center', va='center', color='white', fontsize=12)

        # Mask overlay
        overlay = make_overlay_rgba(
            self.resolved_slices_2d,
            threshold=self.thr_slider.value(),
            opacity_frac=self.opacity_slider.value() / 100.0
        )
        self.overlay_im = self.ax.imshow(
            overlay, interpolation='none', aspect='auto')

        # Color legend
        patches = [Patch(facecolor=MASK_COLORS[i % len(MASK_COLORS)], label=n)
                   for i, n in enumerate(self.names)]
        self.ax.legend(handles=patches, loc='upper right', fontsize=8, framealpha=0.7)

        # Connect mouse events
        self.canvas.mpl_connect('button_press_event',   self._on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event',  self._on_mouse_move)
        self.canvas.mpl_connect('scroll_event',         self._on_scroll)

        self.canvas.draw()

    # ── Slider / combo callbacks ──────────────────────────────────────────────

    def _on_label_changed(self, index):
        self.selected_label = index

    def _on_threshold_changed(self, value):
        self.thr_label.setText(str(value))
        self._refresh_overlay()

    def _on_opacity_changed(self, value):
        self.op_label.setText(f"{value}%")
        self._refresh_overlay()

    def _toggle_overlay(self):
        self._overlay_visible = not self._overlay_visible
        if self._overlay_visible:
            # Re-render in case sliders changed while hidden
            overlay = make_overlay_rgba(
                self.resolved_slices_2d,
                threshold=self.thr_slider.value(),
                opacity_frac=self.opacity_slider.value() / 100.0
            )
            self.overlay_im.set_data(overlay)
        self.overlay_im.set_visible(self._overlay_visible)
        self.canvas.draw_idle()

    def _refresh_overlay(self):
        if not self._overlay_visible:
            return
        overlay = make_overlay_rgba(
            self.resolved_slices_2d,
            threshold=self.thr_slider.value(),
            opacity_frac=self.opacity_slider.value() / 100.0
        )
        self.overlay_im.set_data(overlay)
        self.canvas.draw_idle()

    # ── Mouse event handlers ──────────────────────────────────────────────────

    def _on_mouse_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:          # left – draw
            self._painting   = True
            self._erase_mode = False
            self._paint_at(event)
        elif event.button == 3:        # right – erase
            self._painting   = True
            self._erase_mode = True
            self._paint_at(event)
        elif event.button == 2:        # middle – start pan
            self._pan_start = (event.x, event.y)
            self._pan_xlim  = list(self.ax.get_xlim())
            self._pan_ylim  = list(self.ax.get_ylim())

    def _on_mouse_release(self, event):
        if event.button in (1, 3):
            self._painting = False
        elif event.button == 2:
            self._pan_start = None

    def _on_mouse_move(self, event):
        if self._painting and event.inaxes == self.ax:
            self._paint_at(event)
        elif self._pan_start is not None:
            # Pan: map display-coordinate delta back to data-coordinate delta
            inv  = self.ax.transData.inverted()
            start_d = inv.transform(self._pan_start)
            curr_d  = inv.transform((event.x, event.y))
            dx = start_d[0] - curr_d[0]
            dy = start_d[1] - curr_d[1]
            self.ax.set_xlim([x + dx for x in self._pan_xlim])
            self.ax.set_ylim([y + dy for y in self._pan_ylim])
            self.canvas.draw_idle()

    def _on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        scale = 1.0 / 1.15 if event.button == 'up' else 1.15
        xd, yd = event.xdata, event.ydata
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.set_xlim([xd + (x - xd) * scale for x in xlim])
        self.ax.set_ylim([yd + (y - yd) * scale for y in ylim])
        self.canvas.draw_idle()

    def _paint_at(self, event):
        if event.xdata is None or event.ydata is None:
            return
        x  = int(round(event.xdata))
        y  = int(round(event.ydata))
        r  = max(0, self.brush_size - 1)   # brush radius
        sl = self.resolved_slices_2d[self.selected_label]
        H, W = sl.shape
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        if x0 >= W or x1 <= 0 or y0 >= H or y1 <= 0:
            return

        if self._erase_mode:
            sl[y0:y1, x0:x1] = 0.0
        else:
            # Draw: set selected label to 255 and clear same region in all others
            for j, other in enumerate(self.resolved_slices_2d):
                if j != self.selected_label:
                    other[y0:y1, x0:x1] = 0.0
            sl[y0:y1, x0:x1] = 255.0

        self._refresh_overlay()

    # ── Contrast dialog ───────────────────────────────────────────────────────

    def _open_contrast_dialog(self):
        if ContrastCurveDialog is None or self.mri_slice_raw is None:
            return

        dialog = ContrastCurveDialog(
            self.mri_slice_raw,
            parent=self,
            initial_control_points=self.contrast_points
        )

        def _on_contrast(func):
            self.contrast_func   = func
            self.contrast_points = list(dialog.control_points)
            disp = self._apply_contrast(self.mri_slice_raw)
            self.bg_im.set_data(disp)
            self.bg_im.set_clim(float(disp.min()), float(disp.max()))
            self.canvas.draw_idle()

        dialog.contrast_changed.connect(_on_contrast)
        dialog.exec_()

    # ── Confirm/Save ──────────────────────────────────────────────────────────

    def _on_confirm(self):
        threshold = self.thr_slider.value()
        try:
            outputs = apply_threshold_to_resolved(self.resolved_3d, threshold)
            os.makedirs(self.output_dir, exist_ok=True)
            saved = []
            for out_data, nii, name in zip(outputs, self.niis, self.names):
                # Build a fresh header from data + affine (no stale orientation
                # metadata from the original file after canonical reorientation).
                out_nii = nib.Nifti1Image(out_data.astype(np.uint8), nii.affine)
                path = os.path.join(self.output_dir, f"{name}_binarized.nii.gz")
                nib.save(out_nii, path)
                saved.append(os.path.basename(path))
            QMessageBox.information(
                self, "Saved",
                f"Saved {len(saved)} mask(s) with threshold = {threshold}:\n"
                + "\n".join(saved)
                + f"\n\nOutput folder:\n{self.output_dir}"
            )
            self.accept()
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Error",
                                 f"Saving failed:\n{e}\n\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────────────
# Main application window
# ─────────────────────────────────────────────────────────────────────────────

class MaskBinarizerApp(QWidget):
    def __init__(self, working_dir=None):
        super().__init__()
        self.setWindowTitle("Mask Binarizer")
        self.setMinimumSize(720, 680)
        self._mri_path   = None

        # Derive default paths from the working_dir passed by the master script
        # (expected to be the stain sub-directory, e.g. <block>/MBP_CR343/).
        if working_dir and os.path.isdir(working_dir):
            tr = os.path.join(working_dir, "Transformation_results")
            self._default_mask_dir  = tr if os.path.isdir(tr) else working_dir
            mri_candidate = os.path.join(working_dir, "..", "MRI")
            self._default_mri_dir   = os.path.normpath(mri_candidate)
            self._output_dir        = os.path.join(tr, "Binarized_Masks")
        else:
            self._default_mask_dir  = ""
            self._default_mri_dir   = ""
            self._output_dir        = None

        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)

        title = QLabel("Mask Binarizer")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)

        subtitle = QLabel(
            "Binarize and merge annotation masks with conflict resolution.\n"
            "Shared voxels: highest value wins (kept unchanged).  "
            "Non-shared voxels: threshold applied."
        )
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        # ── Input masks ───────────────────────────────────────────────────────
        mask_group = QGroupBox("Input Masks (select ≥ 2)")
        mask_layout = QVBoxLayout(mask_group)
        self.mask_list = QListWidget()
        self.mask_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.mask_list.setMinimumHeight(130)
        mask_layout.addWidget(self.mask_list)
        btn_row = QHBoxLayout()
        for label, slot in [("Add masks …",     self._add_masks),
                             ("Remove selected", self._remove_selected),
                             ("Clear all",       self.mask_list.clear)]:
            b = QPushButton(label)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        mask_layout.addLayout(btn_row)
        root.addWidget(mask_group)

        # ── Reference MRI ─────────────────────────────────────────────────────
        mri_group = QGroupBox("Reference MRI (for Threshold Evaluation)")
        mri_layout = QHBoxLayout(mri_group)
        self.mri_label = QLabel("No MRI file selected")
        self.mri_label.setStyleSheet("color: gray;")
        self.mri_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        mri_layout.addWidget(self.mri_label)
        for label, slot in [("Browse …", self._choose_mri), ("Clear", self._clear_mri)]:
            b = QPushButton(label)
            b.clicked.connect(slot)
            mri_layout.addWidget(b)
        root.addWidget(mri_group)

        # ── Settings ──────────────────────────────────────────────────────────
        settings_group = QGroupBox("Settings")
        settings_layout = QHBoxLayout(settings_group)
        settings_layout.addWidget(QLabel("Threshold (0–255):"))
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 255)
        self.threshold_spin.setValue(30)
        self.threshold_spin.setToolTip(
            "Initial threshold for preview / direct value for Combine/Binarize."
        )
        settings_layout.addWidget(self.threshold_spin)
        settings_layout.addSpacing(24)
        settings_layout.addWidget(QLabel("Output folder:"))
        self.output_label = QLabel(
            self._output_dir if self._output_dir
            else "(auto: Binarized_Masks/ next to first mask)"
        )
        self.output_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.output_label.setStyleSheet("" if self._output_dir else "color: gray;")
        settings_layout.addWidget(self.output_label)
        browse_out = QPushButton("Browse …")
        browse_out.clicked.connect(self._choose_output_dir)
        settings_layout.addWidget(browse_out)
        root.addWidget(settings_group)

        # ── Action buttons ────────────────────────────────────────────────────
        action_row = QHBoxLayout()
        action_row.setSpacing(10)

        self.binarize_btn = QPushButton("Combine / Binarize")
        self.binarize_btn.setFont(QFont("Arial", 10, QFont.Bold))
        self.binarize_btn.setMinimumHeight(40)
        self.binarize_btn.setToolTip(
            "Resolve conflicts and apply threshold directly. Saves without preview."
        )
        self.binarize_btn.clicked.connect(self._run_binarize)
        action_row.addWidget(self.binarize_btn)

        self.preview_btn = QPushButton("Combine / Threshold Evaluation")
        self.preview_btn.setFont(QFont("Arial", 10, QFont.Bold))
        self.preview_btn.setMinimumHeight(40)
        self.preview_btn.setToolTip(
            "Resolve conflicts then open interactive preview with\n"
            "threshold, opacity, contrast, and manual annotation controls."
        )
        self.preview_btn.clicked.connect(self._run_preview)
        action_row.addWidget(self.preview_btn)

        root.addLayout(action_row)

        # ── Log ───────────────────────────────────────────────────────────────
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Courier", 9))
        self.log_box.setMinimumHeight(140)
        log_layout.addWidget(self.log_box)
        root.addWidget(log_group)

    # ── File selection ────────────────────────────────────────────────────────

    def _add_masks(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select annotation mask files", self._default_mask_dir,
            "NIfTI files (*.nii *.nii.gz);;All files (*)"
        )
        existing = {self.mask_list.item(i).data(Qt.UserRole)
                    for i in range(self.mask_list.count())}
        for p in paths:
            if p not in existing:
                item = QListWidgetItem(os.path.basename(p))
                item.setData(Qt.UserRole, p)
                item.setToolTip(p)
                self.mask_list.addItem(item)
        if self._output_dir is None and self.mask_list.count() > 0:
            first = self.mask_list.item(0).data(Qt.UserRole)
            self._output_dir = os.path.join(os.path.dirname(first), "Binarized_Masks")
            self.output_label.setText(self._output_dir)
            self.output_label.setStyleSheet("")

    def _remove_selected(self):
        for item in self.mask_list.selectedItems():
            self.mask_list.takeItem(self.mask_list.row(item))

    def _choose_mri(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select reference MRI file", self._default_mri_dir,
            "NIfTI files (*.nii *.nii.gz);;All files (*)"
        )
        if path:
            self._mri_path = path
            self.mri_label.setText(os.path.basename(path))
            self.mri_label.setToolTip(path)
            self.mri_label.setStyleSheet("")

    def _clear_mri(self):
        self._mri_path = None
        self.mri_label.setText("No MRI file selected")
        self.mri_label.setStyleSheet("color: gray;")

    def _choose_output_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select output folder", self._output_dir or "")
        if d:
            self._output_dir = d
            self.output_label.setText(d)
            self.output_label.setStyleSheet("")

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _get_mask_paths(self):
        return [self.mask_list.item(i).data(Qt.UserRole)
                for i in range(self.mask_list.count())]

    def _resolved_output_dir(self):
        if self._output_dir:
            return self._output_dir
        paths = self._get_mask_paths()
        return (os.path.join(os.path.dirname(paths[0]), "Binarized_Masks")
                if paths else "Binarized_Masks")

    def _log(self, msg):
        self.log_box.append(msg)
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum())

    def _set_buttons_enabled(self, enabled):
        self.binarize_btn.setEnabled(enabled)
        self.preview_btn.setEnabled(enabled)

    # ── Combine / Binarize ────────────────────────────────────────────────────

    def _run_binarize(self):
        paths = self._get_mask_paths()
        if len(paths) < 1:
            QMessageBox.warning(self, "Too few masks",
                                "Please add at least 1 mask file.")
            return
        output_dir = self._resolved_output_dir()
        threshold  = self.threshold_spin.value()
        self.log_box.clear()
        self._log(f"Combine / Binarize  (threshold = {threshold})")
        self._log(f"Output folder: {output_dir}\n")
        self._set_buttons_enabled(False)
        self._worker = CombineBinarizeWorker(paths, threshold, output_dir)
        self._worker.log_signal.connect(self._log)
        self._worker.done_signal.connect(self._on_binarize_done)
        self._worker.start()

    def _on_binarize_done(self, success, msg):
        self._set_buttons_enabled(True)
        if success:
            QMessageBox.information(self, "Done", msg)
        else:
            QMessageBox.critical(self, "Error", f"Processing failed:\n{msg}")

    # ── Combine / Threshold Evaluation ────────────────────────────────────────

    def _run_preview(self):
        paths = self._get_mask_paths()
        if len(paths) < 1:
            QMessageBox.warning(self, "Too few masks",
                                "Please add at least 1 mask file.")
            return
        output_dir = self._resolved_output_dir()
        self.log_box.clear()
        self._log("Combine / Threshold Evaluation – loading and resolving …")
        if self._mri_path:
            self._log(f"MRI: {os.path.basename(self._mri_path)}")
        else:
            self._log("No MRI selected — preview will show masks only.")
        self._log(f"Output folder: {output_dir}\n")
        self._set_buttons_enabled(False)
        self._preview_worker = CombineForPreviewWorker(paths, self._mri_path, output_dir)
        self._preview_worker.log_signal.connect(self._log)
        self._preview_worker.result_ready.connect(self._on_preview_data_ready)
        self._preview_worker.error_signal.connect(self._on_preview_error)
        self._preview_worker.start()

    def _on_preview_data_ready(self, data):
        self._set_buttons_enabled(True)
        self._log("\nOpening preview dialog …")
        dialog = ThresholdPreviewDialog(
            resolved_slices_2d = data['resolved_slices_2d'],
            resolved_3d        = data['resolved_3d'],
            niis               = data['niis'],
            names              = data['names'],
            mri_slice_raw      = data['mri_slice_raw'],
            contrast_points    = data['contrast_points'],
            output_dir         = data['output_dir'],
            opt_axis           = data['opt_axis'],
            opt_idx            = data['opt_idx'],
            initial_threshold  = self.threshold_spin.value(),
            parent             = self,
        )
        dialog.exec_()

    def _on_preview_error(self, msg):
        self._set_buttons_enabled(True)
        QMessageBox.critical(self, "Error", f"Processing failed:\n{msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--working-dir", default=None,
                        help="Stain working directory passed by the master script.")
    known, remaining = parser.parse_known_args()

    app = QApplication.instance() or QApplication(sys.argv[:1] + remaining)
    window = MaskBinarizerApp(working_dir=known.working_dir)
    window.show()
    sys.exit(app.exec_())
