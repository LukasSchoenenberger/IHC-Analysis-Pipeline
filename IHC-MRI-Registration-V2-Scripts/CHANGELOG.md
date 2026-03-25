# Changelog (v1 → v2)

## General
- All GUI scripts migrated from tkinter to PyQt5
- File dialogs replaced with zenity (Linux-native) with PyQt5 fallback
- Scripts reorganized into subscripts/ and configs/ subdirectories

## Master Script
- Complete rewrite: tkinter → PyQt5 GUI with resizable splitter layout
- Batch processing for Transformer, TIF/PNG→NIfTI, and Split Annotation Masks
  across multiple blocks and stain directories
- Configurable batch settings with persistent JSON configs (file renaming,
  copy-to-Density_Maps/Annotation_Masks, directory type selection)
- Built-in documentation dialog with folder structure reference
- "Create Folder Structure" function for setting up new projects interactively
- Processing log with colour-coded messages and pop-out to separate window
- Annotation mask manager for QuPath download script generation

## Slice Matching
- Rewritten in PyQt5.
- New contrast curve dialog with interactive point-based curve editing and
  histogram visualization
- Changed from multistep process to single step, that allows you to make all adjustments simultaneously.
- Removed automatic detection utility, as it rarely matched to the correct slice.
- Contrast-adjusted images used in saved preview PNG

## Linear Registration
- Rewritten in PyQt5 + pyqtgraph
- Contrast adjustment functionality added
- Transform controls with +/- nudge buttons, mouse controls or direct text input

## Combined Slice Matching + Linear Registration (new)
- Two-in-one GUI combining slice matching and linear registration in a
  single window (left panel: slice matching, right panel: registration)
- Transform output fully compatible with the separate Linear Registration
  script and Transformation.py

## Non-Linear Registration
- Complete rewrite: tkinter + matplotlib → PyQt5 + pyqtgraph with
  GPU-accelerated rendering
- Save/load landmarks functionality added.
- New interaction modes: Landmark, Line, Ruler, Move Landmark, Move Line
  - Line-to-landmark generation with even spacing along drawn polylines
  - Move Landmark mode: drag landmarks to reposition
  - Move Line mode: free move or endpoint-constrained along original polyline
- Added contrast adjusment for MRI and IHC image
- Added overlay functions: 
  - allows the higher resolution and colored overview of the WSI to be overlayed on the ihc image for more accurate landmark placement. 
  - An overview of an already co-registered WSI can be overlayed on the MRI image.
  - The overlay images are saved in non_linear_registration_results/

## Mask Binarizer (replaces Threshold Evaluation)
- Complete rewrite with expanded functionality
- Allows for multiple masks to be binarized simultaneously:
  - Per-voxel conflict resolution: highest value wins for shared voxels,
  threshold applied to the rest
- Two modes: Combine/Binarize (direct) and Combine/Threshold Evaluation
  (interactive preview with MRI overlay)
- Manual mask editing: draw (left-click) / erase (right-click) per label
  with adjustable brush size and exclusive label assignment
- Interactive view: mouse wheel zoom, middle-button pan, S-key overlay toggle

## Transformation
- File dialogs migrated from tkinter to zenity + PyQt5 fallback
- Added projective transform support alongside existing affine path

## Segmentation Splitting
- Per-label splitting: each unique segmentation label processed independently
  (previously all labels collapsed to binary)
- Patch size included in output filenames
- Batch mode via --files CLI argument

## TIF/PNG to NIfTI Conversion
- Added PNG support alongside TIFF
- Batch mode via --annotations_dir CLI argument
- File dialogs migrated from tkinter to zenity + PyQt5 fallback

## Download Masks Generator
- Input changed from plain text file to JSON config
- Output (groovy script) placed in the working directory
