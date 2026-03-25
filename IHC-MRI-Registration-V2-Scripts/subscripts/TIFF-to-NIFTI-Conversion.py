import os
import sys
import argparse
import numpy as np
import json
import nibabel as nib
from skimage import io, transform
from skimage.color import rgb2gray
import subprocess
import shutil
import glob
import traceback

class AnnotationToNiftiConverter:
    def __init__(self):
        self._has_zenity = shutil.which('zenity') is not None

        # Default paths
        self.base_dir = os.getcwd()  # Current working directory
        self.annotations_dir = os.path.join(self.base_dir, "Annotation_masks")
        self.match_results_dir = os.path.join(self.base_dir, "Match_slice_results")
        
        # Initialize variables
        self.reference_nifti = None
        self.matching_info = None
        
    def _open_file_dialog(self, title, start_dir, file_filter="All files (*.*)"):
        """Return a selected file path using zenity, or '' if cancelled."""
        if self._has_zenity:
            cmd = ['zenity', '--file-selection', f'--title={title}',
                   f'--filename={start_dir}/_']
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
                if result.returncode == 0:
                    return result.stdout.strip()
                return ''
            except Exception:
                pass
        from PyQt5.QtWidgets import QApplication, QFileDialog
        app = QApplication.instance() or QApplication([])
        path, _ = QFileDialog.getOpenFileName(
            None, title, start_dir, file_filter,
            options=QFileDialog.DontUseNativeDialog
        )
        return path

    def _open_dir_dialog(self, title, start_dir):
        """Return a selected directory path using zenity, or '' if cancelled."""
        if self._has_zenity:
            cmd = ['zenity', '--file-selection', '--directory',
                   f'--title={title}', f'--filename={start_dir}/_']
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout.strip()
                return ''
            except Exception:
                pass
        from PyQt5.QtWidgets import QApplication, QFileDialog
        app = QApplication.instance() or QApplication([])
        chosen = QFileDialog.getExistingDirectory(
            None, title, start_dir,
            QFileDialog.DontUseNativeDialog | QFileDialog.ShowDirsOnly
        )
        return chosen

    def select_annotations_folder(self):
        """Select the folder containing annotation masks"""
        selected_dir = self._open_dir_dialog(
            "Select the folder containing annotation masks",
            self.annotations_dir
        )

        if selected_dir:
            self.annotations_dir = selected_dir
            print(f"Selected annotations directory: {self.annotations_dir}")
            return True
        else:
            print("No directory selected. Using default directory.")
            # Check if default directory exists
            if not os.path.exists(self.annotations_dir):
                print(f"Default directory '{self.annotations_dir}' does not exist. Exiting.")
                return False
            return True
            
    def get_slice_index(self):
        """Get slice index from matching info, supporting both JSON formats.

        Supports:
        - 'final_best_slice': {'index': N} (matplotlib Slice-Matching.py)
        - 'selected_slice': {'index': N} (PyQt5 Slice-MatchingV2.py)
        """
        if 'final_best_slice' in self.matching_info:
            return self.matching_info['final_best_slice']['index']
        elif 'selected_slice' in self.matching_info:
            return self.matching_info['selected_slice']['index']
        else:
            raise KeyError("Neither 'final_best_slice' nor 'selected_slice' found in matching info")

    def load_matching_info(self):
        """Load matching information from the matching_info.json file.

        Compatible with both Slice-Matching.py (matplotlib) and Slice-MatchingV2.py (PyQt5) outputs.
        """
        # Look for full_section_overview_matching_info.json in the matching results directory
        json_pattern = os.path.join(self.match_results_dir, "*full_section_overview*matching_info.json")
        json_files = glob.glob(json_pattern)

        if not json_files:
            # If not found, look for any matching_info.json file
            json_pattern = os.path.join(self.match_results_dir, "*matching_info.json")
            json_files = glob.glob(json_pattern)

            if not json_files:
                print(f"Error: No matching info JSON files found in {self.match_results_dir}")
                return False

        # Use the first matching file found
        selected_json = json_files[0]

        try:
            with open(selected_json, 'r') as f:
                self.matching_info = json.load(f)

            print(f"Loaded matching info from: {selected_json}")
            print(f"Anatomical plane: {self.matching_info['anatomical_plane']}")

            # Get slice index (compatible with both JSON formats)
            slice_idx = self.get_slice_index()
            print(f"Best slice index: {slice_idx}")
            print(f"Orientation: {self.matching_info['orientation']['description']}")

            # Also load the reference MRI NIfTI file
            mri_path = self.matching_info['mri_file']
            if not os.path.exists(mri_path):
                # Stored path is from a different OS/machine — try relative to working dir
                relative_candidate = os.path.join(self.base_dir, os.path.basename(mri_path))
                if os.path.exists(relative_candidate):
                    mri_path = relative_candidate
                    print(f"Resolved MRI path relative to working directory: {mri_path}")
                else:
                    # Ask the user to locate the file manually
                    print(f"Reference MRI not found at: {mri_path}")
                    mri_path = self._open_file_dialog(
                        "Locate the reference MRI NIfTI file",
                        self.base_dir,
                        "NIfTI files (*.nii *.nii.gz);;All files (*.*)"
                    )
                    if not mri_path:
                        print("Error: No reference MRI file selected.")
                        return False
            self.reference_nifti = nib.load(mri_path)
            print(f"Loaded reference MRI: {mri_path}")

            return True

        except Exception as e:
            print(f"Error: Failed to load matching info: {str(e)}")
            traceback.print_exc()
            return False
    
    def apply_orientation(self, image, orientation_desc):
        """
        Apply the specified orientation to the image.

        Compatible with both orientation description formats:
        - Long format (Slice-Matching.py): "Rotated 90° clockwise + flipped horizontally"
        - Short format (Slice-MatchingV2.py): "Rot 90 CW + Flip H"
        """
        oriented_img = image.copy()
        desc = orientation_desc.lower()  # Normalize to lowercase for matching

        # Check for rotation first (must be applied before flips)
        is_rotated_90 = any(x in desc for x in ["rotated 90", "rot 90 cw", "90° clockwise"])

        # Check for flips
        has_flip_h = any(x in desc for x in ["flipped horizontally", "flip h", "flipped h"])
        has_flip_v = any(x in desc for x in ["flipped vertically", "flip v", "flipped v"])

        # Handle combined H+V flip patterns
        if "h+v" in desc or "horizontally and vertically" in desc or "both" in desc:
            has_flip_h = True
            has_flip_v = True

        # Apply rotation first if needed
        if is_rotated_90:
            oriented_img = np.rot90(oriented_img, k=-1)  # k=-1 for clockwise

        # Then apply flips
        if has_flip_h:
            oriented_img = np.fliplr(oriented_img)
        if has_flip_v:
            oriented_img = np.flipud(oriented_img)

        return oriented_img
    
    def create_nifti_from_mask(self, mask, target_shape, plane, slice_idx, orientation_desc):
        """
        Create a NIfTI file from a mask image, placing it at the correct slice position
        
        Args:
            mask: The mask image array
            target_shape: The shape of the target volume (from reference NIfTI)
            plane: The anatomical plane index (0=Sagittal, 1=Coronal, 2=Axial)
            slice_idx: The slice index to place the mask
            orientation_desc: The orientation description
            
        Returns:
            nibabel.Nifti1Image: The created NIfTI image
        """
        # Get dimensions for the appropriate plane
        if plane == 0:  # Sagittal
            slice_shape = (target_shape[1], target_shape[2])
        elif plane == 1:  # Coronal
            slice_shape = (target_shape[0], target_shape[2])
        else:  # Axial
            slice_shape = (target_shape[0], target_shape[1])
        
        # Resize mask to match slice dimensions
        mask_resized = transform.resize(mask, slice_shape, anti_aliasing=True, preserve_range=True)
        
        # Apply orientation
        mask_oriented = self.apply_orientation(mask_resized, orientation_desc)
        
        # Create an empty volume with the same dimensions as the reference NIfTI
        mask_volume = np.zeros(target_shape)
        
        # Insert the mask at the specified slice position
        if plane == 0:  # Sagittal
            mask_volume[slice_idx, :, :] = mask_oriented
        elif plane == 1:  # Coronal
            mask_volume[:, slice_idx, :] = mask_oriented
        else:  # Axial
            mask_volume[:, :, slice_idx] = mask_oriented
        
        # Create a new NIfTI image with the mask data
        new_nifti = nib.Nifti1Image(mask_volume, self.reference_nifti.affine, self.reference_nifti.header)
        
        return new_nifti
        
    def process_annotations(self):
        """Process all .tif files in the annotations directory"""
        # Find all .tif, .tiff and .png files in the annotations directory
        tif_pattern = os.path.join(self.annotations_dir, "*.tif")
        tiff_pattern = os.path.join(self.annotations_dir, "*.tiff")
        png_pattern = os.path.join(self.annotations_dir, "*.png")
        all_files = glob.glob(tif_pattern) + glob.glob(tiff_pattern) + glob.glob(png_pattern)

        if not all_files:
            print(f"No .tif, .tiff or .png files found in {self.annotations_dir}")
            return
        
        # Extract needed info from the matching info
        try:
            plane = self.matching_info['plane_index']
            slice_idx = self.get_slice_index()  # Compatible with both JSON formats
            orientation_desc = self.matching_info['orientation']['description']
            target_shape = self.reference_nifti.shape
        except KeyError as e:
            print(f"Error: Missing key in matching info: {str(e)}")
            return
        
        # Process each .tif file
        success_count = 0
        error_count = 0
        
        for tif_file in all_files:
            try:
                print(f"Processing: {tif_file}")
                
                # Load the mask
                mask = io.imread(tif_file)
                
                # Determine the image type and convert appropriately
                if len(mask.shape) > 2:
                    # RGB image
                    mask_gray = rgb2gray(mask)
                    
                    # Check if it's likely a binary mask (only 2 values)
                    unique_values = np.unique(mask_gray)
                    if len(unique_values) <= 2:
                        # Convert to binary (0 and 1)
                        mask_processed = (mask_gray > 0).astype(np.float32)
                    else:
                        # Keep grayscale but normalize
                        mask_processed = mask_gray
                else:
                    # Already grayscale
                    # Check if it's likely a binary mask
                    unique_values = np.unique(mask)
                    if len(unique_values) <= 2:
                        # Convert to binary (0 and 1)
                        mask_processed = (mask > 0).astype(np.float32)
                    else:
                        # Keep grayscale
                        mask_processed = mask
                
                # Create NIfTI from mask
                mask_nifti = self.create_nifti_from_mask(
                    mask_processed, target_shape, plane, slice_idx, orientation_desc
                )
                
                # Save the NIfTI file with the same name as the input but with .nii.gz extension
                output_path = os.path.splitext(tif_file)[0] + ".nii.gz"
                nib.save(mask_nifti, output_path)
                
                print(f"Saved: {output_path}")
                success_count += 1
                
            except Exception as e:
                print(f"Error processing {tif_file}: {str(e)}")
                traceback.print_exc()
                error_count += 1
        
        print(f"Processing complete: {success_count} file(s) succeeded, {error_count} error(s).")
        
    def run(self):
        """Main function to run the converter"""
        try:
            print("=== TIF/PNG to NIfTI Annotation Mask Converter ===")
            
            # Select the annotations folder
            if not self.select_annotations_folder():
                return
            
            # Load matching info
            if not self.load_matching_info():
                return
            
            # Process annotations
            self.process_annotations()
            
            print("Conversion complete!")
            
        except Exception as e:
            print(f"Error: An unexpected error occurred: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIF/PNG to NIfTI Annotation Mask Converter")
    parser.add_argument("--annotations_dir", default=None,
                        help="Path to the annotations/density maps folder (skips folder selection dialog)")
    args = parser.parse_args()

    converter = AnnotationToNiftiConverter()
    if args.annotations_dir:
        converter.annotations_dir = os.path.abspath(args.annotations_dir)
        converter.silent = True  # suppress the completion popup in batch mode
        print(f"Non-interactive mode: annotations_dir = {converter.annotations_dir}")
        if converter.load_matching_info():
            converter.process_annotations()
        print("Conversion complete!")
    else:
        converter.run()
