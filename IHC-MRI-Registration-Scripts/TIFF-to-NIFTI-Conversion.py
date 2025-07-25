import os
import numpy as np
import json
import nibabel as nib
from skimage import io, transform
from skimage.color import rgb2gray
import tkinter as tk
from tkinter import filedialog, messagebox
import glob
import traceback

class AnnotationToNiftiConverter:
    def __init__(self):
        # Initialize GUI
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the root window
        
        # Default paths
        self.base_dir = os.getcwd()  # Current working directory
        self.annotations_dir = os.path.join(self.base_dir, "Annotation_masks")
        self.match_results_dir = os.path.join(self.base_dir, "Match_slice_results")
        
        # Initialize variables
        self.reference_nifti = None
        self.matching_info = None
        
    def select_annotations_folder(self):
        """Select the folder containing annotation masks"""
        selected_dir = filedialog.askdirectory(
            title="Select the folder containing annotation masks",
            initialdir=self.annotations_dir
        )
        
        if selected_dir:
            self.annotations_dir = selected_dir
            print(f"Selected annotations directory: {self.annotations_dir}")
            return True
        else:
            print("No directory selected. Using default directory.")
            # Check if default directory exists
            if not os.path.exists(self.annotations_dir):
                response = messagebox.askquestion(
                    "Create Directory", 
                    f"Default directory '{self.annotations_dir}' does not exist. Create it?")
                if response == 'yes':
                    os.makedirs(self.annotations_dir, exist_ok=True)
                    print(f"Created directory: {self.annotations_dir}")
                    return True
                else:
                    print("Directory not created. Exiting.")
                    return False
            return True
            
    def load_matching_info(self):
        """Load matching information from the full_section_overview_matching_info.json file"""
        # Look for full_section_overview_matching_info.json in the matching results directory
        json_pattern = os.path.join(self.match_results_dir, "*full_section_overview*matching_info.json")
        json_files = glob.glob(json_pattern)
        
        if not json_files:
            # If not found, look for any matching_info.json file
            json_pattern = os.path.join(self.match_results_dir, "*matching_info.json")
            json_files = glob.glob(json_pattern)
            
            if not json_files:
                messagebox.showerror("Error", f"No matching info JSON files found in {self.match_results_dir}")
                return False
        
        # Use the first matching file found
        selected_json = json_files[0]
        
        try:
            with open(selected_json, 'r') as f:
                self.matching_info = json.load(f)
                
            print(f"Loaded matching info from: {selected_json}")
            print(f"Anatomical plane: {self.matching_info['anatomical_plane']}")
            print(f"Best slice index: {self.matching_info['final_best_slice']['index']}")
            print(f"Orientation: {self.matching_info['orientation']['description']}")
            
            # Also load the reference MRI NIfTI file
            mri_path = self.matching_info['mri_file']
            if os.path.exists(mri_path):
                self.reference_nifti = nib.load(mri_path)
                print(f"Loaded reference MRI: {mri_path}")
            else:
                messagebox.showerror("Error", f"Reference MRI file not found: {mri_path}")
                return False
                
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load matching info: {str(e)}")
            traceback.print_exc()
            return False
    
    def apply_orientation(self, image, orientation_desc):
        """
        Apply the specified orientation to the image.
        Matches the orientation descriptions from the HistologyMatcher class.
        """
        oriented_img = image.copy()
        
        if "Original" in orientation_desc:
            # No changes
            pass
        elif "Flipped horizontally and vertically" in orientation_desc:
            # Flipped both horizontally and vertically
            oriented_img = np.flipud(np.fliplr(oriented_img))
        elif "Rotated 90° clockwise + flipped horizontally" in orientation_desc:
            # Rotated 90° clockwise + flipped horizontally
            oriented_img = np.rot90(oriented_img, k=-1)  # k=-1 for clockwise
            oriented_img = np.fliplr(oriented_img)
        elif "Rotated 90° clockwise + flipped vertically" in orientation_desc:
            # Rotated 90° clockwise + flipped vertically
            oriented_img = np.rot90(oriented_img, k=-1)  # k=-1 for clockwise
            oriented_img = np.flipud(oriented_img)
        elif "Rotated 90° clockwise + flipped both" in orientation_desc:
            # Rotated 90° clockwise + flipped both ways
            oriented_img = np.rot90(oriented_img, k=-1)  # k=-1 for clockwise
            oriented_img = np.flipud(np.fliplr(oriented_img))
        elif "Flipped horizontally" in orientation_desc:
            # Flipped horizontally
            oriented_img = np.fliplr(oriented_img)
        elif "Flipped vertically" in orientation_desc:
            # Flipped vertically
            oriented_img = np.flipud(oriented_img)
        elif "Rotated 90° clockwise" in orientation_desc:
            # Rotated 90° clockwise
            oriented_img = np.rot90(oriented_img, k=-1)  # k=-1 for clockwise
        
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
        # Find all .tif and .tiff files in the annotations directory
        tif_pattern = os.path.join(self.annotations_dir, "*.tif")
        tiff_pattern = os.path.join(self.annotations_dir, "*.tiff")
        all_files = glob.glob(tif_pattern) + glob.glob(tiff_pattern)
        
        if not all_files:
            messagebox.showinfo("Info", f"No .tif or .tiff files found in {self.annotations_dir}")
            return
        
        # Extract needed info from the matching info
        try:
            plane = self.matching_info['plane_index']
            slice_idx = self.matching_info['final_best_slice']['index']
            orientation_desc = self.matching_info['orientation']['description']
            target_shape = self.reference_nifti.shape
        except KeyError as e:
            messagebox.showerror("Error", f"Missing key in matching info: {str(e)}")
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
        
        # Show summary message
        messagebox.showinfo(
            "Processing Complete", 
            f"Processed {success_count} files successfully.\n"
            f"Encountered errors in {error_count} files."
        )
        
    def run(self):
        """Main function to run the converter"""
        try:
            print("=== TIF to NIfTI Annotation Mask Converter ===")
            
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
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    converter = AnnotationToNiftiConverter()
    converter.run()
