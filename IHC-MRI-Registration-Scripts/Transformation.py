import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy import interpolate
import pickle
import argparse
import glob
import tkinter as tk
from tkinter import filedialog, messagebox
import time

class TransformApplier:
    def __init__(self, binarize=False, threshold=None):
        """
        Initialize transformer with binarization and threshold options
        
        Args:
            binarize: Whether to binarize masks after transformation
            threshold: Intensity threshold value (pixels below this will be set to zero)
        """
        self.binarize = binarize
        self.threshold = threshold
        
        # Initialize file paths
        self.masks_folder = None
        self.linear_transform_path = None
        self.nonlinear_transform_path = None
        self.deformation_field_path = None
        self.output_dir = None
        
        # Initialize GUI components
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the root window

    def select_files(self):
        """
        Select input folder and find transformation files automatically
        
        Returns:
            bool: True if selection successful, False otherwise
        """
        # Ask for the masks folder
        self.masks_folder = filedialog.askdirectory(
            title="Select folder containing segmentation masks"
        )
        
        if not self.masks_folder:
            print("No masks folder selected. Exiting.")
            return False
        
        print(f"Selected masks folder: {self.masks_folder}")
        
        # Try to find linear and non-linear transform files automatically
        result = self.find_transform_files()
        if not result:
            print("Could not find necessary transform files. Exiting.")
            return False
        
        # Create output directory
        parent_dir = os.path.dirname(self.masks_folder)
        self.output_dir = os.path.join(parent_dir, "Transformation_results")
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output will be saved to: {self.output_dir}")
        
        return True
    
    def find_transform_files(self):
        """
        Find linear and non-linear transform files in the standard folders
        
        Returns:
            bool: True if all required files found, False otherwise
        """
        # Get parent directory
        parent_dir = os.path.dirname(self.masks_folder)
        
        # Check for Linear_registration_results folder
        linear_dir = os.path.join(parent_dir, "Linear_registration_results")
        if not os.path.isdir(linear_dir):
            print(f"Linear registration results directory not found: {linear_dir}")
            return self.select_transform_files_manually()
        
        # Check for Non-linear_registration_results folder
        nonlinear_dir = os.path.join(parent_dir, "Non-linear_registration_results")
        if not os.path.isdir(nonlinear_dir):
            print(f"Non-linear registration results directory not found: {nonlinear_dir}")
            return self.select_transform_files_manually()
        
        # Try to find linear transform
        linear_transform_files = glob.glob(os.path.join(linear_dir, "transform_*.tfm"))
        if not linear_transform_files:
            # Try .pkl extension
            linear_transform_files = glob.glob(os.path.join(linear_dir, "transform_*.pkl"))
        
        if not linear_transform_files:
            print("Could not find linear transform file")
            return self.select_transform_files_manually()
        
        self.linear_transform_path = linear_transform_files[0]
        print(f"Found linear transform: {self.linear_transform_path}")
        
        # Try to find non-linear transform
        self.nonlinear_transform_path = os.path.join(nonlinear_dir, "nonlinear_transform.pkl")
        self.deformation_field_path = os.path.join(nonlinear_dir, "deformation_field.pkl")
        
        if not os.path.exists(self.nonlinear_transform_path) or not os.path.exists(self.deformation_field_path):
            print("Could not find non-linear transform or deformation field files")
            return self.select_transform_files_manually()
        
        print(f"Found non-linear transform: {self.nonlinear_transform_path}")
        print(f"Found deformation field: {self.deformation_field_path}")
        
        return True
    
    def select_transform_files_manually(self):
        """
        Manual selection of transform files if automatic detection fails
        
        Returns:
            bool: True if selection successful, False otherwise
        """
        # Ask user to select linear transform file
        self.linear_transform_path = filedialog.askopenfilename(
            title="Select linear transform file",
            filetypes=[("Transform files", "*.tfm;*.pkl")]
        )
        
        if not self.linear_transform_path:
            print("No linear transform file selected. Exiting.")
            return False
        
        # Ask user to select non-linear transform file
        self.nonlinear_transform_path = filedialog.askopenfilename(
            title="Select non-linear transform file",
            filetypes=[("Pickle files", "*.pkl")]
        )
        
        if not self.nonlinear_transform_path:
            print("No non-linear transform file selected. Exiting.")
            return False
        
        # Ask user to select deformation field file
        self.deformation_field_path = filedialog.askopenfilename(
            title="Select deformation field file",
            filetypes=[("Pickle files", "*.pkl")]
        )
        
        if not self.deformation_field_path:
            print("No deformation field file selected. Exiting.")
            return False
        
        print(f"Selected linear transform: {self.linear_transform_path}")
        print(f"Selected non-linear transform: {self.nonlinear_transform_path}")
        print(f"Selected deformation field: {self.deformation_field_path}")
        
        return True

    def detect_slices_with_data(self, data, dimension=None):
        """
        Detect which slices in the volume contain data (non-zero values)
        
        Args:
            data: 3D numpy array of image data
            dimension: Specific dimension to check, or None to check all
            
        Returns:
            tuple: (dimension with most data, list of slice indices with data)
        """
        best_dim = 0
        best_slices = []
        max_data_count = 0
        
        # Check each dimension if not specified
        dimensions = [dimension] if dimension is not None else range(3)
        
        for dim in dimensions:
            slices_with_data = []
            
            # Count non-zero pixels in each slice
            if dim == 0:
                num_slices = data.shape[0]
                for i in range(num_slices):
                    if np.count_nonzero(data[i, :, :]) > 1000:  # More than 10 non-zero pixels
                        slices_with_data.append(i)
                total_data = sum([np.count_nonzero(data[i, :, :]) for i in slices_with_data])
            elif dim == 1:
                num_slices = data.shape[1]
                for i in range(num_slices):
                    if np.count_nonzero(data[:, i, :]) > 1000:
                        slices_with_data.append(i)
                total_data = sum([np.count_nonzero(data[:, i, :]) for i in slices_with_data])
            else:  # dim == 2
                num_slices = data.shape[2]
                for i in range(num_slices):
                    if np.count_nonzero(data[:, :, i]) > 1000:
                        slices_with_data.append(i)
                total_data = sum([np.count_nonzero(data[:, :, i]) for i in slices_with_data])
            
            # If this dimension has more data than previously found, update
            if total_data > max_data_count:
                max_data_count = total_data
                best_dim = dim
                best_slices = slices_with_data
        
        return best_dim, best_slices

    def postprocess_data(self, data):
        """
        Apply thresholding and/or binarization to data
        
        Args:
            data: Numpy array of image data
            
        Returns:
            Processed data array
        """
        processed_data = data.copy()
        
        # Apply threshold first if specified
        if self.threshold is not None:
            print(f"Applying intensity threshold of {self.threshold}")
            processed_data[processed_data < self.threshold] = 0
        
        # Then apply binarization if requested
        if self.binarize:
            print("Applying binarization")
            processed_data = (processed_data > 0).astype(data.dtype)
        
        return processed_data

    def apply_linear_transform(self, mask_nii, transform_path, output_path=None):
        """
        Apply a saved linear transform to a segmentation mask or density map.
        Strictly enforces that data remains only in original slices.
        
        Args:
            mask_nii: Nibabel NIFTI image object containing the data
            transform_path: Path to the saved transform file (.tfm or .pkl)
            output_path: Path where the transformed data will be saved
        
        Returns:
            The transformed data as a nibabel NIFTI image
        """
        print(f"Applying linear transform from {transform_path}")
        
        # Load the data and remember which slices originally had data
        mask_data = mask_nii.get_fdata()
        original_data_copy = mask_data.copy()  # Keep a copy for later comparison
        
        # Load transform info
        if transform_path.endswith('.tfm'):
            # Load SimpleITK transform
            transform = sitk.ReadTransform(transform_path)
            
            # Also need slice info from the pickle file
            pickle_path = transform_path.replace('.tfm', '.pkl')
            if os.path.exists(pickle_path):
                with open(pickle_path, 'rb') as f:
                    transform_dict = pickle.load(f)
                    slice_dimension = transform_dict['slice_info']['dimension']
                    slice_index = transform_dict['slice_info']['slice_index']
            else:
                # Try to find slice info in the same directory
                dir_path = os.path.dirname(transform_path)
                info_path = os.path.join(dir_path, "registration_info.pkl")
                if os.path.exists(info_path):
                    with open(info_path, 'rb') as f:
                        info_dict = pickle.load(f)
                        slice_dimension = info_dict.get('slice_dimension', 0)
                        slice_index = info_dict.get('slice_index', 0)
                else:
                    # Find slices with data instead of using defaults
                    print("Slice info not found, detecting slice with data...")
                    slice_dimension, slice_indices = self.detect_slices_with_data(mask_data)
                    if not slice_indices:
                        print("No data found in the mask!")
                        return mask_nii  # Return original if no data found
                    slice_index = slice_indices[0]  # Use first slice with data
                    print(f"Detected data in dimension {slice_dimension}, slice {slice_index}")
        else:
            # Load from pickle
            with open(transform_path, 'rb') as f:
                transform_dict = pickle.load(f)
                slice_dimension = transform_dict['slice_info']['dimension']
                slice_index = transform_dict['slice_info']['slice_index']
            
            # Create SimpleITK transform
            transform = sitk.AffineTransform(transform_dict['dimension'])
            transform.SetParameters(transform_dict['parameters'])
            transform.SetFixedParameters(transform_dict['fixed_parameters'])
        
        # Create transformed data with same shape as original
        transformed_data = np.zeros_like(mask_data)
        
        # Get the exact indices of slices that contain data in the original
        original_slice_indices = []
        if slice_dimension == 0:
            for i in range(mask_data.shape[0]):
                if np.count_nonzero(mask_data[i, :, :]) > 0:
                    original_slice_indices.append(i)
        elif slice_dimension == 1:
            for i in range(mask_data.shape[1]):
                if np.count_nonzero(mask_data[:, i, :]) > 0:
                    original_slice_indices.append(i)
        else:  # slice_dimension == 2
            for i in range(mask_data.shape[2]):
                if np.count_nonzero(mask_data[:, :, i]) > 0:
                    original_slice_indices.append(i)
        
        print(f"Original data found in exactly {len(original_slice_indices)} slices in dimension {slice_dimension}: {original_slice_indices}")
        
        # If no slices found with data, use the slice index from transform
        if not original_slice_indices:
            original_slice_indices = [slice_index]
        
        # Process only the original slices with data
        for idx in original_slice_indices:
            # Extract the slice to transform
            if slice_dimension == 0:
                mask_slice = mask_data[idx, :, :]
            elif slice_dimension == 1:
                mask_slice = mask_data[:, idx, :]
            else:  # 2
                mask_slice = mask_data[:, :, idx]
            
            # Skip slices with minimal data
            if np.count_nonzero(mask_slice) < 10:
                continue
            
            # Convert to SimpleITK image
            mask_sitk = sitk.GetImageFromArray(mask_slice.astype(mask_data.dtype))
            
            # Create resample filter
            reference_size = mask_sitk.GetSize()
            reference_spacing = mask_sitk.GetSpacing()
            reference_origin = mask_sitk.GetOrigin()
            reference_direction = mask_sitk.GetDirection()
            
            resampler = sitk.ResampleImageFilter()
            resampler.SetSize(reference_size)
            resampler.SetOutputSpacing(reference_spacing)
            resampler.SetOutputOrigin(reference_origin)
            resampler.SetOutputDirection(reference_direction)
            
            # Use nearest neighbor interpolation to avoid value interpolation
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetTransform(transform)
            
            # Apply transform
            transformed_sitk = resampler.Execute(mask_sitk)
            transformed_slice = sitk.GetArrayFromImage(transformed_sitk)
            
            # Insert transformed slice back ONLY to the original slice location
            if slice_dimension == 0:
                transformed_data[idx, :, :] = transformed_slice
            elif slice_dimension == 1:
                transformed_data[:, idx, :] = transformed_slice
            else:  # 2
                transformed_data[:, :, idx] = transformed_slice
        
        # Verify that data is only in the original slices
        for i in range(mask_data.shape[slice_dimension]):
            if i not in original_slice_indices:
                # Zero out any data that might have appeared in non-original slices
                if slice_dimension == 0:
                    transformed_data[i, :, :] = 0
                elif slice_dimension == 1:
                    transformed_data[:, i, :] = 0
                else:  # slice_dimension == 2
                    transformed_data[:, :, i] = 0
        
        # Create NIFTI image with the same header and affine as the input
        transformed_nii = nib.Nifti1Image(transformed_data, mask_nii.affine, mask_nii.header)
        
        # Save result if output path provided
        if output_path:
            nib.save(transformed_nii, output_path)
            print(f"Saved transformed data to: {output_path}")
        
        # Final verification
        verify_slices = []
        for i in range(transformed_data.shape[slice_dimension]):
            if slice_dimension == 0:
                if np.count_nonzero(transformed_data[i, :, :]) > 0:
                    verify_slices.append(i)
            elif slice_dimension == 1:
                if np.count_nonzero(transformed_data[:, i, :]) > 0:
                    verify_slices.append(i)
            else:  # slice_dimension == 2
                if np.count_nonzero(transformed_data[:, :, i]) > 0:
                    verify_slices.append(i)
        
        print(f"After transformation, data found in {len(verify_slices)} slices: {verify_slices}")
        if set(verify_slices) != set(original_slice_indices):
            print("WARNING: Transformed data is in different slices than the original!")
            print(f"Slices only in original: {set(original_slice_indices) - set(verify_slices)}")
            print(f"Slices only in transformed: {set(verify_slices) - set(original_slice_indices)}")
        
        return transformed_nii

    def apply_nonlinear_transform(self, mask_nii, transform_path, deformation_field_path, output_path=None):
        """
        Apply a saved non-linear transform to a segmentation mask.
        Strictly enforces that data remains only in original slices.
        
        Args:
            mask_nii: Nibabel NIFTI image object containing the segmentation mask
            transform_path: Path to the saved non-linear transform parameters
            deformation_field_path: Path to the saved deformation field
            output_path: Path where the transformed mask will be saved
        
        Returns:
            The transformed mask as a nibabel NIFTI image
        """
        print(f"Applying non-linear transform from {transform_path}")
        
        # Load the mask data
        mask_data = mask_nii.get_fdata()
        
        # Load transform parameters
        with open(transform_path, 'rb') as f:
            transform_params = pickle.load(f)
        
        # Get slice info
        slice_dimension = transform_params['slice_info']['dimension']
        slice_index = transform_params['slice_info']['slice_index']
        
        # Load deformation field
        with open(deformation_field_path, 'rb') as f:
            deformation_field = pickle.load(f)
        
        # Get the grid coordinates
        x_orig = deformation_field['x_orig']
        y_orig = deformation_field['y_orig']
        x_warped = deformation_field['x_warped']
        y_warped = deformation_field['y_warped']
        
        # Create transformed data with same shape as original
        transformed_data = np.zeros_like(mask_data)
        
        # Get the exact indices of slices that contain data in the original
        original_slice_indices = []
        if slice_dimension == 0:
            for i in range(mask_data.shape[0]):
                if np.count_nonzero(mask_data[i, :, :]) > 0:
                    original_slice_indices.append(i)
        elif slice_dimension == 1:
            for i in range(mask_data.shape[1]):
                if np.count_nonzero(mask_data[:, i, :]) > 0:
                    original_slice_indices.append(i)
        else:  # slice_dimension == 2
            for i in range(mask_data.shape[2]):
                if np.count_nonzero(mask_data[:, :, i]) > 0:
                    original_slice_indices.append(i)
        
        print(f"Original data found in exactly {len(original_slice_indices)} slices in dimension {slice_dimension}: {original_slice_indices}")
        
        # If no slices found with data, use the slice index from transform
        if not original_slice_indices:
            original_slice_indices = [slice_index]
        
        # Process only the original slices with data
        for idx in original_slice_indices:
            # Extract the slice to transform
            if slice_dimension == 0:
                mask_slice = mask_data[idx, :, :]
            elif slice_dimension == 1:
                mask_slice = mask_data[:, idx, :]
            else:  # 2
                mask_slice = mask_data[:, :, idx]
            
            # Skip slices with minimal data
            if np.count_nonzero(mask_slice) < 10:
                continue
            
            # Ensure shape matches
            if mask_slice.shape != deformation_field['shape']:
                print(f"Warning: Mask slice shape {mask_slice.shape} doesn't match deformation field shape {deformation_field['shape']}")
                # Resize the mask slice to match deformation field shape
                from skimage import transform as skimage_transform
                mask_slice = skimage_transform.resize(
                    mask_slice, 
                    deformation_field['shape'], 
                    order=0,  # nearest neighbor interpolation for masks
                    preserve_range=True,
                    anti_aliasing=False
                ).astype(mask_data.dtype)
            
            # For segmentation masks, use nearest-neighbor interpolation to preserve label values
            # We're going backward from warped coordinates to original coordinates
            # to sample the mask at the correct locations
            height, width = mask_slice.shape
            grid_y, grid_x = np.mgrid[0:height, 0:width]
            
            # Flatten the arrays for griddata
            points = np.vstack([x_warped.ravel(), y_warped.ravel()]).T
            xi = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
            
            # Get mask values
            values = mask_slice.flatten()
            
            # Use griddata with nearest-neighbor interpolation for segmentation masks
            transformed_mask = interpolate.griddata(points, values, xi, method='nearest')
            transformed_mask = transformed_mask.reshape(height, width)
            
            # Convert to same data type as original
            transformed_mask = transformed_mask.astype(mask_data.dtype)
            
            # Insert transformed slice back ONLY to the original slice location
            if slice_dimension == 0:
                transformed_data[idx, :, :] = transformed_mask
            elif slice_dimension == 1:
                transformed_data[:, idx, :] = transformed_mask
            else:  # 2
                transformed_data[:, :, idx] = transformed_mask
        
        # Verify that data is only in the original slices
        for i in range(mask_data.shape[slice_dimension]):
            if i not in original_slice_indices:
                # Zero out any data that might have appeared in non-original slices
                if slice_dimension == 0:
                    transformed_data[i, :, :] = 0
                elif slice_dimension == 1:
                    transformed_data[:, i, :] = 0
                else:  # slice_dimension == 2
                    transformed_data[:, :, i] = 0
        
        # Create NIFTI image with the same header and affine as the input
        transformed_nii = nib.Nifti1Image(transformed_data, mask_nii.affine, mask_nii.header)
        
        # Save result if output path provided
        if output_path:
            nib.save(transformed_nii, output_path)
            print(f"Saved non-linear transformed mask to: {output_path}")
        
        # Final verification
        verify_slices = []
        for i in range(transformed_data.shape[slice_dimension]):
            if slice_dimension == 0:
                if np.count_nonzero(transformed_data[i, :, :]) > 0:
                    verify_slices.append(i)
            elif slice_dimension == 1:
                if np.count_nonzero(transformed_data[:, i, :]) > 0:
                    verify_slices.append(i)
            else:  # slice_dimension == 2
                if np.count_nonzero(transformed_data[:, :, i]) > 0:
                    verify_slices.append(i)
        
        print(f"After transformation, data found in {len(verify_slices)} slices: {verify_slices}")
        if set(verify_slices) != set(original_slice_indices):
            print("WARNING: Transformed data is in different slices than the original!")
            print(f"Slices only in original: {set(original_slice_indices) - set(verify_slices)}")
            print(f"Slices only in transformed: {set(verify_slices) - set(original_slice_indices)}")
        
        return transformed_nii

    def process_mask(self, mask_path):
        """
        Apply both linear and non-linear transforms sequentially to a mask.
        
        Args:
            mask_path: Path to the mask NIFTI file
            
        Returns:
            Path to final transformed mask
        """
        print(f"\nProcessing mask: {mask_path}")
        
        try:
            # Load the mask
            mask_nii = nib.load(mask_path)
            
            # Get filename without extension
            mask_filename = os.path.basename(mask_path)
            mask_name = os.path.splitext(mask_filename)[0]
            if mask_name.endswith('.nii'):
                mask_name = os.path.splitext(mask_name)[0]
            
            # Create output suffixes based on processing options
            suffixes = []
            if self.threshold is not None:
                suffixes.append(f"thr{self.threshold}")
            if self.binarize:
                suffixes.append("bin")
            
            suffix_str = "_" + "_".join(suffixes) if suffixes else ""
            
            # Final output path only
            final_output_path = os.path.join(self.output_dir, f"{mask_name}_transformed{suffix_str}.nii.gz")
            
            # Step 1: Apply linear transform (no intermediate save)
            print(f"Applying linear transform to {mask_filename}...")
            linear_mask_nii = self.apply_linear_transform(
                mask_nii, 
                self.linear_transform_path, 
                None  # Don't save intermediate result
            )
            
            # Check if linear transform produced any data
            linear_data = linear_mask_nii.get_fdata()
            if np.count_nonzero(linear_data) == 0:
                print(f"WARNING: Linear transform produced empty output for {mask_filename}")
                
                # Save original data as a fallback if linear transform failed
                fallback_output_path = os.path.join(self.output_dir, f"{mask_name}_original.nii.gz")
                nib.save(mask_nii, fallback_output_path)
                print(f"Saved original (untransformed) mask to: {fallback_output_path}")
                return fallback_output_path
            
            # Step 2: Apply non-linear transform to the linearly transformed mask
            print(f"Applying non-linear transform to linearly transformed {mask_filename}...")
            nonlinear_mask_nii = self.apply_nonlinear_transform(
                linear_mask_nii,
                self.nonlinear_transform_path, 
                self.deformation_field_path,
                None  # Don't save intermediate result
            )
            
            # Check if non-linear transform produced any data
            nonlinear_data = nonlinear_mask_nii.get_fdata()
            if np.count_nonzero(nonlinear_data) == 0:
                print(f"WARNING: Non-linear transform produced empty output for {mask_filename}")
                
                # If non-linear transform failed but linear succeeded, use linear result
                if np.count_nonzero(linear_data) > 0:
                    print(f"Using linear transform result as final output")
                    nonlinear_data = linear_data
                    nonlinear_mask_nii = linear_mask_nii
            
            # Step 3: Apply post-processing to the fully transformed result
            print(f"Applying post-processing to transformed {mask_filename}...")
            processed_data = self.postprocess_data(nonlinear_data)
            
            # Save the final processed result only
            processed_nii = nib.Nifti1Image(processed_data, nonlinear_mask_nii.affine, nonlinear_mask_nii.header)
            nib.save(processed_nii, final_output_path)
            print(f"Saved processed transformed mask to: {final_output_path}")
            
            print(f"Successfully processed: {mask_filename}")
            
            return final_output_path
            
        except Exception as e:
            print(f"Error processing mask {mask_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def process_all_masks(self):
        """
        Process all NIFTI files in the masks folder
        
        Returns:
            List of processed mask paths
        """
        # Find all NIFTI files in the folder
        nifti_files = []
        for ext in ['.nii', '.nii.gz']:
            nifti_files.extend(glob.glob(os.path.join(self.masks_folder, f'*{ext}')))
        
        if not nifti_files:
            print(f"No NIFTI files found in {self.masks_folder}")
            return []
        
        print(f"Found {len(nifti_files)} NIFTI files in {self.masks_folder}")
        
        # Process each file
        processed_files = []
        for mask_path in nifti_files:
            output_path = self.process_mask(mask_path)
            if output_path:
                processed_files.append(output_path)
        
        return processed_files

    def run(self):
        """
        Run the transformation application process
        
        Returns:
            True if successful, False otherwise
        """
        try:
            start_time = time.time()
            
            # Select files
            if not self.select_files():
                return False
            
            # Process all masks
            processed_files = self.process_all_masks()
            
            elapsed_time = time.time() - start_time
            print(f"\nTransformation complete! {len(processed_files)} files processed in {elapsed_time:.2f} seconds")
            print(f"Results saved to: {self.output_dir}")
            
            return True
            
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            return False

# Standalone execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply registration transforms to segmentation masks')
    parser.add_argument('--masks', help='Path to folder containing segmentation mask NIFTI files')
    parser.add_argument('--linear', help='Path to the saved linear transform file (.pkl or .tfm)')
    parser.add_argument('--nonlinear', help='Path to the saved non-linear transform parameters (.pkl)')
    parser.add_argument('--deformation', help='Path to the saved deformation field (.pkl)')
    parser.add_argument('--output', help='Output directory for transformed masks')
    parser.add_argument('--binarize', action='store_true', help='Binarize masks after transformation')
    parser.add_argument('--threshold', type=float, help='Intensity threshold value (pixels below will be set to zero)')
    
    args = parser.parse_args()
    
    # If command line arguments provided, use them
    if args.masks and args.linear and args.nonlinear and args.deformation and args.output:
        transformer = TransformApplier(binarize=args.binarize, threshold=args.threshold)
        transformer.masks_folder = args.masks
        transformer.linear_transform_path = args.linear
        transformer.nonlinear_transform_path = args.nonlinear
        transformer.deformation_field_path = args.deformation
        transformer.output_dir = args.output
        os.makedirs(transformer.output_dir, exist_ok=True)
        transformer.process_all_masks()
    else:
        # Interactive mode
        transformer = TransformApplier(
            binarize=args.binarize if args is not None else False,
            threshold=args.threshold if args is not None else None
        )
        transformer.run()
