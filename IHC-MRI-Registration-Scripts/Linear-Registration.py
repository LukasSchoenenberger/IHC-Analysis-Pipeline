import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button
import tkinter as tk
from tkinter import filedialog, messagebox
from skimage import transform, exposure, filters, feature
from skimage.metrics import structural_similarity
from scipy.ndimage import gaussian_filter
import cv2
import SimpleITK as sitk
import time
from pathlib import Path
import glob
import pickle

class AutoRegistration:
    def __init__(self):
        # Initialize file paths
        self.mri_path = None
        self.ihc_path = None
        
        # Initialize data
        self.mri_nifti = None
        self.ihc_nifti = None
        self.mri_data = None
        self.ihc_data = None
        
        # Slice information
        self.slice_index = None
        self.slice_axis = None
        
        # Registration results
        self.registered_images = []
        self.method_names = []
        self.transforms = []
        self.metrics = None
        
        # User selected method
        self.selected_method_index = None
        
        # Initialize GUI
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the root window
    
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
        """Apply preprocessing steps to an image"""
        if img.dtype != np.float32 and img.dtype != np.float64:
            img = img.astype(np.float32)
        img_norm = self.normalize_image(img)
        img_eq = exposure.equalize_hist(img_norm)
        img_enhanced = self.enhance_edges(img_eq)
        return img_enhanced

    def extract_slice(self, volume, slice_idx, axis=2):
        """Extract a specific slice from the volume along a given axis"""
        if axis == 0:
            return volume[slice_idx, :, :]
        elif axis == 1:
            return volume[:, slice_idx, :]
        else:  # axis == 2
            return volume[:, :, slice_idx]

    def find_non_empty_slices(self, nifti_data):
        """Find all non-empty slices in the volume and return their indices and axis"""
        results = []
        
        for axis in range(3):
            dim_size = nifti_data.shape[axis]
            
            for slice_idx in range(dim_size):
                slice_data = self.extract_slice(nifti_data, slice_idx, axis)
                nonzero_count = np.count_nonzero(slice_data)
                
                if nonzero_count > 100:  # More than 100 non-zero pixels
                    results.append((slice_idx, axis, nonzero_count))
        
        # Sort by number of non-zero pixels (descending)
        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def normalized_mutual_information(self, img1, img2, bins=32):
        """Calculate normalized mutual information between two images"""
        hist_2d, _, _ = np.histogram2d(
            img1.flatten(), img2.flatten(), bins=bins, range=[[0, 1], [0, 1]]
        )
        
        hist_2d_normalized = hist_2d / np.sum(hist_2d)
        pmargin1 = np.sum(hist_2d_normalized, axis=1)
        pmargin2 = np.sum(hist_2d_normalized, axis=0)
        
        eps = np.finfo(float).eps
        hist_2d_normalized = np.maximum(hist_2d_normalized, eps)
        pmargin1 = np.maximum(pmargin1, eps)
        pmargin2 = np.maximum(pmargin2, eps)
        
        h1 = -np.sum(pmargin1 * np.log2(pmargin1))
        h2 = -np.sum(pmargin2 * np.log2(pmargin2))
        h12 = -np.sum(hist_2d_normalized * np.log2(hist_2d_normalized))
        
        mi = h1 + h2 - h12
        nmi = 2 * mi / (h1 + h2) if (h1 + h2) > 0 else 0
        
        return nmi

    def register_with_nmi(self, fixed_image, moving_image):
        """Perform affine registration using Normalized Mutual Information metric"""
        print("  - Setting up NMI registration...")
        # Convert to SimpleITK images
        fixed_sitk = sitk.GetImageFromArray(fixed_image.astype(np.float32))
        moving_sitk = sitk.GetImageFromArray(moving_image.astype(np.float32))
        
        # Setup registration method
        registration_method = sitk.ImageRegistrationMethod()
        
        # Use normalized mutual information as the similarity metric
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.01)
        
        # Setup optimizer
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, 
                                                          numberOfIterations=100,
                                                          convergenceMinimumValue=1e-6,
                                                          convergenceWindowSize=10)
        registration_method.SetOptimizerScalesFromPhysicalShift()
        
        # Setup transform
        transform = sitk.AffineTransform(2)
        registration_method.SetInitialTransform(transform)
        
        # Perform registration
        print("  - Executing NMI registration...")
        final_transform = registration_method.Execute(fixed_sitk, moving_sitk)
        
        # Apply transform
        print("  - Applying transform...")
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_sitk)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(final_transform)
        
        # Get the transformed image
        transformed_sitk = resampler.Execute(moving_sitk)
        transformed_image = sitk.GetArrayFromImage(transformed_sitk)
        
        return transformed_image, final_transform

    def register_with_ncc(self, fixed_image, moving_image):
        """Perform affine registration using Normalized Cross-Correlation metric"""
        print("  - Setting up NCC registration...")
        # Convert to SimpleITK images
        fixed_sitk = sitk.GetImageFromArray(fixed_image.astype(np.float32))
        moving_sitk = sitk.GetImageFromArray(moving_image.astype(np.float32))
        
        # Setup registration method
        registration_method = sitk.ImageRegistrationMethod()
        
        # Use normalized cross-correlation as the similarity metric
        registration_method.SetMetricAsCorrelation()
        
        # Setup optimizer
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, 
                                                          numberOfIterations=100,
                                                          convergenceMinimumValue=1e-6,
                                                          convergenceWindowSize=10)
        registration_method.SetOptimizerScalesFromPhysicalShift()
        
        # Setup transform
        transform = sitk.AffineTransform(2)
        registration_method.SetInitialTransform(transform)
        
        # Perform registration
        print("  - Executing NCC registration...")
        final_transform = registration_method.Execute(fixed_sitk, moving_sitk)
        
        # Apply transform
        print("  - Applying transform...")
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_sitk)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(final_transform)
        
        # Get the transformed image
        transformed_sitk = resampler.Execute(moving_sitk)
        transformed_image = sitk.GetArrayFromImage(transformed_sitk)
        
        return transformed_image, final_transform

    def register_with_hog(self, fixed_image, moving_image):
        """Perform registration using Histogram of Oriented Gradients features"""
        print("  - Computing HOG features...")
        # Normalize and convert to uint8 for OpenCV
        fixed_uint8 = (self.normalize_image(fixed_image) * 255).astype(np.uint8)
        moving_uint8 = (self.normalize_image(moving_image) * 255).astype(np.uint8)
        
        # Calculate HOG features and visualizations
        fd_fixed, hog_img_fixed = feature.hog(
            fixed_uint8, orientations=8, pixels_per_cell=(16, 16),
            cells_per_block=(1, 1), visualize=True, feature_vector=True
        )
        
        fd_moving, hog_img_moving = feature.hog(
            moving_uint8, orientations=8, pixels_per_cell=(16, 16),
            cells_per_block=(1, 1), visualize=True, feature_vector=True
        )
        
        # Normalize HOG images for visualization and registration
        hog_img_fixed_norm = self.normalize_image(hog_img_fixed) * 255
        hog_img_moving_norm = self.normalize_image(hog_img_moving) * 255
        
        print("  - Finding transformation using ECC...")
        # Use Enhanced Correlation Coefficient algorithm with HOG images
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
        
        try:
            # Use ECC with the HOG images
            (cc, warp_matrix) = cv2.findTransformECC(
                hog_img_fixed_norm.astype(np.uint8),
                hog_img_moving_norm.astype(np.uint8),
                warp_matrix, 
                cv2.MOTION_AFFINE, 
                criteria
            )
        except cv2.error as e:
            print(f"    Warning: ECC registration failed - {e}")
            print("    Using identity transform instead")
            # Fall back to identity transform if ECC fails
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        # Apply the transformation to the original moving image
        print("  - Applying transform...")
        height, width = fixed_image.shape
        transformed_image = cv2.warpAffine(moving_image, warp_matrix, (width, height))
        
        return transformed_image, warp_matrix

    def identity_transform(self, fixed_image, moving_image):
        """
        Create an identity transform (no registration)
        
        Args:
            fixed_image: The reference image
            moving_image: The image to be registered (will be returned unchanged)
            
        Returns:
            moving_image: The original moving image unchanged
            identity_transform: An identity transform
        """
        print("  - Using identity transform (no registration)...")
        
        # Create an identity transform
        identity_matrix = np.eye(2, 3, dtype=np.float32)
        
        # Return the original moving image and the identity transform
        return moving_image, identity_matrix

    def save_registered_nifti(self, original_nifti, registered_slice, slice_idx, axis, output_path):
        """Save the registered slice as a new NIfTI file"""
        # Create a new volume with the same dimensions as the original
        new_data = np.zeros_like(original_nifti.get_fdata())
        
        # Insert the registered slice at the correct position
        if axis == 0:
            new_data[slice_idx, :, :] = registered_slice
        elif axis == 1:
            new_data[:, slice_idx, :] = registered_slice
        else:  # axis == 2
            new_data[:, :, slice_idx] = registered_slice
        
        # Create a new NIfTI image
        new_nifti = nib.Nifti1Image(new_data, original_nifti.affine, original_nifti.header)
        
        # Save to file
        nib.save(new_nifti, output_path)
        print(f"Saved registered NIfTI to: {output_path}")
        
        return new_nifti

    def save_transform_parameters(self, transform, method_name, output_dir):
        """Save transform parameters to a text file and a SimpleITK transform file"""
        # First, save human-readable parameters to a text file
        output_path = os.path.join(output_dir, f"transform_{method_name.lower()}.txt")
        
        if method_name in ['NMI', 'NCC']:
            # For SimpleITK transforms
            params = transform.GetParameters()
            fixed_params = transform.GetFixedParameters()
            
            with open(output_path, 'w') as f:
                f.write(f"Transform type: {method_name} Affine\n")
                f.write(f"Parameters: {params}\n")
                f.write(f"Fixed parameters: {fixed_params}\n")
            
            # Direct save as SimpleITK transform file
            sitk_path = os.path.join(output_dir, f"transform_{method_name.lower()}.tfm")
            sitk.WriteTransform(transform, sitk_path)
            
        else:
            # For OpenCV transforms (warp matrices) or identity transform
            with open(output_path, 'w') as f:
                f.write(f"Transform type: {method_name} Affine\n")
                f.write(f"Warp matrix:\n")
                f.write(f"{transform}\n")
            
            # Convert OpenCV warp matrix to SimpleITK transform
            # For OpenCV transforms, we need to create a new SimpleITK transform
            sitk_transform = sitk.AffineTransform(2)
            
            # OpenCV warp matrix is a 2×3 matrix in the form:
            # [a11 a12 b1]
            # [a21 a22 b2]
            # SimpleITK parameters are in the order: [a11, a21, a12, a22, b1, b2]
            if isinstance(transform, np.ndarray) and transform.shape == (2, 3):
                # Convert parameters to float64 explicitly to match SimpleITK's expectations
                parameters = [float(transform[0, 0]), float(transform[1, 0]), 
                            float(transform[0, 1]), float(transform[1, 1]), 
                            float(transform[0, 2]), float(transform[1, 2])]
                
                # Set parameters using the proper vector type
                params_vector = sitk.VectorDouble(parameters)
                sitk_transform.SetParameters(params_vector)
                
                # Save as SimpleITK transform file
                sitk_path = os.path.join(output_dir, f"transform_{method_name.lower()}.tfm")
                sitk.WriteTransform(sitk_transform, sitk_path)
                
                # Also save slice information in a pickle file for the transformation application
                slice_info = {
                    "slice_info": {
                        "dimension": self.slice_axis,
                        "slice_index": self.slice_index
                    }
                }
                
                # Save as pickle for compatibility with transform application script
                pickle_path = os.path.join(output_dir, f"transform_{method_name.lower()}.pkl")
                with open(pickle_path, 'wb') as f:
                    transform_dict = {
                        'dimension': 2,
                        'parameters': parameters,
                        'fixed_parameters': [0.0, 0.0],  # Default center of rotation
                        'slice_info': slice_info["slice_info"]
                    }
                    pickle.dump(transform_dict, f)
            else:
                print(f"Warning: Transform for {method_name} is not in the expected 2x3 format")
        
        print(f"Saved {method_name} transform parameters to: {output_path}")
        print(f"Saved {method_name} transform as SimpleITK file: {os.path.join(output_dir, f'transform_{method_name.lower()}.tfm')}")

    def create_comparison_visualization(self, fixed_image, moving_image, registered_images, method_names, output_path=None, show=True):
        """Create a visualization comparing original and registered images"""
        num_methods = len(registered_images)
        fig, axes = plt.subplots(2, num_methods + 1, figsize=(5 * (num_methods + 1), 10))
        
        # Display fixed (reference) image
        axes[0, 0].imshow(fixed_image, cmap='gray')
        axes[0, 0].set_title("Reference MRI Slice", fontsize=14)
        axes[0, 0].axis('off')
        
        # Display moving (original IHC) image
        axes[1, 0].imshow(moving_image, cmap='gray')
        axes[1, 0].set_title("Original IHC Slice", fontsize=14)
        axes[1, 0].axis('off')
        
        # Display each registered image
        for i, (method_name, reg_img) in enumerate(zip(method_names, registered_images)):
            # Display registered image
            axes[0, i+1].imshow(reg_img, cmap='gray')
            axes[0, i+1].set_title(f"Registered IHC ({method_name})", fontsize=14)
            axes[0, i+1].axis('off')
            
            # Create overlay
            overlay = np.zeros((*fixed_image.shape, 3))
            overlay[:,:,0] = self.normalize_image(reg_img)  # Red channel = registered
            overlay[:,:,1] = self.normalize_image(fixed_image)  # Green channel = reference
            
            axes[1, i+1].imshow(overlay)
            axes[1, i+1].set_title(f"Overlay (Red=IHC, Green=MRI)", fontsize=14)
            axes[1, i+1].axis('off')
        
        plt.tight_layout()
        plt.suptitle("Registration Method Comparison", fontsize=16)
        plt.subplots_adjust(top=0.92)
        
        # Save figure if path is provided
        if output_path:
            plt.savefig(output_path, dpi=300)
            print(f"Saved visualization to: {output_path}")
        
        # Show the figure if requested
        if show:
            plt.show(block=False)
            
        return fig

    def show_method_selection_UI(self, fixed_image, moving_image, registered_images, method_names, metrics):
        """Show UI for the user to select the best registration method"""
        # Create a figure with enough space for the images and controls
        fig = plt.figure(figsize=(15, 15))  # Increased vertical space
        
        num_methods = len(method_names)
        
        # Create a grid with more vertical spacing
        gs = plt.GridSpec(4, num_methods, height_ratios=[1.2, 1.2, 1.2, 0.3], hspace=0.4)
        
        # Create axes for all images
        axes_top = []    # For registered images
        axes_middle = [] # For overlays
        axes_bottom = [] # For difference images
        
        # Create all axes
        for i in range(num_methods):
            axes_top.append(fig.add_subplot(gs[0, i]))
            axes_middle.append(fig.add_subplot(gs[1, i]))
            axes_bottom.append(fig.add_subplot(gs[2, i]))
        
        # Display each registered image
        for i, (method_name, reg_img) in enumerate(zip(method_names, registered_images)):
            # Row 1: Registered image
            axes_top[i].imshow(reg_img, cmap='gray')
            axes_top[i].set_title(f"{method_name}", fontsize=14, pad=15)  # Added more padding
            axes_top[i].axis('off')
            
            # Row 2: Overlay 
            overlay = np.zeros((*fixed_image.shape, 3))
            overlay[:,:,0] = self.normalize_image(reg_img)  # Red channel = registered
            overlay[:,:,1] = self.normalize_image(fixed_image)  # Green channel = reference
            
            axes_middle[i].imshow(overlay)
            axes_middle[i].set_title(f"Overlay", fontsize=14, pad=15)  # Added more padding
            axes_middle[i].axis('off')
            
            # Row 3: Difference image to highlight misalignments
            diff = np.abs(self.normalize_image(fixed_image) - self.normalize_image(reg_img))
            axes_bottom[i].imshow(diff, cmap='hot')
            axes_bottom[i].set_title(f"Difference", fontsize=14, pad=15)  # Added more padding
            axes_bottom[i].axis('off')
            
            # Add metric values as text
            if metrics:
                ssim_val = metrics[method_name]['SSIM']
                nmi_val = metrics[method_name]['NMI']
                # Add metrics between registered image and overlay with more space
                text_y_pos = -0.17  # Increased negative value to move text down
                axes_top[i].text(0.5, text_y_pos, 
                               f"SSIM: {ssim_val:.4f}\nNMI: {nmi_val:.4f}", 
                               ha='center', transform=axes_top[i].transAxes, fontsize=12)
        
        # Add radio buttons for selection
        ax_radio = plt.axes([0.3, 0.05, 0.4, 0.05])
        radio = RadioButtons(ax_radio, method_names)
        
        # Add a confirm button
        ax_button = plt.axes([0.75, 0.05, 0.2, 0.05])
        button = Button(ax_button, 'Confirm Selection', color='lightgreen', hovercolor='darkgreen')
        
        # Initialize selection
        self.selected_method_index = 0
        selection_made = [False]  # Use a list to allow modification in nested function
        
        # Create visual indicators of selection - one for each column spanning all three rows
        selection_rects = []
        for i in range(num_methods):
            # Calculate the combined position of all three images in this column
            # Use the actual position data from axes
            pos_top = axes_top[i].get_position()
            pos_middle = axes_middle[i].get_position()
            pos_bottom = axes_bottom[i].get_position()
            
            # Use min/max to get the outer bounds of all three axes in this column
            x_min = min(pos_top.x0, pos_middle.x0, pos_bottom.x0) - 0.005  # Add small margin
            x_max = max(pos_top.x0 + pos_top.width, 
                       pos_middle.x0 + pos_middle.width,
                       pos_bottom.x0 + pos_bottom.width) + 0.005  # Add small margin
            
            y_min = pos_bottom.y0 - 0.01  # Add small margin at bottom
            y_max = pos_top.y0 + pos_top.height + 0.03  # Add larger margin at top for text
            
            # Create rectangle for selection highlighting
            rect = plt.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                fill=False, edgecolor='green', linewidth=0,
                transform=fig.transFigure, zorder=1000
            )
            selection_rects.append(rect)
            fig.add_artist(rect)
        
        def highlight_selection(idx):
            # Reset all rectangles
            for rect in selection_rects:
                rect.set_linewidth(0)
            
            # Highlight the selected one
            selection_rects[idx].set_linewidth(3)
            fig.canvas.draw_idle()
        
        # Initialize
        highlight_selection(0)
        
        # Define callbacks
        def on_method_select(label):
            idx = method_names.index(label)
            self.selected_method_index = idx
            highlight_selection(idx)
            print(f"Selected method: {label}")
        
        def on_confirm(_):
            selection_made[0] = True
            plt.close(fig)
        
        # Add key press handler
        def on_key(event):
            if event.key in [str(i+1) for i in range(len(method_names))]:
                idx = int(event.key) - 1
                if 0 <= idx < len(method_names):
                    radio.set_active(idx)
                    self.selected_method_index = idx
                    highlight_selection(idx)
            elif event.key == 'enter':
                selection_made[0] = True
                plt.close(fig)
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        # Connect callbacks
        radio.on_clicked(on_method_select)
        button.on_clicked(on_confirm)
        
        # Add a descriptive title
        fig.suptitle("Select the Best Registration Method", fontsize=18, y=0.98)
        
        # Add instructions
        fig.text(0.5, 0.01, 
                 "Instructions: Compare registration methods and select the best one.\n" + 
                 "You can use number keys (1-4) to select and Enter to confirm.",
                 ha='center', fontsize=12)
        
        # Show the figure and wait for user interaction
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.show(block=True)
        
        if not selection_made[0]:
            print("No selection made, using the first method.")
            
        return self.selected_method_index

    def evaluate_registration(self, fixed_image, moving_image, registered_images, method_names):
        """Calculate and print registration quality metrics"""
        print("\n-- Registration Quality Evaluation --")
        print(f"{'Method':<10} {'SSIM':<10} {'NMI':<10}")
        print("-" * 30)
        
        # Calculate metrics for original moving image
        ssim_orig = structural_similarity(fixed_image, moving_image, data_range=1.0)
        nmi_orig = self.normalized_mutual_information(fixed_image, moving_image)
        print(f"{'Original':<10} {ssim_orig:.4f} {nmi_orig:.4f}")
        
        # Calculate metrics for each registered image
        metrics_dict = {'Original': {'SSIM': ssim_orig, 'NMI': nmi_orig}}
        
        for method_name, reg_img in zip(method_names, registered_images):
            ssim_val = structural_similarity(fixed_image, reg_img, data_range=1.0)
            nmi_val = self.normalized_mutual_information(fixed_image, reg_img)
            print(f"{method_name:<10} {ssim_val:.4f} {nmi_val:.4f}")
            metrics_dict[method_name] = {'SSIM': ssim_val, 'NMI': nmi_val}
        
        return metrics_dict

    def create_metrics_visualization(self, metrics, output_path):
        """Create a bar chart visualization of registration metrics"""
        methods = list(metrics.keys())
        ssim_values = [metrics[m]['SSIM'] for m in methods]
        nmi_values = [metrics[m]['NMI'] for m in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, ssim_values, width, label='SSIM')
        rects2 = ax.bar(x + width/2, nmi_values, width, label='NMI')
        
        ax.set_xlabel('Registration Method')
        ax.set_ylabel('Metric Value')
        ax.set_title('Registration Quality Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        
        # Add value labels on bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        
        fig.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved metrics visualization to: {output_path}")

    def select_files(self):
        """Select MRI and IHC input files using file dialogs"""
        # Look for IHC slice nifti first in the Match_slice_results folder
        match_slice_dir = None
        
        # Check if directory exists
        current_dir = os.path.abspath(os.getcwd())
        potential_dirs = [
            os.path.join(current_dir, "Match_slice_results"),
            os.path.join(os.path.dirname(current_dir), "Match_slice_results"),
        ]
        
        for d in potential_dirs:
            if os.path.isdir(d):
                match_slice_dir = d
                break
        
        # If directory exists, look for IHC nifti files
        ihc_path = None
        if match_slice_dir:
            ihc_files = glob.glob(os.path.join(match_slice_dir, "*_in_block.nii.gz"))
            if ihc_files:
                ihc_path = ihc_files[0]  # Use the first one found
                print(f"Found IHC file from previous step: {ihc_path}")
        
        # If not found automatically, let user select
        if not ihc_path:
            print("No IHC file found from previous step. Please select manually.")
            
            # Make sure the window is updated
            self.root.update()
            
            ihc_path = filedialog.askopenfilename(
                title="Select IHC NIfTI file",
                filetypes=[
                    ("NIfTI files", "*.nii.gz *.nii"),
                    ("NIfTI files (*.nii.gz)", "*.nii.gz"),
                    ("NIfTI files (*.nii)", "*.nii"),
                    ("All files", "*.*")
                ],
                initialdir=os.getcwd(),
                parent=self.root
            )
        
        if not ihc_path:
            print("No IHC file selected. Exiting.")
            return False
        
        self.ihc_path = ihc_path
        
        # Look for the FLASH MRI reference file in the parent directory
        parent_dir = os.path.dirname(os.path.dirname(ihc_path)) if match_slice_dir else os.path.dirname(ihc_path)
        
        # Look specifically for FLASH files or any MRI nifti file
        flash_mri_files = glob.glob(os.path.join(parent_dir, "*FLASH*.nii.gz"))
        
        # Set initial directory for file dialog
        initial_dir = parent_dir
        
        # Set an initial file if a FLASH file was found
        initial_file = ""
        if flash_mri_files:
            initial_file = flash_mri_files[0]
            print(f"Suggesting MRI file: {os.path.basename(initial_file)}")
        
        # Always ask for MRI file selection
        print("Please select the reference MRI NIfTI file...")
        
        # Make sure the window is updated
        self.root.update()
        
        mri_path = filedialog.askopenfilename(
            title="Select reference MRI NIfTI file",
            initialdir=initial_dir,
            initialfile=os.path.basename(initial_file) if initial_file else "",
            filetypes=[
                ("NIfTI files", "*.nii.gz *.nii"),
                ("NIfTI files (*.nii.gz)", "*.nii.gz"),
                ("NIfTI files (*.nii)", "*.nii"),
                ("All files", "*.*")
            ],
            parent=self.root
        )
        
        if not mri_path:
            print("No MRI file selected. Exiting.")
            return False
        
        self.mri_path = mri_path
        print(f"Using MRI file: {self.mri_path}")
        
        return True

    def run(self):
        """Main function to run the automatic registration"""
        try:
            if not self.select_files():
                return
                
            print("\n===== Automatic Affine Registration =====")
            start_time = time.time()
            
            # Load NIfTI files
            print("Loading NIfTI files...")
            self.mri_nifti = nib.load(self.mri_path)
            self.ihc_nifti = nib.load(self.ihc_path)
            
            self.mri_data = self.mri_nifti.get_fdata()
            self.ihc_data = self.ihc_nifti.get_fdata()
            
            print(f"MRI dimensions: {self.mri_data.shape}")
            print(f"IHC dimensions: {self.ihc_data.shape}")
            
            # Find all non-empty slices in the IHC NIfTI
            print("Finding non-empty slices in IHC volume...")
            non_empty_slices = self.find_non_empty_slices(self.ihc_data)
            
            if not non_empty_slices:
                print("No non-empty slices found in IHC volume. Exiting.")
                return
            
            # Use the slice with the most non-zero pixels
            self.slice_index, self.slice_axis, nonzero_count = non_empty_slices[0]
            print(f"Found non-empty IHC slice at index {self.slice_index} along axis {self.slice_axis} with {nonzero_count} non-zero pixels")
            
            # Extract the IHC slice
            ihc_slice = self.extract_slice(self.ihc_data, self.slice_index, self.slice_axis)
            print(f"IHC slice shape: {ihc_slice.shape}")
            
            # Extract the corresponding MRI slice
            mri_slice = self.extract_slice(self.mri_data, self.slice_index, self.slice_axis)
            print(f"MRI slice shape: {mri_slice.shape}")
            
            # Check if MRI slice has valid data
            if np.count_nonzero(mri_slice) < 100:
                print("Warning: MRI slice appears to be nearly empty. This may affect registration quality.")
                
                # Show selected slices and ask user if they want to continue
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(ihc_slice, cmap='gray')
                plt.title("IHC Slice")
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(mri_slice, cmap='gray')
                plt.title("MRI Slice (appears empty)")
                plt.axis('off')
                
                plt.tight_layout()
                plt.show(block=False)
                
                # Ask if user wants to continue
                continue_response = messagebox.askyesno(
                    "Empty MRI Slice", 
                    "The MRI slice appears to be empty or contain very little data. Continue anyway?"
                )
                
                plt.close()
                
                if not continue_response:
                    print("Registration cancelled by user.")
                    return
            
            # Create output directory
            base_dir = os.path.dirname(self.ihc_path)
            parent_dir = os.path.dirname(base_dir)
            self.output_dir = os.path.join(parent_dir, "Linear_registration_results")
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Check if dimensions match, resize if needed
            if ihc_slice.shape != mri_slice.shape:
                print(f"Warning: IHC slice shape {ihc_slice.shape} does not match MRI slice shape {mri_slice.shape}")
                print("Resizing IHC slice to match MRI slice dimensions")
                ihc_slice = transform.resize(ihc_slice, mri_slice.shape, anti_aliasing=True)
                print(f"Resized IHC slice shape: {ihc_slice.shape}")
            
            # Normalize images for visualization and registration
            mri_slice_norm = self.normalize_image(mri_slice)
            ihc_slice_norm = self.normalize_image(ihc_slice)
            
            # Apply preprocessing to enhance features
            print("Applying preprocessing to enhance features...")
            mri_slice_processed = self.preprocess_image(mri_slice_norm)
            ihc_slice_processed = self.preprocess_image(ihc_slice_norm)
            
            print("\n-- Performing registrations with different metrics --")
            
            # Perform registration using different metrics
            self.registered_images = []
            self.method_names = []
            self.transforms = []
            
            # Wrap each registration method in a try-except to handle potential errors
            # 1. Normalized Mutual Information (NMI)
            try:
                print("\n1. Registering with Normalized Mutual Information (NMI)...")
                nmi_registered, nmi_transform = self.register_with_nmi(mri_slice_processed, ihc_slice_processed)
                self.registered_images.append(nmi_registered)
                self.method_names.append("NMI")
                self.transforms.append(nmi_transform)
            except Exception as e:
                print(f"Error with NMI registration: {e}")
                print("Skipping NMI method")
            
            # 2. Normalized Cross-Correlation (NCC)
            try:
                print("\n2. Registering with Normalized Cross-Correlation (NCC)...")
                ncc_registered, ncc_transform = self.register_with_ncc(mri_slice_processed, ihc_slice_processed)
                self.registered_images.append(ncc_registered)
                self.method_names.append("NCC")
                self.transforms.append(ncc_transform)
            except Exception as e:
                print(f"Error with NCC registration: {e}")
                print("Skipping NCC method")
            
            # 3. Histogram of Oriented Gradients (HOG)
            try:
                print("\n3. Registering with Histogram of Oriented Gradients (HOG)...")
                hog_registered, hog_transform = self.register_with_hog(mri_slice_processed, ihc_slice_processed)
                self.registered_images.append(hog_registered)
                self.method_names.append("HOG")
                self.transforms.append(hog_transform)
            except Exception as e:
                print(f"Error with HOG registration: {e}")
                print("Skipping HOG method")
            
            # 4. No Registration (Identity Transform) - replacing SSIM
            try:
                print("\n4. No Registration (Identity Transform)...")
                identity_registered, identity_transform = self.identity_transform(mri_slice_processed, ihc_slice_processed)
                self.registered_images.append(identity_registered)
                self.method_names.append("None")
                self.transforms.append(identity_transform)
            except Exception as e:
                print(f"Error with identity transform: {e}")
                print("Skipping No Registration option")
            
            # If no methods worked, exit
            if not self.registered_images:
                print("All registration methods failed. Please check your input data.")
                return
            
            # Evaluate registration quality for methods that worked
            print("\nEvaluating registration quality...")
            self.metrics = self.evaluate_registration(mri_slice_norm, ihc_slice_norm, self.registered_images, self.method_names)
            
            # Create metrics visualization
            metrics_viz_path = os.path.join(self.output_dir, "registration_metrics.png")
            self.create_metrics_visualization(self.metrics, metrics_viz_path)
            
            # Create the comparison visualization (but don't display it)
            print("\nCreating comparison visualization...")
            viz_path = os.path.join(self.output_dir, "registration_comparison.png")
            self.create_comparison_visualization(
                mri_slice_norm, ihc_slice_norm, self.registered_images, self.method_names, viz_path, show=False
            )
            
            # Show user interface for method selection
            print("\nPlease select the best registration method...")
            selected_idx = self.show_method_selection_UI(
                mri_slice_norm, ihc_slice_norm, self.registered_images, self.method_names, self.metrics
            )
            
            # Get the selected method
            selected_method = self.method_names[selected_idx]
            selected_image = self.registered_images[selected_idx]
            selected_transform = self.transforms[selected_idx]
            
            print(f"\nSelected method: {selected_method}")
            
            # Save only the selected registration result
            print(f"\nSaving selected {selected_method} registration result...")
            
            # Rescale back to original intensity range if needed
            if selected_image.max() <= 1.0 and ihc_slice.max() > 1.0:
                selected_image_rescaled = selected_image * (np.max(ihc_slice) - np.min(ihc_slice)) + np.min(ihc_slice)
            else:
                selected_image_rescaled = selected_image
            
            # Save as NIfTI
            output_path = os.path.join(self.output_dir, f"ihc_to_mri_affine.nii.gz")
            self.save_registered_nifti(self.ihc_nifti, selected_image_rescaled, self.slice_index, self.slice_axis, output_path)
            
            # Save selected transform parameters
            self.save_transform_parameters(selected_transform, selected_method, self.output_dir)
            
            # Create a visual result of just the selected method - save without displaying
            output_comparison_path = os.path.join(self.output_dir, f"selected_{selected_method}_result.png")
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original IHC
            axes[0].imshow(ihc_slice_norm, cmap='gray')
            axes[0].set_title("Original IHC", fontsize=14)
            axes[0].axis('off')
            
            # Registered IHC
            axes[1].imshow(selected_image, cmap='gray')
            axes[1].set_title(f"Registered IHC ({selected_method})", fontsize=14)
            axes[1].axis('off')
            
            # Reference MRI
            axes[2].imshow(mri_slice_norm, cmap='gray')
            axes[2].set_title("Reference MRI", fontsize=14)
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_comparison_path, dpi=300)
            plt.close(fig)
            
            # Final summary
            elapsed_time = time.time() - start_time
            print("\n===== REGISTRATION COMPLETE =====")
            print(f"Selected method: {selected_method}")
            print(f"Output saved to: {output_path}")
            print(f"Transform saved to: {os.path.join(self.output_dir, f'transform_{selected_method.lower()}.tfm')}")
            print(f"Total processing time: {elapsed_time:.2f} seconds")
            print("\nProceed to non-linear registration step")
            
            return output_path
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            print(f"Error: {str(e)}")
            return None

if __name__ == "__main__":
    registration = AutoRegistration()
    registration.run()
