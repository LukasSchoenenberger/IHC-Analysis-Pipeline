import os
import numpy as np
import nibabel as nib
from skimage import io, transform, exposure, filters, feature
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
# Use scipy for normalized cross-correlation since some skimage versions don't have it
from scipy.signal import correlate
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from scipy.ndimage import gaussian_filter
import time
import re
import json
import cv2

class HistologyMatcher:
    def __init__(self):
        # Initialize file paths
        self.mri_path = None
        self.histology_path = None
        
        # Initialize data
        self.mri_nifti = None
        self.mri_data = None
        self.histology_img = None
        self.histology_original = None
        
        # Results of slice matching
        self.optimal_plane = None
        self.best_slice_idx = None
        self.oriented_histology = None
        self.best_orientation_desc = None
        
        # For manual orientation selection
        self.manual_selection_made = False
        self.selected_orientation_idx = None
        
        # For slice selection options
        self.selected_slice_option = None
        self.selected_metric = None
        
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
        """Apply all preprocessing steps to an image"""
        if img.dtype != np.float32 and img.dtype != np.float64:
            img = img.astype(np.float32)
        img_norm = self.normalize_image(img)
        img_eq = exposure.equalize_hist(img_norm)
        img_enhanced = self.enhance_edges(img_eq)
        return img_enhanced

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
    
    def structural_similarity_metric(self, img1, img2):
        """Calculate structural similarity between two images"""
        if img1.shape != img2.shape:
            img2 = transform.resize(img2, img1.shape, anti_aliasing=True)
        
        # ssim returns the mean SSIM as first output, ignore the full SSIM map
        similarity, _ = ssim(img1, img2, data_range=1.0, full=True)
        return similarity
    
    def normalized_cross_correlation_metric(self, img1, img2):
        """Calculate normalized cross-correlation between two images"""
        if img1.shape != img2.shape:
            img2 = transform.resize(img2, img1.shape, anti_aliasing=True)
        
        # Implement normalized cross-correlation manually
        # Subtract mean
        img1_norm = img1 - np.mean(img1)
        img2_norm = img2 - np.mean(img2)
        
        # Calculate correlation
        numerator = np.sum(img1_norm * img2_norm)
        denominator = np.sqrt(np.sum(img1_norm**2) * np.sum(img2_norm**2))
        
        # Avoid division by zero
        if denominator == 0:
            return 0
            
        ncc_value = numerator / denominator
        return ncc_value
    
    def feature_similarity(self, img1, img2):
        """Calculate similarity based on feature matching using ORB (always available in OpenCV)"""
        # Ensure images are in the right format for OpenCV
        img1_cv = (img1 * 255).astype(np.uint8)
        img2_cv = (img2 * 255).astype(np.uint8)
        
        # Resize if needed
        if img1_cv.shape != img2_cv.shape:
            img2_cv = cv2.resize(img2_cv, (img1_cv.shape[1], img1_cv.shape[0]))
        
        try:
            # Use ORB which is always available
            orb = cv2.ORB_create()
            
            # Detect keypoints and compute descriptors
            kp1, des1 = orb.detectAndCompute(img1_cv, None)
            kp2, des2 = orb.detectAndCompute(img2_cv, None)
            
            # If no keypoints were found, return zero similarity
            if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
                return 0.0
                
            # Create BFMatcher with Hamming distance (for binary descriptors like ORB)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
            # Match descriptors
            matches = bf.match(des1, des2)
            
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Calculate similarity based on number of good matches
            # and their average distance (lower distance = better match)
            if len(matches) > 0:
                # Take top 10% of matches or at least 10 matches if available
                num_good_matches = max(min(int(len(matches) * 0.1), len(matches)), min(10, len(matches)))
                good_matches = matches[:num_good_matches]
                
                # Average distance of good matches (lower is better)
                avg_distance = sum(m.distance for m in good_matches) / len(good_matches)
                
                # Normalize to a similarity score (0-1, higher is better)
                max_distance = 100.0  # Arbitrary reference value
                similarity = 1.0 - min(avg_distance / max_distance, 1.0)
                
                # Adjust by number of matches relative to keypoints
                match_ratio = len(good_matches) / min(len(kp1), len(kp2)) if min(len(kp1), len(kp2)) > 0 else 0
                
                # Combine both factors
                similarity = 0.5 * similarity + 0.5 * match_ratio
            else:
                similarity = 0.0
            
            return similarity
        except Exception as e:
            print(f"Error in feature matching: {e}")
            return 0.0

    def hog_similarity(self, img1, img2, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        """Calculate similarity based on Histogram of Oriented Gradients features"""
        if img1.shape != img2.shape:
            raise ValueError("Images must have the same shape for HOG similarity calculation")
        
        hog1 = feature.hog(img1, orientations=orientations, pixels_per_cell=pixels_per_cell,
                          cells_per_block=cells_per_block, block_norm='L2-Hys',
                          visualize=False, feature_vector=True)
        
        hog2 = feature.hog(img2, orientations=orientations, pixels_per_cell=pixels_per_cell,
                          cells_per_block=cells_per_block, block_norm='L2-Hys',
                          visualize=False, feature_vector=True)
        
        hog1_norm = hog1 / np.linalg.norm(hog1) if np.linalg.norm(hog1) > 0 else hog1
        hog2_norm = hog2 / np.linalg.norm(hog2) if np.linalg.norm(hog2) > 0 else hog2
        
        similarity = np.dot(hog1_norm, hog2_norm)
        
        return similarity

    def calculate_all_metrics(self, mri_slice, histology_img):
        """
        Calculate all similarity metrics between a MRI slice and histology image
        
        Args:
            mri_slice: The MRI slice
            histology_img: The histology image
            
        Returns:
            dict: Dictionary containing all similarity metrics
        """
        # Preprocess images
        processed_mri = self.preprocess_image(mri_slice)
        
        # Ensure histology has same shape as MRI slice
        if histology_img.shape != processed_mri.shape:
            histology_img = transform.resize(histology_img, processed_mri.shape, anti_aliasing=True)
        
        # Calculate all metrics
        metrics = {
            'NMI': self.normalized_mutual_information(histology_img, processed_mri),
            'NCC': self.normalized_cross_correlation_metric(histology_img, processed_mri),
            'SSIM': self.structural_similarity_metric(histology_img, processed_mri),
            'Feature': self.feature_similarity(histology_img, processed_mri)
        }
        
        return metrics

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
        orientations.append((flipped_hv, "Flipped horizontally and vertically"))
        
        # Rotated 90° clockwise
        rotated_90 = np.rot90(image, k=-1)  # k=-1 for clockwise
        orientations.append((rotated_90, "Rotated 90° clockwise"))
        
        # Rotated 90° clockwise + flipped horizontally
        rotated_90_flipped_h = np.fliplr(rotated_90)
        orientations.append((rotated_90_flipped_h, "Rotated 90° clockwise + flipped horizontally"))
        
        # Rotated 90° clockwise + flipped vertically
        rotated_90_flipped_v = np.flipud(rotated_90)
        orientations.append((rotated_90_flipped_v, "Rotated 90° clockwise + flipped vertically"))
        
        # Rotated 90° clockwise + flipped both ways
        rotated_90_flipped_hv = np.flipud(np.fliplr(rotated_90))
        orientations.append((rotated_90_flipped_hv, "Rotated 90° clockwise + flipped both ways"))
        
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

    def find_best_slice_using_metric(self, mri_data, histology, plane, metric='nmi'):
        """Find the best matching MRI slice for the histology image using the specified metric"""
        if plane == 0:  # Sagittal
            dim_size = mri_data.shape[0]
        elif plane == 1:  # Coronal
            dim_size = mri_data.shape[1]
        else:  # Axial
            dim_size = mri_data.shape[2]
        
        print(f"Finding best slice using {metric.upper()} in the {['Sagittal', 'Coronal', 'Axial'][plane]} plane...")
        
        # Select the appropriate similarity function based on metric
        if metric.lower() == 'nmi':
            similarity_func = self.normalized_mutual_information
        elif metric.lower() == 'ncc':
            similarity_func = self.normalized_cross_correlation_metric
        elif metric.lower() == 'ssim':
            similarity_func = self.structural_similarity_metric
        elif metric.lower() == 'feature' or metric.lower() == 'sift':
            similarity_func = self.feature_similarity
        else:
            print(f"Unknown metric: {metric}. Using NMI as default.")
            similarity_func = self.normalized_mutual_information
        
        best_slice_idx = -1
        best_score = -float('inf')
        best_slice_data = None
        
        for slice_idx in range(dim_size):
            if plane == 0:  # Sagittal
                mri_slice = mri_data[slice_idx, :, :]
            elif plane == 1:  # Coronal
                mri_slice = mri_data[:, slice_idx, :]
            else:  # Axial
                mri_slice = mri_data[:, :, slice_idx]
            
            if np.count_nonzero(mri_slice) < 100:
                continue
            
            processed_mri = self.preprocess_image(mri_slice)
            
            try:
                score = similarity_func(histology, processed_mri)
                
                if score > best_score:
                    best_score = score
                    best_slice_idx = slice_idx
                    best_slice_data = mri_slice.copy()
                
                if slice_idx % 20 == 0 or slice_idx == dim_size - 1:
                    print(f"Processed slice {slice_idx+1}/{dim_size} - Current best: Slice {best_slice_idx} ({best_score:.4f})")
                    
            except Exception as e:
                print(f"Error processing slice {slice_idx}: {e}")
        
        print(f"Best slice using {metric.upper()}: Slice {best_slice_idx} with score: {best_score:.4f}")
        
        return best_slice_idx, best_score, best_slice_data

    # For backward compatibility
    def find_best_slice_using_nmi(self, mri_data, histology, plane):
        """Find the best matching MRI slice for the histology image using NMI"""
        return self.find_best_slice_using_metric(mri_data, histology, plane, metric='nmi')

    def find_best_orientation_using_hog(self, histology_img, reference_slice):
        """Find the optimal orientation of the histology image using HOG similarity"""
        print("\nFinding optimal orientation using HOG similarity...")
        
        orientations = self.generate_all_orientations(histology_img)
        self.all_orientations = orientations  # Save for manual selection later
        
        best_idx = -1
        best_score = -float('inf')
        best_oriented_img = None
        best_description = ""
        
        for i, (oriented_img, description) in enumerate(orientations):
            if oriented_img.shape != reference_slice.shape:
                oriented_img = transform.resize(oriented_img, reference_slice.shape, anti_aliasing=True)
            
            try:
                score = self.hog_similarity(oriented_img, reference_slice)
                print(f"Orientation {i+1}: {description} - HOG similarity: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_idx = i
                    best_oriented_img = oriented_img
                    best_description = description
            except Exception as e:
                print(f"Error calculating HOG similarity for orientation {i+1}: {e}")
        
        print(f"\nBest orientation using HOG: {best_description} with score: {best_score:.4f}")
        
        return best_oriented_img, best_description, best_score, best_idx

    def extract_slice(self, mri_data, plane, slice_idx):
        """Extract a specific slice from the MRI volume"""
        if plane == 0:  # Sagittal
            return mri_data[slice_idx, :, :]
        elif plane == 1:  # Coronal
            return mri_data[:, slice_idx, :]
        else:  # Axial
            return mri_data[:, :, slice_idx]

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

    def show_orientation_selection_UI(self, histology_resized, first_slice_data):
        """Show UI to select the best orientation manually"""
        from matplotlib.widgets import RadioButtons, Button
        
        orientations = self.all_orientations
        
        # Pre-resize all orientations to match MRI slice dimensions
        resized_orientations = []
        for oriented_img, desc in orientations:
            if oriented_img.shape != first_slice_data.shape:
                resized_img = transform.resize(oriented_img, first_slice_data.shape, anti_aliasing=True)
            else:
                resized_img = oriented_img
            resized_orientations.append((resized_img, desc))
        
        # Create figure with more space at the bottom for controls
        fig = plt.figure(figsize=(15, 15))  # Increased figure height
        
        # Set the grid spec with more space between rows
        gs = fig.add_gridspec(4, 3, height_ratios=[1.2, 1.2, 1.2, 0.3], hspace=0.4)
        
        # Create axes for images
        axs = []
        for row in range(3):
            for col in range(3):
                axs.append(fig.add_subplot(gs[row, col]))
        
        # Display MRI slice in the center
        axs[4].imshow(first_slice_data, cmap='gray')
        axs[4].set_title("MRI Slice (Reference)", fontsize=14, pad=15)  # Increased padding
        axs[4].axis('off')
        
        # Display all orientations around it
        for i, (oriented_img, desc) in enumerate(resized_orientations):
            idx = i if i < 4 else i + 1  # Skip the center cell (4)
            axs[idx].imshow(oriented_img, cmap='gray')
            axs[idx].set_title(f"{i+1}: {desc}", fontsize=14, pad=15)  # Increased font size and padding
            axs[idx].axis('off')
        
        # Create space for larger, more readable selection widgets at bottom
        ax_selections = fig.add_subplot(gs[3, :])
        ax_selections.axis('off')
        
        # Add colored rectangles as frames around each image for highlighting selection
        # Get the exact extents of the image areas
        selection_frames = []
        for i in range(8):
            idx = i if i < 4 else i + 1  # Skip the center cell (4)
            
            # Get the position from the axes but adjust for better fit
            pos = axs[idx].get_position()
            
            # Create a rectangle that precisely matches the image
            rect = plt.Rectangle(
                (pos.x0, pos.y0),  # Bottom left corner
                pos.width,        # Width
                pos.height,       # Height
                fill=False, 
                edgecolor='green', 
                linewidth=0,
                transform=fig.transFigure,
                zorder=1000  # Ensure rectangle is on top
            )
            selection_frames.append(rect)
            fig.add_artist(rect)
        
        # Create radio buttons with better spacing
        radios_ax = fig.add_axes([0.1, 0.05, 0.6, 0.1])
        radios_ax.axis('off')
        
        # Create custom labels that include orientation descriptions
        radio_labels = [f"Option {i+1}: {desc}" for i, (_, desc) in enumerate(orientations)]
        radio = RadioButtons(
            radios_ax, 
            radio_labels,
            activecolor='green'
        )
        
        # Set label font size (this works reliably across matplotlib versions)
        for label in radio.labels:
            label.set_fontsize(12)
        
        # Manually adjust the vertical positions of labels for better spacing
        for i, label in enumerate(radio.labels):
            # Get the current y position
            pos = label.get_position()
            # Set new position with increased vertical spacing
            # Distribute over the vertical space with larger gaps
            new_y = 0.9 - (i * 0.15)  # Increased from typical 0.1 spacing
            label.set_position((pos[0], new_y))
        
        # Create a more prominent confirm button
        button_ax = fig.add_axes([0.75, 0.05, 0.2, 0.05])
        button = Button(button_ax, 'Confirm Selection', color='lightgreen', hovercolor='darkgreen')
        
        # Store selection variables
        self.selected_orientation_idx = 0  # Default to first orientation
        self.manual_selection_made = False
        
        # Highlight the selected option with a colored border
        def highlight_selection(idx):
            # Reset all frames
            for frame in selection_frames:
                frame.set_linewidth(0)
            
            # Highlight the selected one
            selection_frames[idx].set_linewidth(4)
            fig.canvas.draw_idle()
        
        # Call once to highlight the initial selection
        highlight_selection(0)
        
        # Define callbacks
        def on_orientation_select(label):
            # Extract the option number from the label
            idx = int(label.split()[1].replace(':', '')) - 1
            self.selected_orientation_idx = idx
            highlight_selection(idx)
            print(f"Selected orientation: {orientations[idx][1]} (index {idx})")
        
        def on_confirm(_):
            self.manual_selection_made = True
            plt.close(fig)
        
        # Connect callbacks
        radio.on_clicked(on_orientation_select)
        button.on_clicked(on_confirm)
        
        # Add key press handler to allow using number keys 1-8 for selection
        def on_key(event):
            if event.key in ['1', '2', '3', '4', '5', '6', '7', '8']:
                idx = int(event.key) - 1
                radio.set_active(idx)
                self.selected_orientation_idx = idx
                highlight_selection(idx)
            elif event.key == 'enter':
                self.manual_selection_made = True
                plt.close(fig)
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        plt.suptitle("Select the Best Orientation for Histology Image", fontsize=18, y=0.98)  # Adjusted position
        plt.tight_layout(rect=[0, 0.15, 1, 0.95])  # Adjust layout but leave room for controls
        plt.show(block=True)  # This will block until the figure is closed
        
        if not self.manual_selection_made:
            print("No manual selection made, using automatic selection.")
            return None
            
        # Return the selected orientation
        selected_image, selected_desc = orientations[self.selected_orientation_idx]
        
        # Resize to match reference slice if needed
        if selected_image.shape != first_slice_data.shape:
            selected_image = transform.resize(selected_image, first_slice_data.shape, anti_aliasing=True)
            
        print(f"Manual selection: {selected_desc}")
        return selected_image, selected_desc, self.selected_orientation_idx

    def present_slice_options(self, histology_img, current_slice_idx, current_slice_data, current_metric="NMI"):
        """
        Present options to the user after finding the best slice:
        1. Confirm current best slice
        2. Try another similarity metric
        3. Manually select a slice
        
        Displays all four similarity metrics (NMI, NCC, SSIM, Feature) for the current slice
        to help with decision making.
        
        Returns:
        option (int): 1 for confirm, 2 for another metric, 3 for manual selection
        new_metric (str or None): If option 2 is selected, the new metric to use
        """
        # Create a simple dialog window
        option_window = tk.Toplevel()
        option_window.title("Slice Selection Options")
        option_window.geometry("800x600")
        option_window.lift()
        option_window.focus_force()
        
        # Center on screen
        screen_width = option_window.winfo_screenwidth()
        screen_height = option_window.winfo_screenheight()
        window_width = 800
        window_height = 600
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        option_window.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Create frame for the images
        images_frame = tk.Frame(option_window)
        images_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Create frames for the histology and MRI images
        histology_frame = tk.Frame(images_frame)
        histology_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
        
        mri_frame = tk.Frame(images_frame)
        mri_frame.pack(side=tk.RIGHT, padx=10, fill=tk.BOTH, expand=True)
        
        # Display images using matplotlib
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        
        # Histology image
        fig_histology = plt.Figure(figsize=(4, 4), tight_layout=True)
        ax_histology = fig_histology.add_subplot(111)
        ax_histology.imshow(histology_img, cmap='gray')
        ax_histology.set_title("Histology Image")
        ax_histology.axis('off')
        
        canvas_histology = FigureCanvasTkAgg(fig_histology, master=histology_frame)
        canvas_histology.draw()
        canvas_histology.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # MRI slice
        fig_mri = plt.Figure(figsize=(4, 4), tight_layout=True)
        ax_mri = fig_mri.add_subplot(111)
        ax_mri.imshow(current_slice_data, cmap='gray')
        ax_mri.set_title(f"Best MRI Slice (using {current_metric})\nSlice {current_slice_idx}")
        ax_mri.axis('off')
        
        canvas_mri = FigureCanvasTkAgg(fig_mri, master=mri_frame)
        canvas_mri.draw()
        canvas_mri.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Options frame
        options_frame = tk.Frame(option_window)
        options_frame.pack(pady=20, fill=tk.X)
        
        # Calculate all similarity metrics for current slice
        all_metrics = self.calculate_all_metrics(current_slice_data, histology_img)
        
        # Information label
        info_label = tk.Label(
            options_frame, 
            text=f"Current best slice: {current_slice_idx} (using {current_metric})",
            font=("Arial", 12)
        )
        info_label.pack(pady=5)
        
        # Display all metrics for better comparison
        metrics_frame = tk.Frame(options_frame, relief=tk.RIDGE, borderwidth=1)
        metrics_frame.pack(pady=5, fill=tk.X, padx=20)
        
        metrics_title = tk.Label(
            metrics_frame,
            text="All similarity metrics for this slice:",
            font=("Arial", 11, "bold"),
            anchor="w"
        )
        metrics_title.pack(fill=tk.X, padx=10, pady=5)
        
        # Create a grid of metric values
        for i, (metric_name, score) in enumerate(all_metrics.items()):
            metric_frame = tk.Frame(metrics_frame)
            metric_frame.pack(fill=tk.X, padx=10)
            
            # Highlight the current metric being used
            is_current = metric_name.upper() == current_metric.upper()
            font_style = ("Arial", 10, "bold") if is_current else ("Arial", 10)
            bg_color = "#e6ffe6" if is_current else "#f0f0f0"
            
            label = tk.Label(
                metric_frame,
                text=f"{metric_name}:",
                font=font_style,
                width=8,
                anchor="w",
                bg=bg_color
            )
            label.pack(side=tk.LEFT, padx=(10, 0), pady=2)
            
            value = tk.Label(
                metric_frame,
                text=f"{score:.4f}",
                font=font_style,
                bg=bg_color
            )
            value.pack(side=tk.LEFT, padx=5, pady=2)
            
            # Add a note if this is the current metric
            if is_current:
                note = tk.Label(
                    metric_frame,
                    text="(current metric)",
                    font=("Arial", 10, "italic"),
                    fg="green",
                    bg=bg_color
                )
                note.pack(side=tk.LEFT, padx=5, pady=2)
        
        # Question label
        question_label = tk.Label(
            options_frame, 
            text="What would you like to do?",
            font=("Arial", 12, "bold")
        )
        question_label.pack(pady=10)
        
        # Option variables
        self.selected_slice_option = tk.IntVar(value=1)  # Default to confirm
        self.selected_metric = tk.StringVar(value="NCC")  # Default alternative metric
        
        # Buttons frame
        buttons_frame = tk.Frame(options_frame)
        buttons_frame.pack(pady=10)
        
        # Option 1: Confirm
        confirm_btn = tk.Button(
            buttons_frame,
            text="1. Confirm - Use this slice",
            font=("Arial", 11),
            width=30,
            height=2,
            command=lambda: self.set_option_and_close(option_window, 1)
        )
        confirm_btn.pack(pady=5)
        
        # Option 2: Try another metric
        # Create a frame for the metric options
        metric_frame = tk.Frame(buttons_frame)
        metric_frame.pack(pady=5)
        
        metric_btn = tk.Button(
            metric_frame,
            text="2. Try another similarity metric:",
            font=("Arial", 11),
            width=30,
            height=2,
            command=lambda: self.set_option_and_close(option_window, 2)
        )
        metric_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Create radio buttons for metrics
        metrics = ["NCC", "SSIM", "Feature"]
        if current_metric.upper() == "NMI":
            # If current metric is NMI, allow any other metric
            pass
        else:
            # If current metric is not NMI, add NMI as an option
            metrics = ["NMI"] + [m for m in metrics if m.upper() != current_metric.upper()]
        
        # Remove the current metric from options
        metrics = [m for m in metrics if m.upper() != current_metric.upper() and m.upper() != "SIFT"]
        
        # Create a frame for radio buttons
        radio_frame = tk.Frame(metric_frame)
        radio_frame.pack(side=tk.LEFT)
        
        for i, metric in enumerate(metrics):
            rb = tk.Radiobutton(
                radio_frame,
                text=metric,
                variable=self.selected_metric,
                value=metric
            )
            rb.pack(anchor=tk.W)
            if i == 0:
                rb.select()  # Select the first option by default
        
        # Option 3: Manual selection
        manual_btn = tk.Button(
            buttons_frame,
            text="3. Manually select a slice",
            font=("Arial", 11),
            width=30,
            height=2,
            command=lambda: self.set_option_and_close(option_window, 3)
        )
        manual_btn.pack(pady=5)
        
        # Wait for the window to be closed
        option_window.protocol("WM_DELETE_WINDOW", lambda: self.set_option_and_close(option_window, 1))
        option_window.wait_window()
        
        # Return the selected option and metric
        selected_option = self.selected_slice_option.get()
        selected_metric = self.selected_metric.get() if selected_option == 2 else None
        
        return selected_option, selected_metric
    
    def set_option_and_close(self, window, option):
        """Set the selected option and close the window"""
        self.selected_slice_option.set(option)
        window.destroy()
    
    def manually_select_slice(self, mri_data, plane, histology_img):
        """
        Allow the user to manually select the best slice using a slider UI
        
        For each slice viewed, all similarity metrics are calculated and displayed
        to help the user make an informed selection.
        
        Returns:
        int: The manually selected slice index
        """
        # Determine dimensions for the given plane
        if plane == 0:  # Sagittal
            dim_size = mri_data.shape[0]
        elif plane == 1:  # Coronal
            dim_size = mri_data.shape[1]
        else:  # Axial
            dim_size = mri_data.shape[2]
        
        # Create a figure with enough space for the slider
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Display histology image on the left
        ax1.imshow(histology_img, cmap='gray')
        ax1.set_title("Histology Image")
        ax1.axis('off')
        
        # Initial slice (middle of the volume)
        initial_slice_idx = dim_size // 2
        current_slice = self.extract_slice(mri_data, plane, initial_slice_idx)
        
        # Display initial MRI slice on the right
        mri_img = ax2.imshow(current_slice, cmap='gray')
        ax2.set_title(f"MRI Slice {initial_slice_idx}")
        ax2.axis('off')
        
        # Add a slider for slice navigation
        ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
        slider = Slider(
            ax_slider, 
            'Slice', 
            0, 
            dim_size - 1, 
            valinit=initial_slice_idx, 
            valstep=1
        )
        
        # Create a text box for showing metrics
        metrics_ax = plt.axes([0.55, 0.85, 0.4, 0.1])
        metrics_ax.axis('off')
        metrics_text = metrics_ax.text(0, 0, "", fontsize=9, verticalalignment='top', 
                                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Function to update the slice display and calculate metrics
        def update(val):
            slice_idx = int(slider.val)
            slice_data = self.extract_slice(mri_data, plane, slice_idx)
            mri_img.set_data(slice_data)
            
            # Calculate and display metrics for this slice
            metrics = self.calculate_all_metrics(slice_data, histology_img)
            metrics_str = "Similarity Metrics:\n"
            for metric_name, score in metrics.items():
                metrics_str += f"{metric_name}: {score:.4f}\n"
            metrics_text.set_text(metrics_str)
            
            ax2.set_title(f"MRI Slice {slice_idx}")
            fig.canvas.draw_idle()
        
        slider.on_changed(update)
        
        # Add a button to confirm selection
        ax_button = plt.axes([0.8, 0.05, 0.1, 0.03])
        confirm_button = Button(ax_button, 'Confirm', color='lightgreen', hovercolor='darkgreen')
        
        # Add input box for directly entering slice number
        direct_input_ax = plt.axes([0.12, 0.05, 0.08, 0.03])
        direct_input_button = Button(direct_input_ax, 'Enter #', color='lightblue', hovercolor='blue')
        
        # Selected slice (initialize with slider value)
        selected_slice = [int(slider.val)]
        
        # Callback for confirm button
        def confirm(event):
            selected_slice[0] = int(slider.val)
            plt.close(fig)
        
        confirm_button.on_clicked(confirm)
        
        # Callback for direct input button
        def enter_slice_number(event):
            try:
                slice_num = simpledialog.askinteger(
                    "Enter Slice Number", 
                    f"Enter slice number (0-{dim_size-1}):",
                    minvalue=0, 
                    maxvalue=dim_size-1
                )
                
                if slice_num is not None:
                    slider.set_val(slice_num)
                    selected_slice[0] = slice_num
            except Exception as e:
                print(f"Error entering slice number: {e}")
        
        direct_input_button.on_clicked(enter_slice_number)
        
        # Add keyboard shortcuts (arrow keys for navigation, Enter to confirm)
        def key_press(event):
            if event.key == 'right':
                current_val = int(slider.val)
                if current_val < dim_size - 1:
                    slider.set_val(current_val + 1)
            elif event.key == 'left':
                current_val = int(slider.val)
                if current_val > 0:
                    slider.set_val(current_val - 1)
            elif event.key == 'enter':
                selected_slice[0] = int(slider.val)
                plt.close(fig)
        
        fig.canvas.mpl_connect('key_press_event', key_press)
        
        plt.suptitle("Manual Slice Selection", fontsize=16)
        plt.subplots_adjust(bottom=0.15)  # Make room for the slider
        plt.show()
        
        return selected_slice[0]

    def create_visualization(self, histology_original, histology_oriented, orientation_desc, 
                            first_slice, first_slice_idx, final_slice, final_slice_idx,
                            output_path, metric_name="NMI", all_metrics=None):
        """Create a visualization showing all steps of the process with all similarity metrics"""
        # Create figure with increased size and proper spacing
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Add more spacing between subplots
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        
        # Original histology
        axes[0, 0].imshow(histology_original, cmap='gray')
        axes[0, 0].set_title("1. Original Histology", fontsize=12, pad=10)
        axes[0, 0].axis('off')
        
        # First best slice using the selected metric
        axes[0, 1].imshow(first_slice, cmap='gray')
        axes[0, 1].set_title(f"2. Initial Best Slice ({metric_name})\nSlice {first_slice_idx}", fontsize=12, pad=10)
        axes[0, 1].axis('off')
        
        # Reoriented histology
        axes[1, 0].imshow(histology_oriented, cmap='gray')
        axes[1, 0].set_title(f"3. Reoriented Histology\n{orientation_desc}", fontsize=12, pad=10)
        axes[1, 0].axis('off')
        
        # Final best slice with reoriented histology
        axes[1, 1].imshow(final_slice, cmap='gray')
        
        # Use a more compact format for metrics to avoid title overlap
        if all_metrics:
            # Calculate the maximum length of metrics text to avoid overlap
            metrics_text = []
            for metric, score in all_metrics.items():
                metrics_text.append(f"{metric}: {score:.4f}")
            
            # Format metrics on a separate line with smaller font
            metrics_line = "  ".join(metrics_text)
            
            # Create a title with the slice info on one line, metrics on another
            title = f"4. Final Best Slice ({metric_name})\nSlice {final_slice_idx}\n{metrics_line}"
        else:
            title = f"4. Final Best Slice ({metric_name})\nSlice {final_slice_idx}"
        
        axes[1, 1].set_title(title, fontsize=12, pad=15)
        axes[1, 1].axis('off')
        
        # Ensure proper layout for display
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save with tight bounding box to ensure proper cropping
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        # Return the figure object instead of closing it
        return fig
    
    def block_name_from_path(self, path):
        """Extract block name from path using regex patterns"""
        # Try to extract a block pattern like "Block_XX_YY_Z" or similar
        patterns = [
            r'Block[_\s]*(\d+[_\s]*\d+[_\s]*\d+)',  # Block_XX_YY_Z
            r'Block[_\s]*(\d+[_\s]*\d+)',           # Block_XX_YY
            r'Block[_\s]*(\d+)',                    # Block_XX
            r'([bB]\d+)',                           # B12 or b12
        ]
        
        for pattern in patterns:
            match = re.search(pattern, path)
            if match:
                return match.group(1).replace(' ', '_')
        
        # If no block pattern is found, use the directory name
        dir_name = os.path.basename(os.path.dirname(path))
        if dir_name and dir_name != "." and dir_name != "":
            return dir_name
            
        # Last resort: return "block"
        return "block"

    def select_files(self):
        """Select MRI and histology input files using file dialogs"""
        # Get the MRI NIfTI file - make sure to show files in list mode
        root = self.root
        if root is None or not root.winfo_exists():
            # Create a new root if needed
            root = tk.Tk()
            root.withdraw()
        
        # Use a more direct approach with file dialog
        self.root.update()  # Make sure the window is updated
        
        self.mri_path = filedialog.askopenfilename(
            title="Select MRI NIfTI file",
            filetypes=[
                ("NIfTI files", "*.nii *.nii.gz"),
                ("NIfTI files (*.nii)", "*.nii"),
                ("NIfTI files (*.nii.gz)", "*.nii.gz"),
                ("All files", "*.*")
            ],
            initialdir=os.getcwd(),  # Start in the current working directory
            parent=self.root
        )
        
        if not self.mri_path:
            print("No MRI file selected. Exiting.")
            return False
        
        # Get the histology image file
        # Use the directory of the MRI file as the initial directory for histology selection
        initial_dir = os.path.dirname(self.mri_path) if self.mri_path else os.getcwd()
        
        self.histology_path = filedialog.askopenfilename(
            title="Select Histology image file",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.tif *.tiff *.png"), 
                ("JPEG files", "*.jpg *.jpeg"),
                ("TIFF files", "*.tif *.tiff"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ],
            initialdir=initial_dir,
            parent=self.root
        )
        
        if not self.histology_path:
            print("No histology file selected. Exiting.")
            return False
        
        return True

    def save_matching_info(self, first_slice_idx, first_slice_score, 
                          final_slice_idx, final_slice_score, 
                          orientation_desc, output_path, metric_name="NMI", all_metrics=None):
        """
        Save the slice matching and orientation information to a JSON file
        
        Args:
            first_slice_idx: Index of the initial best slice
            first_slice_score: Score of the initial best slice
            final_slice_idx: Index of the final best slice after reorientation
            final_slice_score: Score of the final best slice
            orientation_desc: Description of the best orientation
            output_path: Path to save the info file
            metric_name: The similarity metric used for matching
            all_metrics: Dictionary containing all similarity metrics for the final slice
        """
        # Create a dictionary with all the relevant information
        info = {
            "mri_file": self.mri_path,
            "histology_file": self.histology_path,
            "anatomical_plane": ["Sagittal", "Coronal", "Axial"][int(self.optimal_plane)],
            "plane_index": int(self.optimal_plane),
            "similarity_metric": metric_name,
            "initial_best_slice": {
                "index": int(first_slice_idx),
                "score": float(first_slice_score)
            },
            "orientation": {
                "description": orientation_desc,
                "manual_selection": bool(self.manual_selection_made)
            },
            "final_best_slice": {
                "index": int(final_slice_idx),
                "score": float(final_slice_score)
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add all metrics if provided
        if all_metrics:
            info["all_metrics"] = {k: float(v) for k, v in all_metrics.items()}
        
        # Define a custom encoder to handle NumPy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                import numpy as np
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        
        # Save to JSON file with the custom encoder
        with open(output_path, 'w') as f:
            json.dump(info, f, indent=4, cls=NumpyEncoder)
            
        print(f"Matching information saved to: {output_path}")

    def run(self):
        """Main function to run the histology matcher"""
        try:
            if not self.select_files():
                return
                
            print("\n===== MRI-Histology Slice Matching =====")
            start_time = time.time()
            
            # Load MRI
            self.mri_nifti = nib.load(self.mri_path)
            self.mri_data = self.mri_nifti.get_fdata()
            print(f"MRI dimensions: {self.mri_data.shape}")
            
            # Load histology
            self.histology_img = io.imread(self.histology_path)
            print(f"Original histology dimensions: {self.histology_img.shape}")
            
            # Store original for visualization
            self.histology_original = self.histology_img.copy()
            
            # Convert to grayscale if RGB
            if len(self.histology_img.shape) > 2:
                self.histology_img = rgb2gray(self.histology_img)
                print(f"Converted histology to grayscale: {self.histology_img.shape}")
            
            # Find the optimal plane
            self.optimal_plane = self.find_optimal_plane(self.mri_data)
            
            # Get dimensions for the appropriate plane
            if self.optimal_plane == 0:  # Sagittal
                target_shape = (self.mri_data.shape[1], self.mri_data.shape[2])
            elif self.optimal_plane == 1:  # Coronal
                target_shape = (self.mri_data.shape[0], self.mri_data.shape[2])
            else:  # Axial
                target_shape = (self.mri_data.shape[0], self.mri_data.shape[1])
            
            print(f"Resizing histology to match MRI slice dimensions: {target_shape}")
            
            # Resize histology to match MRI slice dimensions
            histology_resized = transform.resize(self.histology_img, target_shape, anti_aliasing=True)
            
            # Create output directory
            base_dir = os.path.dirname(self.histology_path)
            self.output_dir = os.path.join(base_dir, "Match_slice_results")
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Preprocess histology for similarity measures
            histology_processed = self.preprocess_image(histology_resized)
            
            # STEP 1: Find the best slice using NMI (initially)
            print("\n-- STEP 1: Finding the best matching slice using NMI --")
            
            # Track the current metric being used
            current_metric = "NMI"
            selected_option = 0
            
            # Loop for slice selection with different metrics or manual selection
            while True:
                if selected_option == 0:  # First run or after orientation
                    # Use the current metric to find the best slice
                    first_slice_idx, first_slice_score, first_slice_data = self.find_best_slice_using_metric(
                        self.mri_data, histology_processed, self.optimal_plane, metric=current_metric.lower()
                    )
                elif selected_option == 2:  # Try another metric
                    # Use the selected new metric
                    print(f"\n-- Finding the best matching slice using {current_metric} --")
                    first_slice_idx, first_slice_score, first_slice_data = self.find_best_slice_using_metric(
                        self.mri_data, histology_processed, self.optimal_plane, metric=current_metric.lower()
                    )
                elif selected_option == 3:  # Manual selection
                    print("\n-- Manual slice selection --")
                    first_slice_idx = self.manually_select_slice(
                        self.mri_data, self.optimal_plane, histology_processed
                    )
                    first_slice_data = self.extract_slice(self.mri_data, self.optimal_plane, first_slice_idx)
                    # Since this is manual selection, we don't have a score
                    first_slice_score = 0.0
                    current_metric = "Manual"
                
                # Present the options to the user
                print("\n-- Presenting slice selection options --")
                selected_option, new_metric = self.present_slice_options(
                    histology_processed, first_slice_idx, first_slice_data, current_metric
                )
                
                # Process the user's choice
                if selected_option == 1:  # Confirm
                    print(f"Confirming slice {first_slice_idx} using {current_metric}")
                    break
                elif selected_option == 2:  # Try another metric
                    current_metric = new_metric
                    print(f"Selected new metric: {current_metric}")
                    # Continue loop with new metric
                elif selected_option == 3:  # Manual selection
                    print("Selected manual slice selection")
                    # Continue loop with manual selection
            
            # Preprocess the best slice for orientation detection
            first_slice_processed = self.preprocess_image(first_slice_data)
            
            # STEP 2: Find the best orientation using HOG
            print("\n-- STEP 2: Finding the best orientation using HOG --")
            best_oriented_img, orientation_desc, orientation_score, best_idx = self.find_best_orientation_using_hog(
                histology_processed, first_slice_processed
            )
            
            # Show the orientation options and allow manual selection
            print("\n-- Showing orientation options for manual review --")
            manual_selection = self.show_orientation_selection_UI(histology_resized, first_slice_data)
            
            if manual_selection:
                # Use manually selected orientation
                self.oriented_histology, self.best_orientation_desc, selected_idx = manual_selection
                print(f"Using manually selected orientation: {self.best_orientation_desc}")
                
                # Also reorient the original resized histology (for display)
                histology_resized_oriented = self.all_orientations[selected_idx][0]
                if histology_resized_oriented.shape != target_shape:
                    histology_resized_oriented = transform.resize(histology_resized_oriented, target_shape, anti_aliasing=True)
            else:
                # Use automatically detected orientation
                self.oriented_histology = best_oriented_img
                self.best_orientation_desc = orientation_desc
                
                # Also reorient the original resized histology (for display)
                histology_resized_oriented = self.all_orientations[best_idx][0]
                if histology_resized_oriented.shape != target_shape:
                    histology_resized_oriented = transform.resize(histology_resized_oriented, target_shape, anti_aliasing=True)
            
            # STEP 3: Find the best slice again with the reoriented histology
            print(f"\n-- STEP 3: Finding the best slice with reoriented histology using {current_metric} --")
            
            # Reset selected_option to restart the slice selection flow with the reoriented histology
            selected_option = 0
            
            # Loop for slice selection with different metrics or manual selection (with reoriented histology)
            while True:
                if selected_option == 0:  # First run after reorientation
                    # Use the current metric to find the best slice
                    final_slice_idx, final_slice_score, final_slice_data = self.find_best_slice_using_metric(
                        self.mri_data, self.oriented_histology, self.optimal_plane, metric=current_metric.lower()
                    )
                elif selected_option == 2:  # Try another metric
                    # Use the selected new metric
                    print(f"\n-- Finding the best matching slice using {current_metric} --")
                    final_slice_idx, final_slice_score, final_slice_data = self.find_best_slice_using_metric(
                        self.mri_data, self.oriented_histology, self.optimal_plane, metric=current_metric.lower()
                    )
                elif selected_option == 3:  # Manual selection
                    print("\n-- Manual slice selection --")
                    final_slice_idx = self.manually_select_slice(
                        self.mri_data, self.optimal_plane, self.oriented_histology
                    )
                    final_slice_data = self.extract_slice(self.mri_data, self.optimal_plane, final_slice_idx)
                    # Since this is manual selection, we don't have a score
                    final_slice_score = 0.0
                    current_metric = "Manual"
                
                # Present the options to the user
                print("\n-- Presenting slice selection options --")
                selected_option, new_metric = self.present_slice_options(
                    self.oriented_histology, final_slice_idx, final_slice_data, current_metric
                )
                
                # Process the user's choice
                if selected_option == 1:  # Confirm
                    print(f"Confirming slice {final_slice_idx} using {current_metric}")
                    break
                elif selected_option == 2:  # Try another metric
                    current_metric = new_metric
                    print(f"Selected new metric: {current_metric}")
                    # Continue loop with new metric
                elif selected_option == 3:  # Manual selection
                    print("Selected manual slice selection")
                    # Continue loop with manual selection
            
            # Store the best slice index
            self.best_slice_idx = final_slice_idx
            
            # Calculate all metrics for the final slice
            print("\n-- Calculating all similarity metrics for final slice --")
            all_metrics = self.calculate_all_metrics(final_slice_data, self.oriented_histology)
            print("All similarity metrics for the final slice:")
            for metric, score in all_metrics.items():
                print(f"{metric}: {score:.4f}")
            
            # STEP 4: Create the visualization
            print("\n-- STEP 4: Creating visualization --")
            visualization_path = os.path.join(self.output_dir, "histology_matching_process.png")
            fig = self.create_visualization(
                histology_resized,
                histology_resized_oriented,
                self.best_orientation_desc,
                first_slice_data,
                first_slice_idx,
                final_slice_data,
                final_slice_idx,
                visualization_path,
                metric_name=current_metric,
                all_metrics=all_metrics
            )
            
            # Show the visualization and wait for confirmation
            plt.suptitle("Histology Matching Results - Press ENTER to continue or ESC to cancel", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Add a key press event handler
            def on_key(event):
                if event.key == 'enter':
                    plt.close(fig)
                    self.manual_selection_made = True
                elif event.key == 'escape':
                    plt.close(fig)
                    self.manual_selection_made = False
            
            fig.canvas.mpl_connect('key_press_event', on_key)
            
            # Show plot and wait for user input
            plt.show(block=True)
            
            if not self.manual_selection_made:
                print("Process canceled by user.")
                return
            
            print(f"Visualization saved to {visualization_path}")
            
            # STEP 5: Create a new NIfTI file containing only the histology
            print("\n-- STEP 5: Creating standalone histology NIfTI file --")
            # Normalize histology values to match MRI range for better visualization
            mri_slice = self.extract_slice(self.mri_data, self.optimal_plane, final_slice_idx)
            mri_range = (np.percentile(mri_slice[mri_slice > 0], 1), np.percentile(mri_slice[mri_slice > 0], 99))
            histology_normalized = (histology_resized_oriented - np.min(histology_resized_oriented)) / (np.max(histology_resized_oriented) - np.min(histology_resized_oriented))
            histology_normalized = histology_normalized * (mri_range[1] - mri_range[0]) + mri_range[0]
            
            # Create a standalone histology NIfTI file
            histology_nifti = self.create_histology_nifti(self.mri_nifti, histology_normalized, self.optimal_plane, final_slice_idx)
            
            # Save the histology NIfTI file
            histology_filename = os.path.splitext(os.path.basename(self.histology_path))[0]
            # Remove .nii if part of the filename
            if histology_filename.lower().endswith('.nii'):
                histology_filename = os.path.splitext(histology_filename)[0]
                
            # Create output filename with the pattern: originalname_in_block.nii.gz
            output_nifti_path = os.path.join(self.output_dir, f"{histology_filename}_in_block.nii.gz")
            nib.save(histology_nifti, output_nifti_path)
            print(f"Standalone histology NIfTI file saved to {output_nifti_path}")
            
            # STEP 6: Save matching information to a separate file
            print("\n-- STEP 6: Saving matching information --")
            info_path = os.path.join(self.output_dir, f"{histology_filename}_matching_info.json")
            self.save_matching_info(
                first_slice_idx, first_slice_score,
                final_slice_idx, final_slice_score,
                self.best_orientation_desc, info_path,
                metric_name=current_metric,
                all_metrics=all_metrics
            )
            
            # Final summary
            elapsed_time = time.time() - start_time
            print("\n===== FINAL RESULTS =====")
            print(f"Optimal anatomical plane: {['Sagittal', 'Coronal', 'Axial'][self.optimal_plane]}")
            print(f"Initial best slice ({current_metric}): {first_slice_idx}" + 
                  (f" (Score: {first_slice_score:.4f})" if current_metric != "Manual" else ""))
            print(f"Best orientation: {self.best_orientation_desc}")
            print(f"Final best slice ({current_metric}): {final_slice_idx}" + 
                  (f" (Score: {final_slice_score:.4f})" if current_metric != "Manual" else ""))
            print("All similarity metrics for final slice:")
            for metric, score in all_metrics.items():
                print(f"  {metric}: {score:.4f}")
            print(f"Output saved to: {output_nifti_path}")
            print(f"Matching info saved to: {info_path}")
            print(f"Total processing time: {elapsed_time:.2f} seconds")
            print("\nAnalysis complete!")
            
            # Return the output path for potential use by other scripts
            return output_nifti_path
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            print(f"Error: {str(e)}")
            return None

if __name__ == "__main__":
    matcher = HistologyMatcher()
    matcher.run()
