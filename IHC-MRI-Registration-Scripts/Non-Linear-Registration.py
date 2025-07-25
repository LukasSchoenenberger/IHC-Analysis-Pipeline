import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
import SimpleITK as sitk
from scipy import interpolate
from matplotlib.patches import ConnectionPatch
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox
import glob

class NonlinearRegistration:
    def __init__(self):
        # Initialize file paths
        self.ihc_path = None
        self.mri_path = None
        self.registered_ihc_path = None
        
        # Initialize data
        self.ihc_nii = None
        self.mri_nii = None
        self.ihc_data = None
        self.mri_data = None
        self.registered_ihc_data = None
        
        # Image slices
        self.ihc_slice = None
        self.mri_slice = None
        self.slice_dimension = None
        self.slice_index = None
        
        # Initialize landmarks
        self.landmarks_ihc = []
        self.landmarks_mri = []
        self.connection_lines = []
        self.active_image = 'ihc'  # Start with IHC
        
        # Initialize GUI
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the root window

    def select_files(self):
        """Select input files using file dialogs with smart defaults"""
        # Try to find the linearly registered IHC image first
        linear_reg_dir = None
        current_dir = os.path.abspath(os.getcwd())
        
        # Check common locations for Linear_registration_results folder
        potential_dirs = [
            os.path.join(current_dir, "Linear_registration_results"),
            os.path.join(os.path.dirname(current_dir), "Linear_registration_results"),
        ]
        
        for d in potential_dirs:
            if os.path.isdir(d):
                linear_reg_dir = d
                break
        
        # Find registered IHC file
        registered_ihc_path = None
        if linear_reg_dir:
            reg_files = glob.glob(os.path.join(linear_reg_dir, "ihc_to_mri_affine.nii.gz"))
            if reg_files:
                registered_ihc_path = reg_files[0]
                print(f"Found linearly registered IHC file: {registered_ihc_path}")
            
        # If not found automatically, let user select
        if not registered_ihc_path:
            print("No linearly registered IHC file found. Please select manually.")
            
            # Make sure the window is updated
            self.root.update()
            
            registered_ihc_path = filedialog.askopenfilename(
                title="Select linearly registered IHC NIfTI file",
                filetypes=[
                    ("NIfTI files", "*.nii.gz *.nii"),
                    ("NIfTI files (*.nii.gz)", "*.nii.gz"),
                    ("NIfTI files (*.nii)", "*.nii"),
                    ("All files", "*.*")
                ],
                initialdir=current_dir,
                parent=self.root
            )
        
        if not registered_ihc_path:
            print("No registered IHC file selected. Exiting.")
            return False
                
        self.registered_ihc_path = registered_ihc_path
        
        # Use the path to determine original IHC and MRI files
        parent_dir = os.path.dirname(os.path.dirname(registered_ihc_path))
        
        # Find the original IHC file (should be in Match_slice_results)
        ihc_path = None
        match_slice_dir = os.path.join(parent_dir, "Match_slice_results")
        if os.path.isdir(match_slice_dir):
            ihc_files = glob.glob(os.path.join(match_slice_dir, "*_in_block.nii.gz"))
            if ihc_files:
                ihc_path = ihc_files[0]
                print(f"Found original IHC file: {ihc_path}")
        
        # If not found automatically, let user select
        if not ihc_path:
            print("No original IHC file found. Please select manually.")
            
            # Make sure the window is updated
            self.root.update()
            
            ihc_path = filedialog.askopenfilename(
                title="Select original IHC NIfTI file",
                filetypes=[
                    ("NIfTI files", "*.nii.gz *.nii"),
                    ("NIfTI files (*.nii.gz)", "*.nii.gz"),
                    ("NIfTI files (*.nii)", "*.nii"),
                    ("All files", "*.*")
                ],
                initialdir=parent_dir,
                parent=self.root
            )
        
        if not ihc_path:
            print("No original IHC file selected. Exiting.")
            return False
                
        self.ihc_path = ihc_path
        
        # Look for FLASH MRI file in the parent directory
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

    def extract_slice(self, volume, slice_idx, axis):
        """Extract a specific slice from the volume along a given axis"""
        if axis == 0:
            return volume[slice_idx, :, :]
        elif axis == 1:
            return volume[:, slice_idx, :]
        else:  # axis == 2
            return volume[:, :, slice_idx]

    def find_non_empty_slice(self, data):
        """Find a slice with non-zero content in the given volume"""
        # Check each dimension
        for axis in range(3):
            dim_size = data.shape[axis]
            for slice_idx in range(dim_size):
                slice_data = self.extract_slice(data, slice_idx, axis)
                if np.count_nonzero(slice_data) > 1000:  # More than 100 non-zero pixels
                    return slice_idx, axis
        
        # Default to center of axial plane if no non-empty slice found
        return data.shape[2] // 2, 2

    def normalize_image(self, img):
        """Normalize image to range [0, 1]"""
        img_min, img_max = np.min(img), np.max(img)
        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        return img

    def nonlinear_registration_UI(self):
        """Show UI for landmark-based non-linear registration"""
        from matplotlib.widgets import RadioButtons, Button
        
        # Create figure with more space at the bottom for controls
        fig = plt.figure(figsize=(14, 10))
        
        # Set up grid with space for controls
        gs = fig.add_gridspec(3, 2, height_ratios=[5, 5, 1])
        
        # Create axes for the images
        ax_ihc = fig.add_subplot(gs[0, 0])
        ax_mri = fig.add_subplot(gs[0, 1])
        
        # Add axes for the instructions and landmark list
        ax_instructions = fig.add_subplot(gs[1, :])
        ax_instructions.axis('off')
        
        # Add controls area at bottom
        ax_controls = fig.add_subplot(gs[2, :])
        ax_controls.axis('off')
        
        # Get slices to display
        if self.slice_dimension == 0:
            ihc_img_data = self.registered_ihc_data[self.slice_index, :, :]
            mri_img_data = self.mri_data[self.slice_index, :, :]
        elif self.slice_dimension == 1:
            ihc_img_data = self.registered_ihc_data[:, self.slice_index, :]
            mri_img_data = self.mri_data[:, self.slice_index, :]
        else:  # 2
            ihc_img_data = self.registered_ihc_data[:, :, self.slice_index]
            mri_img_data = self.mri_data[:, :, self.slice_index]
        
        # Normalize for display
        ihc_img_data = self.normalize_image(ihc_img_data)
        mri_img_data = self.normalize_image(mri_img_data)
        
        # Display images
        self.ihc_img = ax_ihc.imshow(ihc_img_data, cmap='gray')
        ax_ihc.set_title("Linearly Registered IHC Image\n(Place landmarks here first)", fontsize=12)
        ax_ihc.axis('off')
        
        self.mri_img = ax_mri.imshow(mri_img_data, cmap='gray')
        ax_mri.set_title("MRI Reference Image", fontsize=12)
        ax_mri.axis('off')
        
        # Add radio buttons for active image selection
        radio_ax = plt.axes([0.05, 0.08, 0.15, 0.05])
        self.radio = RadioButtons(radio_ax, ['IHC Image', 'MRI Image'])
        self.radio.on_clicked(self.set_active_image)
        
        # Add "Perform Registration" button
        button_ax = plt.axes([0.25, 0.08, 0.3, 0.05])
        self.register_button = Button(button_ax, 'Perform Non-Linear Registration', color='lightgreen', hovercolor='darkgreen')
        self.register_button.on_clicked(self.perform_nonlinear_registration)
        
        # Add "Reset Landmarks" button
        reset_ax = plt.axes([0.6, 0.08, 0.15, 0.05])
        self.reset_button = Button(reset_ax, 'Reset Landmarks', color='lightcoral', hovercolor='red')
        self.reset_button.on_clicked(self.reset_landmarks)
        
        # Add instructions
        instructions_text = (
            "Non-Linear Registration Instructions:\n\n"
            "1. Place landmarks alternately (on IHC first, then corresponding point on MRI)\n"
            "2. Use mouse wheel to zoom, right-click + drag to pan\n"
            "3. Press Backspace to remove last landmark pair\n"
            "4. Press Enter or click 'Perform Non-Linear Registration' button when done\n"
            "5. For best results, use at least 20-30 well-distributed landmarks"
        )
        ax_instructions.text(0.5, 0.5, instructions_text,
                         ha='center', va='center', fontsize=12,
                         bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.5))
        
        # Connect event handlers
        self.fig = fig
        self.ax_ihc = ax_ihc
        self.ax_mri = ax_mri
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        # For panning
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        self.pan_active = False
        self.pan_start = None
        
        # Add "Current Landmarks" display
        self.landmark_text = ax_instructions.text(0.5, 0.2, "Current Landmarks: 0 pairs",
                                              ha='center', va='center', fontsize=12,
                                              color='blue')
        
        plt.tight_layout(rect=[0, 0.15, 1, 0.95])
        plt.suptitle("Landmark-Based Non-Linear Registration", fontsize=16, y=0.98)
        plt.show()

    def set_active_image(self, label):
        """Set which image is active for landmark placement"""
        if label == 'IHC Image':
            self.active_image = 'ihc'
        else:
            self.active_image = 'mri'
        print(f"Active image set to: {self.active_image}")

    def on_press(self, event):
        """Handle mouse button press events for panning"""
        if event.button == 3:  # Right mouse button
            self.pan_active = True
            self.pan_start = (event.x, event.y)
    
    def on_release(self, event):
        """Handle mouse button release events for panning"""
        if event.button == 3:  # Right mouse button
            self.pan_active = False
    
    def on_motion(self, event):
        """Handle mouse motion events for panning"""
        if self.pan_active and event.inaxes:
            dx = event.x - self.pan_start[0]
            dy = event.y - self.pan_start[1]
            
            # Update the view limits
            x_min, x_max = event.inaxes.get_xlim()
            y_min, y_max = event.inaxes.get_ylim()
            
            # Scale the movement based on the current view
            scale_x = (x_max - x_min) / event.inaxes.get_window_extent().width
            scale_y = (y_max - y_min) / event.inaxes.get_window_extent().height
            
            event.inaxes.set_xlim(x_min - dx * scale_x, x_max - dx * scale_x)
            event.inaxes.set_ylim(y_min + dy * scale_y, y_max + dy * scale_y)
            
            self.pan_start = (event.x, event.y)
            self.fig.canvas.draw_idle()

    def on_click(self, event):
        """Handle mouse click events for landmark placement"""
        # Ignore right-clicks (used for panning) and clicks outside axes
        if event.button != 1 or event.inaxes not in [self.ax_ihc, self.ax_mri]:
            return
            
        # Place landmark based on active image
        if event.inaxes == self.ax_ihc and self.active_image == 'ihc':
            # Check if we need to match the number of landmarks
            if len(self.landmarks_ihc) > len(self.landmarks_mri):
                print("Please place corresponding landmark on MRI first")
                return
                
            self.landmarks_ihc.append((event.xdata, event.ydata))
            landmark_idx = len(self.landmarks_ihc)
            self.ax_ihc.plot(event.xdata, event.ydata, 'ro', markersize=8)
            self.ax_ihc.text(event.xdata, event.ydata + 5, str(landmark_idx), 
                             color='yellow', fontweight='bold', ha='center')
            
            print(f"Landmark {landmark_idx} set on IHC: ({event.xdata:.1f}, {event.ydata:.1f})")
            
            # Automatically switch to MRI for next landmark
            self.active_image = 'mri'
            self.radio.set_active(1)
            
        elif event.inaxes == self.ax_mri and self.active_image == 'mri':
            # Only allow if we have a corresponding IHC landmark
            if len(self.landmarks_ihc) <= len(self.landmarks_mri):
                print("Please place landmark on IHC first")
                return
                
            self.landmarks_mri.append((event.xdata, event.ydata))
            landmark_idx = len(self.landmarks_mri)
            self.ax_mri.plot(event.xdata, event.ydata, 'ro', markersize=8)
            self.ax_mri.text(event.xdata, event.ydata + 5, str(landmark_idx), 
                             color='yellow', fontweight='bold', ha='center')
            
            print(f"Landmark {landmark_idx} set on MRI: ({event.xdata:.1f}, {event.ydata:.1f})")
            
            # Draw connection between corresponding landmarks
            ihc_x, ihc_y = self.landmarks_ihc[landmark_idx - 1]
            mri_x, mri_y = self.landmarks_mri[landmark_idx - 1]
            
            con = ConnectionPatch(
                xyA=(ihc_x, ihc_y), xyB=(mri_x, mri_y),
                coordsA="data", coordsB="data",
                axesA=self.ax_ihc, axesB=self.ax_mri,
                color="yellow", alpha=0.5, linestyle="dashed"
            )
            self.fig.add_artist(con)
            self.connection_lines.append(con)
            
            # Automatically switch back to IHC for next landmark
            self.active_image = 'ihc'
            self.radio.set_active(0)
            
            # Update landmark count
            self.landmark_text.set_text(f"Current Landmarks: {landmark_idx} pairs")
            
        self.fig.canvas.draw()

    def on_key(self, event):
        """Handle keyboard events"""
        if event.key == 'enter':
            self.perform_nonlinear_registration(event)
        elif event.key == 'escape':
            plt.close(self.fig)
        elif event.key == 'backspace':
            self.remove_last_landmark()
    
    def on_scroll(self, event):
        """Handle mouse scroll events for zooming"""
        # Handle zooming
        ax = event.inaxes
        if ax is None:
            return
            
        # Get the center point (mouse position)
        if event.xdata is None or event.ydata is None:
            # If mouse is outside data range, use center of axes
            x_center = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
            y_center = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
        else:
            x_center = event.xdata
            y_center = event.ydata
        
        # Get the x and y limits
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        
        # Calculate current range
        x_range = (x_max - x_min) / 2
        y_range = (y_max - y_min) / 2
        
        # Calculate zoom factor - fixed regardless of direction
        # Make sure we can zoom both in and out
        if event.button == 'up':  # Scroll up = zoom in
            scale_factor = 0.8  # More aggressive zoom in
        else:  # Scroll down = zoom out
            scale_factor = 1.25  # More aggressive zoom out
        
        # Calculate the new ranges
        new_x_range = x_range * scale_factor
        new_y_range = y_range * scale_factor
        
        # Set new limits centered on mouse position
        ax.set_xlim(x_center - new_x_range, x_center + new_x_range)
        ax.set_ylim(y_center - new_y_range, y_center + new_y_range)
        
        # Redraw canvas
        self.fig.canvas.draw_idle()

    def remove_last_landmark(self):
        """Remove the last set of corresponding landmarks"""
        # Remove the last set of corresponding landmarks
        if len(self.landmarks_mri) == len(self.landmarks_ihc):
            # Remove last landmark from both
            if self.landmarks_ihc:
                self.landmarks_ihc.pop()
                self.landmarks_mri.pop()
                
                # Remove the last connection line
                if self.connection_lines:
                    self.connection_lines[-1].remove()
                    self.connection_lines.pop()
                
                # Redraw the landmarks
                self._redraw_landmarks()
                landmark_idx = len(self.landmarks_ihc)
                self.landmark_text.set_text(f"Current Landmarks: {landmark_idx} pairs")
                print("Removed last landmark pair")
                
        elif len(self.landmarks_ihc) > len(self.landmarks_mri):
            # Only remove from IHC
            if self.landmarks_ihc:
                self.landmarks_ihc.pop()
                self._redraw_landmarks()
                print("Removed last IHC landmark")
    
    def reset_landmarks(self, event=None):
        """Reset all landmarks"""
        self.landmarks_ihc = []
        self.landmarks_mri = []
        
        # Remove connection lines
        for line in self.connection_lines:
            line.remove()
        self.connection_lines = []
        
        # Redraw
        self._redraw_landmarks()
        self.landmark_text.set_text("Current Landmarks: 0 pairs")
        print("All landmarks reset")

    def _redraw_landmarks(self):
        """Redraw all landmarks after changes"""
        # Clear axes and redraw
        self.ax_ihc.clear()
        self.ax_mri.clear()
        
        # Get slices to display
        if self.slice_dimension == 0:
            ihc_img_data = self.registered_ihc_data[self.slice_index, :, :]
            mri_img_data = self.mri_data[self.slice_index, :, :]
        elif self.slice_dimension == 1:
            ihc_img_data = self.registered_ihc_data[:, self.slice_index, :]
            mri_img_data = self.mri_data[:, self.slice_index, :]
        else:  # 2
            ihc_img_data = self.registered_ihc_data[:, :, self.slice_index]
            mri_img_data = self.mri_data[:, :, self.slice_index]
        
        # Normalize for display
        ihc_img_data = self.normalize_image(ihc_img_data)
        mri_img_data = self.normalize_image(mri_img_data)
        
        # Redisplay images
        self.ax_ihc.imshow(ihc_img_data, cmap='gray')
        self.ax_mri.imshow(mri_img_data, cmap='gray')
        
        # Set titles
        self.ax_ihc.set_title("Linearly Registered IHC Image\n(Place landmarks here first)", fontsize=12)
        self.ax_mri.set_title("MRI Reference Image", fontsize=12)
        self.ax_ihc.axis('off')
        self.ax_mri.axis('off')
        
        # Redraw landmarks
        for i, (x, y) in enumerate(self.landmarks_ihc):
            self.ax_ihc.plot(x, y, 'ro', markersize=8)
            self.ax_ihc.text(x, y + 5, str(i+1), color='yellow', fontweight='bold', ha='center')
        
        for i, (x, y) in enumerate(self.landmarks_mri):
            self.ax_mri.plot(x, y, 'ro', markersize=8)
            self.ax_mri.text(x, y + 5, str(i+1), color='yellow', fontweight='bold', ha='center')
        
        # Redraw connection lines
        self.connection_lines = []  # Clear old references
        for i, (ihc_pt, mri_pt) in enumerate(zip(self.landmarks_ihc, self.landmarks_mri)):
            con = ConnectionPatch(
                xyA=ihc_pt, xyB=mri_pt,
                coordsA="data", coordsB="data",
                axesA=self.ax_ihc, axesB=self.ax_mri,
                color="yellow", alpha=0.5, linestyle="dashed"
            )
            self.fig.add_artist(con)
            self.connection_lines.append(con)
        
        self.fig.canvas.draw()

    def perform_nonlinear_registration(self, event=None):
        """Perform non-linear (thin-plate spline) registration using landmarks"""
        # Sanity checks
        if len(self.landmarks_ihc) < 3 or len(self.landmarks_mri) < 3:
            messagebox.showwarning("Not Enough Landmarks", 
                                  "Please place at least 3 landmark pairs before registration.")
            return
        
        if len(self.landmarks_ihc) != len(self.landmarks_mri):
            messagebox.showwarning("Unmatched Landmarks", 
                                  "Equal number of landmarks required on both images.")
            return
        
        if len(self.landmarks_ihc) < 5:
            proceed = messagebox.askyesno("Few Landmarks", 
                                         "Less than 5 landmarks may produce poor results. Continue anyway?")
            if not proceed:
                return
        
        print("Starting non-linear (thin-plate spline) registration...")
        
        # Extract the correct slice
        if self.slice_dimension == 0:
            ihc_img = self.registered_ihc_data[self.slice_index, :, :]
            mri_img = self.mri_data[self.slice_index, :, :]
        elif self.slice_dimension == 1:
            ihc_img = self.registered_ihc_data[:, self.slice_index, :]
            mri_img = self.mri_data[:, self.slice_index, :]
        else:  # 2
            ihc_img = self.registered_ihc_data[:, :, self.slice_index]
            mri_img = self.mri_data[:, :, self.slice_index]
        
        # Convert landmarks to numpy arrays
        src_pts = np.array(self.landmarks_ihc)
        dst_pts = np.array(self.landmarks_mri)
        
        try:
            # Use scipy's Radial Basis Function with thin plate spline
            # We need to separate x, y coordinates for Rbf input
            src_x = src_pts[:,0]
            src_y = src_pts[:,1]
            dst_x = dst_pts[:,0]
            dst_y = dst_pts[:,1]
            
            # Smoothness parameter
            smoothness = 0.0
            
            # Create the x and y interpolation functions using thin-plate spline radial basis
            print("Creating RBF interpolators with thin-plate spline basis...")
            tps_x = interpolate.Rbf(src_x, src_y, dst_x, function='thin_plate', smooth=smoothness)
            tps_y = interpolate.Rbf(src_x, src_y, dst_y, function='thin_plate', smooth=smoothness)
            
            # Save the TPS transformation parameters
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
            
            # Create a mesh grid for the source image
            height, width = ihc_img.shape
            print(f"Image dimensions: {width}x{height}")
            
            # Create grid of all pixel coordinates
            grid_y, grid_x = np.mgrid[0:height, 0:width]
            
            # Apply the thin-plate spline mapping to each pixel
            print("Applying thin-plate spline transformation to all pixels...")
            x_warped = tps_x(grid_x, grid_y)
            y_warped = tps_y(grid_x, grid_y)
            
            # Save the deformation field for future use
            deformation_field = {
                'x_orig': grid_x,
                'y_orig': grid_y,
                'x_warped': x_warped,
                'y_warped': y_warped,
                'shape': ihc_img.shape
            }
            
            # Stack coordinates and flatten for griddata
            points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
            xi = np.vstack([x_warped.ravel(), y_warped.ravel()]).T
            
            # Interpolate the image values
            print("Interpolating warped image...")
            values = ihc_img.flatten()
            
            # Use griddata to resample the image at the warped locations
            tps_result = interpolate.griddata(xi, values, points, method='linear', fill_value=0)
            tps_result = tps_result.reshape(height, width)
            
            # Fill in any NaN values with nearest-neighbor interpolation
            if np.any(np.isnan(tps_result)):
                print("Filling NaN values with nearest-neighbor interpolation...")
                nan_mask = np.isnan(tps_result)
                nn_interp = interpolate.griddata(xi, values, points, method='nearest')
                tps_result[nan_mask] = nn_interp.reshape(height, width)[nan_mask]
            
            # Prepare a new volume for the final registered data
            nonlinear_result = np.zeros_like(self.registered_ihc_data)
            
            # Insert the registered slice in the correct dimension
            if self.slice_dimension == 0:
                nonlinear_result[self.slice_index, :, :] = tps_result
            elif self.slice_dimension == 1:
                nonlinear_result[:, self.slice_index, :] = tps_result
            else:  # 2
                nonlinear_result[:, :, self.slice_index] = tps_result
            
            # Create output directory
            base_dir = os.path.dirname(self.registered_ihc_path)
            parent_dir = os.path.dirname(base_dir)
            output_dir = os.path.join(parent_dir, "Non-linear_registration_results")
            os.makedirs(output_dir, exist_ok=True)
            
            # Show registration result
            self._show_registration_result(ihc_img, tps_result, mri_img, output_dir)
            
            # Save the result
            output_filename = os.path.join(output_dir, "ihc_to_mri_nonlinear.nii.gz")
            self._save_registration_result(nonlinear_result, output_filename)
            
            # Save the non-linear transform parameters and deformation field
            self._save_nonlinear_transform(tps_params, deformation_field, output_dir)
            
            print("\nNon-linear registration complete!")
            print(f"Registered image saved to: {output_filename}")
            print(f"Transform and deformation field saved to: {output_dir}")
            
            # Close window
            plt.close(self.fig)
            
            # Show a message box to inform the user
            messagebox.showinfo("Registration Complete", 
                              f"Non-linear registration successful!\n\nResults saved to:\n{output_dir}")
            
            return output_filename
            
        except Exception as e:
            print(f"Error during non-linear registration: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Registration failed: {str(e)}")
            return None
    
    def _save_nonlinear_transform(self, tps_params, deformation_field, output_dir):
        """Save the non-linear transform parameters and deformation field"""
        # Save TPS parameters
        tps_path = os.path.join(output_dir, "nonlinear_transform.pkl")
        with open(tps_path, 'wb') as f:
            pickle.dump(tps_params, f)
        
        # Save deformation field separately (could be large)
        deformation_path = os.path.join(output_dir, "deformation_field.pkl")
        with open(deformation_path, 'wb') as f:
            pickle.dump(deformation_field, f)
        
        print(f"Non-linear transform saved to: {tps_path}")
        print(f"Deformation field saved to: {deformation_path}")
    
    def _show_registration_result(self, original_ihc, warped_ihc, reference_mri, output_dir):
        """Show and save the registration result with comparison to reference"""
        # Create a nice visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Top left: Linear registered IHC
        axes[0, 0].imshow(self.normalize_image(original_ihc), cmap='gray')
        axes[0, 0].set_title('Linear Registered IHC', fontsize=14)
        axes[0, 0].axis('off')
        
        # Top right: MRI reference
        axes[0, 1].imshow(self.normalize_image(reference_mri), cmap='gray')
        axes[0, 1].set_title('MRI Reference', fontsize=14)
        axes[0, 1].axis('off')
        
        # Bottom left: Non-linear registered IHC
        axes[1, 0].imshow(self.normalize_image(warped_ihc), cmap='gray')
        axes[1, 0].set_title('Non-linear Registered IHC', fontsize=14)
        axes[1, 0].axis('off')
        
        # Bottom right: Overlay to show alignment
        overlay = np.zeros((warped_ihc.shape[0], warped_ihc.shape[1], 3))
        overlay[:,:,0] = self.normalize_image(warped_ihc)  # Red channel = registered
        overlay[:,:,1] = self.normalize_image(reference_mri)  # Green channel = reference
        
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Overlay (Red=IHC, Green=MRI)', fontsize=14)
        axes[1, 1].axis('off')
        
        # Add landmard dots to all images
        for i, (x, y) in enumerate(self.landmarks_ihc):
            axes[0, 0].plot(x, y, 'ro', markersize=6)
            
        for i, (x, y) in enumerate(self.landmarks_mri):
            axes[0, 1].plot(x, y, 'ro', markersize=6)
            axes[1, 1].plot(x, y, 'wo', markersize=4)
        
        # Tight layout
        plt.tight_layout()
        plt.suptitle('Non-linear Registration Result', fontsize=16, y=0.98)
        
        # Save figure
        output_path = os.path.join(output_dir, "nonlinear_registration_result.png")
        plt.savefig(output_path, dpi=300)
        print(f"Visualization saved to: {output_path}")
        
        # Show figure in a new window
        plt.show(block=False)
    
    def _save_registration_result(self, registered_volume, filename):
        """Save the registered result as a NIFTI file"""
        # Create the NIFTI image with original header and affine
        new_nii = nib.Nifti1Image(registered_volume, self.ihc_nii.affine, self.ihc_nii.header)
        
        # Save to file
        nib.save(new_nii, filename)
        print(f"Saved registered image to: {filename}")
        
    def run(self):
        """Main entry point to run the non-linear registration"""
        try:
            # Select input files
            if not self.select_files():
                return
                
            print("\n===== Non-linear Registration =====")
            print("Loading files...")
            
            # Load the images
            self.mri_nii = nib.load(self.mri_path)
            self.ihc_nii = nib.load(self.ihc_path)
            self.registered_ihc_nii = nib.load(self.registered_ihc_path)
            
            self.mri_data = self.mri_nii.get_fdata()
            self.ihc_data = self.ihc_nii.get_fdata()
            self.registered_ihc_data = self.registered_ihc_nii.get_fdata()
            
            print(f"MRI dimensions: {self.mri_data.shape}")
            print(f"Original IHC dimensions: {self.ihc_data.shape}")
            print(f"Registered IHC dimensions: {self.registered_ihc_data.shape}")
            
            # Find the non-empty slice
            print("Finding non-empty slice...")
            self.slice_index, self.slice_dimension = self.find_non_empty_slice(self.registered_ihc_data)
            print(f"Using slice {self.slice_index} along dimension {self.slice_dimension}")
            
            # Start the UI
            print("Starting landmark-based registration UI...")
            self.nonlinear_registration_UI()
            
            return True
            
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            return False

if __name__ == "__main__":
    registration = NonlinearRegistration()
    registration.run()
