import os
import sys
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import time
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

# Simple logging
def log(msg):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")

class FileSelectionDialog:
    """GUI dialog for selecting mask and MRI files"""
    
    def __init__(self, parent=None):
        """Initialize the file selection dialog"""
        self.result = {'mask_path': None, 'mri_path': None}
        
        # Create main window
        if parent is None:
            self.root = tk.Tk()
            self.created_root = True
        else:
            self.root = tk.Toplevel(parent)
            self.created_root = False
            
        self.root.title("Threshold Evaluator - File Selection")
        self.root.geometry("600x400")
        self.root.minsize(500, 300)
        
        # Center the window
        self.center_window()
        
        # Create the interface
        self.create_widgets()
        
        # Make modal if parent exists
        if parent is not None:
            self.root.transient(parent)
            self.root.grab_set()
    
    def center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
    
    def create_widgets(self):
        """Create the dialog widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Threshold Evaluator - File Selection", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Instructions
        instructions = ttk.Label(main_frame, 
                                text="Select a mask file and optionally an MRI file for threshold evaluation.",
                                font=("Arial", 10), justify=tk.CENTER)
        instructions.pack(pady=(0, 30))
        
        # Manual mask selection
        mask_manual_frame = ttk.LabelFrame(main_frame, text="Select Mask File", padding="15")
        mask_manual_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.selected_mask_var = tk.StringVar(value="No mask file selected")
        ttk.Label(mask_manual_frame, textvariable=self.selected_mask_var, 
                 font=("Arial", 10)).pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Button(mask_manual_frame, text="Browse for Mask File...", 
                  command=self.browse_mask_file).pack(anchor=tk.W)
        
        # Manual MRI selection
        mri_manual_frame = ttk.LabelFrame(main_frame, text="Select MRI File (Optional)", padding="15")
        mri_manual_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.selected_mri_var = tk.StringVar(value="No MRI file selected")
        ttk.Label(mri_manual_frame, textvariable=self.selected_mri_var, 
                 font=("Arial", 10)).pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Button(mri_manual_frame, text="Browse for MRI File...", 
                  command=self.browse_mri_file).pack(anchor=tk.W)
        
        # Clear buttons
        clear_frame = ttk.Frame(main_frame)
        clear_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Button(clear_frame, text="Clear Mask Selection", 
                  command=self.clear_mask_selection).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(clear_frame, text="Clear MRI Selection", 
                  command=self.clear_mri_selection).pack(side=tk.LEFT)
        
        # Note about MRI being optional
        note_label = ttk.Label(main_frame, 
                              text="Note: MRI file is optional. You can evaluate the mask alone.",
                              font=("Arial", 9, "italic"), foreground="gray")
        note_label.pack(pady=(10, 20))
        
        # Bottom buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="Cancel", command=self.cancel).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="OK", command=self.ok).pack(side=tk.RIGHT)
    
    def browse_mask_file(self):
        """Browse for mask file manually"""
        file_path = filedialog.askopenfilename(
            title="Select Mask File",
            filetypes=[
                ("NIfTI files", "*.nii *.nii.gz"),
                ("NIfTI compressed", "*.nii.gz"),
                ("NIfTI uncompressed", "*.nii"),
                ("All files", "*.*")
            ],
            initialdir=os.getcwd()
        )
        
        if file_path:
            self.selected_mask_var.set(os.path.basename(file_path))
            # Store the full path
            self.manual_mask_path = file_path
        else:
            self.manual_mask_path = None
    
    def browse_mri_file(self):
        """Browse for MRI file manually"""
        file_path = filedialog.askopenfilename(
            title="Select MRI File",
            filetypes=[
                ("NIfTI files", "*.nii *.nii.gz"),
                ("NIfTI compressed", "*.nii.gz"),
                ("NIfTI uncompressed", "*.nii"),
                ("All files", "*.*")
            ],
            initialdir=os.getcwd()
        )
        
        if file_path:
            self.selected_mri_var.set(os.path.basename(file_path))
            # Store the full path
            self.manual_mri_path = file_path
        else:
            self.manual_mri_path = None
    
    def clear_mask_selection(self):
        """Clear manual mask selection"""
        self.selected_mask_var.set("No mask file selected")
        self.manual_mask_path = None
    
    def clear_mri_selection(self):
        """Clear manual MRI selection"""
        self.selected_mri_var.set("No MRI file selected")
        self.manual_mri_path = None
    
    def get_selected_files(self):
        """Get the currently selected files"""
        mask_path = getattr(self, 'manual_mask_path', None)
        mri_path = getattr(self, 'manual_mri_path', None)
        return mask_path, mri_path
    
    def ok(self):
        """Handle OK button click"""
        mask_path, mri_path = self.get_selected_files()
        
        if mask_path is None:
            messagebox.showerror("No Mask Selected", 
                               "Please select a mask file before proceeding.")
            return
        
        if not os.path.exists(mask_path):
            messagebox.showerror("File Not Found", 
                               f"The selected mask file does not exist:\n{mask_path}")
            return
        
        if mri_path and not os.path.exists(mri_path):
            messagebox.showerror("File Not Found", 
                               f"The selected MRI file does not exist:\n{mri_path}")
            return
        
        # Store results
        self.result['mask_path'] = mask_path
        self.result['mri_path'] = mri_path
        
        # Close dialog
        self.root.destroy()
    
    def cancel(self):
        """Handle Cancel button click"""
        self.result = {'mask_path': None, 'mri_path': None}
        self.root.destroy()
    
    def show(self):
        """Show the dialog and return the result"""
        # Initialize manual paths
        self.manual_mask_path = None
        self.manual_mri_path = None
        
        # Show the dialog
        if self.created_root:
            self.root.mainloop()
        else:
            self.root.wait_window()
        
        return self.result

class SimpleThresholdEvaluator:
    def __init__(self, mask_path=None, mri_path=None):
        self.mask_path = mask_path
        self.mri_path = mri_path
        
        # Initialize data containers
        self.mask_data = None
        self.mri_data = None
        
        # Initialize visualization variables
        self.show_mask = True
        self.mask_slice = None
        self.mri_slice = None
        self.value_text = None
    
    def load_data(self):
        """Load the mask and MRI data"""
        try:
            # Load mask data
            log(f"Loading mask: {self.mask_path}")
            mask_nii = nib.load(self.mask_path)
            self.mask_data = mask_nii.get_fdata()
            
            log(f"Mask dimensions: {self.mask_data.shape}")
            log(f"Mask data type: {self.mask_data.dtype}")
            non_zero = np.count_nonzero(self.mask_data)
            log(f"Non-zero voxels in mask: {non_zero}")
            
            if non_zero == 0:
                log("WARNING: Mask contains no non-zero voxels!")
            
            value_range = (np.min(self.mask_data), np.max(self.mask_data))
            log(f"Mask value range: {value_range[0]} to {value_range[1]}")
            
            # Load MRI data if available
            if self.mri_path and os.path.exists(self.mri_path):
                log(f"Loading MRI: {self.mri_path}")
                mri_nii = nib.load(self.mri_path)
                self.mri_data = mri_nii.get_fdata()
                
                log(f"MRI dimensions: {self.mri_data.shape}")
                log(f"MRI data type: {self.mri_data.dtype}")
            else:
                log("No reference MRI available")
            
            return True
            
        except Exception as e:
            log(f"Error loading data: {str(e)}")
            import traceback
            log(traceback.format_exc())
            return False

    def find_slice_with_data(self):
        """Find which slice contains data in the mask"""
        log("Finding slice with data...")
        # Find the dimension with the most non-zero pixels
        best_dim = 0
        max_nonzero = 0
        slice_idx = 0
        
        for dim in range(3):
            if dim == 0:
                slices = [self.mask_data[i, :, :] for i in range(self.mask_data.shape[0])]
            elif dim == 1:
                slices = [self.mask_data[:, i, :] for i in range(self.mask_data.shape[1])]
            else:  # dim == 2
                slices = [self.mask_data[:, :, i] for i in range(self.mask_data.shape[2])]
            
            # Count nonzero pixels in each slice
            nonzero_counts = [np.count_nonzero(s) for s in slices]
            max_count_idx = np.argmax(nonzero_counts)
            log(f"Dimension {dim}: max non-zero count is {nonzero_counts[max_count_idx]} at index {max_count_idx}")
            
            if nonzero_counts[max_count_idx] > max_nonzero:
                max_nonzero = nonzero_counts[max_count_idx]
                best_dim = dim
                slice_idx = max_count_idx
        
        log(f"Found most data in dimension {best_dim}, slice {slice_idx} with {max_nonzero} non-zero voxels")
        
        return best_dim, slice_idx

    def extract_slice(self, data, dimension, slice_idx):
        """Extract a 2D slice from a 3D volume"""
        if dimension == 0:
            return data[slice_idx, :, :]
        elif dimension == 1:
            return data[:, slice_idx, :]
        else:  # dimension == 2
            return data[:, :, slice_idx]

    def toggle_mask(self, event):
        """Toggle mask overlay visibility"""
        if event.key == 'c':
            self.show_mask = not self.show_mask
            if hasattr(self, 'overlay_img') and self.overlay_img is not None:
                self.overlay_img.set_visible(self.show_mask)
                plt.gcf().canvas.draw_idle()
                log(f"Mask overlay visibility toggled to: {self.show_mask}")

    def show_pixel_value(self, event):
        """Show pixel value under cursor"""
        if hasattr(self, 'mask_img') and event.inaxes == self.mask_img.axes:
            x, y = int(np.round(event.xdata)), int(np.round(event.ydata))
            h, w = self.mask_slice.shape
            
            if 0 <= x < w and 0 <= y < h:
                value = self.mask_slice[y, x]
                
                # Update text
                if self.value_text is None:
                    self.value_text = plt.text(
                        0.02, 0.98, f"Value: {value:.2f}", 
                        transform=self.mask_img.axes.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                    )
                else:
                    self.value_text.set_text(f"Value: {value:.2f}")
                
                plt.gcf().canvas.draw_idle()

    def update_threshold(self, val):
        """Update the threshold value for display"""
        if not hasattr(self, 'threshold_slider'):
            return
            
        threshold = self.threshold_slider.val
        
        # Update mask display based on threshold
        mask_above_threshold = np.copy(self.mask_slice)
        mask_above_threshold[mask_above_threshold < threshold] = 0
        
        # Update mask image
        if hasattr(self, 'mask_img') and self.mask_img is not None:
            self.mask_img.set_array(mask_above_threshold)
        
        # Update overlay if MRI is available
        if hasattr(self, 'mri_img') and hasattr(self, 'overlay_img') and \
           self.mri_slice is not None and self.overlay_img is not None:
            # Create binary mask for overlay
            binary_mask = (mask_above_threshold > 0).astype(float)
            
            # Update overlay image with new thresholded data
            self.overlay_img.set_array(binary_mask)
        
        # Redraw the figure
        plt.gcf().canvas.draw_idle()

    def visualize(self):
        """Create the visualization of mask and MRI with threshold control"""
        log("Creating visualization...")
        # Find the slice with the most data
        dimension, slice_idx = self.find_slice_with_data()
        
        # Extract the slice from the mask
        self.mask_slice = self.extract_slice(self.mask_data, dimension, slice_idx)
        log(f"Extracted mask slice shape: {self.mask_slice.shape}")
        
        # Calculate min and max for color scaling
        vmin = np.min(self.mask_slice)
        vmax = np.max(self.mask_slice)
        if vmin == vmax:
            vmin = 0
            vmax = 1 if vmax == 0 else vmax
        log(f"Mask slice value range: {vmin} to {vmax}")
        
        # Extract the same slice from MRI if available
        self.mri_slice = None
        if self.mri_data is not None:
            # Check if dimensions match
            if self.mri_data.shape[dimension] > slice_idx:
                self.mri_slice = self.extract_slice(self.mri_data, dimension, slice_idx)
                log(f"Extracted MRI slice shape: {self.mri_slice.shape}")
            else:
                log(f"MRI dimension mismatch: MRI has shape {self.mri_data.shape}, "
                    f"cannot extract slice {slice_idx} from dimension {dimension}")
        
        # Create the visualization
        log("Setting up matplotlib figure...")
        plt.close('all')  # Close any existing plots
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))
        
        # Plot MRI with mask overlay
        if self.mri_slice is not None:
            log("Plotting MRI with mask overlay...")
            # Display MRI
            self.mri_img = axs[0].imshow(self.mri_slice, cmap='gray', interpolation='none')
            
            # Create binary mask for overlay
            binary_mask = (self.mask_slice > 0).astype(float)
            
            # Use a direct green color overlay that's clearly visible
            # We use the Greens colormap and adjust alpha for visibility
            self.overlay_img = axs[0].imshow(
                binary_mask,
                cmap='Greens',
                alpha=0.7,
                interpolation='none',
                vmin=0,
                vmax=1
            )
            
            log(f"Overlay created with shape: {binary_mask.shape}, non-zero elements: {np.count_nonzero(binary_mask)}")
            self.overlay_img.set_visible(self.show_mask)
            
            axs[0].set_title(f"MRI with Mask Overlay\n(Press 'c' to toggle mask)")
        else:
            log("No MRI slice available, showing placeholder...")
            axs[0].text(0.5, 0.5, "MRI not available", 
                        ha='center', va='center', transform=axs[0].transAxes)
            axs[0].set_title("MRI not available")
        
        # Plot mask alone - use a distinct colormap
        log("Plotting mask...")
        self.mask_img = axs[1].imshow(
            self.mask_slice, 
            cmap='viridis',  # Using viridis which has good green/yellow contrast
            interpolation='none',
            vmin=vmin,
            vmax=vmax
        )
        axs[1].set_title("Mask with Values\n(Hover to see pixel values)")
        
        # Format the axes
        for ax in axs:
            ax.set_axis_off()
        
        # Add threshold slider
        log("Adding threshold slider...")
        plt.subplots_adjust(bottom=0.2)
        threshold_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
        
        non_zero_mask = self.mask_slice[self.mask_slice > 0]
        if len(non_zero_mask) > 0:
            max_value = np.max(self.mask_slice)
            min_value = np.min(non_zero_mask)
            initial_threshold = min_value + (max_value - min_value) / 2
        else:
            max_value = 1.0
            min_value = 0.0
            initial_threshold = 0.5
        
        self.threshold_slider = Slider(
            threshold_ax, 'Threshold', 
            min_value, max_value, 
            valinit=initial_threshold,
            valstep=0.1
        )
        self.threshold_slider.on_changed(self.update_threshold)
        
        # Display information about threshold
        info_text = (
            f"Min: {min_value:.2f}, Max: {max_value:.2f}\n"
            f"Pixels below threshold will be set to zero"
        )
        plt.figtext(0.25, 0.05, info_text, ha='left')
        
        # Connect events
        log("Connecting event handlers...")
        fig.canvas.mpl_connect('key_press_event', self.toggle_mask)
        fig.canvas.mpl_connect('motion_notify_event', self.show_pixel_value)
        
        # Set window title
        mask_name = os.path.basename(self.mask_path)
        plt.suptitle(f"Threshold Evaluator - {mask_name}", fontsize=14)
        
        # Show the plot
        log("Displaying plot...")
        plt.tight_layout(rect=[0, 0.15, 1, 0.95])  # Adjust layout for the slider
        plt.show()
        log("Plot window closed")

    def run(self):
        """Main function to run the threshold evaluator"""
        try:
            log("Starting Simple Threshold Evaluator")
            
            # Load the data
            if not self.load_data():
                log("Data loading failed")
                return False
            
            # Visualize
            self.visualize()
            
            log("Threshold Evaluator completed successfully")
            return True
            
        except Exception as e:
            log(f"Unexpected error: {str(e)}")
            import traceback
            log(traceback.format_exc())
            return False

def select_files_gui():
    """Show GUI for file selection and return selected paths"""
    log("Opening file selection dialog...")
    
    # Create and show the file selection dialog
    dialog = FileSelectionDialog()
    result = dialog.show()
    
    return result['mask_path'], result['mri_path']

def main():
    parser = argparse.ArgumentParser(description='Simple Threshold Evaluator for NIfTI files')
    parser.add_argument('--mask', help='Path to mask file')
    parser.add_argument('--mri', help='Path to reference MRI file')
    parser.add_argument('--folder', help='Folder containing transformed mask files')
    parser.add_argument('--gui', action='store_true', help='Use GUI for file selection (default when run from master script)')
    
    args = parser.parse_args()
    
    mask_path = args.mask
    mri_path = args.mri
    
    # If no arguments provided or GUI flag set, use GUI selection
    if (not mask_path and not args.folder) or args.gui:
        log("Using GUI for file selection...")
        mask_path, mri_path = select_files_gui()
        
        if not mask_path:
            log("No mask file selected. Exiting.")
            return
    
    # Legacy command-line mode (kept for backward compatibility)
    elif args.folder or not mask_path:
        # Original command-line logic here (kept as fallback)
        if args.folder:
            folder = args.folder
        else:
            # Try to find Transformation_results folder in current directory
            current_dir = os.getcwd()
            folder = os.path.join(current_dir, "Transformation_results")
            if not os.path.exists(folder):
                log("No folder specified and Transformation_results not found. Using GUI...")
                mask_path, mri_path = select_files_gui()
                
                if not mask_path:
                    log("No mask file selected. Exiting.")
                    return
            else:
                # Use the original console-based selection (for backward compatibility)
                log("Using command-line file selection...")
                
                # Find mask files
                mask_files = []
                pattern = os.path.join(folder, "*mask_transformed.nii.gz")
                mask_files = glob.glob(pattern)
                
                if not mask_files:
                    pattern = os.path.join(folder, "*transformed*.nii.gz")
                    mask_files = glob.glob(pattern)
                
                if not mask_files:
                    log(f"No transformed mask files found in {folder}")
                    return
                
                # For command-line mode, just use the first file found
                mask_path = mask_files[0]
                log(f"Using first mask file found: {os.path.basename(mask_path)}")
                
                # Find MRI files
                base_dir = folder
                if "Transformation_results" in base_dir:
                    base_dir = os.path.dirname(base_dir)
                
                mri_files = []
                for ext in [".nii.gz", ".nii"]:
                    pattern = os.path.join(base_dir, f"*{ext}")
                    files = glob.glob(pattern)
                    mri_files.extend(files)
                
                if mri_files:
                    mri_path = mri_files[0]
                    log(f"Using first MRI file found: {os.path.basename(mri_path)}")
                else:
                    mri_path = None
                    log("No MRI files found")
    
    # Run the evaluator
    log(f"Using mask file: {mask_path}")
    log(f"Using MRI file: {mri_path if mri_path else 'None'}")
    
    evaluator = SimpleThresholdEvaluator(mask_path, mri_path)
    success = evaluator.run()
    
    if success:
        log("Threshold evaluation completed successfully")
    else:
        log("Threshold evaluation failed")

if __name__ == "__main__":
    main()
