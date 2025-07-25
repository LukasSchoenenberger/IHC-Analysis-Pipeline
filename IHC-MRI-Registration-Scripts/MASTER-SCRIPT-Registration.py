import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, BooleanVar, DoubleVar, StringVar, scrolledtext
import importlib.util
import subprocess
import threading
from datetime import datetime
from PIL import Image, ImageTk
import signal
import time
import webbrowser


class AnnotationManagerDialog:
    """Dialog for managing annotation mask names"""
    
    def __init__(self, parent, existing_annotations=None):
        self.parent = parent
        self.result = None
        self.annotations = existing_annotations[:] if existing_annotations else []
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Annotation Mask Manager")
        self.dialog.geometry("500x600")
        self.dialog.minsize(400, 500)
        self.dialog.resizable(True, True)
        
        # Make dialog modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (500 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (600 // 2)
        self.dialog.geometry(f"500x600+{x}+{y}")
        
        self.setup_dialog()
        
        # Handle window close
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)
        
        # Focus on entry field
        self.entry.focus_set()
    
    def setup_dialog(self):
        """Setup the dialog interface"""
        main_frame = ttk.Frame(self.dialog, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title and description
        title_label = ttk.Label(main_frame, text="Annotation Mask Manager", 
                               font=('Segoe UI', 14, 'bold'))
        title_label.pack(pady=(0, 10))
        
        desc_label = ttk.Label(main_frame, 
                              text="Add annotation mask names for QuPath script generation.\n"
                                   "These names should match your QuPath annotation classifications.",
                              font=('Segoe UI', 9),
                              wraplength=450)
        desc_label.pack(pady=(0, 15))
        
        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="Add New Annotation", padding="10")
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Entry and add button
        entry_frame = ttk.Frame(input_frame)
        entry_frame.pack(fill=tk.X)
        
        ttk.Label(entry_frame, text="Annotation Name:").pack(anchor='w')
        
        entry_button_frame = ttk.Frame(entry_frame)
        entry_button_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.entry_var = tk.StringVar()
        self.entry = ttk.Entry(entry_button_frame, textvariable=self.entry_var, font=('Segoe UI', 10))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        add_btn = ttk.Button(entry_button_frame, text="Add", command=self.add_annotation)
        add_btn.pack(side=tk.RIGHT)
        
        # Bind Enter key to add annotation
        self.entry.bind('<Return>', lambda e: self.add_annotation())
        
        # Current annotations section
        list_frame = ttk.LabelFrame(main_frame, text="Current Annotations", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Listbox with scrollbar
        listbox_frame = ttk.Frame(list_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create listbox and scrollbar
        self.listbox = tk.Listbox(listbox_frame, font=('Segoe UI', 10), height=10)
        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scrollbar.set)
        
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # List management buttons
        list_btn_frame = ttk.Frame(list_frame)
        list_btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        remove_btn = ttk.Button(list_btn_frame, text="Remove Selected", command=self.remove_annotation)
        remove_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        clear_btn = ttk.Button(list_btn_frame, text="Clear All", command=self.clear_all)
        clear_btn.pack(side=tk.LEFT)
        
        # Load sample annotations button
        load_sample_btn = ttk.Button(list_btn_frame, text="Load Sample", command=self.load_sample_annotations)
        load_sample_btn.pack(side=tk.RIGHT)
        
        # Bottom buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        cancel_btn = ttk.Button(button_frame, text="Cancel", command=self.on_cancel)
        cancel_btn.pack(side=tk.LEFT)
        
        ok_btn = ttk.Button(button_frame, text="Generate Script", command=self.on_ok)
        ok_btn.pack(side=tk.RIGHT)
        
        # Status label
        self.status_var = tk.StringVar()
        self.status_label = ttk.Label(button_frame, textvariable=self.status_var, 
                                     font=('Segoe UI', 9), foreground='gray')
        self.status_label.pack(side=tk.RIGHT, padx=(0, 10))
        
        # Populate existing annotations
        self.refresh_listbox()
        self.update_status()
    
    def add_annotation(self):
        """Add a new annotation to the list"""
        name = self.entry_var.get().strip()
        
        if not name:
            messagebox.showwarning("Empty Name", "Please enter an annotation name.", parent=self.dialog)
            return
        
        if name in self.annotations:
            messagebox.showwarning("Duplicate Name", f"'{name}' already exists in the list.", parent=self.dialog)
            return
        
        # Validate annotation name (basic validation)
        if any(char in name for char in ['\n', '\r', '\t']):
            messagebox.showerror("Invalid Name", "Annotation names cannot contain newlines or tabs.", parent=self.dialog)
            return
        
        self.annotations.append(name)
        self.refresh_listbox()
        self.entry_var.set("")  # Clear entry
        self.entry.focus_set()  # Keep focus on entry
        self.update_status()
    
    def remove_annotation(self):
        """Remove selected annotation from the list"""
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showinfo("No Selection", "Please select an annotation to remove.", parent=self.dialog)
            return
        
        index = selection[0]
        removed_name = self.annotations.pop(index)
        self.refresh_listbox()
        self.update_status()
        
        # Select next item if available
        if self.annotations:
            new_index = min(index, len(self.annotations) - 1)
            self.listbox.selection_set(new_index)
    
    def clear_all(self):
        """Clear all annotations"""
        if self.annotations:
            if messagebox.askyesno("Clear All", "Are you sure you want to remove all annotations?", parent=self.dialog):
                self.annotations.clear()
                self.refresh_listbox()
                self.update_status()
    
    def load_sample_annotations(self):
        """Load sample annotation names"""
        sample_annotations = ["WM", "WM_Lesion", "GM", "Surrounding_WM", "WML_Perilesion"]
        
        # Ask if user wants to replace or add to existing
        if self.annotations:
            response = messagebox.askyesnocancel(
                "Load Sample", 
                "Do you want to replace existing annotations (Yes) or add to them (No)?", 
                parent=self.dialog
            )
            if response is None:  # Cancel
                return
            elif response:  # Yes - replace
                self.annotations.clear()
        
        # Add sample annotations (avoiding duplicates)
        for annotation in sample_annotations:
            if annotation not in self.annotations:
                self.annotations.append(annotation)
        
        self.refresh_listbox()
        self.update_status()
    
    def refresh_listbox(self):
        """Refresh the listbox with current annotations"""
        self.listbox.delete(0, tk.END)
        for annotation in self.annotations:
            self.listbox.insert(tk.END, annotation)
    
    def update_status(self):
        """Update status message"""
        count = len(self.annotations)
        if count == 0:
            self.status_var.set("No annotations added")
        elif count == 1:
            self.status_var.set("1 annotation")
        else:
            self.status_var.set(f"{count} annotations")
    
    def on_ok(self):
        """Handle OK button click"""
        if not self.annotations:
            if not messagebox.askyesno("No Annotations", 
                                     "No annotations have been added. Continue anyway?", 
                                     parent=self.dialog):
                return
        
        self.result = self.annotations[:]
        self.dialog.destroy()
    
    def on_cancel(self):
        """Handle Cancel button click"""
        self.result = None
        self.dialog.destroy()


class RegistrationPipeline:
    def __init__(self, root):
        self.root = root
        self.root.title("IHC-MRI Registration Pipeline")
        self.root.geometry("1400x900")  # Larger window to match nice version
        self.root.minsize(1200, 800)
        
        # Variables
        self.binarize_masks = BooleanVar(value=False)
        self.apply_threshold = BooleanVar(value=False)
        self.threshold_value = DoubleVar(value=10.0)
        self.logs = []
        self.running_thread = None
        
        # For tracking current process
        self.current_step = None
        self.current_process = None  # Store current subprocess
        self.cancelled = False
        self.process_lock = threading.Lock()  # Thread safety for process access
        
        # Script paths (will be determined at runtime)
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.scripts = {
            "slice_matcher": os.path.join(self.script_dir, "Slice-Matching.py"),
            "linear_registration": os.path.join(self.script_dir, "Linear-Registration.py"),
            "nonlinear_registration": os.path.join(self.script_dir, "Non-Linear-Registration.py"),
            "transformer": os.path.join(self.script_dir, "Transformation.py"),
            "tif_to_nifti": os.path.join(self.script_dir, "TIFF-to-NIFTI-Conversion.py"),
            "threshold_evaluator": os.path.join(self.script_dir, "Threshold-Evaluation.py"),
            "segmentation_splitter": os.path.join(self.script_dir, "Segmentation-Splitting.py"),
            "mri_sequence_registration": os.path.join(self.script_dir, "MRI-to-MRI-Registration.py"),
            "download_masks_generator": os.path.join(self.script_dir, "Download-Masks-Generator.py")
        }
        
        # Setup enhanced GUI
        self.setup_enhanced_gui()
        
        # Check if scripts exist
        self.check_scripts()

    def setup_styling(self):
        """Set up modern styling for the application."""
        # Configure ttk styles for better appearance
        style = ttk.Style()
        
        # Try to use a modern theme
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')
        elif 'alt' in available_themes:
            style.theme_use('alt')
        
        # Define modern color scheme
        self.colors = {
            'primary': '#2E86AB',      # Professional blue
            'secondary': '#A23B72',    # Accent purple
            'accent': '#F18F01',       # Orange highlight
            'background': '#F8F9FA',   # Light gray background
            'surface': '#FFFFFF',      # White surfaces
            'text': '#2C3E50',         # Dark text
            'text_light': '#7F8C8D',   # Light gray text
            'success': '#27AE60',      # Green for success
            'error': '#E74C3C'         # Red for errors
        }
        
        # Configure enhanced button styles
        style.configure('Primary.TButton',
                       font=('Segoe UI', 10),
                       padding=(12, 6))
        
        style.configure('Accent.TButton',
                       font=('Segoe UI', 10, 'bold'),
                       padding=(12, 6))
        
        # Configure label styles
        style.configure('Title.TLabel',
                       font=('Segoe UI', 14, 'bold'),
                       foreground=self.colors['primary'])
        
        style.configure('Heading.TLabel',
                       font=('Segoe UI', 11, 'bold'),
                       foreground=self.colors['text'])
        
        style.configure('Info.TLabel',
                       font=('Segoe UI', 9),
                       foreground=self.colors['text_light'])
        
        # Configure root window background
        self.root.configure(bg=self.colors['background'])

    def setup_enhanced_gui(self):
        """Set up enhanced GUI with better aesthetics."""
        # Apply modern styling first
        self.setup_styling()
        
        # Main container with better padding
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Enhanced title section
        title_frame = ttk.Frame(main_container)
        title_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = ttk.Label(
            title_frame, 
            text="IHC-MRI Registration Pipeline", 
            style='Title.TLabel'
        )
        title_label.pack(pady=10)
        
        # Description
        desc_text = (
            "Multi-step registration pipeline for aligning IHC images to MRI space.\n"
            "Run individual steps in the indicated order for successful registration."
        )
        desc_label = ttk.Label(title_frame, text=desc_text, style='Info.TLabel', wraplength=800)
        desc_label.pack(pady=5)
        
        # MAIN CONTENT AREA - Split layout
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # LEFT SIDE - Processing content (70% width)
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # RIGHT SIDE - Information panel (30% width)
        right_frame = ttk.Frame(content_frame, width=620)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)  # Prevent auto-shrinking

        
        # CREATE INFORMATION PANEL
        self.create_info_panel(right_frame)
        
        # Create main processing area
        self.create_processing_area(left_frame)
        
        # ENHANCED LOG AND STATUS AREA
        self.create_log_area(main_container)

    def open_website(self, event=None):
        """Open the project website when logo is clicked."""
        import webbrowser
        import subprocess
        import os
        import time
        
        url = "https://dbe.unibas.ch/en/research/imaging-modelling-diagnosis/"
        
        self.log(f"Attempting to open: {url}")
        
        # SOLUTION: Change to user's home directory to avoid permission issues
        original_cwd = os.getcwd()
        try:
            # Change to a directory we know has proper permissions
            safe_directory = os.path.expanduser("~")  # User's home directory
            os.chdir(safe_directory)
            self.log(f"Changed working directory to: {safe_directory}")
            
            # Now try to open the browser
            result = webbrowser.open(url, new=2)  # new=2 for new tab
            
            if result:
                self.log("Browser opened successfully!")
                return True
            else:
                self.log("Browser command failed", error=True)
                
        except Exception as e:
            self.log(f"Error opening browser: {e}", error=True)
            
        finally:
            # Always restore the original working directory
            try:
                os.chdir(original_cwd)
                self.log(f"Restored working directory to: {original_cwd}")
            except Exception as e:
                self.log(f"Warning: Could not restore working directory: {e}", error=True)
        
        # Fallback: Try direct subprocess with explicit working directory
        try:
            self.log("Trying subprocess with explicit working directory...")
            result = subprocess.Popen(
                ["firefox", url],
                cwd=os.path.expanduser("~"),  # Run from home directory
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            self.log(f"Firefox subprocess started successfully with PID: {result.pid}")
            return True
            
        except Exception as e:
            self.log(f"Subprocess method failed: {e}", error=True)
        
        # If all else fails, show URL to user
        try:
            messagebox.showinfo(
                "Website URL", 
                f"Please copy this URL to your browser:\n\n{url}",
                parent=self.root
            )
            self.log("Displayed URL to user for manual opening")
        except Exception as e:
            self.log(f"Could not show URL dialog: {e}", error=True)
        
        return False

    def create_info_panel(self, parent):
        """Create an informational panel with pipeline overview."""
        info_frame = ttk.LabelFrame(parent, text="Pipeline Overview", padding=(15, 15))
        info_frame.pack(fill='both', expand=True)

        # Create canvas and scrollbar
        canvas = tk.Canvas(info_frame, bg=self.colors['background'], highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        # Create a frame inside the canvas
        scrollable_frame = ttk.Frame(canvas)

        # Resize inner frame when canvas size changes - FORCE IT TO FILL WIDTH AND HEIGHT
        def resize_inner_frame(event):
            # Make the scrollable frame fill the entire canvas width
            canvas.itemconfig("inner", width=event.width)
            # Also try to make it fill the height if content is smaller
            canvas_height = event.height
            required_height = scrollable_frame.winfo_reqheight()
            if required_height < canvas_height:
                canvas.itemconfig("inner", height=canvas_height)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", resize_inner_frame)

        # Create window in canvas
        canvas_frame = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", tags="inner")

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Add content to scrollable_frame
        title_label = ttk.Label(scrollable_frame, text="Registration Workflow", style='Title.TLabel')
        title_label.pack(pady=(0, 15))

        workflow_text = (
            "REGISTRATION PIPELINE:\n"
            "• Slice Matching: Find best matching MRI slice\n"
            "• Linear Registration: Apply 2D affine registration\n"
            "• Non-Linear Registration: Apply deformable registration\n"
            "• Apply Transformation: Apply final transformations to images\n\n"
            "UTILITY TOOLS:\n"
            "• TIF to NIfTI: Convert image formats\n"
            "• Threshold Evaluation: Optimize intensity thresholds\n"
            "• Mask Splitting: Compartmentalize annotation masks\n"
            "• MRI Registration: Co-register MRI sequences\n"
            "• Generate Download Script: Create QuPath annotation download script"
        )

        workflow_label = ttk.Label(scrollable_frame, text=workflow_text,
                                   font=('Segoe UI', 9),
                                   justify='left', wraplength=530)
        workflow_label.pack(padx=10, pady=10, anchor='w')

        session_frame = ttk.LabelFrame(scrollable_frame, text="Session Info", padding=(10, 5))
        session_frame.pack(fill='x', padx=10, pady=10)

        script_count = len([f for f in os.listdir(self.script_dir) if f.endswith('.py')]) if os.path.exists(self.script_dir) else 0
        session_info = f"""Started: {datetime.now().strftime('%H:%M:%S')}
        Working Dir: {os.path.basename(self.script_dir)}
        Scripts Available: {script_count}
        Pipeline Version: 1.1"""

        session_label = ttk.Label(session_frame, text=session_info,
                                  font=('Segoe UI', 8),
                                  foreground=self.colors['text_light'])
        session_label.pack(anchor='w')
        
        # Insert logo image if available
        image_path = os.path.join(self.script_dir, "logo.png")
        if os.path.exists(image_path):
            try:
                pil_img = Image.open(image_path)
                resample = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.ANTIALIAS
                pil_img = pil_img.resize((500, 200), resample)
                self.info_image = ImageTk.PhotoImage(pil_img)

                # Create a frame to center the logo properly - MAKE IT EXPAND TO FILL REMAINING SPACE
                logo_frame = ttk.Frame(scrollable_frame)
                logo_frame.pack(fill='both', expand=True, padx=10, pady=(15, 15))  # Added expand=True
                
                # Use grid to center the image in the frame
                logo_frame.grid_columnconfigure(0, weight=1)
                logo_frame.grid_rowconfigure(0, weight=1)  # Added this to center vertically too
                image_label = ttk.Label(logo_frame, image=self.info_image, cursor="hand2")
                image_label.grid(row=0, column=0)

                # Make the logo clickable
                image_label.bind("<Button-1>", self.open_website)

                # Optional: Add hover effect
                def on_enter(event):
                    image_label.config(relief="raised", borderwidth=2)
                def on_leave(event):
                    image_label.config(relief="flat", borderwidth=0)

                image_label.bind("<Enter>", on_enter)
                image_label.bind("<Leave>", on_leave)
            except Exception as e:
                self.log(f"Failed to load logo.png: {e}", error=True)
        else:
            self.log("logo.png not found in script directory.")

    def create_processing_area(self, parent):
        """Create the main processing area with enhanced styling."""
        # Main scrollable frame
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        
        # Individual pipeline steps
        steps_frame = self.create_enhanced_section_frame(
            scrollable_frame,
            "Individual Pipeline Steps",
            "Run specific steps of the registration pipeline independently"
        )
        
        step_buttons = [
            ("1. Run Slice Matcher", self.run_slice_matcher, "Find corresponding tissue sections between modalities"),
            ("2. Run Linear Registration", self.run_linear_registration, "Apply rigid body transformations for initial alignment"),
            ("3. Run Non-Linear Registration", self.run_nonlinear_registration, "Apply deformable registration for precise alignment"),
            ("4. Run Transformer", self.run_transformer, "Apply final transformations to other images and masks")
        ]
        
        for text, command, description in step_buttons:
            self.create_enhanced_button(steps_frame, text, command, description=description)
        
        # Utility tools section
        utility_frame = self.create_enhanced_section_frame(
            scrollable_frame,
            "Utility Tools",
            "Tools for optional pipeline steps"
        )
        
        utility_buttons = [
            ("Convert TIF to NIfTI", self.run_tif_to_nifti, "Convert TIFF annotation masks to NIfTI format"),
            ("Evaluate Threshold", self.run_threshold_evaluator, "Evaluate intensity threshold for mask correction"),
            ("Split Annotation Masks", self.run_segmentation_splitter, "Split annotation masks into smaller compartments"),
            ("Register MRI Sequences", self.run_mri_sequence_registration, "Register between different MRI sequence types"),
            ("Generate Download Script", self.run_download_script_generator, "Create QuPath script for downloading annotation masks")
        ]
        
        for text, command, description in utility_buttons:
            self.create_enhanced_button(utility_frame, text, command, description=description)
        
        # Options section
        options_frame = self.create_enhanced_section_frame(
            scrollable_frame,
            "Processing Options",
            "Configure transformation and post-processing parameters"
        )
        
        # Binarize option
        binarize_frame = ttk.Frame(options_frame)
        binarize_frame.pack(fill="x", pady=5)
        ttk.Checkbutton(
            binarize_frame, 
            text="Binarize images after transformation",
            variable=self.binarize_masks
        ).pack(anchor='w')
        ttk.Label(binarize_frame, text="Convert transformed masks to binary format", 
                 style='Info.TLabel').pack(anchor='w', padx=(20, 0))
        
        # Threshold options
        threshold_frame = ttk.Frame(options_frame)
        threshold_frame.pack(fill=tk.X, pady=5)
        
        threshold_check_frame = ttk.Frame(threshold_frame)
        threshold_check_frame.pack(fill="x")
        
        # Threshold checkbox
        ttk.Checkbutton(
            threshold_check_frame,
            text="Apply intensity threshold",
            variable=self.apply_threshold
        ).pack(side=tk.LEFT)
        
        # Threshold value
        ttk.Label(threshold_check_frame, text="Value:").pack(side=tk.LEFT, padx=(10, 5))
        threshold_entry = ttk.Entry(
            threshold_check_frame,
            textvariable=self.threshold_value,
            width=8
        )
        threshold_entry.pack(side=tk.LEFT)
        
        desc_frame = ttk.Frame(threshold_frame)
        desc_frame.pack(fill="x")
        ttk.Label(desc_frame, text="Apply intensity threshold to remove interpolation artifacts", 
                 style='Info.TLabel').pack(anchor='w', padx=(20, 0))

    def create_enhanced_section_frame(self, parent, title, description=""):
        """Create a labeled frame section"""
        # Main section frame
        section_frame = ttk.LabelFrame(parent, text="", padding=(15, 10))
        section_frame.pack(fill="x", expand=False, padx=15, pady=8)
        
        # Header frame with title and description
        header_frame = ttk.Frame(section_frame)
        header_frame.pack(fill="x", pady=(0, 10))
        
        # Title
        title_label = ttk.Label(header_frame, text=title, style='Heading.TLabel')
        title_label.pack(anchor="w")
        
        # Description if provided
        if description:
            desc_label = ttk.Label(header_frame, text=description, style='Info.TLabel', wraplength=600)
            desc_label.pack(anchor="w", pady=(2, 0))
        
        # Content frame for buttons/controls
        content_frame = ttk.Frame(section_frame)
        content_frame.pack(fill="x")
        
        return content_frame

    def create_enhanced_button(self, parent, text, command, style='Primary.TButton', description=""):
        """Create a button with optional description."""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill="x", pady=3)
        
        # Main button
        button = ttk.Button(button_frame, text=text, command=command, style=style)
        button.pack(side="left", padx=(0, 10))
        
        # Description label if provided
        if description:
            desc_label = ttk.Label(button_frame, text=description, style='Info.TLabel', wraplength=500)
            desc_label.pack(side="left", anchor="w")
        
        return button

    def create_log_area(self, parent):
        """Create enhanced log and status area."""
        bottom_frame = ttk.Frame(parent)
        bottom_frame.pack(fill=tk.X, pady=(15, 0))
        
        # Enhanced log display with dark theme
        log_frame = ttk.LabelFrame(bottom_frame, text="Processing Log", padding=(10, 5))
        log_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=8,
                                                 font=('Consolas', 9),
                                                 bg='#2C3E50', fg='#ECF0F1',
                                                 insertbackground='#ECF0F1',
                                                 selectbackground='#3498DB')
        self.log_text.pack(fill=tk.X, padx=5, pady=5)
        self.log_text.config(state=tk.DISABLED)
        
        # Enhanced status bar
        status_frame = ttk.Frame(bottom_frame)
        status_frame.pack(fill=tk.X)
        
        # Status with visual indicator
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Select processing steps to begin")
        
        status_label_frame = ttk.LabelFrame(status_frame, text="Status")
        status_label_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.status_label = ttk.Label(status_label_frame, textvariable=self.status_var,
                                     font=('Segoe UI', 10), padding=(10, 8))
        self.status_label.pack(fill=tk.X)
        
        # Enhanced progress bar
        progress_frame = ttk.LabelFrame(status_frame, text="Progress")
        progress_frame.pack(side=tk.RIGHT)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                           mode='indeterminate', length=250)
        self.progress_bar.pack(padx=10, pady=8)
        
        # Cancel button with enhanced styling
        self.cancel_btn = ttk.Button(
            progress_frame, 
            text="Cancel", 
            command=self.cancel_operation,
            state=tk.DISABLED,
            style='Accent.TButton'
        )
        self.cancel_btn.pack(pady=5)
    
    def check_scripts(self):
        """Check if all required scripts exist"""
        missing_scripts = []
        for name, path in self.scripts.items():
            if not os.path.exists(path):
                missing_scripts.append(f"{name}: {path}")
        
        if missing_scripts:
            error_message = "The following scripts were not found:\n\n" + "\n".join(missing_scripts)
            self.log(error_message, error=True)
            messagebox.showerror("Missing Scripts", error_message)
    
    def log(self, message, error=False):
        """Enhanced log update with better styling - thread safe."""
        # Use after to ensure this runs in the main thread
        self.root.after(0, self._log_main_thread, message, error)
    
    def _log_main_thread(self, message, error=False):
        """Enhanced log update with better styling - main thread only."""
        try:
            # Check if widgets still exist
            if not hasattr(self, 'log_text') or not self.log_text.winfo_exists():
                return
                
            # Add timestamp
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            self.log_text.config(state=tk.NORMAL)
            
            if error:
                # Red color for errors
                self.log_text.insert(tk.END, f"[{timestamp}] ERROR: {message}\n")
                self.log_text.tag_add("error", "end-2l linestart", "end-1l lineend")
                self.log_text.tag_configure("error", foreground="#E74C3C", font=('Consolas', 9, 'bold'))
            elif "successfully" in message.lower() or "completed" in message.lower():
                # Green color for success
                self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
                self.log_text.tag_add("success", "end-2l linestart", "end-1l lineend")
                self.log_text.tag_configure("success", foreground="#27AE60", font=('Consolas', 9))
            elif "progress:" in message.lower() or "processing" in message.lower() or "running" in message.lower():
                # Blue color for progress
                self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
                self.log_text.tag_add("progress", "end-2l linestart", "end-1l lineend")  
                self.log_text.tag_configure("progress", foreground="#3498DB", font=('Consolas', 9))
            else:
                # Default white color
                self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
            self.logs.append(message)
            print(message)  # Also print to console
            self.root.update_idletasks()
        except tk.TclError:
            # Widget was destroyed, ignore
            pass
        except Exception as e:
            # Print error but don't crash
            print(f"Log error: {e}")
    
    def update_status(self, message, progress=None):
        """Update status message and progress bar - thread safe"""
        self.root.after(0, self._update_status_main_thread, message, progress)
    
    def _update_status_main_thread(self, message, progress=None):
        """Update status message and progress bar - main thread only"""
        try:
            if hasattr(self, 'status_var') and self.status_var:
                self.status_var.set(message)
            if progress is not None and hasattr(self, 'progress_var') and self.progress_var:
                self.progress_var.set(progress)
        except tk.TclError:
            # Widget was destroyed, ignore
            pass
        except Exception as e:
            print(f"Status update error: {e}")
    
    def enable_buttons(self, enabled=True):
        """Enable or disable buttons during processing - thread safe"""
        self.root.after(0, self._enable_buttons_main_thread, enabled)
    
    def _enable_buttons_main_thread(self, enabled=True):
        """Enable or disable buttons during processing - main thread only"""
        try:
            # Cancel button is enabled only when processing
            if hasattr(self, 'cancel_btn') and self.cancel_btn.winfo_exists():
                self.cancel_btn.configure(state=tk.DISABLED if enabled else tk.NORMAL)
            
            # Handle progress bar
            if hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists():
                if not enabled:
                    self.progress_bar.start(10)
                else:
                    self.progress_bar.stop()
        except tk.TclError:
            # Widget was destroyed, ignore
            pass
        except Exception as e:
            print(f"Button enable error: {e}")
    
    def cancel_operation(self):
        """Cancel the current operation - improved version"""
        self.log("Cancel requested by user...")
        
        with self.process_lock:
            # Set cancellation flag
            self.cancelled = True
            
            # Terminate the current subprocess if it exists
            if self.current_process is not None:
                self.log("Terminating current subprocess...")
                try:
                    # Try graceful termination first
                    if self.current_process.poll() is None:  # Process is still running
                        self.current_process.terminate()
                        
                        # Give it a moment to terminate gracefully
                        try:
                            self.current_process.wait(timeout=3)
                            self.log("Process terminated gracefully")
                        except subprocess.TimeoutExpired:
                            # Force kill if it doesn't terminate
                            self.log("Process didn't terminate gracefully, forcing kill...")
                            self.current_process.kill()
                            self.current_process.wait()
                            self.log("Process forcefully killed")
                    
                    self.current_process = None
                    
                except Exception as e:
                    self.log(f"Error terminating process: {e}", error=True)
        
        # Update UI
        self.update_status("Cancelling operation...", 0)
    
    def run_script(self, script_path, args=None):
        """Run a Python script as a subprocess with improved cancellation handling"""
        cmd = [sys.executable, script_path]
        if args:
            cmd.extend(args)
        
        self.log(f"Running: {' '.join(cmd)}")
        
        try:
            with self.process_lock:
                if self.cancelled:
                    return False
                
                self.current_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    bufsize=1,
                    # On Windows, use CREATE_NEW_PROCESS_GROUP for better signal handling
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
                )
            
            # Read output with frequent cancellation checks
            output_lines = []
            error_lines = []
            
            while True:
                # Check for cancellation more frequently
                if self.cancelled:
                    self.log("Operation cancelled by user")
                    return False
                
                # Check if process is still running
                poll_result = self.current_process.poll()
                if poll_result is not None:
                    # Process has finished, read any remaining output
                    remaining_stdout, remaining_stderr = self.current_process.communicate()
                    if remaining_stdout:
                        for line in remaining_stdout.splitlines():
                            if line.strip():
                                self.log(line.strip())
                    if remaining_stderr:
                        for line in remaining_stderr.splitlines():
                            if line.strip():
                                self.log(f"stderr: {line.strip()}", error=True)
                    break
                
                # Read available output without blocking for too long
                try:
                    # Use a short timeout to avoid blocking
                    import select
                    
                    if hasattr(select, 'select'):  # Unix-like systems
                        ready, _, _ = select.select([self.current_process.stdout], [], [], 0.1)
                        if ready:
                            line = self.current_process.stdout.readline()
                            if line:
                                self.log(line.strip())
                    else:  # Windows - just read with a small delay
                        try:
                            # Try to read a line with a timeout simulation
                            line = self.current_process.stdout.readline()
                            if line:
                                self.log(line.strip())
                            else:
                                time.sleep(0.1)  # Small delay to prevent busy waiting
                        except:
                            time.sleep(0.1)
                            
                except Exception as e:
                    # If there's an error reading, just continue
                    time.sleep(0.1)
            
            # Clean up
            with self.process_lock:
                return_code = self.current_process.returncode if self.current_process else -1
                self.current_process = None
            
            if self.cancelled:
                return False
                
            return return_code == 0
            
        except Exception as e:
            self.log(f"Error running script: {e}", error=True)
            with self.process_lock:
                self.current_process = None
            return False
    
    def run_in_thread(self, func, *args, **kwargs):
        """Run a function in a separate thread"""
        self.cancelled = False
        self.enable_buttons(False)
        
        # Define the thread target function
        def thread_target():
            try:
                result = func(*args, **kwargs)
                if not self.cancelled:
                    self.root.after(100, lambda: self.on_thread_complete(result))
                else:
                    self.root.after(100, lambda: self.on_thread_complete(False, cancelled=True))
            except Exception as e:
                self.log(f"Error: {str(e)}", error=True)
                import traceback
                self.log(traceback.format_exc(), error=True)
                self.root.after(100, lambda: self.on_thread_complete(False, error=str(e)))
        
        # Start the thread
        self.running_thread = threading.Thread(target=thread_target)
        self.running_thread.daemon = True
        self.running_thread.start()
    
    def on_thread_complete(self, success, cancelled=False, error=None):
        """Handle thread completion"""
        self.enable_buttons(True)
        
        # Clean up process reference
        with self.process_lock:
            self.current_process = None
            self.cancelled = False
        
        if cancelled:
            self.update_status("Operation cancelled", 0)
            self.log("Operation was cancelled")
        elif success:
            self.update_status("Operation completed successfully", 100)
            self.log("Operation completed successfully")
        else:
            if error:
                self.update_status(f"Error: {error}", 0)
            else:
                self.update_status("Operation failed", 0)
            self.log("Operation failed", error=True)
    
    def import_script_as_module(self, script_path, module_name):
        """Import a script as a module to call its functions directly"""
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    def load_existing_annotations(self):
        """Load existing annotations from annotation_list.txt if it exists"""
        annotation_file = os.path.join(self.script_dir, "annotation_list.txt")
        annotations = []
        
        if os.path.exists(annotation_file):
            try:
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    annotations = [line.strip() for line in f.readlines() if line.strip()]
                self.log(f"Loaded {len(annotations)} existing annotations from annotation_list.txt")
            except Exception as e:
                self.log(f"Error reading existing annotation_list.txt: {e}", error=True)
        
        return annotations
    
    def save_annotation_list(self, annotations):
        """Save annotation list to annotation_list.txt"""
        annotation_file = os.path.join(self.script_dir, "annotation_list.txt")
        
        try:
            with open(annotation_file, 'w', encoding='utf-8') as f:
                for annotation in annotations:
                    f.write(annotation + "\n")
            
            self.log(f"Saved {len(annotations)} annotations to annotation_list.txt")
            self.log(f"File saved to: {annotation_file}")
            return True
            
        except Exception as e:
            self.log(f"Error saving annotation_list.txt: {e}", error=True)
            messagebox.showerror("Save Error", f"Could not save annotation list:\n{e}", parent=self.root)
            return False
    
    def run_download_script_generator(self):
        """Run the download script generator with annotation management"""
        self.current_step = "download_script_generator"
        self.update_status("Managing annotation list...", 25)
        self.log("Starting download script generator...")
        
        # Load existing annotations
        existing_annotations = self.load_existing_annotations()
        
        # Open annotation manager dialog
        dialog = AnnotationManagerDialog(self.root, existing_annotations)
        self.root.wait_window(dialog.dialog)
        
        # Check if user confirmed or cancelled
        if dialog.result is None:
            self.log("Annotation management cancelled by user")
            self.update_status("Ready - Select processing steps to begin", 0)
            return
        
        annotations = dialog.result
        
        if not annotations:
            self.log("No annotations provided - creating empty annotation_list.txt")
        
        # Save the annotation list
        if not self.save_annotation_list(annotations):
            self.update_status("Failed to save annotation list", 0)
            return
        
        # Now run the download masks generator script
        self.update_status("Running Download Masks Generator...", 75)
        self.log("Launching Download-Masks-Generator.py...")
        
        def run_generator():
            return self.run_script(self.scripts["download_masks_generator"])
        
        self.run_in_thread(run_generator)
    
    def run_slice_matcher(self):
        """Run the slice matcher step"""
        self.current_step = "slice_matcher"
        self.update_status("Running slice matcher...", 25)
        self.log("Starting slice matcher step...")
        self.run_in_thread(self.run_script, self.scripts["slice_matcher"])
    
    def run_linear_registration(self):
        """Run the linear registration step"""
        self.current_step = "linear_registration"
        self.update_status("Running linear registration...", 50)
        self.log("Starting linear registration step...")
        self.run_in_thread(self.run_script, self.scripts["linear_registration"])
    
    def run_nonlinear_registration(self):
        """Run the non-linear registration step"""
        self.current_step = "nonlinear_registration"
        self.update_status("Running non-linear registration...", 75)
        self.log("Starting non-linear registration step...")
        self.run_in_thread(self.run_script, self.scripts["nonlinear_registration"])
    
    def run_transformer(self):
        """Run the transformer step"""
        self.current_step = "transformer"
        self.update_status("Running transformer...", 90)
        self.log("Starting transformer step...")
        
        # Add arguments based on selected options
        args = []
        
        # Add --binarize flag if option is checked
        if self.binarize_masks.get():
            args.append("--binarize")
        
        # Add --threshold value if option is checked
        if self.apply_threshold.get():
            try:
                threshold_value = float(self.threshold_value.get())
                args.append("--threshold")
                args.append(str(threshold_value))
                self.log(f"Using intensity threshold: {threshold_value}")
            except ValueError:
                self.log("Invalid threshold value. Using default.")
        
        # Remove the --intermediate argument since it's not supported by Transformation.py
        # The script already saves only the final result by default
        
        self.run_in_thread(self.run_script, self.scripts["transformer"], args)

    
    def run_tif_to_nifti(self):
        """Run the TIF to NIfTI converter step"""
        self.current_step = "tif_to_nifti"
        self.update_status("Running TIF to NIfTI converter...", 50)
        self.log("Starting TIF to NIfTI annotation mask conversion...")
        self.run_in_thread(self.run_script, self.scripts["tif_to_nifti"])
    
    def run_threshold_evaluator(self):
        """Run the threshold evaluator tool"""
        self.current_step = "threshold_evaluator"
        self.update_status("Running threshold evaluator...", 50)
        self.log("Starting threshold evaluator tool...")
        self.run_in_thread(self.run_script, self.scripts["threshold_evaluator"])
    
    def run_mri_sequence_registration(self):
        """Run the MRI sequence registration utility"""
        self.current_step = "mri_sequence_registration"
        self.update_status("Running MRI sequence registration...", 50)
        self.log("Starting MRI sequence registration utility...")
        self.run_in_thread(self.run_script, self.scripts["mri_sequence_registration"])
    
    def run_segmentation_splitter(self):
        """Run the segmentation splitter utility"""
        self.current_step = "segmentation_splitter"
        self.update_status("Running segmentation splitter...", 50)
        self.log("Starting segmentation splitter utility...")
        
        # The script will now use GUI file selection automatically
        self.log("Opening file selection dialog for segmentation splitting...")
        self.log("Results will be saved to: Splitted_annotation_masks")
        
        self.run_in_thread(self.run_script, self.scripts["segmentation_splitter"])
    
    def run_complete_pipeline(self):
        """Run the complete pipeline in sequence"""
        self.update_status("Running complete pipeline...", 10)
        self.log("Starting complete pipeline...")
        
        def run_pipeline_thread():
            # Run each step in sequence
            if not self.run_script(self.scripts["slice_matcher"]):
                return False
                
            if self.cancelled:
                return False
                
            self.update_status("Slice matcher completed, running linear registration...", 25)
            if not self.run_script(self.scripts["linear_registration"]):
                return False
                
            if self.cancelled:
                return False
                
            self.update_status("Linear registration completed, running non-linear registration...", 50)
            if not self.run_script(self.scripts["nonlinear_registration"]):
                return False
                
            if self.cancelled:
                return False
                
            self.update_status("Non-linear registration completed, running transformer...", 75)
            
            # Add arguments based on selected options
            transformer_args = []
            
            # Add --binarize flag if option is checked
            if self.binarize_masks.get():
                transformer_args.append("--binarize")
            
            # Add --threshold value if option is checked
            if self.apply_threshold.get():
                try:
                    threshold_value = float(self.threshold_value.get())
                    transformer_args.append("--threshold")
                    transformer_args.append(str(threshold_value))
                    self.log(f"Using intensity threshold: {threshold_value}")
                except ValueError:
                    self.log("Invalid threshold value. Using default.")
            
            # Remove the --intermediate argument since it's not supported by Transformation.py
            # The script already saves only the final result by default
                
            if not self.run_script(self.scripts["transformer"], transformer_args):
                return False
            
            return True
            
        self.run_in_thread(run_pipeline_thread)

# Main application
def main():
    root = tk.Tk()
    app = RegistrationPipeline(root)
    root.mainloop()

if __name__ == "__main__":
    main()
