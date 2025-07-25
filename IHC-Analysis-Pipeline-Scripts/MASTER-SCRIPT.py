#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Script for IHC Pipeline GUI
----------------------------------
This script serves as the entry point for the IHC Pipeline GUI application.
It creates a GUI with three main pages (Preprocessing, Parameter-Estimation, Detection)
and integrates all individual processing scripts.
"""

import os
import sys
import shutil
import glob
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import configparser
import logging
import threading
import queue
import time
from datetime import datetime
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
import webbrowser

# Optional drag and drop support
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DRAG_DROP_AVAILABLE = True
except ImportError:
    DRAG_DROP_AVAILABLE = False
    print("Warning: tkinterdnd2 not available. Drag and drop functionality will be disabled.")

def setup_directories_and_files():
    """Set up required directories and relocate script files."""
    # Get the base directory (where this script is located)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define required directories
    directories = [
        os.path.join(base_dir, "Data"),
        os.path.join(base_dir, "Parameters"), 
        os.path.join(base_dir, "Results"),
        os.path.join(base_dir, "Scripts"),
        os.path.join(base_dir, "Logs")
    ]
    
    # Create directories if they don't exist
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    # Get the name of this script
    master_script_name = os.path.basename(__file__)
    
    # Find all Python and YAML files in the base directory
    python_files = glob.glob(os.path.join(base_dir, "*.py"))
    yaml_files = glob.glob(os.path.join(base_dir, "*.yml"))
    yaml_files.extend(glob.glob(os.path.join(base_dir, "*.yaml")))
    
    # Combine the lists
    script_files = python_files + yaml_files
    
    # Move files to the Scripts directory, except the master script
    scripts_dir = os.path.join(base_dir, "Scripts")
    for file_path in script_files:
        file_name = os.path.basename(file_path)
        if file_name != master_script_name:
            destination = os.path.join(scripts_dir, file_name)
            try:
                shutil.move(file_path, destination)
                print(f"Moved {file_name} to Scripts directory")
            except Exception as e:
                print(f"Error moving {file_name}: {str(e)}")

# Run setup before initializing the application
setup_directories_and_files()

# Configure logging
log_dir = "Logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"ihc_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("IHC_Pipeline")

class StainNormalizationPercentilesDialog:
    """Dialog for selecting percentiles for stain normalization"""
    
    def __init__(self, parent):
        """
        Initialize the dialog
        
        Args:
            parent: Parent window
        """
        self.result = None  # Will contain the selected percentiles if OK is clicked
        self.parent = parent
        
        # Create a new top-level dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.dialog.title("Select Stain Normalization Percentiles")
        self.dialog.geometry("600x400")
        self.dialog.minsize(550, 350)
        
        # Prevent closing the dialog directly
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)
        
        # Create the main layout
        self.create_widgets()
        
        # Center the dialog relative to the parent window
        self.center_dialog()
        
        # Wait for the dialog to be closed before returning
        parent.wait_window(self.dialog)
    
    def create_widgets(self):
        """Create dialog widgets"""
        # Create main frame with padding
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title and instructions
        title_label = ttk.Label(main_frame, text="Select Percentiles for Stain Normalization", 
                               font=("TkDefaultFont", 12, "bold"))
        title_label.pack(pady=(0, 15))
        
        instruction_text = ("Choose which percentile to use for normalizing each stain type.\n"
                           "Higher percentiles (95-99.9) capture more of the intensity distribution,\n"
                           "while lower percentiles (1-75) focus on the core stain values.")
        
        instruction_label = ttk.Label(main_frame, text=instruction_text, wraplength=500, justify=tk.CENTER)
        instruction_label.pack(pady=(0, 20))
        
        # Create selection frame
        selection_frame = ttk.LabelFrame(main_frame, text="Percentile Selection")
        selection_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Available percentile options
        percentile_options = [
            "percentile_1", "percentile_50", "percentile_75", 
            "percentile_95", "percentile_97.5", "percentile_99", "percentile_99.9"
        ]
        
        # Create variables for each stain type
        self.nuclei_var = tk.StringVar(value="percentile_95")
        self.myelin_var = tk.StringVar(value="percentile_95")
        self.microglia_var = tk.StringVar(value="percentile_95")
        
        # Create selection rows
        row_padding = 15
        
        # Nuclei selection
        nuclei_frame = ttk.Frame(selection_frame)
        nuclei_frame.pack(fill=tk.X, padx=20, pady=row_padding)
        
        ttk.Label(nuclei_frame, text="Nuclei (H&E):", width=15, font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT)
        nuclei_combo = ttk.Combobox(nuclei_frame, textvariable=self.nuclei_var, 
                                   values=percentile_options, state="readonly", width=15)
        nuclei_combo.pack(side=tk.LEFT, padx=10)
        ttk.Label(nuclei_frame, text="Nuclear staining intensity distribution", 
                 font=("TkDefaultFont", 9, "italic")).pack(side=tk.LEFT, padx=10)
        
        # Myelin selection
        myelin_frame = ttk.Frame(selection_frame)
        myelin_frame.pack(fill=tk.X, padx=20, pady=row_padding)
        
        ttk.Label(myelin_frame, text="Myelin (Blue):", width=15, font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT)
        myelin_combo = ttk.Combobox(myelin_frame, textvariable=self.myelin_var, 
                                   values=percentile_options, state="readonly", width=15)
        myelin_combo.pack(side=tk.LEFT, padx=10)
        ttk.Label(myelin_frame, text="Blue staining intensity distribution", 
                 font=("TkDefaultFont", 9, "italic")).pack(side=tk.LEFT, padx=10)
        
        # Microglia selection
        microglia_frame = ttk.Frame(selection_frame)
        microglia_frame.pack(fill=tk.X, padx=20, pady=row_padding)
        
        ttk.Label(microglia_frame, text="Microglia (Brown):", width=15, font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT)
        microglia_combo = ttk.Combobox(microglia_frame, textvariable=self.microglia_var, 
                                      values=percentile_options, state="readonly", width=15)
        microglia_combo.pack(side=tk.LEFT, padx=10)
        ttk.Label(microglia_frame, text="Brown staining intensity distribution", 
                 font=("TkDefaultFont", 9, "italic")).pack(side=tk.LEFT, padx=10)
        
        # Recommendation text
        recommendation_text = ("Recommendation: percentile_95 works well for most cases,\n"
                              "providing good normalization without being too sensitive to outliers.")
        recommendation_label = ttk.Label(main_frame, text=recommendation_text, 
                                        wraplength=500, justify=tk.CENTER,
                                        font=("TkDefaultFont", 9), foreground="blue")
        recommendation_label.pack(pady=(10, 20))
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        # Add buttons
        ttk.Button(button_frame, text="OK", command=self.on_ok, width=12).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.on_cancel, width=12).pack(side=tk.RIGHT, padx=5)
        
        # Add a "Reset to Default" button
        ttk.Button(button_frame, text="Reset to Default", command=self.reset_to_default, width=15).pack(side=tk.LEFT)
    
    def reset_to_default(self):
        """Reset all selections to default (percentile_95)"""
        self.nuclei_var.set("percentile_95")
        self.myelin_var.set("percentile_95")
        self.microglia_var.set("percentile_95")
    
    def center_dialog(self):
        """Center the dialog window on the parent"""
        self.dialog.update_idletasks()
        
        # Get parent and dialog dimensions
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        parent_x = self.parent.winfo_rootx()
        parent_y = self.parent.winfo_rooty()
        
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        # Calculate center position
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        # Set position
        self.dialog.geometry(f"+{x}+{y}")
    
    def on_ok(self):
        """Handle OK button click"""
        # Get selected percentiles
        percentiles = {
            'nuclei': self.nuclei_var.get(),
            'myelin': self.myelin_var.get(),
            'microglia': self.microglia_var.get()
        }
        
        self.result = percentiles
        self.dialog.destroy()
    
    def on_cancel(self):
        """Handle Cancel button click or window close"""
        self.result = None
        self.dialog.destroy()

class ImportReferenceStainPercentilesDialog:
    """Dialog for importing reference stain percentiles either from file or manual entry"""
    
    def __init__(self, parent, parameters_dir):
        """
        Initialize the dialog
        
        Args:
            parent: Parent window
            parameters_dir: Directory where the reference file will be saved
        """
        self.result = False  # Will be True if import was successful
        self.parent = parent
        self.parameters_dir = parameters_dir
        
        # Create a new top-level dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.dialog.title("Import Reference Stain Percentiles")
        self.dialog.geometry("800x600")
        self.dialog.minsize(700, 500)
        
        # Prevent closing the dialog directly
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)
        
        # Initialize variables for manual entry
        self.stain_name_vars = []
        self.percentile_vars = {}
        
        # Percentile levels to include
        self.percentile_levels = ["1", "50", "75", "95", "97.5", "99", "99.9"]
        
        # Create the main layout
        self.create_widgets()
        
        # Center the dialog relative to the parent window
        self.center_dialog()
        
        # Wait for the dialog to be closed before returning
        parent.wait_window(self.dialog)
    
    def create_widgets(self):
        """Create dialog widgets"""
        # Create main frame with padding
        main_frame = ttk.Frame(self.dialog, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title and instructions
        title_label = ttk.Label(main_frame, text="Import Reference Stain Percentiles", 
                               font=("TkDefaultFont", 12, "bold"))
        title_label.pack(pady=(0, 10))
        
        instruction_text = ("Choose one of the following methods to import reference stain percentiles:")
        ttk.Label(main_frame, text=instruction_text, wraplength=750).pack(fill=tk.X, pady=(0, 15))
        
        # Create notebook for two methods
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Method 1: File Import
        self.create_file_import_tab()
        
        # Method 2: Manual Entry
        self.create_manual_entry_tab()
        
        # Buttons frame at the bottom
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Add buttons
        ttk.Button(button_frame, text="Import", command=self.on_import, width=12).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.on_cancel, width=12).pack(side=tk.RIGHT, padx=5)
    
    def create_file_import_tab(self):
        """Create the file import tab"""
        file_frame = ttk.Frame(self.notebook)
        self.notebook.add(file_frame, text="Import from File")
        
        # Instructions for file import
        instructions = ("Select a reference_stain_percentiles.json file to import.\n\n"
                       "The file should contain percentile values for three stains in JSON format:\n\n"
                       "Example structure:\n"
                       "{\n"
                       '  "stain_names": ["Nuclei (H&E)", "Myelin (Blue)", "Microglia (Brown)"],\n'
                       '  "percentile_1": {"Nuclei (H&E)": -0.177, ...},\n'
                       '  "percentile_50": {"Nuclei (H&E)": 0.039, ...},\n'
                       '  ...\n'
                       "}")
        
        instruction_label = ttk.Label(file_frame, text=instructions, wraplength=700, justify=tk.LEFT)
        instruction_label.pack(pady=15, padx=15, anchor="w")
        
        # File selection area
        selection_frame = ttk.LabelFrame(file_frame, text="File Selection")
        selection_frame.pack(fill=tk.X, padx=15, pady=15)
        
        # Browse button and file display
        browse_frame = ttk.Frame(selection_frame)
        browse_frame.pack(fill=tk.X, padx=15, pady=15)
        
        ttk.Button(browse_frame, text="Browse for File...", 
                  command=self.browse_for_file).pack(side=tk.LEFT)
        
        # Selected file display
        self.selected_file_var = tk.StringVar(value="No file selected")
        ttk.Label(browse_frame, textvariable=self.selected_file_var, 
                 font=("TkDefaultFont", 9)).pack(side=tk.LEFT, padx=(15, 0))
        
        self.selected_file_path = None
    
    def create_manual_entry_tab(self):
        """Create the manual entry tab"""
        manual_frame = ttk.Frame(self.notebook)
        self.notebook.add(manual_frame, text="Manual Entry")
        
        # Scrollable frame for manual entry
        canvas = tk.Canvas(manual_frame)
        scrollbar = ttk.Scrollbar(manual_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Instructions for manual entry
        instructions = ("Enter percentile values for three stains. Values can be positive or negative.\n"
                       "Percentiles represent distribution thresholds for stain concentrations.")
        
        ttk.Label(scrollable_frame, text=instructions, wraplength=650, justify=tk.LEFT).pack(
            pady=15, padx=15, anchor="w")
        
        # Stain names section
        stain_names_frame = ttk.LabelFrame(scrollable_frame, text="Stain Names")
        stain_names_frame.pack(fill=tk.X, padx=15, pady=10)
        
        default_stain_names = ["Nuclei (H&E)", "Myelin (Blue)", "Microglia (Brown)"]
        self.stain_name_vars = []
        
        for i, default_name in enumerate(default_stain_names):
            name_frame = ttk.Frame(stain_names_frame)
            name_frame.pack(fill=tk.X, padx=10, pady=5)
            ttk.Label(name_frame, text=f"Stain {i+1}:", width=10).pack(side=tk.LEFT)
            name_var = tk.StringVar(value=default_name)
            ttk.Entry(name_frame, textvariable=name_var, width=30).pack(side=tk.LEFT, padx=(5, 0))
            self.stain_name_vars.append(name_var)
        
        # Percentiles section
        percentiles_frame = ttk.LabelFrame(scrollable_frame, text="Percentile Values")
        percentiles_frame.pack(fill=tk.X, padx=15, pady=10)
        
        # Create a grid for percentile entries
        # Header row
        header_frame = ttk.Frame(percentiles_frame)
        header_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(header_frame, text="Percentile", width=12).grid(row=0, column=0, padx=5)
        for i, stain_name in enumerate(default_stain_names):
            ttk.Label(header_frame, text=stain_name, width=18).grid(row=0, column=i+1, padx=5)
        
        # Initialize percentile variables
        self.percentile_vars = {}
        
        # Default values from your example
        default_values = {
            "1": [-0.177, -0.123, -0.104],
            "50": [0.039, 0.009, -0.005],
            "75": [0.068, 0.150, 0.004],
            "95": [0.120, 0.490, 0.011],
            "97.5": [0.165, 0.601, 0.017],
            "99": [0.497, 0.786, 0.329],
            "99.9": [10.237, 1.711, 12.269]
        }
        
        # Create entry rows for each percentile
        for row, percentile in enumerate(self.percentile_levels):
            percentile_frame = ttk.Frame(percentiles_frame)
            percentile_frame.pack(fill=tk.X, padx=10, pady=2)
            
            # Percentile label
            ttk.Label(percentile_frame, text=f"{percentile}%", width=12).grid(row=0, column=0, padx=5)
            
            # Entry fields for each stain
            self.percentile_vars[percentile] = []
            default_vals = default_values.get(percentile, [0.0, 0.0, 0.0])
            
            for col in range(3):
                var = tk.StringVar(value=str(default_vals[col]))
                entry = ttk.Entry(percentile_frame, textvariable=var, width=18)
                entry.grid(row=0, column=col+1, padx=5)
                
                # Validation for numeric entries
                vcmd = (self.dialog.register(self.validate_numeric), '%P')
                entry.config(validate='key', validatecommand=vcmd)
                
                self.percentile_vars[percentile].append(var)
    
    def validate_numeric(self, value):
        """Validate numeric input (allows positive and negative floats)"""
        if value == "" or value == "-":
            return True
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def center_dialog(self):
        """Center the dialog window on the parent"""
        self.dialog.update_idletasks()
        
        # Get parent and dialog dimensions
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        parent_x = self.parent.winfo_rootx()  
        parent_y = self.parent.winfo_rooty()
        
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        # Calculate center position
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        # Set position
        self.dialog.geometry(f"+{x}+{y}")
    
    def browse_for_file(self):
        """Open file browser to select reference file"""
        file_path = filedialog.askopenfilename(
            title="Select Reference Stain Percentiles File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            parent=self.dialog
        )
        
        if file_path:
            self.selected_file_path = file_path
            self.selected_file_var.set(os.path.basename(file_path))
    
    def copy_reference_file(self, source_path):
        """Copy the reference file to the parameters directory"""
        output_path = os.path.join(self.parameters_dir, "reference_stain_percentiles.json")
        
        try:
            # Ensure parameters directory exists
            os.makedirs(self.parameters_dir, exist_ok=True)
            
            # Copy the file
            shutil.copy2(source_path, output_path)
            
            return output_path
        except Exception as e:
            raise Exception(f"Error copying file: {str(e)}")
    
    def save_reference_file(self, stain_names, percentile_data):
        """Save the reference percentiles to a JSON file"""
        output_path = os.path.join(self.parameters_dir, "reference_stain_percentiles.json")
        
        try:
            # Ensure parameters directory exists
            os.makedirs(self.parameters_dir, exist_ok=True)
            
            # Build the JSON structure
            json_data = {
                "stain_names": stain_names
            }
            
            # Add percentile data
            for percentile in self.percentile_levels:
                percentile_key = f"percentile_{percentile}"
                json_data[percentile_key] = {}
                
                for i, stain_name in enumerate(stain_names):
                    json_data[percentile_key][stain_name] = percentile_data[percentile][i]
            
            # Write JSON file
            import json
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=4)
            
            return output_path
        except Exception as e:
            raise Exception(f"Error saving file: {str(e)}")
    
    def on_import(self):
        """Handle import button click"""
        current_tab = self.notebook.index(self.notebook.select())
        
        try:
            if current_tab == 0:  # File import tab
                if not self.selected_file_path:
                    messagebox.showerror("Error", "Please select a file to import", parent=self.dialog)
                    return
                
                # Simply copy the selected file to Parameters folder
                output_path = self.copy_reference_file(self.selected_file_path)
                
                # Show success message for file copy
                messagebox.showinfo("Success", 
                                  f"Reference stain percentiles file imported successfully!\n\n"
                                  f"Source: {os.path.basename(self.selected_file_path)}\n"
                                  f"Saved to: {output_path}", 
                                  parent=self.dialog)
                
            else:  # Manual entry tab
                # Collect stain names
                stain_names = []
                for i, name_var in enumerate(self.stain_name_vars):
                    name = name_var.get().strip()
                    if not name:
                        messagebox.showerror("Error", f"Please enter a name for Stain {i+1}", parent=self.dialog)
                        return
                    stain_names.append(name)
                
                # Collect percentile data
                percentile_data = {}
                for percentile in self.percentile_levels:
                    percentile_data[percentile] = []
                    for i, var in enumerate(self.percentile_vars[percentile]):
                        value_str = var.get().strip()
                        if not value_str:
                            messagebox.showerror("Error", f"Please enter a value for {stain_names[i]} at percentile {percentile}%", parent=self.dialog)
                            return
                        
                        try:
                            value = float(value_str)
                            percentile_data[percentile].append(value)
                        except ValueError:
                            messagebox.showerror("Error", f"Invalid numeric value for {stain_names[i]} at percentile {percentile}%", parent=self.dialog)
                            return
                
                # Save the reference file from manual entry
                output_path = self.save_reference_file(stain_names, percentile_data)
                
                # Show success message for manual entry
                messagebox.showinfo("Success", 
                                  f"Reference stain percentiles created successfully!\n\n"
                                  f"Stains: {', '.join(stain_names)}\n"
                                  f"Saved to: {output_path}", 
                                  parent=self.dialog)
            
            self.result = True
            self.dialog.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self.dialog)
    
    def on_cancel(self):
        """Handle Cancel button click or window close"""
        self.result = False
        self.dialog.destroy()

class ImportReferenceStainVectorsDialog:
    """Dialog for importing reference stain vectors either from file or manual entry"""
    
    def __init__(self, parent, parameters_dir):
        """
        Initialize the dialog
        
        Args:
            parent: Parent window
            parameters_dir: Directory where the reference file will be saved
        """
        self.result = False  # Will be True if import was successful
        self.parent = parent
        self.parameters_dir = parameters_dir
        
        # Create a new top-level dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.dialog.title("Import Reference Stain Vectors")
        self.dialog.geometry("700x500")
        self.dialog.minsize(600, 400)
        
        # Prevent closing the dialog directly
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)
        
        # Initialize variables for manual entry
        self.stain_vars = []
        
        # Create the main layout
        self.create_widgets()
        
        # Center the dialog relative to the parent window
        self.center_dialog()
        
        # Wait for the dialog to be closed before returning
        parent.wait_window(self.dialog)
    
    def create_widgets(self):
        """Create dialog widgets"""
        # Create main frame with padding
        main_frame = ttk.Frame(self.dialog, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title and instructions
        title_label = ttk.Label(main_frame, text="Import Reference Stain Vectors", 
                               font=("TkDefaultFont", 12, "bold"))
        title_label.pack(pady=(0, 10))
        
        instruction_text = ("Choose one of the following methods to import reference stain vectors:")
        ttk.Label(main_frame, text=instruction_text, wraplength=650).pack(fill=tk.X, pady=(0, 15))
        
        # Create notebook for two methods
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Method 1: File Import
        self.create_file_import_tab()
        
        # Method 2: Manual Entry
        self.create_manual_entry_tab()
        
        # Buttons frame at the bottom
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Add buttons
        ttk.Button(button_frame, text="Import", command=self.on_import, width=12).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.on_cancel, width=12).pack(side=tk.RIGHT, padx=5)
    
    def create_file_import_tab(self):
        """Create the file import tab"""
        file_frame = ttk.Frame(self.notebook)
        self.notebook.add(file_frame, text="Import from File")
        
        # Instructions for file import
        instructions = ("Select a reference_stain_vectors.txt file to import.\n\n"
                       "The file should contain RGB values for three stains in the following format:\n"
                       "Stain_Name_1: R G B\n"
                       "Stain_Name_2: R G B\n"
                       "Stain_Name_3: R G B\n\n"
                       "Example:\n"
                       "Hematoxylin: 0.650 0.704 0.286\n"
                       "Myelin: 0.072 0.990 0.105\n"
                       "Microglia: 0.268 0.570 0.776")
        
        instruction_label = ttk.Label(file_frame, text=instructions, wraplength=600, justify=tk.LEFT)
        instruction_label.pack(pady=15, padx=15, anchor="w")
        
        # File selection area
        selection_frame = ttk.LabelFrame(file_frame, text="File Selection")
        selection_frame.pack(fill=tk.X, padx=15, pady=15)
        
        # Browse button and file display
        browse_frame = ttk.Frame(selection_frame)
        browse_frame.pack(fill=tk.X, padx=15, pady=15)
        
        ttk.Button(browse_frame, text="Browse for File...", 
                  command=self.browse_for_file).pack(side=tk.LEFT)
        
        # Selected file display
        self.selected_file_var = tk.StringVar(value="No file selected")
        ttk.Label(browse_frame, textvariable=self.selected_file_var, 
                 font=("TkDefaultFont", 9)).pack(side=tk.LEFT, padx=(15, 0))
        
        self.selected_file_path = None
    
    def create_manual_entry_tab(self):
        """Create the manual entry tab"""
        manual_frame = ttk.Frame(self.notebook)
        self.notebook.add(manual_frame, text="Manual Entry")
        
        # Scrollable frame for manual entry
        canvas = tk.Canvas(manual_frame)
        scrollbar = ttk.Scrollbar(manual_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Instructions for manual entry
        instructions = ("Enter the RGB values for three stains. RGB values should be normalized (0.0 to 1.0).\n"
                       "Common stain types: Nuclei (H&E), Myelin (Blue), Microglia (Brown)")
        
        ttk.Label(scrollable_frame, text=instructions, wraplength=550, justify=tk.LEFT).pack(
            pady=15, padx=15, anchor="w")
        
        # Create entry fields for three stains with CORRECT names
        self.stain_vars = []
        stain_defaults = [
            ("Nuclei (H&E)", "0.9436", "0.2920", "0.1562"),
            ("Myelin (Blue)", "0.7306", "0.6608", "0.1722"),
            ("Microglia (Brown)", "0.2161", "0.6447", "0.7332")
        ]
        
        for i, (default_name, r, g, b) in enumerate(stain_defaults):
            stain_frame = ttk.LabelFrame(scrollable_frame, text=f"Stain {i+1}")
            stain_frame.pack(fill=tk.X, padx=15, pady=10)
            
            # Stain name
            name_frame = ttk.Frame(stain_frame)
            name_frame.pack(fill=tk.X, padx=10, pady=5)
            ttk.Label(name_frame, text="Name:", width=10).pack(side=tk.LEFT)
            name_var = tk.StringVar(value=default_name)
            ttk.Entry(name_frame, textvariable=name_var, width=20).pack(side=tk.LEFT, padx=(5, 0))
            
            # RGB values
            rgb_frame = ttk.Frame(stain_frame)
            rgb_frame.pack(fill=tk.X, padx=10, pady=5)
            ttk.Label(rgb_frame, text="RGB:", width=10).pack(side=tk.LEFT)
            
            r_var = tk.StringVar(value=r)
            g_var = tk.StringVar(value=g)
            b_var = tk.StringVar(value=b)
            
            ttk.Label(rgb_frame, text="R:").pack(side=tk.LEFT, padx=(5, 2))
            r_entry = ttk.Entry(rgb_frame, textvariable=r_var, width=8)
            r_entry.pack(side=tk.LEFT, padx=(0, 5))
            
            ttk.Label(rgb_frame, text="G:").pack(side=tk.LEFT, padx=(0, 2))
            g_entry = ttk.Entry(rgb_frame, textvariable=g_var, width=8)
            g_entry.pack(side=tk.LEFT, padx=(0, 5))
            
            ttk.Label(rgb_frame, text="B:").pack(side=tk.LEFT, padx=(0, 2))
            b_entry = ttk.Entry(rgb_frame, textvariable=b_var, width=8)
            b_entry.pack(side=tk.LEFT, padx=(0, 5))
            
            # Validation for RGB entries
            vcmd = (self.dialog.register(self.validate_rgb), '%P')
            for entry in [r_entry, g_entry, b_entry]:
                entry.config(validate='key', validatecommand=vcmd)
            
            self.stain_vars.append({
                'name': name_var,
                'r': r_var,
                'g': g_var,
                'b': b_var
            })
    
    def validate_rgb(self, value):
        """Validate RGB input (should be float between 0.0 and 1.0)"""
        if value == "":
            return True
        try:
            float_val = float(value)
            return 0.0 <= float_val <= 1.0
        except ValueError:
            return False
    
    def center_dialog(self):
        """Center the dialog window on the parent"""
        self.dialog.update_idletasks()
        
        # Get parent and dialog dimensions
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        parent_x = self.parent.winfo_rootx()  
        parent_y = self.parent.winfo_rooty()
        
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        # Calculate center position
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        # Set position
        self.dialog.geometry(f"+{x}+{y}")
    
    def browse_for_file(self):
        """Open file browser to select reference file"""
        file_path = filedialog.askopenfilename(
            title="Select Reference Stain Vectors File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            parent=self.dialog
        )
        
        if file_path:
            self.selected_file_path = file_path
            self.selected_file_var.set(os.path.basename(file_path))
    
    def save_reference_file(self, stain_data):
        """Save the reference stain vectors to the parameters directory"""
        output_path = os.path.join(self.parameters_dir, "reference_stain_vectors.txt")
        
        try:
            # Ensure parameters directory exists
            os.makedirs(self.parameters_dir, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write("Estimated stain vectors:\n")
                
                for name, r, g, b in stain_data:
                    f.write(f"{name}: [{r:.4f}, {g:.4f}, {b:.4f}]\n")
            
            return output_path
        except Exception as e:
            raise Exception(f"Error saving file: {str(e)}")
    
    def copy_reference_file(self, source_path):
        """Copy the reference file to the parameters directory"""
        output_path = os.path.join(self.parameters_dir, "reference_stain_vectors.txt")
        
        try:
            # Ensure parameters directory exists
            os.makedirs(self.parameters_dir, exist_ok=True)
            
            # Copy the file
            shutil.copy2(source_path, output_path)
            
            return output_path
        except Exception as e:
            raise Exception(f"Error copying file: {str(e)}")
    
    def on_import(self):
        """Handle import button click"""
        current_tab = self.notebook.index(self.notebook.select())
        
        try:
            if current_tab == 0:  # File import tab
                if not self.selected_file_path:
                    messagebox.showerror("Error", "Please select a file to import", parent=self.dialog)
                    return
                
                # Simply copy the selected file to Parameters folder
                output_path = self.copy_reference_file(self.selected_file_path)
                
                # Show success message for file copy
                messagebox.showinfo("Success", 
                                  f"Reference stain vectors file imported successfully!\n\n"
                                  f"Source: {os.path.basename(self.selected_file_path)}\n"
                                  f"Saved to: {output_path}", 
                                  parent=self.dialog)
                
            else:  # Manual entry tab
                # Collect data from manual entry
                stain_data = []
                for i, stain_vars in enumerate(self.stain_vars):
                    name = stain_vars['name'].get().strip()
                    r_str = stain_vars['r'].get().strip()
                    g_str = stain_vars['g'].get().strip() 
                    b_str = stain_vars['b'].get().strip()
                    
                    if not name:
                        messagebox.showerror("Error", f"Please enter a name for Stain {i+1}", parent=self.dialog)
                        return
                    
                    if not all([r_str, g_str, b_str]):
                        messagebox.showerror("Error", f"Please enter all RGB values for {name}", parent=self.dialog)
                        return
                    
                    try:
                        r, g, b = float(r_str), float(g_str), float(b_str)
                        if not all(0.0 <= val <= 1.0 for val in [r, g, b]):
                            messagebox.showerror("Error", f"RGB values for {name} must be between 0.0 and 1.0", parent=self.dialog)
                            return
                        stain_data.append((name, r, g, b))
                    except ValueError:
                        messagebox.showerror("Error", f"Invalid RGB values for {name}", parent=self.dialog)
                        return
                
                # Save the reference file from manual entry
                output_path = self.save_reference_file(stain_data)
                
                # Show success message for manual entry
                stain_names = [data[0] for data in stain_data]
                messagebox.showinfo("Success", 
                                  f"Reference stain vectors created successfully!\n\n"
                                  f"Stains: {', '.join(stain_names)}\n"
                                  f"Saved to: {output_path}", 
                                  parent=self.dialog)
            
            self.result = True
            self.dialog.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self.dialog)
    
    def on_cancel(self):
        """Handle Cancel button click or window close"""
        self.result = False
        self.dialog.destroy()

class ImportDetectionParametersDialog:
    """Dialog for importing detection parameters for Cell, Microglia, or Myelin detection"""
    
    def __init__(self, parent, parameters_dir, detection_type):
        """
        Initialize the dialog
        
        Args:
            parent: Parent window
            parameters_dir: Directory where the parameter file will be saved
            detection_type: Type of detection ('cell', 'microglia', 'myelin')
        """
        self.result = False  # Will be True if import was successful
        self.parent = parent
        self.parameters_dir = parameters_dir
        self.detection_type = detection_type.lower()
        
        # Define the target filename based on detection type
        self.target_filenames = {
            'cell': 'cell_detection_parameters.json',
            'microglia': 'microglia_detection_parameters.json',
            'myelin': 'myelin_detection_parameters.json'
        }
        
        # Define display names
        self.display_names = {
            'cell': 'Cell Detection',
            'microglia': 'Microglia Detection',
            'myelin': 'Myelin Detection'
        }
        
        # Create a new top-level dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.dialog.title(f"Import {self.display_names[self.detection_type]} Parameters")
        self.dialog.geometry("700x500")
        self.dialog.minsize(600, 400)
        
        # Prevent closing the dialog directly
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)
        
        # Create the main layout
        self.create_widgets()
        
        # Center the dialog relative to the parent window
        self.center_dialog()
        
        # Wait for the dialog to be closed before returning
        parent.wait_window(self.dialog)
    
    def create_widgets(self):
        """Create dialog widgets"""
        # Create main frame with padding
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title and instructions
        title_label = ttk.Label(main_frame, text=f"Import {self.display_names[self.detection_type]} Parameters", 
                               font=("TkDefaultFont", 12, "bold"))
        title_label.pack(pady=(0, 15))
        
        instruction_text = (f"Select a {self.target_filenames[self.detection_type]} file to import.\n\n"
                           f"The file should contain detection parameters in JSON format with the following structure:\n\n"
                           "Example structure:\n"
                           "{\n"
                           '  "detection_method": "method_name",\n'
                           '  "parameters": {\n'
                           '    "threshold": 0.5,\n'
                           '    "min_size": 10,\n'
                           '    "max_size": 1000,\n'
                           '    ...\n'
                           '  }\n'
                           "}")
        
        instruction_label = ttk.Label(main_frame, text=instruction_text, wraplength=650, justify=tk.LEFT)
        instruction_label.pack(pady=(0, 20))
        
        # File selection area
        selection_frame = ttk.LabelFrame(main_frame, text="File Selection")
        selection_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Browse button and file display
        browse_frame = ttk.Frame(selection_frame)
        browse_frame.pack(fill=tk.X, padx=20, pady=20)
        
        ttk.Button(browse_frame, text="Browse for Parameter File...", 
                  command=self.browse_for_file).pack(side=tk.LEFT)
        
        # Selected file display
        self.selected_file_var = tk.StringVar(value="No file selected")
        ttk.Label(browse_frame, textvariable=self.selected_file_var, 
                 font=("TkDefaultFont", 9)).pack(side=tk.LEFT, padx=(15, 0))
        
        self.selected_file_path = None
        
        # Information about parameter types
        info_frame = ttk.LabelFrame(main_frame, text="Parameter Information")
        info_frame.pack(fill=tk.X, pady=(0, 20))
        
        info_text = self.get_parameter_info()
        info_label = ttk.Label(info_frame, text=info_text, wraplength=650, justify=tk.LEFT)
        info_label.pack(padx=20, pady=15)
        
        # Buttons frame at the bottom
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        # Add buttons
        ttk.Button(button_frame, text="Import", command=self.on_import, width=12).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.on_cancel, width=12).pack(side=tk.RIGHT, padx=5)
    
    def get_parameter_info(self):
        """Get information text for the specific detection type"""
        info_texts = {
            'cell': ("Cell Detection Parameters typically include:\n"
                    "• Detection method (e.g., 'blob_detection', 'watershed')\n"
                    "• Threshold values for cell identification\n"
                    "• Size constraints (min/max cell area)\n"
                    "• Morphological parameters\n"
                    "• Filtering criteria"),
            
            'microglia': ("Microglia Detection Parameters typically include:\n"
                         "• Detection method (e.g., 'morphology_based', 'intensity_based')\n"
                         "• Threshold values for microglia identification\n"
                         "• Shape analysis parameters\n"
                         "• Branch detection settings\n"
                         "• Size and intensity filters"),
            
            'myelin': ("Myelin Detection Parameters typically include:\n"
                      "• Detection method (e.g., 'fiber_tracking', 'intensity_threshold')\n"
                      "• Threshold values for myelin fiber identification\n"
                      "• Fiber width and length constraints\n"
                      "• Continuity parameters\n"
                      "• Noise reduction settings")
        }
        
        return info_texts.get(self.detection_type, "Detection parameters for the selected method.")
    
    def center_dialog(self):
        """Center the dialog window on the parent"""
        self.dialog.update_idletasks()
        
        # Get parent and dialog dimensions
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        parent_x = self.parent.winfo_rootx()
        parent_y = self.parent.winfo_rooty()
        
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        # Calculate center position
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        # Set position
        self.dialog.geometry(f"+{x}+{y}")
    
    def browse_for_file(self):
        """Open file browser to select parameter file"""
        file_path = filedialog.askopenfilename(
            title=f"Select {self.display_names[self.detection_type]} Parameters File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            parent=self.dialog
        )
        
        if file_path:
            self.selected_file_path = file_path
            self.selected_file_var.set(os.path.basename(file_path))
    
    def copy_parameter_file(self, source_path):
        """Copy the parameter file to the parameters directory with the correct name"""
        target_filename = self.target_filenames[self.detection_type]
        output_path = os.path.join(self.parameters_dir, target_filename)
        
        try:
            # Ensure parameters directory exists
            os.makedirs(self.parameters_dir, exist_ok=True)
            
            # Copy the file
            shutil.copy2(source_path, output_path)
            
            return output_path
        except Exception as e:
            raise Exception(f"Error copying file: {str(e)}")
    
    def validate_parameter_file(self, file_path):
        """Basic validation of the parameter file format"""
        try:
            import json
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check if it's a dictionary (basic JSON structure validation)
            if not isinstance(data, dict):
                return False, "File must contain a JSON object"
            
            # Optional: Add more specific validation here if needed
            # For now, we just check if it's valid JSON
            return True, "File appears to be valid"
            
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON format: {str(e)}"
        except Exception as e:
            return False, f"Error reading file: {str(e)}"
    
    def on_import(self):
        """Handle import button click"""
        try:
            if not self.selected_file_path:
                messagebox.showerror("Error", "Please select a parameter file to import", parent=self.dialog)
                return
            
            # Validate the file
            is_valid, message = self.validate_parameter_file(self.selected_file_path)
            if not is_valid:
                messagebox.showerror("Invalid File", f"The selected file is not valid:\n{message}", parent=self.dialog)
                return
            
            # Copy the file to the parameters directory
            output_path = self.copy_parameter_file(self.selected_file_path)
            
            # Show success message
            messagebox.showinfo("Success", 
                              f"{self.display_names[self.detection_type]} parameters imported successfully!\n\n"
                              f"Source: {os.path.basename(self.selected_file_path)}\n"
                              f"Saved as: {os.path.basename(output_path)}\n"
                              f"Location: {output_path}", 
                              parent=self.dialog)
            
            self.result = True
            self.dialog.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import parameters:\n{str(e)}", parent=self.dialog)
    
    def on_cancel(self):
        """Handle Cancel button click or window close"""
        self.result = False
        self.dialog.destroy()



class IHCPipelineGUI:
    """Enhanced main application class for the IHC Pipeline GUI with improved aesthetics."""
    
    def __init__(self, root):
        """Initialize the application with the root window."""
        self.root = root
        self.root.title("IHC Analysis Pipeline")
        self.root.geometry("1400x900")  # Larger window
        self.root.minsize(1200, 800)
        
        # Set up configuration
        self.config = configparser.ConfigParser()
        self.config_file = "ihc_pipeline_config.ini"
        self.load_config()
        
        # Define main directory paths
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.scripts_dir = os.path.join(self.base_dir, "Scripts")
        self.data_dir = os.path.join(self.base_dir, "Data")
        self.parameters_dir = os.path.join(self.base_dir, "Parameters")
        self.results_dir = os.path.join(self.base_dir, "Results")
        
        # Create state variables for checkboxes
        self.create_state_variables()
        
        # For managing script execution
        self.process_queue = queue.Queue()
        self.currently_running = False
        self.cancel_requested = False
        
        # Set up the enhanced GUI
        self.setup_enhanced_gui()
        
        # Start the queue monitoring thread
        self.start_queue_monitor()
        
        # Validate overview creation setup on startup
        self.root.after(1000, self.validate_overview_setup)
    
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
    
    def load_config(self):
        """Load configuration from file or create default if not exists."""
        if os.path.exists(self.config_file):
            try:
                self.config.read(self.config_file)
                logger.info(f"Configuration loaded from {self.config_file}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                self.create_default_config()
        else:
            self.create_default_config()
    
    def create_default_config(self):
        """Create a default configuration file."""
        self.config["Paths"] = {
            "scripts_dir": "Scripts",
            "data_dir": "Data",
            "parameters_dir": "Parameters",
            "results_dir": "Results"
        }
        self.config["Parameters"] = {
            "default_tile_size": "256",
            "default_overlap": "32"
        }
        with open(self.config_file, 'w') as configfile:
            self.config.write(configfile)
        logger.info(f"Default configuration created at {self.config_file}")
    
    def create_state_variables(self):
        """Create state variables for checkboxes and other controls."""
        # Checkboxes for Preprocessing page
        self.background_removal_overview_var = tk.BooleanVar(value=False)
        self.l_channel_normalization_overview_var = tk.BooleanVar(value=False)
        self.illumination_correction_overview_var = tk.BooleanVar(value=False)
        self.normalization_overview_var = tk.BooleanVar(value=False)
        
        # Checkboxes for Detection page
        self.cell_density_map_var = tk.BooleanVar(value=False)
        self.cell_detection_overview_var = tk.BooleanVar(value=False)
        self.microglia_density_map_var = tk.BooleanVar(value=False)
        self.microglia_detection_overview_var = tk.BooleanVar(value=False)
        self.myelin_density_map_var = tk.BooleanVar(value=False)
        self.myelin_detection_overview_var = tk.BooleanVar(value=False)
        
        # Variables to store selected file/directory paths
        self.selected_data_dir = tk.StringVar(value=self.data_dir)
        self.selected_output_dir = tk.StringVar(value=self.results_dir)
        self.selected_parameters_dir = tk.StringVar(value=self.parameters_dir)
    
    def setup_enhanced_gui(self):
        """Set up enhanced GUI with better aesthetics."""
        # Apply modern styling first
        self.setup_styling()
        
        # Enable thread safety in tkinter
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Main container with better padding
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Enhanced toolbar
        toolbar_frame = ttk.LabelFrame(main_container, text="Directory Configuration", 
                                      padding=(15, 10))
        toolbar_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Enhanced toolbar buttons with icons
        ttk.Button(toolbar_frame, text="Select Data Directory", 
                   command=self.select_data_directory, style='Primary.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar_frame, text="Select Output Directory", 
                   command=self.select_output_directory, style='Primary.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar_frame, text="Select Parameters Directory", 
                   command=self.select_parameters_directory, style='Primary.TButton').pack(side=tk.LEFT, padx=5)
        
        self.cancel_button = ttk.Button(toolbar_frame, text="Cancel Operation", 
                                       command=self.cancel_operation, state=tk.DISABLED,
                                       style='Accent.TButton')
        self.cancel_button.pack(side=tk.RIGHT, padx=5)
        
        # MAIN CONTENT AREA - Split layout
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # LEFT SIDE - Processing tabs (70% width)
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # RIGHT SIDE - Information panel (30% width)
        right_frame = ttk.Frame(content_frame, width=620)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)  # Prevent auto-shrinking
        
        # CREATE INFORMATION PANEL
        self.create_info_panel(right_frame)
        
        # Enhanced notebook with icons
        self.notebook = ttk.Notebook(left_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create pages with enhanced styling
        self.preprocessing_frame = ttk.Frame(self.notebook)
        self.parameter_estimation_frame = ttk.Frame(self.notebook)  
        self.detection_frame = ttk.Frame(self.notebook)
        
        # Add pages with emoji icons
        self.notebook.add(self.preprocessing_frame, text="Preprocessing")
        self.notebook.add(self.parameter_estimation_frame, text="Parameter Estimation")
        self.notebook.add(self.detection_frame, text="Detection")
        
        # ENHANCED LOG AND STATUS AREA
        bottom_frame = ttk.Frame(main_container)
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
        
        # Setup pages with enhanced styling
        self.setup_preprocessing_page()
        self.setup_parameter_estimation_page()
        self.setup_detection_page()
    
    def open_website(self, event=None):
        """Open the project website when logo is clicked."""
        import webbrowser
        import subprocess
        import os
        import time
        
        url = "https://dbe.unibas.ch/en/research/imaging-modelling-diagnosis/"
        
        self.update_log(f"Attempting to open: {url}")
        
        # SOLUTION: Change to user's home directory to avoid permission issues
        original_cwd = os.getcwd()
        try:
            # Change to a directory we know has proper permissions
            safe_directory = os.path.expanduser("~")  # User's home directory
            os.chdir(safe_directory)
            self.update_log(f"Changed working directory to: {safe_directory}")
            
            # Now try to open the browser
            result = webbrowser.open(url, new=2)  # new=2 for new tab
            
            if result:
                self.update_log("Browser opened successfully!")
                return True
            else:
                self.update_log("Browser command failed", error=True)
                
        except Exception as e:
            self.update_log(f"Error opening browser: {e}", error=True)
            
        finally:
            # Always restore the original working directory
            try:
                os.chdir(original_cwd)
                self.update_log(f"Restored working directory to: {original_cwd}")
            except Exception as e:
                self.update_log(f"Warning: Could not restore working directory: {e}", error=True)
        
        # Fallback: Try direct subprocess with explicit working directory
        try:
            self.update_log("Trying subprocess with explicit working directory...")
            result = subprocess.Popen(
                ["firefox", url],
                cwd=os.path.expanduser("~"),  # Run from home directory
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            self.update_log(f"Firefox subprocess started successfully with PID: {result.pid}")
            return True
            
        except Exception as e:
            self.update_log(f"Subprocess method failed: {e}", error=True)
        
        # If all else fails, show URL to user
        try:
            messagebox.showinfo(
                "Website URL", 
                f"Please copy this URL to your browser:\n\n{url}",
                parent=self.root
            )
            self.update_log("Displayed URL to user for manual opening")
        except Exception as e:
            self.update_log(f"Could not show URL dialog: {e}", error=True)
        
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
        title_label = ttk.Label(scrollable_frame, text="IHC Analysis Pipeline", style='Title.TLabel')
        title_label.pack(pady=(0, 15))

        workflow_text = (
            "PREPROCESSING WORKFLOW:\n"
            "• Metadata Creation - Extract Image Properties\n"
            "• Luminance Normalization - Correct Lightness\n"
            "• Background Removal - Identify and Remove Background\n"
            "• Create Registration Template - For MRI co-registration\n"
            "• Illumination Correction - Correct radial falloff pattern\n"
            "• Stain Processing - Separate and Normalize Stain Channels\n"
            "• Tile Creation - Prepare for Analysis\n\n"
            "PARAMETER ESTIMATION:\n"
            "• Tile Selection - Choose Representative Regions\n"
            "• Detection Parameters - Adjust Algorithm Parameters\n\n"
            "DETECTION:\n"
            "• Cell Detection - Count Nuclei\n"
            "• Microglia Detection - Count Activated Microglia\n"
            "• Myelin Detection - Quantify Myelin\n"
            "• Density Maps - Derive Measurement Densities\n"
            "• Overview - Overview for Validation"
        )

        workflow_label = ttk.Label(scrollable_frame, text=workflow_text,
                                   font=('Segoe UI', 9),
                                   justify='left', wraplength=530)
        workflow_label.pack(padx=10, pady=10, anchor='w')

        session_frame = ttk.LabelFrame(scrollable_frame, text="Session Info", padding=(10, 5))
        session_frame.pack(fill='x', padx=10, pady=10)

        script_count = len([f for f in os.listdir(self.scripts_dir) if f.endswith('.py')]) if os.path.exists(self.scripts_dir) else 0
        session_info = f"""Started: {datetime.now().strftime('%H:%M:%S')}
        Working Dir: {os.path.basename(self.base_dir)}
        Scripts Available: {script_count}
        Pipeline Version: 1.1"""

        session_label = ttk.Label(session_frame, text=session_info,
                                  font=('Segoe UI', 8),
                                  foreground=self.colors['text_light'])
        session_label.pack(anchor='w')
        
        # Insert logo image if available
        image_path = os.path.join(self.base_dir, "logo.png")
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
                self.update_log(f"Failed to load logo.png: {e}", error=True)
        else:
            self.update_log("logo.png not found in script directory.")
    
    def create_enhanced_section_frame(self, parent, title, description=""):
        """Create an enhanced labeled frame section with better styling."""
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
            desc_label = ttk.Label(header_frame, text=description, style='Info.TLabel')
            desc_label.pack(anchor="w", pady=(2, 0))
        
        # Content frame for buttons/controls
        content_frame = ttk.Frame(section_frame)
        content_frame.pack(fill="x")
        
        return content_frame
    
    def create_enhanced_button(self, parent, text, command, style='Primary.TButton', description=""):
        """Create an enhanced button with optional description."""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill="x", pady=3)
        
        # Main button
        button = ttk.Button(button_frame, text=text, command=command, style=style)
        button.pack(side="left", padx=(0, 10))
        
        # Description label if provided
        if description:
            desc_label = ttk.Label(button_frame, text=description, style='Info.TLabel')
            desc_label.pack(side="left", anchor="w")
        
        return button
    
    def on_closing(self):
        """Handle window close event."""
        if self.currently_running:
            if messagebox.askokcancel("Quit", "An operation is still running. Are you sure you want to quit?"):
                self.cancel_requested = True
                self.root.destroy()
        else:
            self.root.destroy()
    
    def select_data_directory(self):
        """Open a dialog to select the data directory."""
        directory = filedialog.askdirectory(initialdir=self.data_dir)
        if directory:
            self.selected_data_dir.set(directory)
            logger.info(f"Selected data directory: {directory}")
            self.update_log(f"Data directory set to: {directory}")
    
    def select_output_directory(self):
        """Open a dialog to select the output directory."""
        directory = filedialog.askdirectory(initialdir=self.results_dir)
        if directory:
            self.selected_output_dir.set(directory)
            logger.info(f"Selected output directory: {directory}")
            self.update_log(f"Output directory set to: {directory}")
    
    def select_parameters_directory(self):
        """Open a dialog to select the parameters directory."""
        directory = filedialog.askdirectory(initialdir=self.parameters_dir)
        if directory:
            self.selected_parameters_dir.set(directory)
            logger.info(f"Selected parameters directory: {directory}")
            self.update_log(f"Parameters directory set to: {directory}")
    
    def update_log(self, message, error=False):
        """Update the log text widget with a message (thread-safe)."""
        # Use after to ensure this runs in the main thread
        self.root.after(0, self._update_log_main_thread, message, error)
    
    def _update_log_main_thread(self, message, error=False):
        """Enhanced log update with better styling."""
        self.log_text.config(state=tk.NORMAL)
        
        # Add timestamp
        timestamp = datetime.now().strftime('%H:%M:%S')
        
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
        elif "progress:" in message.lower() or "processing" in message.lower():
            # Blue color for progress
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.tag_add("progress", "end-2l linestart", "end-1l lineend")  
            self.log_text.tag_configure("progress", foreground="#3498DB", font=('Consolas', 9))
        else:
            # Default white color
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.root.update_idletasks()
    
    def update_status_with_icon(self, message, status_type="info"):
        """Update status with appropriate icon."""
        self.status_var.set(f"{message}")
    
    def cancel_operation(self):
        """Cancel the currently running operation."""
        if self.currently_running:
            self.cancel_requested = True
            self.update_log("Cancellation requested. Waiting for current operation to complete...")
            self.update_status_with_icon("Cancelling operation...", "warning")
            
            # Start a watchdog thread to force-stop if needed
            watchdog = threading.Thread(target=self._cancellation_watchdog)
            watchdog.daemon = True
            watchdog.start()
    
    def _cancellation_watchdog(self):
        """Watchdog to ensure cancellation happens even if a script gets stuck."""
        # Give the script some time to cancel gracefully
        for _ in range(30):  # Wait up to 30 seconds
            if not self.currently_running:
                return  # Operation completed normally
            time.sleep(1)
        
        # If we're still running after the timeout, force reset the state
        if self.currently_running:
            self.update_log("Forced cancellation after timeout", error=True)
            self.root.after(0, self._reset_after_processing)
    
    def open_import_stain_vectors_dialog(self):
        """Open the import reference stain vectors dialog"""
        try:
            dialog = ImportReferenceStainVectorsDialog(self.root, self.selected_parameters_dir.get())
            if dialog.result:
                self.update_log("Reference stain vectors imported successfully")
            else:
                self.update_log("Import cancelled")
        except Exception as e:
            self.update_log(f"Error opening import dialog: {str(e)}", error=True)
            logger.error(f"Error opening import dialog: {str(e)}")
    
    def open_import_stain_percentiles_dialog(self):
        """Open the import reference stain percentiles dialog"""
        try:
            dialog = ImportReferenceStainPercentilesDialog(self.root, self.selected_parameters_dir.get())
            if dialog.result:
                self.update_log("Reference stain percentiles imported successfully")
            else:
                self.update_log("Import cancelled")
        except Exception as e:
            self.update_log(f"Error opening import dialog: {str(e)}", error=True)
            logger.error(f"Error opening import dialog: {str(e)}")
    
    def open_stain_normalization_dialog(self):
        """Open the stain normalization percentiles selection dialog and queue the script"""
        try:
            dialog = StainNormalizationPercentilesDialog(self.root)
            if dialog.result:
                # Get the selected percentiles
                percentiles = dialog.result
                
                # Create arguments for the script
                args = [
                    "--nuclei-percentile", percentiles['nuclei'],
                    "--myelin-percentile", percentiles['myelin'],
                    "--microglia-percentile", percentiles['microglia']
                ]
                
                # Log the selection
                self.update_log(f"Selected percentiles: Nuclei={percentiles['nuclei']}, Myelin={percentiles['myelin']}, Microglia={percentiles['microglia']}")
                
                # Queue the script with the selected percentiles
                self.queue_script("Normalize-Stain-Concentrations.py", args)
                
                # If overview checkbox is checked, queue the overview creation script
                if self.normalization_overview_var.get():
                    self.process_queue.put({
                        "type": "script", 
                        "script_name": "Overview-Creation.py", 
                        "args": ["--step-type", "normalization"]
                    })
                    self.update_log("Queued overview creation for normalization step")
            else:
                self.update_log("Stain normalization cancelled")
        except Exception as e:
            self.update_log(f"Error opening percentile selection dialog: {str(e)}", error=True)
            logger.error(f"Error opening percentile selection dialog: {str(e)}")
    
    def open_import_cell_parameters_dialog(self):
        """Open the import cell detection parameters dialog"""
        try:
            dialog = ImportDetectionParametersDialog(self.root, self.selected_parameters_dir.get(), "cell")
            if dialog.result:
                self.update_log("Cell detection parameters imported successfully")
            else:
                self.update_log("Cell detection parameters import cancelled")
        except Exception as e:
            self.update_log(f"Error opening cell parameters import dialog: {str(e)}", error=True)
            logger.error(f"Error opening cell parameters import dialog: {str(e)}")

    def open_import_microglia_parameters_dialog(self):
        """Open the import microglia detection parameters dialog"""
        try:
            dialog = ImportDetectionParametersDialog(self.root, self.selected_parameters_dir.get(), "microglia")
            if dialog.result:
                self.update_log("Microglia detection parameters imported successfully")
            else:
                self.update_log("Microglia detection parameters import cancelled")
        except Exception as e:
            self.update_log(f"Error opening microglia parameters import dialog: {str(e)}", error=True)
            logger.error(f"Error opening microglia parameters import dialog: {str(e)}")

    def open_import_myelin_parameters_dialog(self):
        """Open the import myelin detection parameters dialog"""
        try:
            dialog = ImportDetectionParametersDialog(self.root, self.selected_parameters_dir.get(), "myelin")
            if dialog.result:
                self.update_log("Myelin detection parameters imported successfully")
            else:
                self.update_log("Myelin detection parameters import cancelled")
        except Exception as e:
            self.update_log(f"Error opening myelin parameters import dialog: {str(e)}", error=True)
            logger.error(f"Error opening myelin parameters import dialog: {str(e)}")
    
    def _show_tile_selection_guidance(self):
        """Show guidance message when tile selection interactive interface opens"""
        try:
            messagebox.showinfo(
                "Tile Selection Interface Opened",
                "The tile selection interface has opened in a separate window.\n\n"
                "Instructions:\n"
                "• Click tiles to select/deselect them for parameter estimation\n"
                "• Press 'm' to toggle between Selection and Navigation modes\n"
                "• Press Enter when you're finished with each selection step\n"
                "• You'll be guided through 3 selection steps:\n"
                "  1. Cell Detection Parameter Estimation\n"
                "  2. Myelin Detection Parameter Estimation\n"
                "  3. Microglia Detection Parameter Estimation\n\n"
                "The main GUI will wait until tile selection is complete.",
                parent=self.root
            )
        except Exception as e:
            # Don't let dialog errors break the workflow
            logger.warning(f"Error showing tile selection guidance: {str(e)}")
    
    def validate_overview_setup(self):
        """Validate that Overview-Creation.py script exists and required directories are set up"""
        overview_script_path = os.path.join(self.scripts_dir, "Overview-Creation.py")
        
        if not os.path.exists(overview_script_path):
            self.update_log("WARNING: Overview-Creation.py script not found in Scripts directory", error=True)
            return False
        
        # Check if Parameters directory exists and has metadata
        metadata_path = os.path.join(self.selected_parameters_dir.get(), "Metadata.csv")
        if not os.path.exists(metadata_path):
            self.update_log("WARNING: Metadata.csv not found. Overview creation may fail without tile grid information.", error=True)
        
        # Ensure Results/Overviews directory exists
        overviews_dir = os.path.join(self.selected_output_dir.get(), "Overviews")
        os.makedirs(overviews_dir, exist_ok=True)
        
        return True
    
    def setup_preprocessing_page(self):
        """Set up the enhanced Preprocessing page."""
        # Main scrollable frame
        main_frame = ttk.Frame(self.preprocessing_frame)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        canvas = tk.Canvas(main_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 1. Metadata Creation Section
        metadata_frame = self.create_enhanced_section_frame(
            scrollable_frame, 
            "Metadata Creation",
            "Extract and organize image properties and dimensions"
        )
        
        self.create_enhanced_button(metadata_frame, "Manual Metadata Creation", 
                                  lambda: self.queue_script("Metadata-GUI.py"),
                                  description="Interactive GUI for manual metadata entry")
        
        self.create_enhanced_button(metadata_frame, "Automatic Metadata Extraction", 
                                  lambda: self.queue_script("Extract-Metadata.py"),
                                  description="Automatically scan and extract image metadata")
        
        # 2. Luminance Normalization Section
        lum_norm_frame = self.create_enhanced_section_frame(
            scrollable_frame,
            "Luminance Normalization",
            "Normalize lighting conditions across and within images"
        )
        
        self.create_enhanced_button(lum_norm_frame, "L-Channel Normalization",
                                  lambda: self.queue_preprocessing_with_overview("L-Channel-Normalization.py", 
                                          self.l_channel_normalization_overview_var.get(), "l-channel"),
                                  description="Normalize luminance channel values")
        
        check_frame = ttk.Frame(lum_norm_frame)
        check_frame.pack(fill="x", pady=2)
        ttk.Checkbutton(check_frame, text="Create L-Channel Overview", 
                       variable=self.l_channel_normalization_overview_var).pack(anchor="w")
        
        # 3. Background Removal Section  
        bg_frame = self.create_enhanced_section_frame(
            scrollable_frame,
            "Background Removal", 
            "Remove unwanted background"
        )
        
        self.create_enhanced_button(bg_frame, "Remove Background",
                                  lambda: self.queue_preprocessing_with_overview("Background-Removal.py", 
                                          self.background_removal_overview_var.get(), "background"),
                                  description="Clean background from images")
        
        # Checkbox with better styling
        check_frame = ttk.Frame(bg_frame)
        check_frame.pack(fill="x", pady=5)
        ttk.Checkbutton(check_frame, text="Create Background Removal Overview", 
                       variable=self.background_removal_overview_var).pack(anchor="w")
        
        # 4. Registration Template Creation Section  
        reg_frame = self.create_enhanced_section_frame(
            scrollable_frame,
            "Create Registration Template", 
            "Generate WSI template for registration"
        )
        
        self.create_enhanced_button(reg_frame, "Create Template",
                                  lambda: self.queue_script("Create-Registration-Template.py"),
                                  description="Create overview with black background")
        
        # 5. Illumination Correction Section
        illum_frame = self.create_enhanced_section_frame(
            scrollable_frame,
            "Illumination Correction",
            "Correct illumination artifacts in images"
        )
        
        self.create_enhanced_button(illum_frame, "Estimate Flat-Field Model",
                                  lambda: self.queue_script("Estimate-Flat-Field-Model.py"),
                                  description="Calculate illumination correction model")
        
        self.create_enhanced_button(illum_frame, "Apply Flat-Field Correction",
                                  lambda: self.queue_preprocessing_with_overview("Apply-Flat-Field-Model.py", 
                                          self.illumination_correction_overview_var.get(), "illumination"),
                                  description="Apply illumination correction to images")
        
        check_frame2 = ttk.Frame(illum_frame)
        check_frame2.pack(fill="x", pady=2)
        ttk.Checkbutton(check_frame2, text="Create Illumination Correction Overview", 
                       variable=self.illumination_correction_overview_var).pack(anchor="w")
        
        # 6. Stain Processing Section
        stain_frame = self.create_enhanced_section_frame(
            scrollable_frame,
            "Stain Processing",
            "Separate and normalize different stain types"
        )
        
        # Stain Vector subframe
        vector_frame = ttk.LabelFrame(stain_frame, text="Stain Vector Estimation", padding=(10, 5))
        vector_frame.pack(fill="x", pady=(0, 5))
        
        self.create_enhanced_button(vector_frame, "Estimate Stain Vectors",
                                  lambda: self.queue_script("EBKSVD.py"),
                                  description="Automatically estimate stain separation vectors")
        
        self.create_enhanced_button(vector_frame, "Import Reference Stain Vectors",
                                  self.open_import_stain_vectors_dialog,
                                  description="Load pre-defined stain vectors from file")
        
        # Stain Percentiles subframe  
        percentile_frame = ttk.LabelFrame(stain_frame, text="Stain Percentiles", padding=(10, 5))
        percentile_frame.pack(fill="x", pady=5)
        
        self.create_enhanced_button(percentile_frame, "Calculate Stain Percentiles",
                                  lambda: self.queue_script("Percentile-Calculation.py"),
                                  description="Calculate stain intensity distributions")
        
        self.create_enhanced_button(percentile_frame, "Import Reference Percentiles", 
                                  self.open_import_stain_percentiles_dialog,
                                  description="Load pre-calculated percentile values")
        
        # Normalization
        self.create_enhanced_button(stain_frame, "Normalize Stain Concentrations",
                                  self.open_stain_normalization_dialog,
                                  description="Normalize stain concentrations between images")
        
        check_frame3 = ttk.Frame(stain_frame)
        check_frame3.pack(fill="x", pady=2)
        ttk.Checkbutton(check_frame3, text="Create Normalization Overview", 
                       variable=self.normalization_overview_var).pack(anchor="w")
        
        # 7. Tile Creation
        tile_frame = self.create_enhanced_section_frame(
            scrollable_frame,
            "Tile Creation",
            "Create small tiles of ca. 100x100 micrometer"
        )
        
        self.create_enhanced_button(tile_frame, "Create Small Tiles",
                                  lambda: self.queue_script("Create-Small-Tiles.py"),
                                  description="Divide images into smaller processing tiles")
    
    def setup_parameter_estimation_page(self):
        """Set up the enhanced Parameter Estimation page."""
        # Main scrollable frame
        main_frame = ttk.Frame(self.parameter_estimation_frame)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        canvas = tk.Canvas(main_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Stain Statistics Calculation
        stats_frame = self.create_enhanced_section_frame(
            scrollable_frame,
            "Stain Statistics Calculation",
            "Calculate global statistics for all stain channels from normalized tiles"
        )
        
        self.create_enhanced_button(stats_frame, "Calculate Stain Statistics",
                                  lambda: self.queue_script("Calculate-Stain-Statistics.py"),
                                  description="Compute global stain concentration statistics for parameter estimation")
        
        # Tile Selection
        tile_frame = self.create_enhanced_section_frame(
            scrollable_frame,
            "Tile Selection",
            "Choose representative tiles for parameter optimization"
        )
        
        self.create_enhanced_button(tile_frame, "Interactive Tile Selection Tool",
                                  lambda: self.queue_script("Tile-Selection-Tool.py"),
                                  description="Visually select tiles for parameter estimation")
        
        # Cell Detection Parameters
        cell_frame = self.create_enhanced_section_frame(
            scrollable_frame,
            "Cell Detection Parameters",
            "Optimize parameters for nuclei detection"
        )
        
        self.create_enhanced_button(cell_frame, "Estimate Cell Parameters",
                                  lambda: self.queue_script("Cell-Detection-Parameter-Estimation.py"),
                                  description="Interactive parameter optimization for cell detection")
        
        self.create_enhanced_button(cell_frame, "Import Cell Parameters",
                                  self.open_import_cell_parameters_dialog,
                                  description="Load pre-optimized cell detection parameters")
        
        # Microglia Detection Parameters
        microglia_frame = self.create_enhanced_section_frame(
            scrollable_frame,
            "Microglia Detection Parameters",
            "Optimize parameters for microglia detection"
        )
        
        self.create_enhanced_button(microglia_frame, "Estimate Microglia Parameters",
                                  lambda: self.queue_script("Microglia-Parameter-Estimation.py"),
                                  description="Interactive parameter optimization for microglia detection")
        
        self.create_enhanced_button(microglia_frame, "Import Microglia Parameters",
                                  self.open_import_microglia_parameters_dialog,
                                  description="Load pre-optimized microglia detection parameters")
        
        # Myelin Detection Parameters
        myelin_frame = self.create_enhanced_section_frame(
            scrollable_frame,
            "Myelin Detection Parameters",
            "Optimize parameters for myelin detection"
        )
        
        self.create_enhanced_button(myelin_frame, "Estimate Myelin Parameters",
                                  lambda: self.queue_script("Myelin-Parameter-Estimation.py"),
                                  description="Interactive parameter optimization for myelin detection")
        
        self.create_enhanced_button(myelin_frame, "Import Myelin Parameters",
                                  self.open_import_myelin_parameters_dialog,
                                  description="Load pre-optimized myelin detection parameters")
    
    def setup_detection_page(self):
        """Set up the enhanced Detection page."""
        # Main scrollable frame
        main_frame = ttk.Frame(self.detection_frame)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        canvas = tk.Canvas(main_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Cell Detection
        cell_frame = self.create_enhanced_section_frame(
            scrollable_frame,
            "Cell Detection",
            "Detect and quantify nuclei"
        )
        
        self.create_enhanced_button(cell_frame, "Run Cell Detection",
                                  lambda: self.queue_detection("Cell-Detection.py"),
                                  style='Accent.TButton',
                                  description="Execute cell detection algorithm")
        
        # Checkboxes for options
        options_frame = ttk.Frame(cell_frame)
        options_frame.pack(fill="x", pady=5)
        ttk.Checkbutton(options_frame, text="Create Cell Density Map", 
                       variable=self.cell_density_map_var).pack(anchor="w")
        ttk.Checkbutton(options_frame, text="Create Cell Detection Overview", 
                       variable=self.cell_detection_overview_var).pack(anchor="w")
        
        # Microglia Detection
        microglia_frame = self.create_enhanced_section_frame(
            scrollable_frame,
            "Microglia Detection",
            "Detect and quantify microglia"
        )
        
        self.create_enhanced_button(microglia_frame, "Run Microglia Detection",
                                  lambda: self.queue_detection("Microglia-Detection.py"),
                                  style='Accent.TButton',
                                  description="Execute microglia detection algorithm")
        
        # Checkboxes for microglia options
        microglia_options_frame = ttk.Frame(microglia_frame)
        microglia_options_frame.pack(fill="x", pady=5)
        ttk.Checkbutton(microglia_options_frame, text="Create Microglia Density Map", 
                       variable=self.microglia_density_map_var).pack(anchor="w")
        ttk.Checkbutton(microglia_options_frame, text="Create Microglia Detection Overview", 
                       variable=self.microglia_detection_overview_var).pack(anchor="w")
        
        # Myelin Detection
        myelin_frame = self.create_enhanced_section_frame(
            scrollable_frame,
            "Myelin Detection",
            "Detect and quantify myelination"
        )
        
        self.create_enhanced_button(myelin_frame, "Run Myelin Detection",
                                  lambda: self.queue_detection("Myelin-Detection.py"),
                                  style='Accent.TButton',
                                  description="Execute myelin detection algorithm")
        
        # Checkboxes for myelin options
        myelin_options_frame = ttk.Frame(myelin_frame)
        myelin_options_frame.pack(fill="x", pady=5)
        ttk.Checkbutton(myelin_options_frame, text="Create Myelin Density Map", 
                       variable=self.myelin_density_map_var).pack(anchor="w")
        ttk.Checkbutton(myelin_options_frame, text="Create Myelin Detection Overview", 
                       variable=self.myelin_detection_overview_var).pack(anchor="w")
    
    def create_section_frame(self, parent, title):
        """Create a labeled frame section with proper spacing."""
        frame = ttk.LabelFrame(parent, text=title)
        frame.pack(fill="x", expand=False, padx=10, pady=5, anchor="w")
        return frame
    
    def queue_script(self, script_name, args=None):
        """Add a script to the execution queue."""
        self.process_queue.put({"type": "script", "script_name": script_name, "args": args})
        self.update_log(f"Queued {script_name} for execution")
        
        # Start processing if not already running
        if not self.currently_running:
            self.process_queue_items()
    
    def queue_preprocessing_with_overview(self, script_name, create_overview, step_type):
        """Queue preprocessing script and overview if requested."""
        # Queue the main script
        self.queue_script(script_name)
        
        # If overview checkbox is checked, queue the overview creation script
        if create_overview:
            self.process_queue.put({
                "type": "script", 
                "script_name": "Overview-Creation.py", 
                "args": ["--step-type", step_type]
            })
            self.update_log(f"Queued overview creation for {step_type} step")
    
    def queue_detection(self, script_name):
        """Queue detection script with appropriate options based on checkbox states."""
        # Determine which detection type we're running
        detection_type = None
        if script_name == "Cell-Detection.py":
            detection_type = "cell"
            density_map = self.cell_density_map_var.get()
            overview = self.cell_detection_overview_var.get()
        elif script_name == "Microglia-Detection.py":
            detection_type = "microglia"
            density_map = self.microglia_density_map_var.get()
            overview = self.microglia_detection_overview_var.get()
        elif script_name == "Myelin-Detection.py":
            detection_type = "myelin"
            density_map = self.myelin_density_map_var.get()
            overview = self.myelin_detection_overview_var.get()
        
        # Queue the main detection script (no special arguments needed)
        self.queue_script(script_name)
        
        # Queue density map creation if checkbox is checked
        if density_map:
            density_script = f"{detection_type.capitalize()}-Density-Map.py"
            self.queue_script(density_script)  # No extra arguments needed
            self.update_log(f"Queued {density_script} for {detection_type} density map creation")
        
        # Queue overview creation using Overview-Creation.py if checkbox is checked
        if overview:
            self.process_queue.put({
                "type": "script", 
                "script_name": "Overview-Creation.py", 
                "args": ["--step-type", detection_type]
            })
            self.update_log(f"Queued overview creation for {detection_type} detection")
    
    def start_queue_monitor(self):
        """Start monitoring the process queue in a separate thread."""
        # Instead of starting a monitoring thread, set up a periodic check
        # This ensures all GUI updates happen in the main thread
        self.check_queue()
    
    def check_queue(self):
        """Periodically check the queue for new items to process."""
        if not self.currently_running and not self.process_queue.empty():
            self.process_queue_items()
        
        # Schedule the next check in 100ms
        self.root.after(100, self.check_queue)
    
    def process_queue_items(self):
        """Process items in the queue one by one."""
        if self.currently_running:
            return  # Already processing
        
        if self.process_queue.empty():
            return  # Nothing to process
        
        self.currently_running = True
        self.cancel_requested = False
        self.cancel_button.config(state=tk.NORMAL)
        
        # Start progress bar in the main thread
        self.progress_bar.start(10)
        self.update_status_with_icon("Processing queued operations...", "processing")
        
        # Create a worker thread to actually process the queue
        worker_thread = threading.Thread(target=self._process_queue_worker)
        worker_thread.daemon = True
        worker_thread.start()
    
    def _process_queue_worker(self):
        """Worker thread that actually processes the queue."""
        try:
            while not self.process_queue.empty() and not self.cancel_requested:
                # Get the next task from queue
                task = self.process_queue.get()
                
                if task["type"] == "script":
                    script_name = task["script_name"]
                    args = task["args"]
                    
                    # Run the script
                    self.run_script_with_progress(script_name, args)
                
                # Mark task as done
                self.process_queue.task_done()
            
        finally:
            # Reset state - use after to ensure it runs on the main thread
            self.root.after(0, self._reset_after_processing)
    
    def _reset_after_processing(self):
        """Reset UI state after processing is complete (runs in main thread)."""
        self.currently_running = False
        self.cancel_requested = False
        self.cancel_button.config(state=tk.DISABLED)
        self.progress_bar.stop()
        self.update_status_with_icon("Ready - Select processing steps to begin", "ready")
        self.update_log("Processing completed successfully")
    
    def run_script_with_progress(self, script_name, args=None):
        """Run a script and monitor its output for progress updates."""
        script_path = os.path.join(self.scripts_dir, script_name)
        
        if not os.path.exists(script_path):
            self.update_log(f"Script not found: {script_path}", error=True)
            return
        
        # Special validation for Overview-Creation.py
        if script_name == "Overview-Creation.py":
            if not self.validate_overview_setup():
                self.update_log("Overview creation setup validation failed", error=True)
                return
        
        try:
            cmd = [sys.executable, script_path]
            
            # Add standard arguments for all scripts
            cmd.extend([
                "--data-dir", self.selected_data_dir.get(),
                "--output-dir", self.selected_output_dir.get(),
                "--parameters-dir", self.selected_parameters_dir.get()
            ])
            
            # Add any additional arguments
            if args:
                cmd.extend(args)
            
            # Update status in main thread
            self.root.after(0, lambda: self.update_status_with_icon(f"Running {script_name}...", "processing"))
            self.update_log(f"Starting {script_name}...")
            
            # Log the full command being executed for debugging
            if script_name == "Overview-Creation.py":
                self.update_log(f"Command: {' '.join(cmd)}")
            
            # Run the script with real-time output reading
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1  # Line buffered
            )
            
            # Function to read output in real-time
            def read_output(pipe, is_stderr=False):
                for line in iter(pipe.readline, ''):
                    if line:
                        line_stripped = line.strip()
                        
                        # Only mark as error if it comes from stderr AND contains an error indicator
                        is_error = is_stderr and any(err_word in line_stripped.lower() for err_word in 
                                                    ["error", "exception", "traceback", "failed"])
                        
                        self.update_log(line_stripped, error=is_error)
                        
                        # Check for progress indicators in output
                        if "Found" in line and "image files" in line:
                            self.root.after(0, lambda ln=line_stripped: self.update_status_with_icon(f"Processing {ln}", "processing"))
                        elif "Progress:" in line or "Collecting" in line or "Normalizing" in line:
                            self.root.after(0, lambda ln=line_stripped: self.update_status_with_icon(f"Status: {ln}", "processing"))
                        elif "Overview creation completed" in line:
                            self.root.after(0, lambda: self.update_status_with_icon("Overview completed", "success"))
                        elif "Starting tile selection workflow" in line:
                            self.root.after(0, lambda: self.update_status_with_icon("Initializing tile selection...", "processing"))
                        elif "Overview creation completed. Starting interactive selection" in line:
                            self.root.after(0, lambda: self.update_status_with_icon("Interactive tile selection opened", "info"))
                            # Show user guidance
                            self.root.after(100, self._show_tile_selection_guidance)
                        elif "Launching interactive tile selection interface" in line:
                            self.root.after(0, lambda: self.update_status_with_icon("Waiting for user tile selection...", "info"))
                        elif "Selection complete" in line and "tiles selected" in line:
                            self.root.after(0, lambda: self.update_status_with_icon("Processing tile selection...", "processing"))
                        elif "All tile selection steps complete" in line:
                            self.root.after(0, lambda: self.update_status_with_icon("Tile selection completed", "success"))
            
            # Create threads to read output
            stdout_thread = threading.Thread(target=read_output, args=(process.stdout, False))
            stderr_thread = threading.Thread(target=read_output, args=(process.stderr, True))
            
            # Set threads as daemon so they exit when the main thread exits
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            
            # Start threads
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for process to complete
            return_code = process.wait()
            
            # Wait for output threads to finish
            stdout_thread.join()
            stderr_thread.join()
            
            if return_code != 0:
                self.update_log(f"Error running {script_name} (Return code: {return_code})", error=True)
                self.root.after(0, lambda: self.update_status_with_icon(f"Error running {script_name}", "error"))
            else:
                self.update_log(f"Successfully completed {script_name}")
                self.root.after(0, lambda: self.update_status_with_icon(f"Completed {script_name}", "success"))
        
        except Exception as e:
            self.update_log(f"Exception running {script_name}: {str(e)}", error=True)
            self.root.after(0, lambda: self.update_status_with_icon(f"Error running {script_name}", "error"))

def main():
    """Main function to run the application."""
    root = tk.Tk()
    app = IHCPipelineGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
