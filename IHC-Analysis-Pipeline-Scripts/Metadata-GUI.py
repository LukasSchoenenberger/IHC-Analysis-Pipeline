#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metadata GUI
------------
This script provides a graphical user interface for creating and editing metadata
for the IHC Pipeline. It allows users to enter values for image dimensions,
background values, and tile dimensions, and saves them in a CSV file.

The script also generates a QuPath groovy script for tile export based on the
metadata values.
"""

import os
import sys
import csv
import tkinter as tk
from tkinter import messagebox
import argparse

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metadata GUI for IHC Pipeline')
    parser.add_argument('--data-dir', type=str, help='Data directory path')
    parser.add_argument('--output-dir', type=str, help='Output directory path')
    parser.add_argument('--parameters-dir', type=str, help='Parameters directory path')
    parser.add_argument('--scripts-dir', type=str, help='Scripts directory path')
    return parser.parse_args()

# Define the columns for the CSV file
columns = [
    "Block-ID", 
    "Pixel-Width", 
    "Pixel-Height", 
    "WSI-Width_Micrometer", 
    "WSI-Width_Pixel", 
    "WSI-Height_Micrometer", 
    "WSI-Height_Pixel", 
    "Tile-Width_Micrometer", 
    "Tile-Width_Pixel", 
    "Tile-Height_Micrometer", 
    "Tile-Height_Pixel", 
    "#Columns", 
    "#Rows", 
    "Background-Value"
]

# Field descriptions for clarity
descriptions = {
    "Block-ID": "Identifier of the block (e.g., 2022_72_23)",
    "Pixel-Width": "Width of a pixel in micrometers",
    "Pixel-Height": "Height of a pixel in micrometers",
    "WSI-Width_Micrometer": "Width of WSI in micrometers",
    "WSI-Width_Pixel": "Width of WSI in pixels",
    "WSI-Height_Micrometer": "Height of WSI in micrometers",
    "WSI-Height_Pixel": "Height of WSI in pixels",
    "Tile-Width_Micrometer": "Width of tile in micrometers",
    "Tile-Width_Pixel": "Width of tile in pixels",
    "Tile-Height_Micrometer": "Height of tile in micrometers",
    "Tile-Height_Pixel": "Height of tile in pixels",
    "#Columns": "Number of columns for export",
    "#Rows": "Number of rows for export",
    "Background-Value": "Background value (leave empty)"
}

class MetadataGUI:
    """GUI for creating and editing metadata for the IHC Pipeline."""
    
    def __init__(self, root, parameters_dir, scripts_dir):
        """Initialize the GUI with the given root window and directories."""
        self.root = root
        self.parameters_dir = parameters_dir
        self.scripts_dir = scripts_dir
        
        # Ensure the parameters directory exists
        os.makedirs(self.parameters_dir, exist_ok=True)
        
        # Set file paths
        self.metadata_csv = os.path.join(self.parameters_dir, "Metadata.csv")
        self.output_groovy_file = os.path.join(self.scripts_dir, "Tile-Export-QuPath.groovy")
        
        # Initialize the GUI elements
        self.setup_gui()
        
        # Load existing metadata if available
        self.load_existing_metadata()
    
    def setup_gui(self):
        """Set up the GUI components."""
        self.root.title("Metadata Entry")
        self.root.geometry("600x650")  # Make the window a bit larger
        
        # Create a frame for the entries with a scrollbar
        frame = tk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Add a canvas with a scrollbar
        canvas = tk.Canvas(frame)
        scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Label at the top
        tk.Label(self.scrollable_frame, text="Enter Metadata Values", font=('Arial', 14, 'bold')).grid(
            row=0, column=0, columnspan=2, pady=10)
        tk.Label(self.scrollable_frame, text="Leave fields blank to keep empty", font=('Arial', 10)).grid(
            row=1, column=0, columnspan=2, pady=5)
        
        # Create labeled entry fields for each column
        self.entry_fields = {}
        for i, col in enumerate(columns):
            tk.Label(self.scrollable_frame, text=f"{col}:", anchor="w", width=20).grid(
                row=i+2, column=0, sticky="w", pady=5)
            entry = tk.Entry(self.scrollable_frame, width=40)
            entry.grid(row=i+2, column=1, pady=5, padx=10, sticky="ew")
            self.entry_fields[col] = entry
            
            # Add field descriptions
            if col in descriptions:
                tk.Label(self.scrollable_frame, text=descriptions[col], 
                         font=('Arial', 8), fg='gray').grid(row=i+2, column=2, sticky="w", pady=5)
        
        # Add save button
        save_button = tk.Button(self.root, text="Save Metadata", command=self.save_metadata, 
                               bg="#4CAF50", fg="white", font=('Arial', 12))
        save_button.pack(pady=20)
    
    def load_existing_metadata(self):
        """Load existing metadata from CSV if available."""
        if os.path.exists(self.metadata_csv):
            try:
                with open(self.metadata_csv, 'r', newline='') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        for col in columns:
                            if col in row:
                                self.entry_fields[col].insert(0, row[col])
                        break  # Only read the first row
            except Exception as e:
                messagebox.showwarning("Warning", f"Error reading existing Metadata.csv: {e}")
    
    def update_groovy_script(self, rows, cols):
        """Create/update the groovy script for QuPath tile export."""
        groovy_template = """// Get the current image from QuPath
def imageData = getCurrentImageData()
def server = imageData.getServer()

// Image dimensions from server
def width = server.getWidth()
def height = server.getHeight()

// *** CONFIGURE ROWS AND COLUMNS HERE ***
def rows = {rows}  // Set from Metadata GUI
def cols = {cols}  // Set from Metadata GUI

// Calculate tile dimensions based on image size and grid
def tileWidth = Math.ceil(width / cols)   // Width for each column
def tileHeight = Math.ceil(height / rows)  // Height for each row

// Create export directory for tiles
def pathTiles = buildFilePath(PROJECT_BASE_DIR, 'Tiles-Medium')  // Export folder name
mkdirs(pathTiles)

// Export all tiles
for (int row = 0; row < rows; row++) {{
    for (int col = 0; col < cols; col++) {{
        // Calculate tile coordinates (without overlap)
        double x = col * tileWidth
        double y = row * tileHeight
        double w = tileWidth
        double h = tileHeight
        
        // Create request for this tile
        def request = RegionRequest.createInstance(
            server.getPath(),
            1,  // Full resolution (downsample = 1)
            x as int,
            y as int,
            w as int,
            h as int
        )
        
        // Export tile
        def tileOutputPath = buildFilePath(pathTiles, String.format('tile_r%d_c%d.tif', row, col))
        writeImageRegion(server, request, tileOutputPath)
        
        print String.format('Exported tile r%d_c%d (%d/%d)\\n', 
            row, col, (row * cols + col + 1), rows * cols)
    }}
}}

print 'Export of all tiles complete!'
print String.format('Exported %d tiles (%d rows x %d columns) to folder: %s', rows * cols, rows, cols, pathTiles)
"""
        try:
            valid_rows = int(rows) if rows.strip() else "/* ROWS NOT AVAILABLE */"
            valid_cols = int(cols) if cols.strip() else "/* COLUMNS NOT AVAILABLE */"
        except ValueError:
            valid_rows = "/* ROWS NOT AVAILABLE */"
            valid_cols = "/* COLUMNS NOT AVAILABLE */"
        
        groovy_content = groovy_template.format(rows=valid_rows, cols=valid_cols)
        
        # Ensure the scripts directory exists
        os.makedirs(os.path.dirname(self.output_groovy_file), exist_ok=True)
        
        with open(self.output_groovy_file, 'w') as file:
            file.write(groovy_content)
    
    def save_metadata(self):
        """Save data from GUI to CSV and update the groovy script."""
        # Collect data from entry fields
        data = {}
        for col in columns:
            value = self.entry_fields[col].get()
            data[col] = value
        
        # Write to CSV
        with open(self.metadata_csv, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerow(data)
        
        # Update groovy script with rows and columns
        self.update_groovy_script(data["#Rows"], data["#Columns"])
        
        messagebox.showinfo("Success", f"Metadata saved to {self.metadata_csv}\nGroovy script saved to {self.output_groovy_file}")
        self.root.destroy()

def main():
    """Main function to run the application."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Determine directories to use
    # Default to predefined paths if not provided via command line
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parameters_dir = args.parameters_dir if args.parameters_dir else os.path.join(base_dir, "..", "Parameters")
    scripts_dir = args.scripts_dir if args.scripts_dir else os.path.join(base_dir, "..", "Scripts")
    
    # Create GUI
    root = tk.Tk()
    app = MetadataGUI(root, parameters_dir, scripts_dir)
    root.mainloop()

if __name__ == "__main__":
    main()
