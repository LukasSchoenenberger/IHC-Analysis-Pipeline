#!/usr/bin/env python3
import os
import sys
import argparse
import glob
import numpy as np
import nibabel as nib
import tkinter as tk
from tkinter import filedialog, messagebox
import SimpleITK as sitk
import tempfile
import shutil

def find_reference_flash(base_dir):
    """
    Find a FLASH MRI sequence in the base directory to use as reference
    """
    # Look for files with FLASH in the name
    flash_files = glob.glob(os.path.join(base_dir, "*FLASH*.nii*"))
    
    if not flash_files:
        # No FLASH file found automatically
        return None
    
    # Return the first matching file
    return flash_files[0]

def select_input_folder_gui():
    """
    Opens a GUI directory dialog to select the input folder containing MRI sequences
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    folder_path = filedialog.askdirectory(
        title="Select Input Folder with MRI Sequences",
        initialdir=os.getcwd()
    )
    
    root.destroy()  # Clean up the root window
    
    return folder_path if folder_path else None

def select_reference_file_gui():
    """
    Opens a GUI file dialog to select the reference FLASH file
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    file_path = filedialog.askopenfilename(
        title="Select Reference FLASH MRI",
        filetypes=[
            ("NIfTI files", "*.nii *.nii.gz"),
            ("All files", "*.*")
        ],
        initialdir=os.getcwd()
    )
    
    root.destroy()  # Clean up the root window
    
    return file_path if file_path else None

def direct_resample(input_file, reference_file, output_file):
    """
    Directly resample the input to match reference dimensions without temporary files
    """
    # Load images with SimpleITK
    input_sitk = sitk.ReadImage(input_file)
    reference_sitk = sitk.ReadImage(reference_file)
    
    # Create a resampler that matches dimensions but keeps content as is
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_sitk)
    
    # Use identity transform - just change dimensions and spacing, not content
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkLinear)
    
    # Perform the resampling
    output_sitk = resampler.Execute(input_sitk)
    
    # Write directly to the output file
    sitk.WriteImage(output_sitk, output_file)
    
    return True

def process_mri_sequence(input_file, reference_file, output_dir, verbose=False):
    """
    Process a single MRI sequence file by matching dimensions to the reference
    """
    print(f"Processing: {input_file}")
    
    # Create output filename
    base_name = os.path.basename(input_file)
    name_parts = os.path.splitext(base_name)
    
    # Handle double extension (.nii.gz)
    if name_parts[1] == '.gz':
        name_parts = os.path.splitext(name_parts[0])
        output_filename = f"{name_parts[0]}_registered.nii.gz"
    else:
        output_filename = f"{name_parts[0]}_registered{name_parts[1]}"
    
    output_path = os.path.join(output_dir, output_filename)
    
    # Use direct resampling with SimpleITK
    success = direct_resample(input_file, reference_file, output_path)
    
    if success:
        print(f"Saved registered image to: {output_path}")
        
        # Print output image information if verbose
        if verbose:
            try:
                out_img = nib.load(output_path)
                print(f"Output image dimensions: {out_img.shape}")
                print(f"Output image voxel size: {out_img.header.get_zooms()}")
            except Exception as e:
                print(f"Warning: Could not read output image info: {e}")
    
    return output_path if success else None

def main():
    parser = argparse.ArgumentParser(description="Register MRI sequences to a reference FLASH sequence")
    parser.add_argument("--input-dir", type=str, default=None, 
                        help="Directory containing input MRI sequences (if not provided, GUI will open)")
    parser.add_argument("--reference", type=str, default=None,
                        help="Path to reference FLASH MRI (optional, will search for FLASH file if not provided)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information during processing")
    parser.add_argument("--gui", action="store_true",
                        help="Force GUI mode for folder selection")
    args = parser.parse_args()
    
    verbose = args.verbose
    
    # Get the base directory (where the script is running from)
    base_dir = os.getcwd()
    
    # Get input directory - use GUI if no argument provided or if GUI flag is set
    input_dir = args.input_dir
    
    if not input_dir or args.gui:
        print("Opening folder selection dialog...")
        input_dir = select_input_folder_gui()
        
        if not input_dir:
            print("No input folder selected. Exiting.")
            return 1
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return 1
    
    print(f"Input directory: {input_dir}")
    
    # Always use "MRI_sequences_registered" as output directory in the base directory
    output_dir = os.path.join(base_dir, "MRI_sequences_registered")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)
    
    print(f"Output directory: {output_dir}")
    
    # Find or select the reference FLASH file
    reference_file = args.reference
    if not reference_file:
        print("Searching for reference FLASH file...")
        reference_file = find_reference_flash(base_dir)
        
        if not reference_file:
            print("No FLASH reference file found automatically.")
            print("Please select the reference FLASH file using the dialog...")
            reference_file = select_reference_file_gui()
            
            if not reference_file:
                print("No reference file selected. Exiting.")
                return 1
    
    # Check if reference file exists
    if not os.path.exists(reference_file):
        print(f"Error: Reference file '{reference_file}' does not exist.")
        return 1
    
    print(f"Using reference file: {reference_file}")
    
    # Print reference image information if verbose
    if verbose:
        try:
            ref_img = nib.load(reference_file)
            print(f"Reference image dimensions: {ref_img.shape}")
            print(f"Reference image voxel size: {ref_img.header.get_zooms()}")
            print(f"Reference image datatype: {ref_img.get_data_dtype()}")
        except Exception as e:
            print(f"Warning: Could not read reference image info: {e}")
    
    # Get all NIfTI files in the input directory
    input_files = []
    for ext in ['.nii', '.nii.gz']:
        pattern = os.path.join(input_dir, f'*{ext}')
        found_files = glob.glob(pattern)
        input_files.extend(found_files)
    
    if not input_files:
        print(f"Error: No NIfTI files (.nii or .nii.gz) found in '{input_dir}'.")
        return 1
    
    print(f"Found {len(input_files)} NIfTI files to process:")
    for i, file_path in enumerate(input_files, 1):
        print(f"  {i}. {os.path.basename(file_path)}")
    
    # Process each input file
    processed_files = []
    failed_files = []
    
    for i, input_file in enumerate(input_files, 1):
        try:
            print(f"\n--- Processing file {i}/{len(input_files)} ---")
            
            # Print input image information if verbose
            if verbose:
                try:
                    in_img = nib.load(input_file)
                    print(f"Input image dimensions: {in_img.shape}")
                    print(f"Input image voxel size: {in_img.header.get_zooms()}")
                    print(f"Input image datatype: {in_img.get_data_dtype()}")
                except Exception as e:
                    print(f"Warning: Could not read input image info: {e}")
            
            output_file = process_mri_sequence(input_file, reference_file, output_dir, verbose)
            if output_file:
                processed_files.append(output_file)
            else:
                failed_files.append(input_file)
            
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            failed_files.append(input_file)
            if verbose:
                import traceback
                traceback.print_exc()
    
    # Print summary
    print(f"\n=== Registration Summary ===")
    print(f"Total files found: {len(input_files)}")
    print(f"Successfully processed: {len(processed_files)}")
    print(f"Failed: {len(failed_files)}")
    
    if failed_files:
        print(f"\nFailed files:")
        for failed_file in failed_files:
            print(f"  - {os.path.basename(failed_file)}")
    
    if processed_files:
        print(f"\nOutput files saved to: {output_dir}")
        print(f"Processed files:")
        for processed_file in processed_files:
            print(f"  - {os.path.basename(processed_file)}")
    
    print("\nRegistration complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
