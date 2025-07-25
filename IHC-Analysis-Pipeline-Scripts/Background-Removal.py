#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Background Removal Script
-------------------------
This script performs complete background removal for IHC Pipeline:
1. Analyzes RGB background values from a reference tile
2. Generates an optical density (OD) histogram and determines optimal threshold
3. Applies background removal to all tiles using this threshold

Part of the IHC Pipeline GUI application.
"""

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
import json
import csv
import glob
import logging
from PIL import Image

# Configure logging
logger = logging.getLogger("Background-Removal")

def analyze_rgb_background(input_dir, parameters_dir):
    """
    Analyze RGB channel values from a reference tile and save to metadata.csv
    
    Args:
        input_dir: Directory containing input tiles
        parameters_dir: Directory containing parameter files
    
    Returns:
        np.array: Background RGB values as [R, G, B]
    """
    # Find reference tile (assumed to be tile_r0_c0.tif)
    ref_tile_path = os.path.join(input_dir, "Tiles-Medium-L-Channel-Normalized", "tile_r0_c0.tif")
    
    if not os.path.exists(ref_tile_path):
        print(f"Reference tile not found: {ref_tile_path}")
        sys.stdout.flush()
        # Use default white background if file not found
        return np.array([255, 255, 255])
    
    try:
        # Open the TIFF image
        img = Image.open(ref_tile_path)
        
        # Convert to RGB if not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert image to numpy array
        img_array = np.array(img)
        
        # Calculate mean for each channel
        red_mean = np.mean(img_array[:, :, 0])
        green_mean = np.mean(img_array[:, :, 1])
        blue_mean = np.mean(img_array[:, :, 2])
        
        bg_values = np.array([red_mean, green_mean, blue_mean])
        
        # Update metadata.csv
        metadata_path = os.path.join(parameters_dir, "Metadata.csv")
        
        if os.path.exists(metadata_path):
            # Read the existing metadata
            rows = []
            with open(metadata_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                fieldnames = reader.fieldnames
                
                # Add Background-Value if not present
                if "Background-Value" not in fieldnames:
                    fieldnames.append("Background-Value")
                
                for row in reader:
                    rows.append(row)
            
            # Update the rows with background values
            for row in rows:
                row["Background-Value"] = f"{red_mean:.2f},{green_mean:.2f},{blue_mean:.2f}"
            
            # Write back to the file
            with open(metadata_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
                
            print(f"Updated metadata.csv with background values: R={red_mean:.2f}, G={green_mean:.2f}, B={blue_mean:.2f}")
            sys.stdout.flush()
        else:
            print(f"Metadata file not found: {metadata_path}")
            print(f"Background values: R={red_mean:.2f}, G={green_mean:.2f}, B={blue_mean:.2f}")
            sys.stdout.flush()
        
        return bg_values
        
    except Exception as e:
        logger.error(f"Error analyzing background: {e}")
        print(f"Error analyzing background: {e}")
        sys.stdout.flush()
        # Use default white background if error occurs
        return np.array([255, 255, 255])

def rgb_to_od(img, background):
    """Convert RGB image to optical density (OD) space"""
    # Ensure floating point operations
    img = img.astype(float)
    background = background.astype(float)
    
    # Prevent log(0) by setting a minimum value
    eps = 1e-6
    img = np.maximum(img, eps)
    
    # Calculate OD
    od = -np.log10(img / background)
    
    return od

def sample_pixels(image, num_samples=5000):
    """Randomly sample pixels from an image"""
    h, w, c = image.shape
    
    # Randomly sample from the entire image
    y_indices = np.random.randint(0, h, num_samples)
    x_indices = np.random.randint(0, w, num_samples)
    
    # Extract the selected pixels
    sampled_pixels = image[y_indices, x_indices]
    
    return sampled_pixels

def process_file(tif_file, num_samples, background):
    """Process a single TIF file and return OD values of sampled pixels"""
    try:
        # Read the image
        img = cv2.imread(tif_file)
        if img is None:
            logger.warning(f"Could not read {tif_file}, skipping")
            return None
        
        # Convert from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Sample pixels
        sampled_pixels = sample_pixels(img, num_samples)
        
        # Convert to OD
        od_pixels = rgb_to_od(sampled_pixels, background)
        
        # Reshape to (n_samples * 3) and return
        return od_pixels.reshape(-1)
        
    except Exception as e:
        logger.error(f"Error processing {tif_file}: {e}")
        return None

def generate_od_histogram(input_dir, parameters_dir, bg_values, num_samples=5000):
    """
    Generate OD histogram and determine optimal threshold
    
    Args:
        input_dir: Directory containing input tiles
        parameters_dir: Directory for output parameter files
        bg_values: Background RGB values as [R, G, B]
        num_samples: Number of pixels to sample per image
    
    Returns:
        float: Determined OD threshold value
    """
    # Get all TIF files in the input directory
    tiles_dir = os.path.join(input_dir, "Tiles-Medium-L-Channel-Normalized")
    tif_files = glob.glob(os.path.join(tiles_dir, "*.tif"))
    tif_files.extend(glob.glob(os.path.join(tiles_dir, "*.tiff")))
    
    if not tif_files:
        logger.warning(f"No TIF files found in {tiles_dir}")
        print(f"No TIF files found in {tiles_dir}")
        sys.stdout.flush()
        return 0.15  # Default threshold
    
    total_files = len(tif_files)
    print(f"Found {total_files} TIF files")
    sys.stdout.flush()
    
    # Determine number of CPU cores to use
    available_cpus = multiprocessing.cpu_count()
    if available_cpus >= 8:
        max_processes = available_cpus - 6  # Leave cores free for GUI on larger machines
    else:
        max_processes = max(1, available_cpus // 2)  # Use half on smaller machines
    
    print(f"Processing using {max_processes} CPU cores")
    sys.stdout.flush()
    
    # Create a partial function with fixed parameters
    process_file_partial = partial(
        process_file,
        num_samples=num_samples,
        background=bg_values
    )
    
    # Process files in parallel
    all_od_values = []
    
    # Use the 'spawn' method for better cross-platform compatibility
    ctx = multiprocessing.get_context('spawn')
    
    with ctx.Pool(processes=max_processes) as pool:
        # Calculate optimal chunk size for balanced load
        chunksize = max(1, total_files // (max_processes * 4))
        
        # Custom progress tracking
        processed_count = 0
        valid_results = []
        
        # Process files with explicit progress reporting
        for result in pool.imap(process_file_partial, tif_files, chunksize=chunksize):
            processed_count += 1
            
            # Log progress every 5% or at least every 50 files
            if processed_count % max(1, min(50, total_files // 20)) == 0 or processed_count == total_files:
                progress_pct = (processed_count / total_files) * 100
                print(f"Histogram progress: {processed_count}/{total_files} files ({progress_pct:.1f}%)")
                sys.stdout.flush()
            
            if result is not None:
                valid_results.append(result)
    
    # Collect non-None results
    if valid_results:
        all_od_values = np.concatenate(valid_results)
    else:
        logger.warning("No data collected for histogram")
        print("No data collected for histogram")
        sys.stdout.flush()
        return 0.15  # Default threshold
    
    print(f"Calculating histogram from {len(all_od_values)} samples...")
    sys.stdout.flush()
    
    # Create histogram
    plt.figure(figsize=(12, 8))
    
    # Remove extreme outliers (values > 2.0 typically indicate errors or artifacts)
    filtered_values = all_od_values[all_od_values < 2.0]
    
    # Generate histogram data to find the largest frequency drop (high to low)
    hist_values, bin_edges = np.histogram(filtered_values, bins=46)
    
    # Find the bin with the largest frequency drop (negative jump)
    freq_jumps = np.diff(hist_values)
    largest_drop_idx = np.argmin(freq_jumps)  # Find the most negative value (largest drop)
    largest_drop_value = bin_edges[largest_drop_idx + 1]  # Add 1 since diff reduces length by 1
    largest_drop_size = freq_jumps[largest_drop_idx]
    
    # Default threshold if no significant drop is found
    threshold = 0.15
    
    # If a significant drop is found, use that as the threshold
    if largest_drop_size < 0:
        threshold = largest_drop_value
        print(f"Determined OD threshold at largest histogram drop: {threshold:.3f}")
        sys.stdout.flush()
    else:
        print(f"No significant frequency drop found, using default threshold: {threshold:.3f}")
        sys.stdout.flush()
    
    # Plot histogram
    plt.hist(filtered_values, bins=46, alpha=0.7, color='steelblue')
    plt.title(f'Global Optical Density Histogram\n({len(tif_files)} images, {num_samples} samples per image)')
    plt.xlabel('Optical Density')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    
    # Add vertical line at the threshold
    plt.axvline(threshold, color='r', linestyle='dashed', linewidth=2)
    plt.text(threshold + 0.05, plt.gca().get_ylim()[1]*0.9, 
            f'Threshold: {threshold:.3f}', 
            color='r', fontweight='bold')
    
    # Save histogram
    output_file = os.path.join(parameters_dir, 'OD-Histogram.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Histogram saved to {output_file}")
    print(f"Histogram saved to {output_file}")
    sys.stdout.flush()
    
    # Save threshold to JSON
    threshold_file = os.path.join(parameters_dir, 'Background-Threshold.json')
    with open(threshold_file, 'w') as f:
        json.dump({"threshold": threshold}, f)
    logger.info(f"Threshold value saved to {threshold_file}")
    print(f"Threshold value saved to {threshold_file}")
    sys.stdout.flush()
    
    return threshold

def identify_background(od, threshold):
    """Identify background pixels based on optical density"""
    od_norm = np.linalg.norm(od, axis=0)
    bg_mask = od_norm <= threshold
    return bg_mask

def process_image_removal(args):
    """
    Process a single image: remove background pixels and save to output directory
    
    Args:
        args: Tuple containing (img_path, output_dir, bg_rgb, od_threshold)
        
    Returns:
        Error message or None on success
    """
    img_path, output_dir, bg_rgb, od_threshold = args
    
    try:
        # Get filename without full path
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, filename)
        
        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            return f"Error reading {img_path}. Skipping."
            
        # Convert from BGR to RGB for processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to optical density
        h, w, c = img_rgb.shape
        img_flat = img_rgb.reshape(h*w, c).T
        img_flat = img_flat.astype(float)
        bg_rgb = bg_rgb.astype(float)
        
        eps = 1e-6
        img_flat = np.maximum(img_flat, eps)
        
        od = -np.log10(img_flat / bg_rgb[:, np.newaxis])
        
        # Identify background pixels
        bg_mask = identify_background(od, threshold=od_threshold)
        
        # Reshape mask to image dimensions
        bg_mask_img = bg_mask.reshape(h, w)
        
        # Create a copy of the image and set background pixels to white
        img_no_bg = img_rgb.copy()
        img_no_bg[bg_mask_img] = [255, 255, 255]
        
        # Convert back to BGR for saving
        img_no_bg_bgr = cv2.cvtColor(img_no_bg, cv2.COLOR_RGB2BGR)
        
        # Save processed image
        cv2.imwrite(output_path, img_no_bg_bgr)
        
        return None  # Success, no error
        
    except Exception as e:
        return f"Error processing {img_path}: {str(e)}"

def apply_background_removal(input_dir, data_dir, bg_values, threshold):
    """
    Apply background removal to all tiles using determined threshold
    
    Args:
        input_dir: Directory containing input tiles
        data_dir: Base data directory to create output folder
        bg_values: Background RGB values as [R, G, B]
        threshold: OD threshold for background detection
    
    Returns:
        int: Number of successfully processed images
    """
    # Input and output directories
    input_tiles_dir = os.path.join(input_dir, "Tiles-Medium-L-Channel-Normalized")
    output_dir = os.path.join(data_dir, "Tiles-Medium-L-Channel-Normalized-BG-Removed")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all image files in input directory
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_tiles_dir, ext)))
    
    if not image_files:
        logger.warning(f"No image files found in {input_tiles_dir}")
        print(f"No image files found in {input_tiles_dir}")
        sys.stdout.flush()
        return 0
    
    total_files = len(image_files)
    print(f"Found {total_files} image files for background removal")
    sys.stdout.flush()
    
    # Determine number of processes to use
    available_cpus = multiprocessing.cpu_count()
    if available_cpus >= 8:
        max_processes = available_cpus - 6  # Leave cores free for GUI on larger machines
    else:
        max_processes = max(1, available_cpus // 2)  # Use half on smaller machines
    
    print(f"Processing using {max_processes} CPU cores")
    sys.stdout.flush()
    
    # Create a partial function with fixed parameters
    args_list = [(img_path, output_dir, bg_values, threshold) for img_path in image_files]
    
    # Use the 'spawn' method for better cross-platform compatibility
    ctx = multiprocessing.get_context('spawn')
    
    # Process images in parallel
    with ctx.Pool(processes=max_processes) as pool:
        # Calculate optimal chunk size for balanced load
        chunksize = max(1, total_files // (max_processes * 4))
        
        # Custom progress tracking
        processed_count = 0
        errors = []
        
        # Process files with explicit progress reporting
        for result in pool.imap(process_image_removal, args_list, chunksize=chunksize):
            processed_count += 1
            
            # Log progress every 5% or at least every 50 files
            if processed_count % max(1, min(50, total_files // 20)) == 0 or processed_count == total_files:
                progress_pct = (processed_count / total_files) * 100
                print(f"Background removal progress: {processed_count}/{total_files} files ({progress_pct:.1f}%)")
                sys.stdout.flush()
            
            if result is not None:
                errors.append(result)
    
    # Report any errors
    if errors:
        logger.warning(f"{len(errors)} errors occurred:")
        for error in errors:
            logger.warning(f"  {error}")
    
    successful = total_files - len(errors)
    print(f"Background removal completed: {successful} successful, {len(errors)} failed")
    print(f"Processed images saved to: {output_dir}")
    sys.stdout.flush()
    
    return successful

def main():
    """Main function to run the background removal process."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Background Removal for IHC Pipeline")
    parser.add_argument("--data-dir", help="Directory containing the image data")
    parser.add_argument("--output-dir", help="Directory for output files")
    parser.add_argument("--parameters-dir", help="Directory containing parameter files")
    parser.add_argument("--sample-percentage", type=int, default=10, 
                        help="Percentage of pixels to sample from each tile (1-100)")
    parser.add_argument("--manual-threshold", type=float, 
                        help="Manual OD threshold value to override automatic calculation")
    args = parser.parse_args()
    
    # Determine paths based on whether called from GUI or standalone
    if args.data_dir and args.output_dir and args.parameters_dir:
        # Called from GUI - use provided paths
        data_dir = args.data_dir
        output_dir = args.output_dir
        parameters_dir = args.parameters_dir
        
        # Set up log directory - use Logs directory at same level as Data, Results, Parameters
        base_dir = os.path.dirname(args.data_dir)  # Get parent directory of Data
        log_dir = os.path.join(base_dir, "Logs")
        os.makedirs(log_dir, exist_ok=True)
    else:
        print("ERROR: Please provide --data-dir, --output-dir, and --parameters-dir")
        sys.stdout.flush()
        return
    
    # Configure logging to file in the Logs directory
    log_file = os.path.join(log_dir, "background_removal.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
    print("=== Background Removal Process Started ===")
    print(f"Input directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Parameters directory: {parameters_dir}")
    print(f"Log file: {log_file}")
    if args.manual_threshold is not None:
        print(f"Manual threshold specified: {args.manual_threshold:.3f}")
    sys.stdout.flush()
    
    logger.info("=== Background Removal Process Started ===")
    logger.info(f"Input directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Parameters directory: {parameters_dir}")
    if args.manual_threshold is not None:
        logger.info(f"Manual threshold specified: {args.manual_threshold:.3f}")
    
    # Step 1: Analyze RGB background values
    print("\n== Step 1: Analyzing RGB background values ==")
    sys.stdout.flush()
    logger.info("Step 1: Analyzing RGB background values")
    bg_values = analyze_rgb_background(data_dir, parameters_dir)
    
    # Step 2: Determine threshold (automatic or manual)
    if args.manual_threshold is not None:
        # Use manual threshold - skip histogram generation
        print(f"\n== Step 2: Using manual threshold: {args.manual_threshold:.3f} ==")
        sys.stdout.flush()
        logger.info(f"Step 2: Using manual threshold: {args.manual_threshold:.3f}")
        threshold = args.manual_threshold
        
        # Save manual threshold to JSON for consistency
        threshold_file = os.path.join(parameters_dir, 'Background-Threshold.json')
        with open(threshold_file, 'w') as f:
            json.dump({"threshold": threshold, "source": "manual"}, f)
        logger.info(f"Manual threshold value saved to {threshold_file}")
        print(f"Manual threshold value saved to {threshold_file}")
        sys.stdout.flush()
    else:
        # Generate OD histogram and determine threshold automatically
        print("\n== Step 2: Generating OD histogram and determining threshold ==")
        sys.stdout.flush()
        logger.info("Step 2: Generating OD histogram and determining threshold")
        threshold = generate_od_histogram(data_dir, parameters_dir, bg_values, 
                                          num_samples=args.sample_percentage * 500)  # Scale by percentage
    
    # Step 3: Apply background removal
    print("\n== Step 3: Applying background removal to all tiles ==")
    sys.stdout.flush()
    logger.info("Step 3: Applying background removal to all tiles")
    apply_background_removal(data_dir, data_dir, bg_values, threshold)
    
    print("\n=== Background Removal Process Completed ===")
    sys.stdout.flush()
    logger.info("=== Background Removal Process Completed ===")

if __name__ == "__main__":
    # For Windows compatibility with multiprocessing
    multiprocessing.freeze_support()
    
    # Configure basic console logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    try:
        main()
    except Exception as e:
        print(f"ERROR: Unhandled exception: {str(e)}")
        sys.stdout.flush()
        logging.error(f"Unhandled exception: {str(e)}", exc_info=True)
