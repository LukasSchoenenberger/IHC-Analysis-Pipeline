#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L-Channel Normalization Script
------------------------------
This script normalizes the L-channel of LAB color space across a set of image tiles.
It calculates global L-channel statistics and applies min-max normalization to each tile.

Part of the IHC Pipeline GUI application.
"""

import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import glob
import multiprocessing
from functools import partial
import json
import logging
import sys

# Configure logging
logger = logging.getLogger("L-Channel-Normalization")

def extract_l_channel(file_path, sample_percentage=10):
    """
    Extract the L channel values from an image
    
    Args:
        file_path: Path to the image file
        sample_percentage: Percentage of pixels to sample (to reduce memory usage)
        
    Returns:
        L channel values as a flat array or None if error
    """
    try:
        # Read the image
        img = cv2.imread(file_path)
        
        if img is None:
            return None
        
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Extract L channel
        l_channel = lab[:,:,0]
        
        # Flatten the array
        flat_l = l_channel.flatten()
        
        # Sample a percentage of pixels to reduce memory usage if needed
        if sample_percentage < 100:
            num_samples = int(len(flat_l) * sample_percentage / 100)
            indices = np.random.choice(len(flat_l), num_samples, replace=False)
            flat_l = flat_l[indices]
        
        return flat_l
    
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return None

def calculate_global_statistics(input_folder, sample_percentage=10, max_processes=None):
    """
    Calculate global L-channel statistics across all tiles
    
    Args:
        input_folder: Directory containing image tiles
        sample_percentage: Percentage of pixels to sample from each tile
        max_processes: Maximum number of processes to use
        
    Returns:
        Dictionary with global min and max values
    """
    # Find all image files in the input folder
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
    
    if not image_files:
        logger.warning(f"No image files found in {input_folder}")
        return None
    
    # Determine number of processes to use
    if max_processes is None:
        available_cpus = multiprocessing.cpu_count()
        # Use all but 1 core for very large machines, or half for smaller machines
        if available_cpus >= 8:
            max_processes = available_cpus - 6
        else:
            max_processes = max(1, available_cpus // 2)
    
    total_files = len(image_files)
    print(f"Found {total_files} image files. Processing using {max_processes} CPU cores.")
    sys.stdout.flush()  
    
    # Create a partial function with fixed parameters
    extract_func = partial(extract_l_channel, sample_percentage=sample_percentage)
    
    # Process all tiles in parallel to collect L-channel values
    all_l_values = []
    
    # Use the 'spawn' method for better cross-platform compatibility
    ctx = multiprocessing.get_context('spawn')
    
    with ctx.Pool(processes=max_processes) as pool:
        
        chunksize = max(1, total_files // (max_processes * 4))
        
        # Custom progress tracking
        processed_count = 0
        valid_results = []
        
        # Process files in batches for better progress reporting
        for i, result in enumerate(pool.imap(extract_func, image_files, chunksize=chunksize)):
            processed_count += 1
            
            # Log progress every 5% or at least every 50 files
            if processed_count % max(1, min(50, total_files // 20)) == 0 or processed_count == total_files:
                progress_pct = (processed_count / total_files) * 100
                print(f"Statistics progress: {processed_count}/{total_files} files ({progress_pct:.1f}%)")
                sys.stdout.flush()
            
            if result is not None:
                valid_results.append(result)
        
        if not valid_results:
            logger.error("No valid L-channel data collected")
            return None
        
        print(f"Combining statistics from {len(valid_results)} valid files...")
        sys.stdout.flush()
        
        # Combine all L values
        all_l_values = np.concatenate(valid_results)
    
    # Calculate global statistics
    print(f"Calculating min/max values from {len(all_l_values)} pixel samples...")
    sys.stdout.flush()
    
    l_min = float(np.min(all_l_values))
    l_max = float(np.max(all_l_values))
    
    # Create statistics dictionary
    statistics = {
        "l_min": l_min,
        "l_max": l_max,
        "num_pixels_analyzed": len(all_l_values),
        "num_tiles_analyzed": len(valid_results)
    }
    
    print(f"Statistics calculation complete. Min: {l_min}, Max: {l_max}")
    sys.stdout.flush()
    
    logger.info(f"Global L-channel statistics:")
    logger.info(f"  Min: {l_min}")
    logger.info(f"  Max: {l_max}")
    logger.info(f"  Total pixels analyzed: {len(all_l_values)}")
    
    return statistics

def normalize_l_channel_global(img, l_min, l_max):
    """
    Normalize the L channel using global min-max statistics
    
    Args:
        img: Input image
        l_min, l_max: Global min and max L values
    
    Returns:
        Normalized image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Split the LAB image into L, A, and B channels
    l, a, b = cv2.split(lab)
    
    # Use min-max normalization on the L channel
    l_norm = np.round(((l.astype(float) - l_min) / (l_max - l_min)) * 255).astype(np.uint8)
    
    # Merge channels
    lab_norm = cv2.merge([l_norm, a, b])
    
    # Convert back to BGR
    img_norm = cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)
    
    return img_norm

def process_single_file(args):
    """
    Process a single file with global statistics and save to output folder
    
    Args:
        args: Tuple containing (file_path, output_folder, l_min, l_max)
    
    Returns:
        Error message or None on success
    """
    file_path, output_folder, l_min, l_max = args
    
    try:
        # Get the filename
        file_name = os.path.basename(file_path)
        
        # Read the image
        img = cv2.imread(file_path)
        
        if img is None:
            return f"Error reading {file_path}. Skipping."
        
        # Normalize the L channel with global statistics
        img_normalized = normalize_l_channel_global(
            img, l_min, l_max
        )
        
        # Save the normalized image
        output_path = os.path.join(output_folder, file_name)
        cv2.imwrite(output_path, img_normalized)
        
        return None  # Success, no error message
    except Exception as e:
        return f"Error processing {file_path}: {str(e)}"

def normalize_tiles_with_global_stats(input_folder, output_folder, statistics, max_processes=None):
    """
    Normalize all tiles using global L-channel statistics
    
    Args:
        input_folder: Input directory containing tiles
        output_folder: Output directory for normalized tiles
        statistics: Dictionary with global L-channel statistics
        max_processes: Maximum number of processes to use
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all image files in the input folder
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
    
    if not image_files:
        logger.warning(f"No image files found in {input_folder}")
        return
    
    # Determine number of processes to use
    if max_processes is None:
        # Revised CPU count logic to use more cores effectively
        available_cpus = multiprocessing.cpu_count()
        # Use all but 1 core for very large machines, or half for smaller machines
        if available_cpus >= 8:
            max_processes = available_cpus - 6
        else:
            max_processes = max(1, available_cpus // 2)
    
    total_files = len(image_files)
    print(f"Starting normalization of {total_files} image files using {max_processes} CPU cores.")
    sys.stdout.flush()  # Force print to show immediately
    
    # Extract statistics
    l_min = statistics['l_min']
    l_max = statistics['l_max']
    
    # Prepare arguments for each file
    args_list = [(file_path, output_folder, l_min, l_max) 
                for file_path in image_files]
    
    # Use the 'spawn' method for better cross-platform compatibility
    ctx = multiprocessing.get_context('spawn')
    
    # Process files in parallel
    with ctx.Pool(processes=max_processes) as pool:
        # Use a smaller chunksize to balance load better
        chunksize = max(1, total_files // (max_processes * 4))
        
        # Custom progress tracking
        processed_count = 0
        errors = []
        
        # Process files with explicit progress reporting
        for i, result in enumerate(pool.imap(process_single_file, args_list, chunksize=chunksize)):
            processed_count += 1
            
            # Log progress every 5% or at least every 50 files
            if processed_count % max(1, min(50, total_files // 20)) == 0 or processed_count == total_files:
                progress_pct = (processed_count / total_files) * 100
                print(f"Normalization progress: {processed_count}/{total_files} files ({progress_pct:.1f}%)")
                sys.stdout.flush()
            
            if result is not None:
                errors.append(result)
    
    # Report any errors
    if errors:
        logger.warning(f"{len(errors)} errors occurred:")
        for error in errors:
            logger.warning(f"  {error}")
    
    print(f"Normalization completed: {total_files - len(errors)} files processed successfully, {len(errors)} errors.")
    sys.stdout.flush()
    
    logger.info(f"All tiles processed and saved to {output_folder}")

def save_statistics(statistics, parameters_dir):
    """Save statistics to a JSON file in the parameters directory"""
    os.makedirs(parameters_dir, exist_ok=True)
    
    stats_file = os.path.join(parameters_dir, "l_channel_statistics.json")
    with open(stats_file, 'w') as f:
        json.dump(statistics, f, indent=4)
    
    logger.info(f"Statistics saved to {stats_file}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Normalize L-channel of images using global statistics')
    parser.add_argument('--data-dir', help='Base data directory')
    parser.add_argument('--output-dir', help='Base output directory')
    parser.add_argument('--parameters-dir', help='Parameters directory')
    parser.add_argument('--input', help='Input folder containing image files (for standalone use)')
    parser.add_argument('--output', help='Output folder for normalized images (for standalone use)')
    parser.add_argument('--sample', type=int, default=10, help='Percentage of pixels to sample from each tile (1-100)')
    
    args = parser.parse_args()
    
    # Determine input/output paths based on whether called from GUI or standalone
    if args.data_dir and args.output_dir and args.parameters_dir:
        # Called from GUI - use default paths
        # Check if Tiles-Medium is already in the path to avoid duplication
        if args.data_dir.endswith('Tiles-Medium'):
            input_folder = args.data_dir
            
            output_folder = os.path.join(os.path.dirname(args.data_dir), "Tiles-Medium-L-Channel-Normalized")
        else:
            input_folder = os.path.join(args.data_dir, "Tiles-Medium")
            
            output_folder = os.path.join(args.data_dir, "Tiles-Medium-L-Channel-Normalized")
        parameters_dir = args.parameters_dir
        
        # Set up log directory 
        base_dir = os.path.dirname(args.data_dir)  # Get parent directory of Data
        log_dir = os.path.join(base_dir, "Logs")
        os.makedirs(log_dir, exist_ok=True)
    elif args.input and args.output:
        # Called standalone with custom paths
        input_folder = args.input
        output_folder = args.output
        parameters_dir = args.parameters_dir if args.parameters_dir else os.path.dirname(args.output)
        
        # Set up log directory for standalone mode - use a Logs directory in the current directory
        log_dir = "Logs"
        os.makedirs(log_dir, exist_ok=True)
    else:
        print("ERROR: Either provide --data-dir, --output-dir, and --parameters-dir OR provide --input and --output")
        sys.stdout.flush()
        return
    
    # Create output and parameter directories if they don't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(parameters_dir, exist_ok=True)
    
    # Configure logging to file in the Logs directory
    log_file = os.path.join(log_dir, "l_channel_normalization.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Calculate global statistics
    print("Starting L-Channel Normalization")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Log file: {log_file}")
    sys.stdout.flush()
    
    logger.info("Starting L-Channel Normalization")
    logger.info(f"Input folder: {input_folder}")
    logger.info(f"Output folder: {output_folder}")
    
    print("Calculating global L-channel statistics...")
    sys.stdout.flush()
    logger.info("Calculating global L-channel statistics...")
    
    statistics = calculate_global_statistics(
        input_folder, 
        sample_percentage=args.sample
    )
    
    if statistics is None:
        print("ERROR: Failed to calculate global statistics. Exiting.")
        sys.stdout.flush()
        logger.error("Failed to calculate global statistics. Exiting.")
        return
    
    # Save statistics to Parameters folder
    print(f"Saving statistics to {parameters_dir}...")
    sys.stdout.flush()
    save_statistics(statistics, parameters_dir)
    
    # Normalize tiles using global statistics
    print("Starting tile normalization...")
    sys.stdout.flush()
    normalize_tiles_with_global_stats(
        input_folder, 
        output_folder, 
        statistics
    )
    
    print("L-Channel Normalization completed successfully")
    sys.stdout.flush()
    logger.info("L-Channel Normalization completed successfully")

if __name__ == "__main__":
    # For Windows compatibility
    multiprocessing.freeze_support()
    # Configure basic console logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    main()
