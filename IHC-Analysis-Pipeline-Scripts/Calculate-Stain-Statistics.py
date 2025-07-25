#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate Stain Statistics Script
--------------------------------
This script calculates global stain statistics for all stain channels (Hematoxylin, Myelin, Microglia)
from normalized small tiles. It processes tiles to compute concentration statistics for each stain
and saves the results to JSON files in the Parameters directory.

Part of the IHC Pipeline GUI application.
"""

import os
import sys
import numpy as np
import tifffile
import cv2
from pathlib import Path
import json
import time
import argparse
import traceback
import multiprocessing
from functools import partial
from scipy.linalg import pinv
import logging

# Configure logging
logger = logging.getLogger("Calculate-Stain-Statistics")

def load_stain_matrix(stain_vectors_path, verbose=False):
    """
    Load stain matrix from reference_stain_vectors.txt file
    
    Args:
        stain_vectors_path: Path to the reference_stain_vectors.txt file
        verbose: Whether to print information about the loaded matrix
        
    Returns:
        tuple: (stain_matrix, stain_names)
    """
    if not Path(stain_vectors_path).exists():
        raise FileNotFoundError(f"Stain vectors file not found: {stain_vectors_path}")
    
    # Parse the stain vectors file
    stain_matrix = []
    stain_names = []
    
    with open(stain_vectors_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip the header line
            line = line.strip()
            if ':' in line and '[' in line:
                # Handle format: "Name: [R, G, B]" or "Name : [R, G, B]"
                parts = line.split(': [')
                if len(parts) != 2:
                    # Try alternative format with space before colon
                    parts = line.split(' : [')
                
                if len(parts) == 2:
                    name = parts[0].strip()
                    stain_names.append(name)
                    
                    vector_str = parts[1].rstrip(']')
                    # Handle both comma-space and just comma separation
                    vector_values = []
                    for val_str in vector_str.split(','):
                        vector_values.append(float(val_str.strip()))
                    
                    vector = np.array(vector_values)
                    stain_matrix.append(vector)
    
    stain_matrix = np.array(stain_matrix)
    
    # Print information if verbose mode is enabled
    if verbose:
        print(f"Loaded stain matrix with shape: {stain_matrix.shape}")
        print("Stain names:", stain_names)
        sys.stdout.flush()
    
    return stain_matrix, stain_names

def rgb_to_od(img, background):
    """
    Convert RGB image to optical density (OD) space
    
    Args:
        img: Input RGB image
        background: Background RGB values
        
    Returns:
        numpy.ndarray: OD values
    """
    h, w, c = img.shape
    img_flat = img.reshape(h*w, c).T
    img_flat = img_flat.astype(float)
    
    eps = 1e-6
    img_flat = np.maximum(img_flat, eps)
    
    od = -np.log10(img_flat / background[:, np.newaxis])
    
    return od

def identify_background(img):
    """
    Create a background mask that identifies white pixels (value 255)
    
    Args:
        img: Original RGB image
        
    Returns:
        numpy.ndarray: Binary mask where True = background, False = foreground
    """
    h, w, _ = img.shape
    img_flat = img.reshape(h*w, 3)
    
    # Identify pixels where all channels have value 255
    bg_mask = np.all(img_flat == 255, axis=1)
    
    return bg_mask

def identify_stain_indices(stain_names):
    """
    Identify the indices for each stain channel based on the stain names
    
    Args:
        stain_names: List of stain names from the stain matrix
        
    Returns:
        tuple: (hematoxylin_idx, myelin_idx, microglia_idx)
    """
    # Initialize indices
    hematoxylin_idx = None
    myelin_idx = None
    microglia_idx = None
    
    # Search for each stain by keywords in name
    for i, name in enumerate(stain_names):
        name_lower = name.lower()
        
        # Check for hematoxylin (nuclei)
        if any(keyword in name_lower for keyword in ['hematoxylin', 'nuclei', 'h&e', 'he']):
            hematoxylin_idx = i
        
        # Check for myelin
        if any(keyword in name_lower for keyword in ['myelin', 'mbp', 'dab', 'blue']):
            myelin_idx = i
        
        # Check for microglia
        if any(keyword in name_lower for keyword in ['microglia', 'cr3/43', 'brown']):
            microglia_idx = i
    
    # Apply fallbacks if any stain was not identified
    num_stains = len(stain_names)
    
    # Fallback for 3-stain case (most common)
    if num_stains >= 3:
        if hematoxylin_idx is None:
            hematoxylin_idx = 0  # Traditionally first stain
        if myelin_idx is None:
            myelin_idx = 1  # Traditionally second stain
        if microglia_idx is None:
            microglia_idx = 2  # Traditionally third stain
    
    # Fallback for 2-stain case
    elif num_stains == 2:
        if hematoxylin_idx is None:
            hematoxylin_idx = 0
        if myelin_idx is None:
            myelin_idx = 1
        if microglia_idx is None:
            microglia_idx = 1  
    
    # Fallback for 1-stain case
    elif num_stains == 1:
        if hematoxylin_idx is None:
            hematoxylin_idx = 0
        if myelin_idx is None:
            myelin_idx = 0
        if microglia_idx is None:
            microglia_idx = 0
    
    return hematoxylin_idx, myelin_idx, microglia_idx

def calculate_stain_statistics(stain_concentration, bg_mask, h, w):
    """
    Calculate statistics for a stain channel
    
    Args:
        stain_concentration: Flattened concentration values
        bg_mask: Background mask
        h, w: Image dimensions
        
    Returns:
        dict: Statistics for the stain
    """
    # Create mask for negative values
    negative_mask = stain_concentration < 0
    
    # Reshape for calculations
    bg_mask_reshaped = bg_mask.reshape(h, w)
    stain_map = stain_concentration.reshape(h, w)
    
    # Count negative pixels in foreground
    negative_count = np.sum(negative_mask & ~bg_mask)
    
    # Update mask to include negative values
    combined_mask = bg_mask | negative_mask
    combined_mask_reshaped = combined_mask.reshape(h, w)
    
    # Create foreground mask (non-background, non-negative)
    foreground_mask = ~combined_mask_reshaped
    
    # Create original foreground mask (without removing negatives)
    original_foreground = ~bg_mask_reshaped
    original_foreground_count = np.sum(original_foreground)
    
    # Calculate statistics on foreground pixels only
    if np.any(foreground_mask):
        foreground_values = stain_map[foreground_mask]
        stain_mean = np.mean(foreground_values)
        stain_median = np.median(foreground_values)
        stain_std = np.std(foreground_values)
        stain_min = np.min(foreground_values)
        stain_max = np.max(foreground_values)
        foreground_count = np.sum(foreground_mask)
    else:
        stain_mean = 0.0
        stain_median = 0.0
        stain_std = 0.0
        stain_min = 0.0
        stain_max = 0.0
        foreground_count = 0
    
    # Calculate percentage of negative pixels
    percent_negative = (negative_count / original_foreground_count * 100) if original_foreground_count > 0 else 0
    
    return {
        'mean': stain_mean,
        'median': stain_median,
        'std': stain_std,
        'min': stain_min,
        'max': stain_max,
        'foreground_count': foreground_count,
        'negative_count': negative_count,
        'percent_negative': percent_negative
    }

def calculate_tile_stats(args):
    """
    Calculate intensity statistics for a tile for all stain channels
    
    Args:
        args: Tuple containing (tile_path, stain_vectors_path)
        
    Returns:
        dict: Tile statistics or None if error
    """
    tile_path, stain_vectors_path = args
    
    try:
        # Load image
        original_img = tifffile.imread(str(tile_path))
        
        # Ensure proper format
        if len(original_img.shape) == 2:
            # Convert grayscale to RGB
            original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        elif len(original_img.shape) == 3 and original_img.shape[2] > 3:
            # Handle multi-channel images by using only RGB channels
            original_img = original_img[:, :, :3]
        
        # Load stain matrix 
        stain_matrix, stain_names = load_stain_matrix(stain_vectors_path, verbose=False)
        
        # Background RGB values
        background = np.array([255, 255, 255])
        
        # Identify background pixels
        bg_mask = identify_background(original_img)
        
        # Convert to optical density
        od = rgb_to_od(original_img, background)
        
        # Calculate stain concentrations
        stain_matrix_T = stain_matrix.T
        concentrations = np.dot(pinv(stain_matrix_T), od)
        
        # Clip concentrations to the -3 to +3 range
        filtered_concentrations = np.clip(concentrations, -3, 3)
        
        # Zero out concentration values for background pixels
        filtered_concentrations[:, bg_mask] = 0
        
        # Identify indices for each stain
        hematoxylin_idx, myelin_idx, microglia_idx = identify_stain_indices(stain_names)
        
        # Get image dimensions
        h, w, _ = original_img.shape
        
        # Calculate statistics for each stain
        hematoxylin_stats = calculate_stain_statistics(
            filtered_concentrations[hematoxylin_idx, :], bg_mask, h, w)
        
        myelin_stats = calculate_stain_statistics(
            filtered_concentrations[myelin_idx, :], bg_mask, h, w)
        
        microglia_stats = calculate_stain_statistics(
            filtered_concentrations[microglia_idx, :], bg_mask, h, w)
        
        # Return combined statistics
        return {
            'hematoxylin': hematoxylin_stats,
            'myelin': myelin_stats,
            'microglia': microglia_stats,
            'total_pixels': h * w,
            'background_pixels': int(np.sum(bg_mask)),
            'foreground_pixels': int(h * w - np.sum(bg_mask))
        }
    except Exception as e:
        return f"Error calculating statistics for {tile_path}: {str(e)}"

def calculate_global_statistics(tile_files, stain_vectors_path, max_processes=None):
    """
    Calculate global statistics for all tiles for all stain channels
    
    Args:
        tile_files: List of all tile paths
        stain_vectors_path: Path to stain vectors file
        max_processes: Maximum number of processes to use
        
    Returns:
        dict: Global statistics for each stain
    """
    # Determine number of processes to use
    if max_processes is None:
        available_cpus = multiprocessing.cpu_count()
        if available_cpus >= 8:
            max_processes = available_cpus - 1  # Leave one core free for GUI
        else:
            max_processes = max(1, available_cpus // 2)  
    
    total_files = len(tile_files)
    print(f"Processing using {max_processes} CPU cores.")
    print(f"Calculating statistics for {total_files} tiles and all stain channels...")
    sys.stdout.flush()
    
    tile_stats = []
    
    # Prepare arguments for multiprocessing
    args_list = [(tile_path, stain_vectors_path) for tile_path in tile_files]
    
    # Use spawn method for cross-platform compatibility
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=max_processes) as pool:
        # Use smaller chunksize for better load distribution
        chunksize = max(1, total_files // (max_processes * 4))
        
        # Process tiles with progress reporting
        processed_count = 0
        errors = []
        
        for result in pool.imap(calculate_tile_stats, args_list, chunksize=chunksize):
            processed_count += 1
            
            # Report progress at regular intervals
            if processed_count % max(1, min(50, total_files // 20)) == 0 or processed_count == total_files:
                progress_pct = (processed_count / total_files) * 100
                print(f"Progress: {processed_count}/{total_files} tiles ({progress_pct:.1f}%)")
                sys.stdout.flush()
            
            if isinstance(result, dict):
                tile_stats.append(result)
            else:
                # Result is an error message
                errors.append(result)
    
    # Report any errors
    if errors:
        print(f"{len(errors)} errors occurred during processing:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")
        sys.stdout.flush()
    
    if not tile_stats:
        print("Failed to calculate statistics for any tiles!")
        sys.stdout.flush()
        return None
    
    print(f"Successfully processed {len(tile_stats)} tiles")
    sys.stdout.flush()
    
    # Initialize containers for channel statistics
    channels = ['hematoxylin', 'myelin', 'microglia']
    global_stats = {}
    
    # Calculate overall background statistics
    total_pixels = sum(stats['total_pixels'] for stats in tile_stats)
    total_background = sum(stats.get('background_pixels', 0) for stats in tile_stats)
    total_foreground = sum(stats.get('foreground_pixels', 0) for stats in tile_stats)
    
    if total_pixels > 0:
        percent_background = (total_background / total_pixels) * 100
        percent_foreground = (total_foreground / total_pixels) * 100
    else:
        percent_background = 0
        percent_foreground = 0
    
    print(f"\nGlobal Image Statistics:")
    print(f"  Total pixels: {total_pixels}")
    print(f"  Background pixels: {total_background} ({percent_background:.2f}%)")
    print(f"  Foreground pixels: {total_foreground} ({percent_foreground:.2f}%)")
    sys.stdout.flush()
    
    # Calculate global statistics for each channel
    for channel in channels:
        # Extract statistics for this channel
        means = [stats[channel]['mean'] for stats in tile_stats]
        medians = [stats[channel]['median'] for stats in tile_stats]
        stds = [stats[channel]['std'] for stats in tile_stats]
        mins = [stats[channel]['min'] for stats in tile_stats]
        maxs = [stats[channel]['max'] for stats in tile_stats]
        
        # Negative pixel statistics
        total_negative = sum(stats[channel]['negative_count'] for stats in tile_stats)
        total_foreground_channel = sum(stats[channel]['foreground_count'] for stats in tile_stats)
        total_foreground_with_neg = total_foreground_channel + total_negative
        percent_negative = (total_negative / total_foreground_with_neg * 100) if total_foreground_with_neg > 0 else 0
        
        # Calculate global statistics
        mean_global = np.mean(means)
        median_global = np.median(means)
        std_global = np.std(means)
        min_global = np.min(mins)
        max_global = np.max(maxs)
        
        # Print summary for this channel
        print(f"\n{channel.capitalize()} Channel:")
        print(f"  Global Mean: {mean_global:.6f}")
        print(f"  Global Median: {median_global:.6f}")
        print(f"  Global Range: {min_global:.6f} - {max_global:.6f}")
        print(f"  Negative Pixels: {total_negative} ({percent_negative:.2f}% of foreground)")
        sys.stdout.flush()
        
        # Store in global stats
        global_stats[channel] = {
            f'global_{channel}_mean': float(mean_global),
            f'global_{channel}_median': float(median_global),
            f'global_{channel}_std': float(std_global),
            f'global_{channel}_min': float(min_global),
            f'global_{channel}_max': float(max_global),
            'total_negative_pixels': int(total_negative),
            'percent_negative': float(percent_negative),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    # Add global pixel statistics to global stats
    global_stats['image'] = {
        'total_pixels': total_pixels,
        'background_pixels': total_background,
        'foreground_pixels': total_foreground,
        'percent_background': float(percent_background),
        'percent_foreground': float(percent_foreground),
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return global_stats

def save_global_statistics(global_stats, params_dir):
    """
    Save global statistics for each channel to separate JSON files
    
    Args:
        global_stats: Dictionary of global statistics by channel
        params_dir: Directory to save the JSON files
    """
    try:
        # Ensure the directory exists
        os.makedirs(params_dir, exist_ok=True)
        
        # Save each channel to its own file
        for channel, stats in global_stats.items():
            output_path = os.path.join(params_dir, f"{channel}_global_stats.json")
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            print(f"Saved {channel} global statistics to {output_path}")
            sys.stdout.flush()
    except Exception as e:
        print(f"Error saving global statistics: {e}")
        sys.stdout.flush()

def natural_sort_key(s):
    """Create key for natural sorting of tile filenames"""
    import re
    def try_int(text):
        try:
            return int(text)
        except ValueError:
            return text
    return [try_int(c) for c in re.split('([0-9]+)', s)]

def main():
    """Main function to calculate global statistics for all stain channels"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Calculate global statistics for all stain channels")
    parser.add_argument("--data-dir", help="Base data directory")
    parser.add_argument("--output-dir", help="Output directory (not used, kept for compatibility)")
    parser.add_argument("--parameters-dir", help="Parameters directory")
    parser.add_argument("--base-dir", help="Base directory (for standalone use)")
    
    args = parser.parse_args()
    
    # Determine paths based on GUI vs standalone mode
    if args.data_dir and args.parameters_dir:
        # Called from GUI
        base_dir = args.data_dir
        parameters_dir = args.parameters_dir
        log_dir = os.path.join(os.path.dirname(args.data_dir), "Logs")
    elif args.base_dir:
        # Called standalone
        base_dir = args.base_dir
        parameters_dir = args.parameters_dir if args.parameters_dir else os.path.join(base_dir, "Parameters")
        log_dir = os.path.join(base_dir, "Logs")
    else:
        print("ERROR: Either provide --data-dir and --parameters-dir OR provide --base-dir")
        sys.stdout.flush()
        return
    
    # Setup input directory path
    input_tiles_dir = os.path.join(base_dir, "Tiles-Medium-L-Channel-Normalized-BG-Removed-Illumination-Corrected-Stain-Normalized-Small-Tiles")
    stain_vectors_path = os.path.join(parameters_dir, "reference_stain_vectors.txt")
    
    # Create directories
    os.makedirs(parameters_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging to file
    log_file = os.path.join(log_dir, "calculate_stain_statistics.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Check if input directory exists
    if not os.path.exists(input_tiles_dir):
        print(f"ERROR: Input directory not found: {input_tiles_dir}")
        sys.stdout.flush()
        logger.error(f"Input directory not found: {input_tiles_dir}")
        return
    
    # Check for stain vectors file
    if not os.path.exists(stain_vectors_path):
        print(f"ERROR: Stain vectors file not found: {stain_vectors_path}")
        sys.stdout.flush()
        logger.error(f"Stain vectors file not found: {stain_vectors_path}")
        return
    
    # Load and print stain matrix info
    try:
        stain_matrix, stain_names = load_stain_matrix(stain_vectors_path, verbose=True)
        print(f"Using stain vectors from: {stain_vectors_path}")
        sys.stdout.flush()
        logger.info(f"Using stain vectors from: {stain_vectors_path}")
    except Exception as e:
        print(f"ERROR loading stain matrix: {e}")
        sys.stdout.flush()
        logger.error(f"Error loading stain matrix: {e}")
        return
    
    # Track start time for overall processing
    start_time = time.time()
    
    print("Starting Calculate Stain Statistics")
    print(f"Input directory: {input_tiles_dir}")
    print(f"Parameters directory: {parameters_dir}")
    print(f"Log file: {log_file}")
    sys.stdout.flush()
    
    logger.info("Starting Calculate Stain Statistics")
    logger.info(f"Input directory: {input_tiles_dir}")
    logger.info(f"Parameters directory: {parameters_dir}")
    
    # Find all tile files
    tile_extensions = ['*.tif', '*.tiff']
    tile_files = []
    
    for ext in tile_extensions:
        tile_files.extend(Path(input_tiles_dir).glob(ext))
    
    if not tile_files:
        print(f"ERROR: No tiles found in {input_tiles_dir}")
        sys.stdout.flush()
        logger.error(f"No tiles found in {input_tiles_dir}")
        return
    
    # Sort files naturally
    tile_files = sorted(tile_files, key=lambda x: natural_sort_key(x.name))
    
    print(f"Found {len(tile_files)} tiles")
    sys.stdout.flush()
    logger.info(f"Found {len(tile_files)} tiles")
    
    # Calculate global statistics
    global_stats = calculate_global_statistics(tile_files, stain_vectors_path)
    if global_stats is None:
        print("ERROR: Failed to calculate global statistics. Aborting.")
        sys.stdout.flush()
        logger.error("Failed to calculate global statistics")
        return
    
    # Save the global statistics for each channel
    print("Saving global statistics...")
    sys.stdout.flush()
    save_global_statistics(global_stats, parameters_dir)
    
    # Print processing time
    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    print("Global statistics calculation complete!")
    sys.stdout.flush()
    
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    logger.info("Global statistics calculation complete!")

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
