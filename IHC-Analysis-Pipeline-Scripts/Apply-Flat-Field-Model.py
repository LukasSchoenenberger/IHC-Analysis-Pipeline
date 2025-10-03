#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply Flat Field Model Script
-----------------------------
This script applies a flat-field correction to image tiles to remove illumination artifacts.
It uses a precomputed flat-field distortion model to correct uneven illumination across tiles.

Part of the IHC Pipeline GUI application.
"""

import os
import sys
import numpy as np
import tifffile as tiff
import glob
import time
import re
import argparse
import multiprocessing
from contextlib import contextmanager
import gc
import logging
from functools import partial

# Configure logging
logger = logging.getLogger("Apply-Flat-Field-Model")

@contextmanager
def tile_loader(tile_path):
    """
    Context manager to load a tile, use it, and then free memory.
    
    Args:
        tile_path (str): Path to the tile file.
        
    Yields:
        ndarray: Loaded tile data.
    """
    try:
        tile = tiff.imread(tile_path)
        yield tile
    finally:
        # Explicitly delete to help with garbage collection
        del tile
        gc.collect()


def apply_flat_field_correction(tile, flat_field_model, ignore_value=255, strength=0.98):
    """
    Apply flat field correction to a tile with brightness preservation,
    but only to pixels that don't have the specified ignore_value.
    
    Args:
        tile (ndarray): Input tile to correct.
        flat_field_model (ndarray): Flat field distortion model.
        ignore_value (int): Pixel value to ignore (typically 255 for background).
        strength (float): Strength of the correction (1.0 = normal, >1.0 = stronger, <1.0 = weaker)
        
    Returns:
        ndarray: Corrected tile.
    """
    # Create a copy of the input tile to avoid modifying the original
    corrected = tile.copy()
    
    # Convert to float32 for calculation
    tile_float = tile.astype(np.float32)
    
    # Create a mask for pixels that are not ignore_value in all channels
    # For RGB images, a pixel is considered background if all three channels have value 255
    if len(tile.shape) == 3 and tile.shape[2] == 3:  # RGB image
        mask_r = tile[:, :, 0] != ignore_value
        mask_g = tile[:, :, 1] != ignore_value
        mask_b = tile[:, :, 2] != ignore_value
        # A pixel is foreground if ANY channel is not 255
        foreground_mask = mask_r | mask_g | mask_b
    else:  # Grayscale image
        foreground_mask = tile != ignore_value
    
    # Count foreground pixels
    num_foreground = np.sum(foreground_mask)
    
    # If there are no foreground pixels, return the original tile
    if num_foreground == 0:
        return corrected
    
    # Calculate average intensity of foreground pixels in original image and flat-field model
    # We only want to use foreground pixels for normalization
    if len(tile.shape) == 3:  # RGB image
        # Process each channel separately
        for c in range(tile.shape[2]):
            # Create a channel-specific mask
            channel_mask = tile[:, :, c] != ignore_value
            
            # If no foreground pixels in this channel, skip
            if np.sum(channel_mask) == 0:
                continue
            
            # Calculate average of foreground pixels for this channel
            tile_avg = np.mean(tile_float[:, :, c][channel_mask])
            flat_field_avg = np.mean(flat_field_model[:, :, c][channel_mask])
            
            # Apply correction only to foreground pixels
            # Create a temporary array for the corrected values
            corrected_channel = tile_float[:, :, c].copy()
            
            # Apply the correction formula with adjustable strength
            epsilon = 1e-10  # To avoid division by zero
            corrected_channel[channel_mask] = (tile_float[:, :, c][channel_mask] / 
                                              ((flat_field_model[:, :, c][channel_mask] ** strength) + epsilon)) * flat_field_avg
            
            # Clip to valid range
            if np.issubdtype(tile.dtype, np.integer):
                max_val = np.iinfo(tile.dtype).max
                corrected_channel = np.clip(corrected_channel, 0, max_val)
            else:
                corrected_channel = np.clip(corrected_channel, 0, 1.0)
            
            # Update only the foreground pixels in the output image
            corrected[:, :, c][channel_mask] = corrected_channel[channel_mask].astype(tile.dtype)
    else:  # Grayscale image
        # Calculate average of foreground pixels
        tile_avg = np.mean(tile_float[foreground_mask])
        flat_field_avg = np.mean(flat_field_model[foreground_mask])
        
        # Apply correction only to foreground pixels with adjustable strength
        epsilon = 1e-10  # To avoid division by zero
        corrected_values = (tile_float[foreground_mask] / 
                           ((flat_field_model[foreground_mask] ** strength) + epsilon)) * flat_field_avg
        
        # Clip to valid range
        if np.issubdtype(tile.dtype, np.integer):
            max_val = np.iinfo(tile.dtype).max
            corrected_values = np.clip(corrected_values, 0, max_val)
        else:
            corrected_values = np.clip(corrected_values, 0, 1.0)
        
        # Update only the foreground pixels in the output image
        corrected[foreground_mask] = corrected_values.astype(tile.dtype)
    
    return corrected


def extract_row_col_from_filename(filename):
    """
    Extract row and column indices from a tile filename.
    
    Args:
        filename (str): Tile filename.
        
    Returns:
        tuple: (row_index, col_index) or (None, None) if extraction fails.
    """
    # Try to match r{row}_c{col} pattern
    match = re.search(r'r(\d+)_c(\d+)', filename)
    if match:
        row = int(match.group(1))
        col = int(match.group(2))
        return row, col
    return None, None


def preprocess_flat_field_model(model, enhance_contrast=0.0):
    """
    Enhance the contrast of the flat field model to make illumination patterns more visible.
    
    Args:
        model (ndarray): The original flat field model.
        enhance_contrast (float): Strength of contrast enhancement (0.0 = none, 1.0 = maximum).
        
    Returns:
        ndarray: The enhanced flat field model.
    """
    if enhance_contrast <= 0:
        return model.copy()
    
    enhanced_model = model.copy().astype(np.float32)
    
    for c in range(3):
        channel = enhanced_model[:, :, c]
        
        # Calculate percentiles for contrast stretching
        p_low = np.percentile(channel, 10 * enhance_contrast)
        p_high = np.percentile(channel, 100 - 10 * enhance_contrast)
        
        # Apply contrast stretching
        if p_high > p_low:
            # Linear contrast stretch
            channel = np.clip((channel - p_low) / (p_high - p_low), 0, 1)
            
            # Blend with original based on enhancement strength
            enhanced_model[:, :, c] = channel * enhance_contrast + model[:, :, c] * (1 - enhance_contrast)
            
            # Normalize to have same mean as original
            orig_mean = np.mean(model[:, :, c])
            enhanced_model[:, :, c] *= orig_mean / np.mean(enhanced_model[:, :, c])
    
    return enhanced_model


def group_tiles_into_batches(tile_paths, grid_size=2):
    """
    Group tile paths into logical batches based on grid alignment.
    Handles any grid size (2x2, 3x3, etc.) and incomplete batches.
    
    Args:
        tile_paths (list): List of paths to tile files.
        grid_size (int): Size of the grid (2 for 2x2, 3 for 3x3, etc.).
        
    Returns:
        list: List of batches, where each batch is a dict mapping (row_mod_grid, col_mod_grid) to (row, col, path) tuples.
    """
    # Extract row and column for each tile
    indexed_tiles = []
    for path in tile_paths:
        filename = os.path.basename(path)
        row, col = extract_row_col_from_filename(filename)
        if row is not None and col is not None:
            indexed_tiles.append((row, col, path))
        else:
            print(f"Warning: Could not extract row/column from {filename}, skipping")
            sys.stdout.flush()
    
    # Sort tiles by row and column
    indexed_tiles.sort()
    
    # Group tiles into batches based on their position in the grid
    batches = {}  # Maps (base_row, base_col) to a dict of positions
    for row, col, path in indexed_tiles:
        # Determine the base row/col for this batch (floor division to get the boundary)
        base_row = (row // grid_size) * grid_size
        base_col = (col // grid_size) * grid_size
        
        # Calculate position within the grid
        row_pos = row % grid_size
        col_pos = col % grid_size
        
        # Add to appropriate batch
        batch_key = (base_row, base_col)
        if batch_key not in batches:
            batches[batch_key] = {}
        
        position_key = (row_pos, col_pos)
        batches[batch_key][position_key] = (row, col, path)
    
    # Convert to list of batches
    batch_list = list(batches.values())
    
    # Count tiles in each batch
    complete_count = sum(1 for batch in batch_list if len(batch) == grid_size * grid_size)
    incomplete_count = len(batch_list) - complete_count
    
    print(f"Grouped {len(indexed_tiles)} tiles into {len(batch_list)} batches (grid size: {grid_size}x{grid_size}):")
    print(f"  - Complete {grid_size}x{grid_size} batches: {complete_count}")
    print(f"  - Incomplete batches: {incomplete_count}")
    sys.stdout.flush()
    
    return batch_list


def determine_grid_size(flat_field_shape, tile_shape):
    """
    Determine the grid size based on the flat field model and tile dimensions.
    
    Args:
        flat_field_shape (tuple): Shape of the flat field model (height, width).
        tile_shape (tuple): Shape of a single tile (height, width).
        
    Returns:
        int: Grid size (2 for 2x2, 3 for 3x3, etc.).
    """
    height_ratio = flat_field_shape[0] / tile_shape[0]
    width_ratio = flat_field_shape[1] / tile_shape[1]
    
    # Round to nearest integer
    height_grid = round(height_ratio)
    width_grid = round(width_ratio)
    
    if height_grid != width_grid:
        print(f"Warning: Non-square grid detected. Height ratio: {height_ratio}, Width ratio: {width_ratio}")
        print(f"Using grid size: {height_grid}x{width_grid}")
        sys.stdout.flush()
    
    return height_grid  # Assuming square grid


def process_single_tile(args):
    """
    Process a single tile with flat field correction
    
    Args:
        args: Tuple containing (position, row, col, tile_path, output_dir, flat_field_region, ignore_value, strength)
        
    Returns:
        Error message string or None on success
    """
    try:
        position, row, col, tile_path, output_dir, flat_field_region, ignore_value, strength = args
        
        # Get the filename
        file_name = os.path.basename(tile_path)
        
        # Prepare output path
        output_path = os.path.join(output_dir, file_name)
        
        # Load tile, apply correction, and save
        with tile_loader(tile_path) as tile:
            # Apply the correction
            corrected_tile = apply_flat_field_correction(
                tile, flat_field_region, ignore_value=ignore_value, strength=strength
            )
            
            # Save the corrected tile
            tiff.imwrite(output_path, corrected_tile)
            
        return None  # Success, no error message
        
    except Exception as e:
        return f"Error processing tile r{row}_c{col}: {str(e)}"


def correct_tiles_with_flat_field_parallel(input_dir, output_dir, flat_field_model_path, 
                                          ignore_value=255, strength=0.98, enhance_contrast=0.0,
                                          max_processes=None):
    """
    Apply flat field correction to tiles in parallel using multiprocessing
    
    Args:
        input_dir (str): Directory containing input tiles.
        output_dir (str): Directory to save corrected tiles.
        flat_field_model_path (str): Path to the flat field model image.
        ignore_value (int): Pixel value to ignore (typically 255 for background).
        strength (float): Strength of the correction (1.0 = normal, >1.0 = stronger, <1.0 = weaker)
        enhance_contrast (float): Amount of contrast enhancement for flat field model (0.0 = none)
        max_processes (int, optional): Maximum number of processes to use for multiprocessing.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the flat field model
    print(f"Loading flat field model from {flat_field_model_path}")
    sys.stdout.flush()
    
    try:
        flat_field_model = tiff.imread(flat_field_model_path).astype(np.float32)
    except Exception as e:
        print(f"ERROR: Failed to load flat field model: {str(e)}")
        sys.stdout.flush()
        logger.error(f"Failed to load flat field model: {str(e)}")
        return
    
    # Get all tile paths
    tile_paths = glob.glob(os.path.join(input_dir, "*.tif")) + glob.glob(os.path.join(input_dir, "*.tiff"))
    if not tile_paths:
        print(f"ERROR: No TIFF files found in {input_dir}")
        sys.stdout.flush()
        logger.error(f"No TIFF files found in {input_dir}")
        return
    
    print(f"Found {len(tile_paths)} tile files to process")
    sys.stdout.flush()
    
    # Load first tile to get dimensions
    try:
        with tile_loader(tile_paths[0]) as first_tile:
            tile_height, tile_width = first_tile.shape[:2]
    except Exception as e:
        print(f"ERROR: Failed to load first tile: {str(e)}")
        sys.stdout.flush()
        logger.error(f"Failed to load first tile: {str(e)}")
        return
    
    # Determine the grid size
    grid_size = determine_grid_size(flat_field_model.shape[:2], (tile_height, tile_width))
    print(f"Auto-detected grid size: {grid_size}x{grid_size}")
    sys.stdout.flush()
    
    # Check if flat field model has correct size
    expected_height = tile_height * grid_size
    expected_width = tile_width * grid_size
    
    if flat_field_model.shape[0] != expected_height or flat_field_model.shape[1] != expected_width:
        print(f"Warning: Flat field model size ({flat_field_model.shape[0]}x{flat_field_model.shape[1]}) " +
                f"does not match expected size ({expected_height}x{expected_width})")
        sys.stdout.flush()
        
        # Resize flat field model if needed
        if flat_field_model.shape[0] >= expected_height and flat_field_model.shape[1] >= expected_width:
            print(f"Cropping flat field model to expected size")
            flat_field_model = flat_field_model[:expected_height, :expected_width]
            sys.stdout.flush()
        else:
            print(f"ERROR: Flat field model is too small, cannot proceed")
            sys.stdout.flush()
            logger.error("Flat field model is too small, cannot proceed")
            return
    
    # Enhance the model contrast if requested
    if enhance_contrast > 0:
        print(f"Enhancing flat field model contrast (level: {enhance_contrast:.2f})...")
        flat_field_model = preprocess_flat_field_model(flat_field_model, enhance_contrast)
        sys.stdout.flush()
    
    # Group tiles into logical batches
    batches = group_tiles_into_batches(tile_paths, grid_size=grid_size)
    
    # CPU core allocation
    if max_processes is None:
        available_cpus = multiprocessing.cpu_count()
        if available_cpus >= 8:
            max_processes = available_cpus - 1  # Leave one core free for GUI
        else:
            max_processes = max(1, available_cpus // 2)  # Use half on smaller machines
    
    print(f"Processing using {max_processes} CPU cores.")
    print(f"Applying flat field correction with strength={strength:.2f} and ignore_value={ignore_value}")
    sys.stdout.flush()
    
    # Use the 'spawn' method for better cross-platform compatibility
    ctx = multiprocessing.get_context('spawn')
    
    # Process batches one by one (tiles within batches are processed in parallel)
    start_time = time.time()
    total_tiles = sum(len(batch) for batch in batches)
    processed_count = 0
    error_count = 0
    
    for batch_idx, batch in enumerate(batches):
        # Show which tiles are in this batch
        positions = {(row % grid_size, col % grid_size): (row, col) for row, col, _ in batch.values()}
        position_str = []
        for row_pos in range(grid_size):
            for col_pos in range(grid_size):
                if (row_pos, col_pos) in positions:
                    row, col = positions[(row_pos, col_pos)]
                    position_str.append(f"r{row}_c{col}")
                else:
                    position_str.append("empty")
        
        print(f"Processing batch {batch_idx+1}/{len(batches)} ({len(batch)}/{grid_size*grid_size} tiles): {', '.join(position_str)}")
        sys.stdout.flush()
        
        # Prepare task arguments for this batch
        task_args = []
        for position, (row, col, tile_path) in batch.items():
            row_pos, col_pos = position
            row_offset = row_pos * tile_height
            col_offset = col_pos * tile_width
            
            # Extract the appropriate region of the flat field model
            flat_field_region = flat_field_model[row_offset:row_offset+tile_height, 
                                                col_offset:col_offset+tile_width]
            
            # Create task argument tuple
            task_args.append((position, row, col, tile_path, output_dir, 
                            flat_field_region, ignore_value, strength))
        
        # Process tiles in this batch in parallel
        with ctx.Pool(processes=max_processes) as pool:
            chunksize = max(1, len(task_args) // (max_processes * 4))
            
            # Process the batch and collect results
            batch_errors = []
            for result in pool.imap(process_single_tile, task_args, chunksize=chunksize):
                processed_count += 1
                
                # Report progress at regular intervals
                if processed_count % max(1, min(50, total_tiles // 20)) == 0 or processed_count == total_tiles:
                    progress_pct = (processed_count / total_tiles) * 100
                    print(f"Progress: {processed_count}/{total_tiles} tiles ({progress_pct:.1f}%)")
                    sys.stdout.flush()
                
                if result is not None:
                    batch_errors.append(result)
                    error_count += 1
            
            # Report any errors in this batch
            if batch_errors:
                for error in batch_errors:
                    print(f"  {error}")
                    sys.stdout.flush()
    
    # Report final statistics
    end_time = time.time()
    print(f"Flat field correction completed in {end_time - start_time:.2f} seconds")
    print(f"Processed {processed_count} tiles with {error_count} errors")
    print(f"Corrected tiles saved to {output_dir}")
    sys.stdout.flush()
    
    # Log completion
    logger.info(f"Flat field correction completed. Processed {processed_count} tiles with {error_count} errors.")


def main():
    """Main function to run the script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Apply flat field correction to image tiles.')
    parser.add_argument('--data-dir', help='Base data directory')
    parser.add_argument('--output-dir', help='Base output directory')
    parser.add_argument('--parameters-dir', help='Parameters directory')
    parser.add_argument('--input', help='Input folder containing image files (for standalone use)')
    parser.add_argument('--output', help='Output folder for corrected images (for standalone use)')
    parser.add_argument('--model', help='Path to flat field model (for standalone use)')
    parser.add_argument('--strength', type=float, default=0.98, 
                      help='Strength of the correction (default: 0.98, higher = stronger)')
    
    args = parser.parse_args()
    
    # Determine input/output paths based on whether called from GUI or standalone
    if args.data_dir and args.output_dir and args.parameters_dir:
        # Called from GUI - use default paths
        input_folder = os.path.join(args.data_dir, "Tiles-Medium-L-Channel-Normalized-BG-Removed")
        output_folder = os.path.join(args.data_dir, "Tiles-Medium-L-Channel-Normalized-BG-Removed-Illumination-Corrected")
        flat_field_model = os.path.join(args.parameters_dir, "flat_field_model.tif")
        
        # Set up log directory - use Logs directory that is at the same level as Data, Results, Parameters
        base_dir = os.path.dirname(args.data_dir)  # Get parent directory of Data
        log_dir = os.path.join(base_dir, "Logs")
        os.makedirs(log_dir, exist_ok=True)
    elif args.input and args.output:
        # Called standalone with custom paths
        input_folder = args.input
        output_folder = args.output
        flat_field_model = args.model if args.model else os.path.join(
            args.parameters_dir if args.parameters_dir else "Parameters", 
            "flat_field_model.tif"
        )
        
        # Set up log directory for standalone mode - use a Logs directory in the current directory
        log_dir = "Logs"
        os.makedirs(log_dir, exist_ok=True)
    else:
        print("ERROR: Either provide --data-dir, --output-dir, and --parameters-dir OR provide --input and --output")
        sys.stdout.flush()
        return
    
    # Create output directories if they don't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Configure logging to file in the Logs directory
    log_file = os.path.join(log_dir, "apply_flat_field_model.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log start of processing
    print(f"Starting Apply Flat Field Model")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Flat field model: {flat_field_model}")
    print(f"Correction strength: {args.strength}")
    print(f"Log file: {log_file}")
    sys.stdout.flush()
    
    logger.info("Starting Apply Flat Field Model")
    logger.info(f"Input folder: {input_folder}")
    logger.info(f"Output folder: {output_folder}")
    logger.info(f"Flat field model: {flat_field_model}")
    
    # Run the correction
    correct_tiles_with_flat_field_parallel(
        input_folder,
        output_folder,
        flat_field_model,
        ignore_value=255,  
        strength=args.strength,
        enhance_contrast=0.0  
    )
    
    print("Apply Flat Field Model completed successfully")
    sys.stdout.flush()
    logger.info("Apply Flat Field Model completed successfully")


if __name__ == "__main__":
    # For Windows compatibility
    multiprocessing.freeze_support()
    main()
