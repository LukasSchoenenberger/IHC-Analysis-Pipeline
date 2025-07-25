#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Small Tiles Script
-------------------------
This script splits larger image tiles into smaller tiles with configurable size.
Padding for edge tiles is done with white pixels (255).

Part of the IHC Pipeline GUI application.
"""

import cv2
import numpy as np
import os
import math
import multiprocessing
import pandas as pd
import argparse
import logging
import sys
import glob
import re
from functools import partial

# Configure logging
logger = logging.getLogger("Create-Small-Tiles")

# Define naming pattern functions at module level so they can be pickled
def pattern_standard(row, col):
    return f'tile_r{row}_c{col}.tif'

def pattern_simple(row, col):
    return f'r{row}_c{col}.tif'

def pattern_corrected_standard(row, col):
    return f'corrected_tile_r{row}_c{col}.tif'

def pattern_corrected_simple(row, col):
    return f'corrected_r{row}_c{col}.tif'

def detect_file_naming_pattern(input_dir):
    """
    Detect the naming pattern used in the input directory.
    Returns a tuple of (pattern_name, pattern_function)
    """
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.tif')]
    if not input_files:
        raise ValueError(f"No .tif files found in input directory: {input_dir}")
    
    # Define standard patterns with their detection regex
    patterns = [
        ('tile_r{row}_c{col}.tif', pattern_standard, r'tile_r(\d+)_c(\d+)\.tif'),
        ('r{row}_c{col}.tif', pattern_simple, r'r(\d+)_c(\d+)\.tif'), 
        ('corrected_tile_r{row}_c{col}.tif', pattern_corrected_standard, r'corrected_tile_r(\d+)_c(\d+)\.tif'),
        ('corrected_r{row}_c{col}.tif', pattern_corrected_simple, r'corrected_r(\d+)_c(\d+)\.tif')
    ]
    
    # Try to match each pattern
    for sample_file in input_files[:10]:  # Check first 10 files
        for pattern_name, pattern_func, pattern_regex in patterns:
            if re.match(pattern_regex, sample_file):
                print(f"Detected file naming pattern: '{pattern_name}'")
                sys.stdout.flush()
                return pattern_name, pattern_func
    
    # If no standard pattern is found, use the default
    print(f"No standard pattern detected. Defaulting to: 'tile_r{{row}}_c{{col}}.tif'")
    sys.stdout.flush()
    return 'tile_r{row}_c{col}.tif', pattern_standard

def determine_input_tile_dimensions(input_dir):
    """
    Determine the dimensions of input tiles by reading the first tile found.
    """
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.tif')]
    if not input_files:
        raise ValueError(f"No .tif files found in input directory: {input_dir}")
    
    # Read the first file to determine dimensions
    first_file = os.path.join(input_dir, input_files[0])
    img = cv2.imread(first_file)
    if img is None:
        raise ValueError(f"Could not read image: {first_file}")
    
    input_tile_height, input_tile_width = img.shape[:2]
    print(f"Determined input tile dimensions: {input_tile_width}x{input_tile_height} pixels")
    sys.stdout.flush()
    
    return input_tile_width, input_tile_height

def get_input_tile_coordinates_for_output_tile(out_row, out_col, output_tile_size, 
                                               input_tile_width, input_tile_height,
                                               medium_rows, medium_cols, total_width, total_height):
    """
    Calculate which input tiles contain pixels for a specific output tile.
    Returns a list of (input_row, input_col, start_y, start_x, end_y, end_x) for each input tile needed.
    """
    # Calculate global pixel coordinates for this output tile
    out_start_y = out_row * output_tile_size
    out_start_x = out_col * output_tile_size
    out_end_y = min((out_row + 1) * output_tile_size, total_height)
    out_end_x = min((out_col + 1) * output_tile_size, total_width)
    
    # Calculate which input tiles contain these pixels
    start_in_row = out_start_y // input_tile_height
    start_in_col = out_start_x // input_tile_width
    end_in_row = (out_end_y - 1) // input_tile_height
    end_in_col = (out_end_x - 1) // input_tile_width
    
    # List to hold input tile coordinates
    input_tiles = []
    
    # Iterate over all input tiles that contain pixels for this output tile
    for in_row in range(start_in_row, end_in_row + 1):
        for in_col in range(start_in_col, end_in_col + 1):
            # Skip if outside the medium tile grid
            if in_row >= medium_rows or in_col >= medium_cols:
                continue
            
            # Calculate pixel coordinates within this input tile
            in_start_y = max(0, out_start_y - in_row * input_tile_height)
            in_start_x = max(0, out_start_x - in_col * input_tile_width)
            in_end_y = min(input_tile_height, out_end_y - in_row * input_tile_height)
            in_end_x = min(input_tile_width, out_end_x - in_col * input_tile_width)
            
            # Add to the list
            input_tiles.append((in_row, in_col, in_start_y, in_start_x, in_end_y, in_end_x))
    
    return input_tiles

def create_single_output_tile(tile_info):
    """
    Create a single output tile by loading and combining parts of input tiles.
    Pads with white pixels (255) where needed.
    
    Args:
        tile_info: Tuple containing all necessary information for creating the tile
        
    Returns:
        Tuple of (success, has_content, error_message)
    """
    try:
        (out_row, out_col, input_dir, output_dir, output_tile_size, 
         input_tile_width, input_tile_height, medium_rows, medium_cols,
         total_width, total_height, pattern_function, alternate_patterns) = tile_info
        
        # Initialize empty output tile with white pixels (255) for padding
        output_tile = np.full((output_tile_size, output_tile_size, 3), 255, dtype=np.uint8)
        
        # Get the list of input tiles needed for this output tile
        input_tiles_info = get_input_tile_coordinates_for_output_tile(
            out_row, out_col, output_tile_size, input_tile_width, input_tile_height,
            medium_rows, medium_cols, total_width, total_height
        )
        
        # Track if this output tile has any content
        has_content = False
        
        # Iterate over each input tile we need to extract from
        for in_row, in_col, in_start_y, in_start_x, in_end_y, in_end_x in input_tiles_info:
            # Calculate where this goes in the output tile
            out_start_y = in_row * input_tile_height + in_start_y - out_row * output_tile_size
            out_start_x = in_col * input_tile_width + in_start_x - out_col * output_tile_size
            
            # Try to load the input tile using the main pattern
            input_tile_path = os.path.join(input_dir, pattern_function(in_row, in_col))
            
            # If the file doesn't exist with the main pattern, try alternative patterns
            if not os.path.exists(input_tile_path):
                found = False
                for alt_pattern_func in alternate_patterns:
                    alt_path = os.path.join(input_dir, alt_pattern_func(in_row, in_col))
                    if os.path.exists(alt_path):
                        input_tile_path = alt_path
                        found = True
                        break
                
                if not found:
                    # Skip if we couldn't find the file with any pattern
                    continue
            
            # Load the input tile
            input_tile = cv2.imread(input_tile_path)
            if input_tile is None:
                continue
                
            # Extract the portion we need
            portion = input_tile[in_start_y:in_end_y, in_start_x:in_end_x]
            
            # Place in output tile
            height, width = portion.shape[:2]
            output_tile[out_start_y:out_start_y+height, out_start_x:out_start_x+width] = portion
            
            # Check if this portion has content (non-white pixels)
            if np.any(portion[:, :, 0] < 255) or np.any(portion[:, :, 1] < 255) or np.any(portion[:, :, 2] < 255):
                has_content = True
        
        # Save the output tile
        output_path = os.path.join(output_dir, f'tile_r{out_row}_c{out_col}.tif')
        success = cv2.imwrite(output_path, output_tile)
        
        if success:
            return (True, has_content, None)
        else:
            return (False, False, f"Failed to save tile: {output_path}")
            
    except Exception as e:
        return (False, False, f"Error creating output tile r{out_row}_c{out_col}: {str(e)}")

def load_metadata(metadata_file):
    """
    Load metadata from CSV file to get grid dimensions
    
    Args:
        metadata_file: Path to the metadata CSV file
        
    Returns:
        tuple: (rows, cols) from the metadata
    """
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(metadata_file)
        
        # Extract rows and columns
        if '#Rows' not in df.columns or '#Columns' not in df.columns:
            raise ValueError("Required columns '#Rows' and '#Columns' not found in metadata file")
        
        rows = int(df['#Rows'].iloc[0])
        cols = int(df['#Columns'].iloc[0])
        
        return rows, cols
        
    except Exception as e:
        raise Exception(f"Error reading metadata file: {str(e)}")

def process_tiles(input_dir, output_dir, medium_rows, medium_cols, output_tile_size=360, max_processes=None):
    """
    Process all tiles, splitting them into smaller tiles using multiprocessing.
    """
    # Determine number of processes to use
    if max_processes is None:
        available_cpus = multiprocessing.cpu_count()
        if available_cpus >= 8:
            max_processes = available_cpus - 6  # Leave cores free for GUI
        else:
            max_processes = max(1, available_cpus // 2)  # Use half on smaller machines
    
    print(f"Processing using {max_processes} CPU cores")
    sys.stdout.flush()
    
    # Detect file naming pattern
    pattern_name, pattern_function = detect_file_naming_pattern(input_dir)
    
    # Store patterns that should be tried when loading files
    alternate_patterns = [
        pattern_standard,
        pattern_simple,
        pattern_corrected_standard,
        pattern_corrected_simple
    ]
    
    # Determine input tile dimensions
    input_tile_width, input_tile_height = determine_input_tile_dimensions(input_dir)
    
    # Calculate total image dimensions
    total_width = input_tile_width * medium_cols
    total_height = input_tile_height * medium_rows
    
    # Calculate how many output tiles we need
    small_cols = math.ceil(total_width / output_tile_size)
    small_rows = math.ceil(total_height / output_tile_size)
    total_tiles = small_rows * small_cols
    
    print(f"Medium grid configuration: {medium_rows}x{medium_cols}")
    print(f"Output tile size: {output_tile_size}x{output_tile_size} pixels")
    print(f"Using file pattern: {pattern_name}")
    print(f"Total image dimensions: {total_width}x{total_height} pixels")
    print(f"Creating {small_rows}×{small_cols} output tiles ({total_tiles} total)")
    sys.stdout.flush()
    
    # Create list of all tile processing tasks
    tile_tasks = []
    for out_row in range(small_rows):
        for out_col in range(small_cols):
            tile_info = (
                out_row, out_col, input_dir, output_dir, output_tile_size,
                input_tile_width, input_tile_height, medium_rows, medium_cols,
                total_width, total_height, pattern_function, alternate_patterns
            )
            tile_tasks.append(tile_info)
    
    # Process tiles in parallel
    print("Creating small tiles...")
    sys.stdout.flush()
    
    # Use the 'spawn' method for better cross-platform compatibility
    ctx = multiprocessing.get_context('spawn')
    
    tiles_saved = 0
    empty_tiles_saved = 0
    content_tiles_saved = 0
    errors = []
    
    with ctx.Pool(processes=max_processes) as pool:
        # Calculate optimal chunk size for balanced load
        chunksize = max(1, total_tiles // (max_processes * 4))
        
        # Process tiles and collect results with explicit progress reporting
        processed_count = 0
        
        for result in pool.imap(create_single_output_tile, tile_tasks, chunksize=chunksize):
            processed_count += 1
            
            # Process result
            success, has_content, error_msg = result
            
            if success:
                tiles_saved += 1
                if has_content:
                    content_tiles_saved += 1
                else:
                    empty_tiles_saved += 1
            elif error_msg:
                errors.append(error_msg)
            
            # Log progress at regular intervals
            if processed_count % max(1, min(100, total_tiles // 20)) == 0 or processed_count == total_tiles:
                progress_pct = (processed_count / total_tiles) * 100
                print(f"Progress: {processed_count}/{total_tiles} tiles ({progress_pct:.1f}%)")
                sys.stdout.flush()
    
    # Report errors if any
    if errors:
        print(f"{len(errors)} errors occurred:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")
        sys.stdout.flush()
    
    # Verify output files
    output_files = [f for f in os.listdir(output_dir) if f.endswith('.tif')]
    
    print(f"Output directory: {output_dir}")
    print(f"{len(output_files)} files created (expected {total_tiles})")
    
    if len(output_files) != total_tiles:
        print(f"Warning: {len(output_files)}/{total_tiles} expected tiles were created")
        
    print(f"\nStatistics:")
    print(f"  - Total output tiles created: {tiles_saved} out of {total_tiles} expected")
    
    # Avoid division by zero
    if tiles_saved > 0:
        print(f"  - Empty tiles (all white): {empty_tiles_saved} ({empty_tiles_saved/tiles_saved*100:.1f}% of total)")
        print(f"  - Tiles with content: {content_tiles_saved} ({content_tiles_saved/tiles_saved*100:.1f}% of total)")
    else:
        print(f"  - Empty tiles (all white): {empty_tiles_saved} (0.0% of total)")
        print(f"  - Tiles with content: {content_tiles_saved} (0.0% of total)")
    
    # Print some example filenames
    if output_files:
        print("\nSample output files:")
        for f in sorted(output_files)[:5]:
            print(f"  - {f}")
    
    print(f"\nProcessing complete! Small tiles saved to: {output_dir}")
    sys.stdout.flush()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Split larger tiles into smaller tiles with white padding')
    parser.add_argument('--data-dir', help='Base data directory')
    parser.add_argument('--output-dir', help='Base output directory') 
    parser.add_argument('--parameters-dir', help='Parameters directory')
    parser.add_argument('--tile-size', type=int, default=360,
                        help='Output tile size in pixels (default: 360)')
    
    # Arguments for standalone mode
    parser.add_argument('--input', help='Input directory containing tiles (for standalone use)')
    parser.add_argument('--output', help='Output directory for small tiles (for standalone use)')
    parser.add_argument('--metadata', help='Path to metadata CSV file (for standalone use)')
    
    args = parser.parse_args()
    
    # Determine input/output paths based on whether called from GUI or standalone
    if args.data_dir and args.output_dir and args.parameters_dir:
        # Called from GUI - use standard paths
        input_folder = os.path.join(args.data_dir, "Tiles-Medium-L-Channel-Normalized-BG-Removed-Illumination-Corrected-Stain-Normalized")
        output_folder = os.path.join(args.data_dir, "Tiles-Medium-L-Channel-Normalized-BG-Removed-Illumination-Corrected-Stain-Normalized-Small-Tiles")
        metadata_file = os.path.join(args.parameters_dir, "Metadata.csv")
        
        # Set up log directory - use Logs directory at same level as Data, Results, Parameters
        base_dir = os.path.dirname(args.data_dir)  # Get parent directory of Data
        log_dir = os.path.join(base_dir, "Logs")
        os.makedirs(log_dir, exist_ok=True)
    elif args.input and args.output and args.metadata:
        # Called standalone with custom paths
        input_folder = args.input
        output_folder = args.output
        metadata_file = args.metadata
        
        # Set up log directory for standalone mode
        log_dir = "Logs"
        os.makedirs(log_dir, exist_ok=True)
    else:
        print("ERROR: Either provide --data-dir, --output-dir, and --parameters-dir OR provide --input, --output, and --metadata")
        sys.stdout.flush()
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Configure logging to file
    log_file = os.path.join(log_dir, "create_small_tiles.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    print("Starting Create Small Tiles")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Metadata file: {metadata_file}")
    print(f"Output tile size: {args.tile_size}x{args.tile_size} pixels")
    print(f"Log file: {log_file}")
    sys.stdout.flush()
    
    logger.info("Starting Create Small Tiles")
    logger.info(f"Input folder: {input_folder}")
    logger.info(f"Output folder: {output_folder}")
    logger.info(f"Metadata file: {metadata_file}")
    logger.info(f"Output tile size: {args.tile_size}x{args.tile_size} pixels")
    
    # Validate input directory
    if not os.path.exists(input_folder):
        print(f"ERROR: Input folder does not exist: {input_folder}")
        sys.stdout.flush()
        logger.error(f"Input folder does not exist: {input_folder}")
        return
    
    # Load metadata to get grid dimensions
    try:
        print("Loading metadata...")
        sys.stdout.flush()
        medium_rows, medium_cols = load_metadata(metadata_file)
        print(f"Loaded metadata: {medium_rows} rows × {medium_cols} columns")
        sys.stdout.flush()
        logger.info(f"Loaded metadata: {medium_rows} rows × {medium_cols} columns")
    except Exception as e:
        print(f"ERROR: Failed to load metadata: {str(e)}")
        sys.stdout.flush()
        logger.error(f"Failed to load metadata: {str(e)}")
        return
    
    # Process tiles
    try:
        print("Processing tiles...")
        sys.stdout.flush()
        process_tiles(input_folder, output_folder, medium_rows, medium_cols, args.tile_size)
        
        print("Create Small Tiles completed successfully")
        sys.stdout.flush()
        logger.info("Create Small Tiles completed successfully")
    except Exception as e:
        print(f"ERROR: Failed to process tiles: {str(e)}")
        sys.stdout.flush()
        logger.error(f"Failed to process tiles: {str(e)}")

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