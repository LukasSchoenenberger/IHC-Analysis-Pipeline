#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Percentile Calculation Script
-----------------------------
This script calculates global percentiles for stain concentrations using the estimated
stain vectors. It processes background-removed and corrected image tiles and calculates
statistical distributions for each stain channel.

Part of the IHC Pipeline GUI application.
"""

import numpy as np
import cv2
import os
import glob
import re
import json
import argparse
import logging
import sys
import multiprocessing
from functools import partial
from scipy.linalg import pinv

# Configure logging
logger = logging.getLogger("Percentile-Calculation")

def load_stain_vectors(stain_vectors_file):
    """
    Load stain vectors from a text file
    
    Args:
        stain_vectors_file: Path to the stain_vectors.txt file
        
    Returns:
        stain_matrix: Numpy array of shape (3, num_stains)
        stain_names: List of stain names
    """
    if not os.path.exists(stain_vectors_file):
        raise FileNotFoundError(f"Stain vectors file not found: {stain_vectors_file}")
    
    stain_matrix = []
    stain_names = []
    
    with open(stain_vectors_file, 'r') as f:
        lines = f.readlines()
        
        # Skip header lines that might contain comments
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Try to extract stain name and vector
            # Format: "Stain_Name: [r, g, b]" or "Stain_Name: r g b"
            if ':' in line:
                parts = line.split(':', 1)
                name = parts[0].strip()
                vector_str = parts[1].strip()
                
                # Handle bracket format [r, g, b]
                if vector_str.startswith('[') and vector_str.endswith(']'):
                    vector_str = vector_str[1:-1]
                    vector = [float(val.strip()) for val in vector_str.split(',')]
                else:
                    # Handle space-separated format r g b
                    vector = [float(val) for val in vector_str.split()]
                
                if len(vector) == 3:
                    stain_matrix.append(vector)
                    stain_names.append(name)
    
    if not stain_matrix:
        raise ValueError(f"No valid stain vectors found in {stain_vectors_file}")
    
    return np.array(stain_matrix).T, stain_names

def process_tile(img_path, stain_matrix, background, sample_percentage=10):
    """
    Process a single tile and calculate stain concentrations, excluding white pixels.
    Skip tiles with more than 50% white pixels entirely.
    
    Args:
        img_path: Path to the image file
        stain_matrix: Stain matrix (shape: 3 x num_stains)
        background: Background RGB values
        sample_percentage: Percentage of pixels to sample from the tile
        
    Returns:
        List of concentrations for each stain or None if error/skipped
    """
    try:
        # Calculate inverse of stain matrix for concentration calculation
        M_inv = pinv(stain_matrix)
        num_stains = stain_matrix.shape[1]
        
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            return None
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create a flat representation of the image
        h, w, c = img.shape
        img_flat = img.reshape(h*w, c)
        total_pixels = h * w
        
        # Identify non-white pixels (not 255 in all channels)
        non_white_mask = ~np.all(img_flat == 255, axis=1)
        non_white_indices = np.where(non_white_mask)[0]
        
        # Calculate percentage of white pixels
        white_pixel_percentage = (total_pixels - len(non_white_indices)) / total_pixels * 100
        
        # Skip tiles with more than 50% white pixels
        if white_pixel_percentage > 50:
            return None
        
        # Skip if no non-white pixels (all background)
        if len(non_white_indices) == 0:
            return None
        
        # Sample percentage of non-white pixels
        if sample_percentage < 100:
            num_samples = max(1, int(len(non_white_indices) * sample_percentage / 100))
            if num_samples < len(non_white_indices):
                sample_indices = np.random.choice(non_white_indices, num_samples, replace=False)
            else:
                sample_indices = non_white_indices
        else:
            sample_indices = non_white_indices
        
        # Extract sampled tissue pixels
        tissue_pixels = img_flat[sample_indices].T  # Transpose to match the expected shape
        
        # Convert to optical density (only tissue pixels)
        od_tissue = -np.log10(np.maximum(tissue_pixels.astype(float), 1e-6) / background[:, np.newaxis])
        
        # Calculate concentrations
        concentrations = np.dot(M_inv, od_tissue)
        
        # Return concentrations for each stain
        return [concentrations[i, :] for i in range(num_stains)]
            
    except Exception as e:
        return None

def calculate_global_percentiles(input_dir, stain_matrix, background, sample_percentage=10, max_processes=None):
    """
    Calculate global percentiles for stain concentrations using multiprocessing
    
    Args:
        input_dir: Directory containing image tiles
        stain_matrix: Stain matrix (shape: 3 x num_stains)
        background: Background RGB values
        sample_percentage: Percentage of pixels to sample from each tile
        max_processes: Maximum number of processes to use
        
    Returns:
        percentiles_dict: Dictionary with percentile values for each stain
    """
    # Find all tiles in input directory
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not image_files:
        print(f"ERROR: No image files found in {input_dir}")
        sys.stdout.flush()
        return None
    
    total_files = len(image_files)
    print(f"Found {total_files} image files")
    sys.stdout.flush()
    
    # Determine number of processes to use
    if max_processes is None:
        available_cpus = multiprocessing.cpu_count()
        if available_cpus >= 8:
            max_processes = available_cpus - 6  # Leave cores free for GUI
        else:
            max_processes = max(1, available_cpus // 2)  # Use half on smaller machines
    
    print(f"Processing using {max_processes} CPU cores")
    sys.stdout.flush()
    
    # Create a partial function with fixed parameters
    process_tile_partial = partial(
        process_tile,
        stain_matrix=stain_matrix,
        background=background,
        sample_percentage=sample_percentage
    )
    
    # Process tiles in parallel
    num_stains = stain_matrix.shape[1]
    all_concentrations = [[] for _ in range(num_stains)]
    
    print("Processing tiles for stain concentration analysis...")
    print("(Skipping tiles with >50% white pixels)")
    sys.stdout.flush()
    
    # Use the 'spawn' method for better cross-platform compatibility
    ctx = multiprocessing.get_context('spawn')
    
    with ctx.Pool(processes=max_processes) as pool:
        # Calculate optimal chunk size for balanced load
        chunksize = max(1, total_files // (max_processes * 4))
        
        # Process tiles and collect results with explicit progress reporting
        processed_count = 0
        valid_results = []
        skipped_count = 0
        
        for result in pool.imap(process_tile_partial, image_files, chunksize=chunksize):
            processed_count += 1
            
            # Log progress at regular intervals
            if processed_count % max(1, min(50, total_files // 20)) == 0 or processed_count == total_files:
                progress_pct = (processed_count / total_files) * 100
                print(f"Progress: {processed_count}/{total_files} tiles ({progress_pct:.1f}%)")
                sys.stdout.flush()
            
            if result is not None:
                valid_results.append(result)
            else:
                skipped_count += 1
        
        # Combine concentrations for each stain
        for stain_conc_list in valid_results:
            for i in range(num_stains):
                all_concentrations[i].append(stain_conc_list[i])
    
    print(f"Analysis completed. {len(valid_results)} tiles processed successfully")
    print(f"{skipped_count} tiles skipped (>50% white pixels or errors)")
    sys.stdout.flush()
    
    # Concatenate all sampled concentrations for each stain
    if not all(all_concentrations):
        print("ERROR: No valid concentration data found")
        sys.stdout.flush()
        return None
        
    concatenated_concentrations = []
    total_pixels = 0
    
    for stain_conc in all_concentrations:
        if stain_conc:  # Check if list is not empty
            concat_conc = np.concatenate(stain_conc)
            concatenated_concentrations.append(concat_conc)
            total_pixels += len(concat_conc)
        else:
            # Handle the case where a stain has no data
            concatenated_concentrations.append(np.array([]))
    
    print(f"Calculating percentiles from {total_pixels // num_stains} total pixels...")
    sys.stdout.flush()
    
    # Calculate multiple percentiles for each stain
    percentile_values = [1, 50, 75, 95, 97.5, 99, 99.9]
    percentiles_dict = {p: [] for p in percentile_values}
    
    for conc in concatenated_concentrations:
        if len(conc) > 0:
            for p in percentile_values:
                percentiles_dict[p].append(float(np.percentile(conc, p)))
        else:
            # Default values if no data
            for p in percentile_values:
                if p <= 50:
                    # Lower percentiles default to 0
                    percentiles_dict[p].append(0.0)
                else:
                    # Upper percentiles default to 1
                    percentiles_dict[p].append(1.0)
    
    print("Percentile calculation completed")
    for p in percentile_values:
        print(f"{p}th percentiles: {[f'{val:.6f}' for val in percentiles_dict[p]]}")
    sys.stdout.flush()
    
    return percentiles_dict

def save_parameters(output_dir, stain_names, percentiles_dict):
    """
    Save parameters to JSON and text files
    
    Args:
        output_dir: Output directory
        stain_names: List of stain names
        percentiles_dict: Dictionary with percentile values for each stain
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create parameters dictionary
    parameters = {
        "stain_names": stain_names,
        "percentile_1": {name: float(perc) for name, perc in zip(stain_names, percentiles_dict[1])},
        "percentile_50": {name: float(perc) for name, perc in zip(stain_names, percentiles_dict[50])},
        "percentile_75": {name: float(perc) for name, perc in zip(stain_names, percentiles_dict[75])},
        "percentile_95": {name: float(perc) for name, perc in zip(stain_names, percentiles_dict[95])},
        "percentile_97.5": {name: float(perc) for name, perc in zip(stain_names, percentiles_dict[97.5])},
        "percentile_99": {name: float(perc) for name, perc in zip(stain_names, percentiles_dict[99])},
        "percentile_99.9": {name: float(perc) for name, perc in zip(stain_names, percentiles_dict[99.9])}
    }
    
    # Save to JSON file
    output_file = os.path.join(output_dir, "stain_percentiles.json")
    with open(output_file, 'w') as f:
        json.dump(parameters, f, indent=4)
    
    print(f"Parameters saved to {output_file}")
    sys.stdout.flush()
    
    # Also save as text file for easy reading
    txt_file = os.path.join(output_dir, "stain_percentiles.txt")
    with open(txt_file, 'w') as f:
        f.write("Global percentiles for stain concentrations:\n\n")
        for p in [1, 50, 75, 95, 97.5, 99, 99.9]:
            f.write(f"--- {p}th percentile ---\n")
            for name, perc in zip(stain_names, percentiles_dict[p]):
                f.write(f"{name}: {perc:.6f}\n")
            f.write("\n")
    
    print(f"Parameters also saved to {txt_file}")
    sys.stdout.flush()
    
    logger.info(f"Stain percentiles saved to {output_file} and {txt_file}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate stain percentiles from processed image tiles')
    parser.add_argument('--data-dir', help='Base data directory')
    parser.add_argument('--output-dir', help='Base output directory')
    parser.add_argument('--parameters-dir', help='Parameters directory')
    parser.add_argument('--input', help='Input directory containing processed tiles (for standalone use)')
    parser.add_argument('--stain-vectors', help='Path to stain vectors file (for standalone use)')
    
    args = parser.parse_args()
    
    # Fixed parameters as specified
    background = np.array([255, 255, 255])  # Always use white background
    sample_percentage = 10  # Always sample 10% of pixels per tile
    
    # Determine input/output paths based on whether called from GUI or standalone
    if args.data_dir and args.output_dir and args.parameters_dir:
        # Called from GUI - use standard paths
        input_folder = os.path.join(args.data_dir, "Tiles-Medium-L-Channel-Normalized-BG-Removed-Illumination-Corrected")
        output_folder = args.parameters_dir
        stain_vectors_file = os.path.join(args.parameters_dir, "stain_vectors.txt")
        
        # Set up log directory - use Logs directory at same level as Data, Results, Parameters
        base_dir = os.path.dirname(args.data_dir)  # Get parent directory of Data
        log_dir = os.path.join(base_dir, "Logs")
        os.makedirs(log_dir, exist_ok=True)
    elif args.input and args.stain_vectors:
        # Called standalone with custom paths
        input_folder = args.input
        output_folder = args.parameters_dir if args.parameters_dir else os.path.dirname(args.input)
        stain_vectors_file = args.stain_vectors
        
        # Set up log directory for standalone mode
        log_dir = "Logs"
        os.makedirs(log_dir, exist_ok=True)
    else:
        print("ERROR: Either provide --data-dir, --output-dir, and --parameters-dir OR provide --input and --stain-vectors")
        sys.stdout.flush()
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Configure logging to file in the Logs directory
    log_file = os.path.join(log_dir, "percentile_calculation.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    print("Starting Percentile Calculation")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Stain vectors file: {stain_vectors_file}")
    print(f"Log file: {log_file}")
    sys.stdout.flush()
    
    logger.info("Starting Percentile Calculation")
    logger.info(f"Input folder: {input_folder}")
    logger.info(f"Output folder: {output_folder}")
    logger.info(f"Stain vectors file: {stain_vectors_file}")
    
    # Validate input directory
    if not os.path.exists(input_folder):
        print(f"ERROR: Input folder does not exist: {input_folder}")
        sys.stdout.flush()
        logger.error(f"Input folder does not exist: {input_folder}")
        return
    
    # Load stain vectors
    try:
        print("Loading stain vectors...")
        sys.stdout.flush()
        stain_matrix, stain_names = load_stain_vectors(stain_vectors_file)
        print(f"Loaded {len(stain_names)} stain vectors: {stain_names}")
        sys.stdout.flush()
        logger.info(f"Loaded stain vectors: {stain_names}")
    except Exception as e:
        print(f"ERROR: Failed to load stain vectors: {str(e)}")
        sys.stdout.flush()
        logger.error(f"Failed to load stain vectors: {str(e)}")
        return
    
    # Calculate global percentiles
    print(f"Calculating global percentiles (sampling {sample_percentage}% of pixels per tile)...")
    print(f"Background values: {background}")
    sys.stdout.flush()
    
    percentiles_dict = calculate_global_percentiles(
        input_folder, 
        stain_matrix, 
        background, 
        sample_percentage=sample_percentage
    )
    
    if percentiles_dict:
        # Save parameters
        print("Saving percentile parameters...")
        sys.stdout.flush()
        save_parameters(output_folder, stain_names, percentiles_dict)
        
        print("Percentile calculation completed successfully")
        sys.stdout.flush()
        logger.info("Percentile calculation completed successfully")
    else:
        print("ERROR: Failed to calculate percentiles")
        sys.stdout.flush()
        logger.error("Failed to calculate percentiles")

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
