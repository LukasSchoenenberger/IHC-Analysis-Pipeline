#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normalize Stain Concentrations Script
------------------------------------
This script normalizes stain concentrations using the Perez et al. approach.
It applies stain normalization to tiles that have already been preprocessed
(L-channel normalized, background removed, and illumination corrected).

Part of the IHC Pipeline GUI application.
"""

import numpy as np
import cv2
import os
import glob
import json
import re
import argparse
from scipy.linalg import pinv
import multiprocessing
from functools import partial
import logging
import sys

# Configure logging
logger = logging.getLogger("Normalize-Stain-Concentrations")

def rgb_to_od(img, background=None):
    """Convert RGB image to optical density (OD) space"""
    if background is None:
        background = np.array([255, 255, 255])
    
    h, w, c = img.shape
    img_flat = img.reshape(h*w, c).T
    img_flat = img_flat.astype(float)
    background = background.astype(float)
    
    eps = 1e-6
    img_flat = np.maximum(img_flat, eps)
    
    od = -np.log10(img_flat / background[:, np.newaxis])
    
    return od

def od_to_rgb(od, background=None):
    """Convert optical density (OD) back to RGB"""
    if background is None:
        background = np.array([255, 255, 255])
    
    rgb_flat = background[:, np.newaxis] * np.power(10, -od)
    rgb_flat = np.clip(rgb_flat, 0, 255)
    
    return rgb_flat

def load_stain_vectors(stain_vectors_file):
    """
    Load stain vectors from a text file
    
    Args:
        stain_vectors_file: Path to the stain_vectors.txt file
        
    Returns:
        stain_matrix: Numpy array of shape (3, num_stains)
        stain_names: List of stain names
    """
    stain_matrix = []
    stain_names = []
    
    print(f"Loading stain vectors from: {stain_vectors_file}")
    sys.stdout.flush()
    
    with open(stain_vectors_file, 'r') as f:
        lines = f.readlines()
        
        print(f"Found {len(lines)} lines in stain vectors file")
        sys.stdout.flush()
        
        # Skip comment lines and empty lines
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            # Skip header lines like "Estimated stain vectors:"
            if line.endswith(':') and '[' not in line:
                print(f"Skipping header line {line_num}: {line}")
                sys.stdout.flush()
                continue
            
            print(f"Processing line {line_num}: {line}")
            sys.stdout.flush()
            
            # Parse line format: "Stain_Name: R G B" or "Stain_Name: [R, G, B]"
            if ':' in line:
                parts = line.split(':', 1)
                name = parts[0].strip()
                vector_str = parts[1].strip()
                
                # Skip if vector string is empty (like header lines)
                if not vector_str:
                    print(f"  Skipping line with empty vector data: {line}")
                    sys.stdout.flush()
                    continue
                
                print(f"  Stain name: '{name}', Vector string: '{vector_str}'")
                sys.stdout.flush()
                
                try:
                    # Handle both formats: "R G B" and "[R, G, B]"
                    if vector_str.startswith('[') and vector_str.endswith(']'):
                        vector_str = vector_str[1:-1]  # Remove brackets
                        vector = [float(val.strip()) for val in vector_str.split(',')]
                    else:
                        vector = [float(val) for val in vector_str.split()]
                    
                    # Validate that we have exactly 3 RGB values
                    if len(vector) != 3:
                        print(f"  Warning: Stain '{name}' has {len(vector)} values instead of 3 RGB values. Skipping.")
                        sys.stdout.flush()
                        continue
                    
                    print(f"  Parsed vector: {vector}")
                    sys.stdout.flush()
                    
                    stain_matrix.append(vector)
                    stain_names.append(name)
                    
                except ValueError as e:
                    print(f"  Error parsing vector for stain '{name}': {e}. Skipping.")
                    sys.stdout.flush()
                    continue
    
    if not stain_matrix:
        raise ValueError(f"No valid stain vectors found in {stain_vectors_file}")
    
    print(f"Successfully loaded {len(stain_matrix)} stain vectors: {stain_names}")
    sys.stdout.flush()
    
    # Ensure we have exactly 3 stains for our pipeline
    if len(stain_matrix) > 3:
        print(f"Warning: Found {len(stain_matrix)} stains, using only the first 3: {stain_names[:3]}")
        sys.stdout.flush()
        stain_matrix = stain_matrix[:3]
        stain_names = stain_names[:3]
    elif len(stain_matrix) < 3:
        raise ValueError(f"Expected 3 stain vectors, but only found {len(stain_matrix)}: {stain_names}")
    
    # Convert to numpy array and transpose to get shape (3, num_stains)
    stain_array = np.array(stain_matrix)
    print(f"Stain matrix shape before transpose: {stain_array.shape}")
    sys.stdout.flush()
    
    return stain_array.T, stain_names

def load_percentiles(percentiles_file):
    """
    Load precomputed percentiles from JSON file
    
    Args:
        percentiles_file: Path to the stain_percentiles.json file
        
    Returns:
        Dictionary with stain names and their percentiles
    """
    with open(percentiles_file, 'r') as f:
        params = json.load(f)
    
    return params

def get_stain_percentiles(stain_names, percentiles_data, percentile_keys):
    """
    Get the requested percentile for each stain
    
    Args:
        stain_names: List of stain names
        percentiles_data: Dictionary with stain percentiles data
        percentile_keys: List of percentile keys to use for each stain
        
    Returns:
        List of percentile values for each stain
    """
    percentiles = []
    
    for i, name in enumerate(stain_names):
        percentile_key = percentile_keys[i]
        
        # Check if the percentile key exists in the data
        if percentile_key not in percentiles_data:
            logger.warning(f"Percentile {percentile_key} not found in percentiles data. Using percentile_99 instead.")
            percentile_key = "percentile_99"
            
        if name not in percentiles_data[percentile_key]:
            logger.error(f"Stain '{name}' not found in {percentile_key} data")
            raise ValueError(f"Stain '{name}' not found in percentiles data")
            
        percentiles.append(percentiles_data[percentile_key][name])
    
    return percentiles

def normalize_single_tile(args):
    """
    Normalize a single tile using Perez et al. approach, excluding white pixels
    
    Args:
        args: Tuple containing (img_path, source_M, source_percentiles, target_M, target_percentiles, background, output_dir)
        
    Returns:
        Error message or None on success
    """
    img_path, source_M, source_percentiles, target_M, target_percentiles, background, output_dir = args
    
    try:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            return f"Could not read {img_path}"
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        
        # Create a flat representation of the image
        img_flat = img.reshape(h*w, c)
        
        # Identify white pixels (background that's already been removed)
        white_mask = np.all(img_flat == 255, axis=1)
        tissue_mask = ~white_mask
        
        # If no tissue pixels, just copy the image and return
        if not np.any(tissue_mask):
            output_filename = os.path.join(output_dir, os.path.basename(img_path))
            cv2.imwrite(output_filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return None
        
        # Calculate inverse of source stain matrix
        source_M_inv = pinv(source_M)
        
        # Convert to optical density
        od = rgb_to_od(img, background)
        
        # Calculate stain concentrations
        conc = np.dot(source_M_inv, od)
        
        # Apply normalization using global percentiles
        norm_factor = np.array(target_percentiles) / np.array(source_percentiles)
        
        # Apply normalization factor to each stain
        norm_conc = conc.copy()
        for i in range(len(norm_factor)):
            norm_conc[i, :] = conc[i, :] * norm_factor[i]
        
        # Reconstruct normalized image in OD space using TARGET stain matrix
        norm_od = np.dot(target_M, norm_conc)
        
        # Convert back to RGB
        norm_rgb_flat = od_to_rgb(norm_od, background)
        
        # Reshape to image
        norm_rgb = norm_rgb_flat.T.reshape(h, w, 3).astype(np.uint8)
        
        # Reshape the white mask to 2D for indexing
        white_mask_2d = white_mask.reshape(h, w)
        
        # Set white pixels back to white in the normalized image
        norm_rgb[white_mask_2d] = [255, 255, 255]
        
        # Save normalized image
        output_filename = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(output_filename, cv2.cvtColor(norm_rgb, cv2.COLOR_RGB2BGR))
        
        return None  # Success
        
    except Exception as e:
        return f"Error processing {img_path}: {str(e)}"

def stain_normalization(input_tiles_dir, output_dir, source_vectors_file, source_percentiles_file, 
                       target_vectors_file=None, target_percentiles_file=None,
                       percentile_mapping=None):
    """
    Normalize all tiles using stain matrices and percentiles
    
    Args:
        input_tiles_dir: Directory containing input tiles
        output_dir: Directory to save normalized tiles
        source_vectors_file: Path to the source stain vectors file
        source_percentiles_file: Path to the source precomputed percentiles file
        target_vectors_file: Path to the target stain vectors file (if None, use source)
        target_percentiles_file: Path to target percentiles file (if None, standardize to 1.0)
        percentile_mapping: Dictionary mapping stain types to percentile keys
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set fixed background values
    background = np.array([255, 255, 255])
    
    # Set default percentile mapping if not provided
    if percentile_mapping is None:
        percentile_mapping = {
            "Nuclei_HE": "percentile_95",
            "Myelin_Blue": "percentile_95", 
            "Microglia_Brown": "percentile_95"
        }
    
    print("Loading source stain vectors...")
    sys.stdout.flush()
    
    # Load source stain matrix
    source_M, source_stain_names = load_stain_vectors(source_vectors_file)
    
    # Load source percentiles data
    source_params = load_percentiles(source_percentiles_file)
    
    # Get percentile keys for each stain
    source_percentile_keys = []
    for name in source_stain_names:
        # Get the appropriate percentile key for this stain
        if name in percentile_mapping:
            source_percentile_keys.append(percentile_mapping[name])
        else:
            logger.warning(f"No percentile mapping for stain '{name}'. Using percentile_99.")
            source_percentile_keys.append("percentile_99")
    
    # Get percentile values for each stain
    source_percentiles = get_stain_percentiles(source_stain_names, source_params, source_percentile_keys)
    
    print(f"Loaded source stain matrix with shape {source_M.shape}")
    print(f"Source stains: {', '.join(source_stain_names)}")
    print(f"Using percentiles for source:")
    for name, key, value in zip(source_stain_names, source_percentile_keys, source_percentiles):
        print(f"  {name}: {key} = {value}")
    sys.stdout.flush()
    
    logger.info(f"Loaded source stain matrix with shape {source_M.shape}")
    logger.info(f"Source stains: {', '.join(source_stain_names)}")
    
    # Load target information
    if target_vectors_file and os.path.exists(target_vectors_file):
        target_M, _ = load_stain_vectors(target_vectors_file)
        print(f"Loaded target stain matrix with shape {target_M.shape}")
        sys.stdout.flush()
        logger.info(f"Loaded target stain matrix with shape {target_M.shape}")
    else:
        target_M = source_M
        print("Using source stain matrix as target")
        sys.stdout.flush()
        logger.info("Using source stain matrix as target")
    
    # Load target percentiles or use default values
    if target_percentiles_file and os.path.exists(target_percentiles_file):
        target_params = load_percentiles(target_percentiles_file)
        target_percentiles = get_stain_percentiles(source_stain_names, target_params, source_percentile_keys)
        
        print(f"Using percentiles for target:")
        for name, key, value in zip(source_stain_names, source_percentile_keys, target_percentiles):
            print(f"  {name}: {key} = {value}")
        sys.stdout.flush()
        logger.info("Using target percentiles from file")
    else:
        # Use standard values as reference (standardize to 1.0)
        target_percentiles = [1.0 for _ in range(len(source_percentiles))]
        print(f"Using default target percentiles: {target_percentiles}")
        sys.stdout.flush()
        logger.info("Using default target percentiles (1.0 for all stains)")
    
    # Find all tiles in input directory
    print("Scanning for image files...")
    sys.stdout.flush()
    
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_tiles_dir, ext)))
    
    if not image_files:
        print(f"ERROR: No image files found in {input_tiles_dir}")
        sys.stdout.flush()
        logger.error(f"No image files found in {input_tiles_dir}")
        return
    
    print(f"Found {len(image_files)} tiles to normalize")
    print(f"Using background RGB values: {background}")
    sys.stdout.flush()
    
    logger.info(f"Found {len(image_files)} tiles to normalize")
    
    # Determine number of processes (always max CPUs - 6)
    available_cpus = multiprocessing.cpu_count()
    if available_cpus >= 8:
        max_processes = available_cpus - 6
    else:
        max_processes = max(1, available_cpus // 2)
    
    print(f"Processing using {max_processes} CPU cores.")
    sys.stdout.flush()
    
    logger.info(f"Using {max_processes} CPU cores for normalization")
    
    # Prepare arguments for parallel processing
    args_list = [(img_path, source_M, source_percentiles, target_M, target_percentiles, background, output_dir)
                 for img_path in image_files]
    
    # Process tiles in parallel
    print("Starting stain normalization...")
    sys.stdout.flush()
    
    # Use spawn method for better cross-platform compatibility
    ctx = multiprocessing.get_context('spawn')
    
    with ctx.Pool(processes=max_processes) as pool:
        # Calculate optimal chunk size for balanced load
        total_files = len(image_files)
        chunksize = max(1, total_files // (max_processes * 4))
        
        # Process tiles with progress reporting
        processed_count = 0
        errors = []
        
        for result in pool.imap(normalize_single_tile, args_list, chunksize=chunksize):
            processed_count += 1
            
            # Log progress at regular intervals
            if processed_count % max(1, min(50, total_files // 20)) == 0 or processed_count == total_files:
                percentage = (processed_count / total_files) * 100
                print(f"Progress: {processed_count}/{total_files} tiles ({percentage:.1f}%)")
                sys.stdout.flush()
            
            if result is not None:
                errors.append(result)
    
    # Report results
    successful = len(image_files) - len(errors)
    
    if errors:
        print(f"{len(errors)} errors occurred:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        sys.stdout.flush()
        
        logger.warning(f"{len(errors)} errors occurred during normalization")
        for error in errors:
            logger.warning(f"  {error}")
    
    print(f"Normalization completed: {successful} successful, {len(errors)} failed")
    print(f"Normalized tiles saved to: {output_dir}")
    sys.stdout.flush()
    
    logger.info(f"Normalization completed: {successful} successful, {len(errors)} failed")
    logger.info(f"Normalized tiles saved to: {output_dir}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Normalize stain concentrations using Perez et al. approach")
    parser.add_argument('--data-dir', help='Base data directory')
    parser.add_argument('--output-dir', help='Base output directory')
    parser.add_argument('--parameters-dir', help='Parameters directory')
    parser.add_argument('--input', help='Input directory with tiles (for standalone use)')
    parser.add_argument('--output', help='Output directory for normalized tiles (for standalone use)')
    
    # Percentile selection arguments
    parser.add_argument('--nuclei-percentile', default="percentile_99.9", 
                       help='Percentile to use for nuclei/H&E stain (default: percentile_99.9)')
    parser.add_argument('--myelin-percentile', default="percentile_99.9",
                       help='Percentile to use for myelin/blue stain (default: percentile_99.9)')
    parser.add_argument('--microglia-percentile', default="percentile_99.9",
                       help='Percentile to use for microglia/brown stain (default: percentile_99.9)')
    
    args = parser.parse_args()
    
    # Determine input/output paths based on whether called from GUI or standalone
    if args.data_dir and args.output_dir and args.parameters_dir:
        # Called from GUI - use default paths
        input_folder = os.path.join(args.data_dir, "Tiles-Medium-L-Channel-Normalized-BG-Removed-Illumination-Corrected")
        output_folder = os.path.join(args.data_dir, "Tiles-Medium-L-Channel-Normalized-BG-Removed-Illumination-Corrected-Stain-Normalized")
        parameters_dir = args.parameters_dir
        
        # Set up log directory
        base_dir = os.path.dirname(args.data_dir)
        log_dir = os.path.join(base_dir, "Logs")
        os.makedirs(log_dir, exist_ok=True)
    elif args.input and args.output:
        # Called standalone with custom paths
        input_folder = args.input
        output_folder = args.output
        parameters_dir = args.parameters_dir if args.parameters_dir else os.path.dirname(args.output)
        
        # Set up log directory for standalone mode
        log_dir = "Logs"
        os.makedirs(log_dir, exist_ok=True)
    else:
        print("ERROR: Either provide --data-dir, --output-dir, and --parameters-dir OR provide --input and --output")
        sys.stdout.flush()
        return
    
    # Create output and parameter directories if they don't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(parameters_dir, exist_ok=True)
    
    # Configure logging to file
    log_file = os.path.join(log_dir, "normalize_stain_concentrations.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Define file paths
    source_vectors_file = os.path.join(parameters_dir, "stain_vectors.txt")
    source_percentiles_file = os.path.join(parameters_dir, "stain_percentiles.json")
    target_vectors_file = os.path.join(parameters_dir, "reference_stain_vectors.txt")
    target_percentiles_file = os.path.join(parameters_dir, "reference_stain_percentiles.json")
    
    # Check if required files exist
    if not os.path.exists(source_vectors_file):
        print(f"ERROR: Source stain vectors file not found: {source_vectors_file}")
        sys.stdout.flush()
        logger.error(f"Source stain vectors file not found: {source_vectors_file}")
        return
    
    if not os.path.exists(source_percentiles_file):
        print(f"ERROR: Source percentiles file not found: {source_percentiles_file}")
        sys.stdout.flush()
        logger.error(f"Source percentiles file not found: {source_percentiles_file}")
        return
    
    # Check if reference files exist
    has_reference_vectors = os.path.exists(target_vectors_file)
    has_reference_percentiles = os.path.exists(target_percentiles_file)
    
    if not has_reference_vectors:
        print(f"Warning: Reference stain vectors file not found at {target_vectors_file}")
        print("Will use source stain vectors as reference")
        target_vectors_file = None
    
    if not has_reference_percentiles:
        print(f"Warning: Reference percentiles file not found at {target_percentiles_file}")
        print("Will use standard values (1.0) as reference percentiles")
        target_percentiles_file = None
    
    # Create percentile mapping from command line arguments using correct stain names
    percentile_mapping = {
        "Nuclei_HE": args.nuclei_percentile,
        "Myelin_Blue": args.myelin_percentile,
        "Microglia_Brown": args.microglia_percentile
    }
    
    print("Starting Stain Concentration Normalization")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Parameters directory: {parameters_dir}")
    print(f"Log file: {log_file}")
    print(f"Percentile mapping: {percentile_mapping}")
    sys.stdout.flush()
    
    logger.info("Starting Stain Concentration Normalization")
    logger.info(f"Input folder: {input_folder}")
    logger.info(f"Output folder: {output_folder}")
    logger.info(f"Parameters directory: {parameters_dir}")
    logger.info(f"Percentile mapping: {percentile_mapping}")
    
    # Perform normalization
    stain_normalization(
        input_folder,
        output_folder,
        source_vectors_file,
        source_percentiles_file,
        target_vectors_file,
        target_percentiles_file,
        percentile_mapping
    )
    
    print("Stain Concentration Normalization completed successfully")
    sys.stdout.flush()
    logger.info("Stain Concentration Normalization completed successfully")

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
