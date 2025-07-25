#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flat Field Model Estimation Script
----------------------------------
This script estimates a flat field distortion model from L-channel normalized tiles.
It identifies the most suitable reference tiles and creates a model to correct
illumination artifacts in the images.

Part of the IHC Pipeline GUI application.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from scipy.ndimage import uniform_filter
import time
import glob
import gc
import logging
import sys
import argparse
import multiprocessing
from contextlib import contextmanager


# Configure logging
logger = logging.getLogger("Flat-Field-Model")


@contextmanager
def tile_loader(tile_path):
    """
    Context manager to load a tile, use it, and then free memory.
    
    Args:
        tile_path (str): Path to the tile file.
        
    Yields:
        ndarray: Loaded tile data.
    """
    tile = tiff.imread(tile_path)
    try:
        yield tile
    finally:
        # Explicitly delete to help with garbage collection
        del tile
        gc.collect()


def calculate_local_coefficient_of_variation(image, window_size=5, mask=None):
    """
    Calculate local coefficient of variation (LCoV) for an image.
    LCoV = local_std / local_mean
    
    Args:
        image (ndarray): Input image.
        window_size (int): Size of the sliding window.
        mask (ndarray, optional): Binary mask of valid pixels (False for pixels to ignore).
        
    Returns:
        ndarray: Local coefficient of variation.
    """
    # Convert to float32 for calculation
    image = image.astype(np.float32)
    
    # Apply mask if provided
    if mask is not None:
        # Create a copy to avoid modifying the original
        image_masked = image.copy()
        # Use mean of valid pixels as replacement value for invalid pixels
        valid_mean = np.mean(image[mask]) if np.any(mask) else 0
        image_masked[~mask] = valid_mean
        image = image_masked
    
    # Calculate local mean
    local_mean = uniform_filter(image, size=window_size)
    
    # Calculate local variance
    local_variance = uniform_filter(image**2, size=window_size) - local_mean**2
    
    # Calculate local standard deviation
    local_std = np.sqrt(np.maximum(local_variance, 0))
    
    # Calculate local coefficient of variation (LCoV)
    epsilon = 1e-10  # To avoid division by zero
    lcov = local_std / (local_mean + epsilon)
    
    # If mask is provided, set LCoV to 0 for ignored pixels
    if mask is not None:
        lcov[~mask] = 0
    
    return lcov





def extract_flat_field_model(adaptive_tiles_dir, output_dir, window_size=5, max_tiles=None, batch_size=10, 
                            min_std_dev=1.0):
    """
    Calculate the flat field model from adaptive tiles in a memory-efficient way.
    
    Args:
        adaptive_tiles_dir (str): Directory containing the adaptive tiles.
        output_dir (str): Directory to save the output flat field model.
        window_size (int): Size of the sliding window for LCoV calculation.
        max_tiles (int): Maximum number of tiles to process (None = all).
        batch_size (int): Number of tiles to process in each batch.
        min_std_dev (float): Minimum standard deviation required for a valid candidate.
        
    Returns:
        ndarray: The calculated flat field model.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Fixed values
    ignore_value = 255  # Fixed value for pixels to ignore
    
    # Get all adaptive tiles
    tile_paths = glob.glob(os.path.join(adaptive_tiles_dir, "*.tif")) + glob.glob(os.path.join(adaptive_tiles_dir, "*.tiff"))
    if not tile_paths:
        logger.error(f"No TIFF files found in {adaptive_tiles_dir}")
        print(f"ERROR: No TIFF files found in {adaptive_tiles_dir}")
        sys.stdout.flush()
        raise ValueError(f"No TIFF files found in {adaptive_tiles_dir}")
    
    # Limit to max_tiles if specified
    if max_tiles is not None and max_tiles < len(tile_paths):
        print(f"Limiting to {max_tiles} tiles out of {len(tile_paths)}")
        sys.stdout.flush()
        tile_paths = tile_paths[:max_tiles]
    else:
        print(f"Processing all {len(tile_paths)} tiles")
        sys.stdout.flush()
    
    print(f"STEP 1: Extracting dimensions and initializing arrays...")
    print(f"NOTE: Ignoring pixels with value {ignore_value} in calculations")
    sys.stdout.flush()
    
    # Get dimensions from first tile
    with tile_loader(tile_paths[0]) as first_tile:
        height, width, channels = first_tile.shape
        print(f"Tile dimensions: {height}x{width}x{channels}")
        sys.stdout.flush()
    
    # Initialize arrays to store candidates for each channel
    num_tiles = len(tile_paths)
    
    # Process tiles in batches to avoid memory issues
    num_batches = (num_tiles + batch_size - 1) // batch_size  # Ceiling division
    
    # Initialize arrays to store sorted candidates
    sorted_candidates_r = np.zeros((num_tiles, height, width), dtype=np.float32)
    sorted_candidates_g = np.zeros((num_tiles, height, width), dtype=np.float32)
    sorted_candidates_b = np.zeros((num_tiles, height, width), dtype=np.float32)
    
    # Initialize masks to track valid (non-ignore_value) pixels
    pixel_valid_count_r = np.zeros((height, width), dtype=np.int32)
    pixel_valid_count_g = np.zeros((height, width), dtype=np.int32)
    pixel_valid_count_b = np.zeros((height, width), dtype=np.int32)
    
    print("STEP 2: Processing tiles in batches to extract channels...")
    sys.stdout.flush()
    start_time = time.time()
    
    # Process tiles in batches
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_tiles)
        batch_size_actual = batch_end - batch_start
        
        print(f"Processing batch {batch_idx+1}/{num_batches} (tiles {batch_start+1}-{batch_end}/{num_tiles})...")
        sys.stdout.flush()
        
        # Initialize temporary arrays for this batch
        batch_candidates_r = np.zeros((batch_size_actual, height, width), dtype=np.float32)
        batch_candidates_g = np.zeros((batch_size_actual, height, width), dtype=np.float32)
        batch_candidates_b = np.zeros((batch_size_actual, height, width), dtype=np.float32)
        
        # Initialize masks for this batch
        batch_masks_r = np.zeros((batch_size_actual, height, width), dtype=bool)
        batch_masks_g = np.zeros((batch_size_actual, height, width), dtype=bool)
        batch_masks_b = np.zeros((batch_size_actual, height, width), dtype=bool)
        
        # Load tiles in this batch
        for i, tile_path in enumerate(tile_paths[batch_start:batch_end]):
            with tile_loader(tile_path) as tile:
                # Extract channels
                r_channel = tile[:, :, 0].astype(np.float32)
                g_channel = tile[:, :, 1].astype(np.float32)
                b_channel = tile[:, :, 2].astype(np.float32)
                
                # Create masks for non-ignore_value pixels
                r_mask = r_channel != ignore_value
                g_mask = g_channel != ignore_value
                b_mask = b_channel != ignore_value
                
                # Store channels and masks
                batch_candidates_r[i] = r_channel
                batch_candidates_g[i] = g_channel
                batch_candidates_b[i] = b_channel
                
                batch_masks_r[i] = r_mask
                batch_masks_g[i] = g_mask
                batch_masks_b[i] = b_mask
                
                # Update valid pixel counts
                pixel_valid_count_r += r_mask
                pixel_valid_count_g += g_mask
                pixel_valid_count_b += b_mask
        
        # Merge batch data into overall arrays
        sorted_candidates_r[batch_start:batch_end] = batch_candidates_r
        sorted_candidates_g[batch_start:batch_end] = batch_candidates_g
        sorted_candidates_b[batch_start:batch_end] = batch_candidates_b
        
        # Free memory
        del batch_candidates_r, batch_candidates_g, batch_candidates_b
        del batch_masks_r, batch_masks_g, batch_masks_b
        gc.collect()
    
    # Print statistics about valid pixel counts
    print("\nValid pixel statistics (non-255 values):")
    print(f"  Red channel: min={np.min(pixel_valid_count_r)} max={np.max(pixel_valid_count_r)} mean={np.mean(pixel_valid_count_r):.2f}")
    print(f"  Green channel: min={np.min(pixel_valid_count_g)} max={np.max(pixel_valid_count_g)} mean={np.mean(pixel_valid_count_g):.2f}")
    print(f"  Blue channel: min={np.min(pixel_valid_count_b)} max={np.max(pixel_valid_count_b)} mean={np.mean(pixel_valid_count_b):.2f}")
    sys.stdout.flush()
    
    print("STEP 3: Sorting pixel intensities at each position (ignoring value 255)...")
    sys.stdout.flush()
    
    # Sort intensities at each pixel position, ignoring ignore_value
    for y in range(height):
        if y % 100 == 0 or y == height - 1:
            percent_done = (y + 1) / height * 100
            elapsed = time.time() - start_time
            eta = elapsed / (y + 1) * (height - y - 1)
            print(f"Sorting rows: {y+1}/{height} ({percent_done:.1f}%) - ETA: {int(eta//60):02d}:{int(eta%60):02d}")
            sys.stdout.flush()
        
        for x in range(width):
            # Get pixel values across all tiles at position (x,y)
            r_values = sorted_candidates_r[:, y, x].copy()
            g_values = sorted_candidates_g[:, y, x].copy()
            b_values = sorted_candidates_b[:, y, x].copy()
            
            # Filter out ignore_value pixels
            r_valid = r_values[r_values != ignore_value]
            g_valid = g_values[g_values != ignore_value]
            b_valid = b_values[b_values != ignore_value]
            
            # Sort valid values in descending order (brightest first)
            if len(r_valid) > 0:
                r_sorted = np.sort(r_valid)[::-1]
                sorted_candidates_r[:len(r_sorted), y, x] = r_sorted
                # Fill remaining slots with median of valid values (to avoid biasing smoothness calculation)
                if len(r_sorted) < num_tiles:
                    r_median = np.median(r_valid)
                    sorted_candidates_r[len(r_sorted):, y, x] = r_median
            
            if len(g_valid) > 0:
                g_sorted = np.sort(g_valid)[::-1]
                sorted_candidates_g[:len(g_sorted), y, x] = g_sorted
                if len(g_sorted) < num_tiles:
                    g_median = np.median(g_valid)
                    sorted_candidates_g[len(g_sorted):, y, x] = g_median
            
            if len(b_valid) > 0:
                b_sorted = np.sort(b_valid)[::-1]
                sorted_candidates_b[:len(b_sorted), y, x] = b_sorted
                if len(b_sorted) < num_tiles:
                    b_median = np.median(b_valid)
                    sorted_candidates_b[len(b_sorted):, y, x] = b_median
    
    print("STEP 4: Calculating smoothness scores...")
    sys.stdout.flush()
    
    # Create masks for the final model (positions with at least one valid value)
    r_valid_mask = pixel_valid_count_r > 0
    g_valid_mask = pixel_valid_count_g > 0
    b_valid_mask = pixel_valid_count_b > 0
    
    # Calculate standard deviation for each candidate (for constraining selection)
    print(f"Calculating standard deviations and applying minimum threshold of {min_std_dev}...")
    sys.stdout.flush()
    std_intensities_r = np.array([np.std(sorted_candidates_r[i][r_valid_mask]) for i in range(num_tiles)])
    std_intensities_g = np.array([np.std(sorted_candidates_g[i][g_valid_mask]) for i in range(num_tiles)])
    std_intensities_b = np.array([np.std(sorted_candidates_b[i][b_valid_mask]) for i in range(num_tiles)])
    
    # Calculate smoothness scores for each candidate
    r_smoothness_scores = np.zeros(num_tiles)
    g_smoothness_scores = np.zeros(num_tiles)
    b_smoothness_scores = np.zeros(num_tiles)
    
    for i in range(num_tiles):
        if (i + 1) % 10 == 0 or i == num_tiles - 1:
            percent_done = (i + 1) / num_tiles * 100
            print(f"Evaluating candidate {i+1}/{num_tiles} ({percent_done:.1f}%)")
            sys.stdout.flush()
        
        # Calculate local coefficient of variation for each channel, using masks to ignore invalid pixels
        lcov_r = calculate_local_coefficient_of_variation(sorted_candidates_r[i], window_size, mask=r_valid_mask)
        lcov_g = calculate_local_coefficient_of_variation(sorted_candidates_g[i], window_size, mask=g_valid_mask)
        lcov_b = calculate_local_coefficient_of_variation(sorted_candidates_b[i], window_size, mask=b_valid_mask)
        
        # Sum LCoV values (only for valid pixels)
        r_smoothness_scores[i] = np.sum(lcov_r[r_valid_mask])
        g_smoothness_scores[i] = np.sum(lcov_g[g_valid_mask])
        b_smoothness_scores[i] = np.sum(lcov_b[b_valid_mask])
        
        # Free memory
        del lcov_r, lcov_g, lcov_b
        gc.collect()
    
    # Create masked arrays to exclude candidates with very low std (too uniform)
    masked_r_scores = r_smoothness_scores.copy()
    masked_g_scores = g_smoothness_scores.copy()
    masked_b_scores = b_smoothness_scores.copy()
    
    # Mask out overly uniform candidates by setting their scores very high
    masked_r_scores[std_intensities_r < min_std_dev] = np.inf
    masked_g_scores[std_intensities_g < min_std_dev] = np.inf
    masked_b_scores[std_intensities_b < min_std_dev] = np.inf
    
    # Count how many candidates were excluded for each channel
    r_excluded = np.sum(std_intensities_r < min_std_dev)
    g_excluded = np.sum(std_intensities_g < min_std_dev)
    b_excluded = np.sum(std_intensities_b < min_std_dev)
    
    print(f"\nExcluded candidates (std dev < {min_std_dev}):")
    print(f"  Red: {r_excluded}/{num_tiles} ({r_excluded/num_tiles*100:.1f}%)")
    print(f"  Green: {g_excluded}/{num_tiles} ({g_excluded/num_tiles*100:.1f}%)")
    print(f"  Blue: {b_excluded}/{num_tiles} ({b_excluded/num_tiles*100:.1f}%)")
    sys.stdout.flush()
    
    # Print statistics about smoothness scores to help diagnose issues
    print("\nSmoothness score statistics:")
    print(f"  Red: min={np.min(r_smoothness_scores):.2f}, max={np.max(r_smoothness_scores):.2f}, "
          f"mean={np.mean(r_smoothness_scores):.2f}, std={np.std(r_smoothness_scores):.2f}")
    print(f"  Green: min={np.min(g_smoothness_scores):.2f}, max={np.max(g_smoothness_scores):.2f}, "
          f"mean={np.mean(g_smoothness_scores):.2f}, std={np.std(g_smoothness_scores):.2f}")
    print(f"  Blue: min={np.min(b_smoothness_scores):.2f}, max={np.max(b_smoothness_scores):.2f}, "
          f"mean={np.mean(b_smoothness_scores):.2f}, std={np.std(b_smoothness_scores):.2f}")
    sys.stdout.flush()
    
    # Find optimal candidate for each channel (minimum LCoV sum) using masked scores
    r_optimal_idx = np.argmin(masked_r_scores)
    g_optimal_idx = np.argmin(masked_g_scores)
    b_optimal_idx = np.argmin(masked_b_scores)
    
    # Check if any channel has no valid candidates
    valid_candidates = True
    if np.isinf(masked_r_scores[r_optimal_idx]):
        print(f"WARNING: No valid red channel candidates with std dev >= {min_std_dev}")
        print(f"Using candidate with highest std dev: {np.max(std_intensities_r):.4f}")
        sys.stdout.flush()
        r_optimal_idx = np.argmax(std_intensities_r)
        valid_candidates = False
        
    if np.isinf(masked_g_scores[g_optimal_idx]):
        print(f"WARNING: No valid green channel candidates with std dev >= {min_std_dev}")
        print(f"Using candidate with highest std dev: {np.max(std_intensities_g):.4f}")
        sys.stdout.flush()
        g_optimal_idx = np.argmax(std_intensities_g)
        valid_candidates = False
        
    if np.isinf(masked_b_scores[b_optimal_idx]):
        print(f"WARNING: No valid blue channel candidates with std dev >= {min_std_dev}")
        print(f"Using candidate with highest std dev: {np.max(std_intensities_b):.4f}")
        sys.stdout.flush()
        b_optimal_idx = np.argmax(std_intensities_b)
        valid_candidates = False
    
    if not valid_candidates:
        print("\nWARNING: Consider lowering the min_std_dev threshold")
        sys.stdout.flush()
    
    print(f"\nOptimal indices - R: {r_optimal_idx}, G: {g_optimal_idx}, B: {b_optimal_idx}")
    sys.stdout.flush()
    
    print(f"Optimal candidates standard deviations:")
    print(f"  Red: {std_intensities_r[r_optimal_idx]:.4f}")
    print(f"  Green: {std_intensities_g[g_optimal_idx]:.4f}")
    print(f"  Blue: {std_intensities_b[b_optimal_idx]:.4f}")
    sys.stdout.flush()
    
    # Get the optimal flat-field distortion model for each channel
    flat_field_r = sorted_candidates_r[r_optimal_idx].copy()
    flat_field_g = sorted_candidates_g[g_optimal_idx].copy()
    flat_field_b = sorted_candidates_b[b_optimal_idx].copy()
    
    # Ensure no ignore_value pixels remain in the final model
    # Replace any remaining ignore_value pixels with local median of valid pixels
    for y in range(height):
        for x in range(width):
            if flat_field_r[y, x] == ignore_value or not r_valid_mask[y, x]:
                # Get valid values in a 5x5 neighborhood
                y_min, y_max = max(0, y-2), min(height, y+3)
                x_min, x_max = max(0, x-2), min(width, x+3)
                neighborhood = flat_field_r[y_min:y_max, x_min:x_max]
                valid_values = neighborhood[neighborhood != ignore_value]
                if len(valid_values) > 0:
                    flat_field_r[y, x] = np.median(valid_values)
                else:
                    # If no valid values in neighborhood, use global median
                    flat_field_r[y, x] = np.median(flat_field_r[flat_field_r != ignore_value])
            
            if flat_field_g[y, x] == ignore_value or not g_valid_mask[y, x]:
                y_min, y_max = max(0, y-2), min(height, y+3)
                x_min, x_max = max(0, x-2), min(width, x+3)
                neighborhood = flat_field_g[y_min:y_max, x_min:x_max]
                valid_values = neighborhood[neighborhood != ignore_value]
                if len(valid_values) > 0:
                    flat_field_g[y, x] = np.median(valid_values)
                else:
                    flat_field_g[y, x] = np.median(flat_field_g[flat_field_g != ignore_value])
            
            if flat_field_b[y, x] == ignore_value or not b_valid_mask[y, x]:
                y_min, y_max = max(0, y-2), min(height, y+3)
                x_min, x_max = max(0, x-2), min(width, x+3)
                neighborhood = flat_field_b[y_min:y_max, x_min:x_max]
                valid_values = neighborhood[neighborhood != ignore_value]
                if len(valid_values) > 0:
                    flat_field_b[y, x] = np.median(valid_values)
                else:
                    flat_field_b[y, x] = np.median(flat_field_b[flat_field_b != ignore_value])
    
    # Free memory
    del sorted_candidates_r, sorted_candidates_g, sorted_candidates_b
    gc.collect()
    
    # Combine channels into a single RGB flat-field model
    flat_field_model = np.stack([flat_field_r, flat_field_g, flat_field_b], axis=2)
    
    # Final check to ensure no ignore_value pixels in the model
    for c in range(3):
        if np.any(flat_field_model[:,:,c] == ignore_value):
            print(f"WARNING: Found {np.sum(flat_field_model[:,:,c] == ignore_value)} pixels with value {ignore_value} in channel {c}")
            sys.stdout.flush()
            # Replace any remaining ignore_value with median
            mask = flat_field_model[:,:,c] == ignore_value
            valid_values = flat_field_model[:,:,c][~mask]
            median_value = np.median(valid_values)
            flat_field_model[:,:,c][mask] = median_value
            print(f"  Replaced with median value: {median_value}")
            sys.stdout.flush()
    
    # Print statistics of the flat-field model
    print(f"\nFlat-field model statistics:")
    print(f"  Red channel: min={np.min(flat_field_model[:,:,0]):.2f}, max={np.max(flat_field_model[:,:,0]):.2f}, mean={np.mean(flat_field_model[:,:,0]):.2f}")
    print(f"  Green channel: min={np.min(flat_field_model[:,:,1]):.2f}, max={np.max(flat_field_model[:,:,1]):.2f}, mean={np.mean(flat_field_model[:,:,1]):.2f}")
    print(f"  Blue channel: min={np.min(flat_field_model[:,:,2]):.2f}, max={np.max(flat_field_model[:,:,2]):.2f}, mean={np.mean(flat_field_model[:,:,2]):.2f}")
    sys.stdout.flush()
    
    # Save the flat-field model
    flat_field_model_path = os.path.join(output_dir, "flat_field_model.tif")
    with tile_loader(tile_paths[0]) as sample_tile:
        model_dtype = sample_tile.dtype
    
    tiff.imwrite(flat_field_model_path, flat_field_model.astype(model_dtype))
    print(f"Saved flat-field model to {flat_field_model_path}")
    sys.stdout.flush()
    
    # Also save individual channel images for easy inspection
    tiff.imwrite(os.path.join(output_dir, "flat_field_model_red.tif"), flat_field_model[:,:,0].astype(model_dtype))
    tiff.imwrite(os.path.join(output_dir, "flat_field_model_green.tif"), flat_field_model[:,:,1].astype(model_dtype))
    tiff.imwrite(os.path.join(output_dir, "flat_field_model_blue.tif"), flat_field_model[:,:,2].astype(model_dtype))
    
    return flat_field_model


def visualize_flat_field_model(flat_field_model, output_dir):
    """
    Create visualizations of the flat-field model.
    
    Args:
        flat_field_model (ndarray): Flat-field distortion model.
        output_dir (str): Directory to save the visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Main visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original flat-field model
    axes[0, 0].imshow(flat_field_model)
    axes[0, 0].set_title('Flat-field Distortion Model')
    axes[0, 0].axis('off')
    
    # Enhanced contrast visualization
    enhanced = flat_field_model.copy().astype(np.float32)
    for c in range(3):
        p2, p98 = np.percentile(enhanced[:,:,c], (2, 98))
        if p98 > p2:
            enhanced[:,:,c] = np.clip((enhanced[:,:,c] - p2) / (p98 - p2), 0, 1)
    
    axes[0, 1].imshow(enhanced)
    axes[0, 1].set_title('Enhanced Contrast')
    axes[0, 1].axis('off')
    
    # Horizontal profile
    middle_row = flat_field_model.shape[0] // 2
    axes[1, 0].plot(flat_field_model[middle_row, :, 0], 'r-', label='Red')
    axes[1, 0].plot(flat_field_model[middle_row, :, 1], 'g-', label='Green')
    axes[1, 0].plot(flat_field_model[middle_row, :, 2], 'b-', label='Blue')
    axes[1, 0].set_title('Horizontal Profile (Middle Row)')
    axes[1, 0].set_xlabel('Column')
    axes[1, 0].set_ylabel('Intensity')
    axes[1, 0].legend()
    
    # Vertical profile
    middle_col = flat_field_model.shape[1] // 2
    axes[1, 1].plot(flat_field_model[:, middle_col, 0], 'r-', label='Red')
    axes[1, 1].plot(flat_field_model[:, middle_col, 1], 'g-', label='Green')
    axes[1, 1].plot(flat_field_model[:, middle_col, 2], 'b-', label='Blue')
    axes[1, 1].set_title('Vertical Profile (Middle Column)')
    axes[1, 1].set_xlabel('Row')
    axes[1, 1].set_ylabel('Intensity')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "flat_field_model_visualization.png"))
    plt.close()
    print(f"Saved visualization to {os.path.join(output_dir, 'flat_field_model_visualization.png')}")
    sys.stdout.flush()
    
    # Channel visualizations
    channels = ['Red', 'Green', 'Blue']
    for i, channel in enumerate(channels):
        plt.figure(figsize=(10, 8))
        
        # Extract this channel
        channel_data = flat_field_model[:, :, i]
        
        # Create heatmap
        plt.imshow(channel_data, cmap='hot')
        plt.colorbar(label='Intensity')
        plt.title(f'{channel} Channel Flat-field Model')
        
        # Save
        output_path = os.path.join(output_dir, f"flat_field_model_{channel.lower()}_channel.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Saved {channel} channel visualization to {output_path}")
        sys.stdout.flush()


def main():
    """Main function to run the script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Estimate flat field distortion model from L-channel normalized tiles.")
    parser.add_argument("--data-dir", help="Base data directory")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--parameters-dir", help="Parameters directory")
    parser.add_argument("--input", help="Input folder containing normalized tiles (for standalone use)")
    parser.add_argument("--output", help="Output folder for flat field model (for standalone use)")
    parser.add_argument("--window-size", type=int, default=5, help="Window size for LCoV calculation (default: 5)")
    parser.add_argument("--max-tiles", type=int, default=None, help="Maximum number of tiles to process (default: all)")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of tiles to process in each batch (default: 10)")
    parser.add_argument("--min-std-dev", type=float, default=1.0, 
                       help="Minimum standard deviation required for a valid candidate (default: 1.0)")
    parser.add_argument("--visualizations", action="store_true", help="Create visualizations")
    
    args = parser.parse_args()
    
    # Determine paths based on whether called from GUI or standalone
    if args.data_dir and args.output_dir and args.parameters_dir:
        # Called from GUI - use default paths
        # Input folder is the L-channel normalized and background-removed tiles directory
        input_folder = os.path.join(args.data_dir, "Tiles-Medium-L-Channel-Normalized-BG-Removed")
        
        # Output folder for model is in Parameters directory
        model_output_folder = args.parameters_dir
        
        # Output folder for visualizations is in Results directory
        visualizations_folder = os.path.join(args.output_dir, "Flat-Field-Model")
        
        # When in GUI mode, always create visualizations regardless of flag
        create_visualizations = True
        
        # Set up log directory - use Logs directory at the same level as Data, Results, Parameters
        base_dir = os.path.dirname(args.data_dir)  # Get parent directory of Data
        log_dir = os.path.join(base_dir, "Logs")
        os.makedirs(log_dir, exist_ok=True)
    elif args.input and args.output:
        # Called standalone with custom paths
        input_folder = args.input
        model_output_folder = args.output
        visualizations_folder = os.path.join(args.output, "visualizations")
        
        # In standalone mode, use the visualizations flag
        create_visualizations = args.visualizations
        
        # Set up log directory for standalone mode
        log_dir = "Logs"
        os.makedirs(log_dir, exist_ok=True)
    else:
        print("ERROR: Either provide --data-dir, --output-dir, and --parameters-dir OR provide --input and --output")
        sys.stdout.flush()
        return 1
    
    # Create output directories if they don't exist
    os.makedirs(model_output_folder, exist_ok=True)
    os.makedirs(visualizations_folder, exist_ok=True)
    
    # Configure logging to file in the Logs directory
    log_file = os.path.join(log_dir, "flat_field_model.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Print configuration
    print("Starting Flat Field Model Estimation")
    print(f"Input folder: {input_folder}")
    print(f"Model output folder: {model_output_folder}")
    print(f"Visualizations folder: {visualizations_folder}")
    print(f"Log file: {log_file}")
    sys.stdout.flush()
    
    logger.info("Starting Flat Field Model Estimation")
    logger.info(f"Input folder: {input_folder}")
    logger.info(f"Model output folder: {model_output_folder}")
    
    # Start timer
    start_time = time.time()
    
    try:
        # Calculate the flat field model
        print(f"Calculating flat field model from tiles in {input_folder}...")
        print(f"Using window size {args.window_size} and minimum standard deviation threshold of {args.min_std_dev}")
        sys.stdout.flush()
        
        flat_field_model = extract_flat_field_model(
            input_folder,
            model_output_folder,
            window_size=args.window_size,
            max_tiles=args.max_tiles,
            batch_size=args.batch_size,
            min_std_dev=args.min_std_dev
        )
        
        # Create visualizations
        print("Creating visualizations...")
        sys.stdout.flush()
        visualize_flat_field_model(flat_field_model, visualizations_folder)
        
        # Report processing time
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Processing completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        sys.stdout.flush()
        logger.info(f"Processing completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        
        return 0
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.stdout.flush()
        logger.error(f"ERROR: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    # For Windows compatibility
    multiprocessing.freeze_support()
    
    # Configure basic console logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    sys.exit(main())
