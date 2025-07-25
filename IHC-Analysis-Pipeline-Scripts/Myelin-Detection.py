#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Myelin Detection Script
----------------------
This script performs myelin detection using Wolf's adaptive thresholding method.
Uses parameters loaded from JSON configuration files.

Part of the IHC Pipeline GUI application.
"""

import os
import numpy as np
import tifffile
import cv2
from pathlib import Path
import re
import json
import traceback
import pandas as pd
import multiprocessing
from functools import partial
import time
import csv
import argparse
from scipy.linalg import pinv
from scipy.ndimage import uniform_filter, label
import logging
import sys

# Configure logging
logger = logging.getLogger("Myelin-Detection")

class MyelinDetector:
    """
    Myelin detection using Wolf's adaptive thresholding method.
    Updated to work with the IHC Pipeline GUI system.
    """
    def __init__(self, base_dir, parameters_dir):
        """
        Initialize the myelin detector
        
        Args:
            base_dir: Base directory containing data
            parameters_dir: Directory containing parameter files
        """
        self.base_dir = Path(base_dir)
        self.parameters_dir = Path(parameters_dir)
        
        # Define input directory following the preprocessing chain
        self.input_dir = self.base_dir / "Tiles-Medium-L-Channel-Normalized-BG-Removed-Illumination-Corrected-Stain-Normalized-Small-Tiles"
        
        # Define parameter files
        self.stain_vectors_path = self.parameters_dir / "reference_stain_vectors.txt"
        self.global_stats_path = self.parameters_dir / "myelin_global_stats.json"
        self.detection_params_path = self.parameters_dir / "myelin_detection_parameters.json"
        
        # Background value for preprocessed images
        self.background_value = 255
        self.background_threshold = 253
        
        # Background RGB values for OD calculation
        self.background = np.array([255, 255, 255])
        
        # Filtering range for stain concentrations
        self.min_value = -3
        self.max_value = 3
        
        # Load configuration
        self.stain_matrix, self.stain_names = self._load_stain_matrix()
        self.global_stats = self._load_global_statistics()
        self.detection_params = self._load_detection_parameters()
        
        # Auto-detect and report myelin channel
        myelin_channel_name = "Unknown"
        myelin_channel_index = 1  # Default fallback
        
        for i, name in enumerate(self.stain_names):
            if 'myelin' in name.lower() or 'blue' in name.lower() or 'mbp' in name.lower() or 'dab' in name.lower():
                myelin_channel_name = name
                myelin_channel_index = i
                break
        
        print(f"Initialized Myelin Detector")
        print(f"Auto-detected myelin channel: '{myelin_channel_name}' (index {myelin_channel_index})")
        print(f"Wolf parameters: alpha={self.detection_params['alpha']}, k={self.detection_params['k']}, min_component_size={self.detection_params['min_component_size']}")
        sys.stdout.flush()
    
    def _load_stain_matrix(self):
        """Load stain matrix from reference_stain_vectors.txt file - improved version"""
        # Default stain matrix as fallback
        default_matrix = np.array([
            [0.8926, 0.4478, 0.0520],  # Nuclei (H&E)
            [0.7329, 0.6569, 0.1770],  # Myelin (Blue)
            [0.2890, 0.6240, 0.7260]   # Microglia (Brown)
        ])
        default_names = ["Nuclei (H&E)", "Myelin (Blue)", "Microglia (Brown)"]
        
        if not self.stain_vectors_path.exists():
            print(f"ERROR: reference_stain_vectors.txt not found at {self.stain_vectors_path}")
            print("Using default stain matrix")
            sys.stdout.flush()
            return default_matrix, default_names
        
        try:
            stain_matrix = []
            stain_names = []
            
            print(f"Loading stain matrix from: {self.stain_vectors_path.absolute()}")
            sys.stdout.flush()
            
            with open(self.stain_vectors_path, 'r') as f:
                lines = f.readlines()
                
            print(f"File contains {len(lines)} lines")
            sys.stdout.flush()
                
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                # Skip empty lines, comments, and header lines
                if not line or line.startswith('#') or ':' not in line:
                    continue
                
                # Skip lines that are just headers (like "WSI Stain Vectors:")
                if line.endswith(':') and '[' not in line:
                    continue
                    
                # Try to parse the line
                if ': [' in line:
                    # Format: "Name: [R, G, B]"
                    parts = line.split(': [')
                    if len(parts) == 2:
                        name = parts[0].strip()
                        vector_str = parts[1].rstrip(']')
                        try:
                            vector = np.array([float(x.strip()) for x in vector_str.split(',')])
                            if len(vector) == 3:
                                stain_names.append(name)
                                stain_matrix.append(vector)
                                print(f"  Parsed line {line_num}: {name} -> {vector}")
                                sys.stdout.flush()
                        except ValueError as e:
                            print(f"  Skipped line {line_num} (parsing error): {line}")
                            sys.stdout.flush()
                            continue
                else:
                    # Format: "Name: R G B"
                    parts = line.split(':')
                    if len(parts) == 2:
                        name = parts[0].strip()
                        values_str = parts[1].strip()
                        try:
                            values = values_str.split()
                            if len(values) == 3:
                                vector = np.array([float(x) for x in values])
                                stain_names.append(name)
                                stain_matrix.append(vector)
                                print(f"  Parsed line {line_num}: {name} -> {vector}")
                                sys.stdout.flush()
                        except ValueError as e:
                            print(f"  Skipped line {line_num} (parsing error): {line}")
                            sys.stdout.flush()
                            continue
            
            if not stain_matrix:
                raise ValueError("No valid stain vectors found in file")
                
            self.stain_matrix = np.array(stain_matrix)
            self.stain_names = stain_names
            
            print(f"Successfully loaded stain matrix with shape: {self.stain_matrix.shape}")
            print(f"Parsed {len(self.stain_names)} stain channels:")
            for i, (name, vector) in enumerate(zip(self.stain_names, self.stain_matrix)):
                print(f"  [{i}] {name}: {vector}")
            sys.stdout.flush()
            
            return self.stain_matrix, self.stain_names
            
        except Exception as e:
            print(f"Error loading stain matrix: {e}")
            print("Using default stain matrix")
            sys.stdout.flush()
            traceback.print_exc()
            return default_matrix, default_names
    
    def _load_global_statistics(self):
        """Load global statistics from JSON file"""
        if not self.global_stats_path.exists():
            print(f"Warning: Global statistics file not found: {self.global_stats_path}")
            sys.stdout.flush()
            return None
        
        try:
            with open(self.global_stats_path, 'r') as f:
                stats = json.load(f)
            
            print(f"Loaded global statistics from: {self.global_stats_path.absolute()}")
            print(f"Global statistics loaded:")
            for key, value in stats.items():
                if key != "timestamp":
                    print(f"  {key}: {value}")
            sys.stdout.flush()
            return stats
            
        except Exception as e:
            print(f"Error loading global statistics: {e}")
            sys.stdout.flush()
            return None
    
    def _load_detection_parameters(self):
        """Load detection parameters from JSON file"""
        # Default parameters
        default_params = {
            'alpha': 0.5,
            'k': 0.0,
            'window_size': 15,
            'component_filtering': True,
            'min_component_size': 20
        }
        
        if not self.detection_params_path.exists():
            print(f"Warning: Detection parameters file not found: {self.detection_params_path}")
            print("Using default parameters")
            sys.stdout.flush()
            return default_params
        
        try:
            with open(self.detection_params_path, 'r') as f:
                params = json.load(f)
            
            # Load parameters from the JSON structure
            wolf_params = {
                'alpha': params.get('alpha', default_params['alpha']),
                'k': params.get('k', default_params['k']),
                'window_size': params.get('window_size', default_params['window_size']),
                'component_filtering': default_params['component_filtering'],  # Always enable for consistency
                'min_component_size': params.get('min_component_size', default_params['min_component_size'])
            }
            
            print(f"Loaded detection parameters from: {self.detection_params_path.absolute()}")
            print(f"Parameters: alpha={wolf_params['alpha']}, k={wolf_params['k']}, window_size={wolf_params['window_size']}, min_component_size={wolf_params['min_component_size']}")
            sys.stdout.flush()
            return wolf_params
            
        except Exception as e:
            print(f"Error loading detection parameters: {e}")
            print("Using default parameters")
            sys.stdout.flush()
            return default_params
    
    def rgb_to_od(self, img):
        """Convert RGB image to optical density (OD) space"""
        h, w, c = img.shape
        img_flat = img.reshape(h*w, c).T
        img_flat = img_flat.astype(float)
        
        eps = 1e-6
        img_flat = np.maximum(img_flat, eps)
        
        od = -np.log10(img_flat / self.background[:, np.newaxis])
        
        return od
    
    def identify_background(self, img):
        """Identify background pixels in preprocessed tiles"""
        h, w, _ = img.shape
        img_flat = img.reshape(h*w, 3)
        
        # Pixels where all channels are very close to or equal to 255
        bg_mask = np.all(img_flat >= self.background_threshold, axis=1)
        
        return bg_mask
    
    def separate_stains(self, img):
        """Separate stains using the stain matrix"""
        h, w, _ = img.shape
        
        # Convert to optical density
        od = self.rgb_to_od(img)
        
        # Identify background pixels
        bg_mask = self.identify_background(img)
        
        # Transpose stain matrix for calculation
        stain_matrix_T = self.stain_matrix.T
        
        # Calculate stain concentrations
        concentrations = np.dot(pinv(stain_matrix_T), od)
        
        # Filter concentrations to the specified range
        filtered_concentrations = np.clip(concentrations, self.min_value, self.max_value)
        
        # Zero out concentration values for background pixels
        filtered_concentrations[:, bg_mask] = 0
        
        return filtered_concentrations, bg_mask
    
    def extract_myelin_channel(self, img):
        """Extract myelin channel from an image"""
        h, w, _ = img.shape
        
        # Separate stains
        concentrations, bg_mask = self.separate_stains(img)
        
        # Find the index of the myelin stain with better auto-detection
        myelin_idx = None
        for i, name in enumerate(self.stain_names):
            if 'myelin' in name.lower() or 'blue' in name.lower() or 'mbp' in name.lower() or 'dab' in name.lower():
                myelin_idx = i
                break
        
        # If we couldn't identify myelin by name, use the second stain as fallback
        if myelin_idx is None:
            myelin_idx = 1 if concentrations.shape[0] > 1 else 0
            print(f"Warning: Could not auto-detect myelin channel by name, using index {myelin_idx}")
            sys.stdout.flush()
        
        # Extract myelin channel
        myelin_concentration = concentrations[myelin_idx, :]
        
        # Extend background mask to include negative myelin values
        negative_myelin_mask = myelin_concentration < 0
        extended_bg_mask = bg_mask | negative_myelin_mask
        
        # Ensure background has zero concentration
        myelin_concentration[extended_bg_mask] = 0
        
        return myelin_concentration, extended_bg_mask, h, w
    
    def compute_local_statistics(self, image, window_size, bg_mask):
        """Compute local mean and standard deviation for each pixel"""
        # Create a copy with background set to 0
        img_masked = image.copy()
        img_masked[bg_mask] = 0
        
        # Calculate local mean
        sum_img = uniform_filter(img_masked, size=window_size)
        
        # Create mask array
        mask = (~bg_mask).astype(float)
        
        # Count valid pixels in each neighborhood
        count = uniform_filter(mask, size=window_size)
        count = np.maximum(count, 1e-6)
        
        # Calculate mean
        mean = sum_img / count
        
        # Calculate local variance
        sum_sq_img = uniform_filter(img_masked**2, size=window_size)
        mean_sq = sum_sq_img / count
        variance = np.maximum(mean_sq - mean**2, 0.0)
        
        # Calculate standard deviation
        std = np.sqrt(variance)
        
        # Set background values to 0
        mean[bg_mask] = 0
        std[bg_mask] = 0
            
        return mean, std
    
    def filter_small_components(self, binary, min_size):
        """Remove connected components smaller than min_size"""
        if min_size <= 0:
            return binary
        
        # Find connected components using OpenCV for consistency with parameter estimation
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary.astype(np.uint8), connectivity=8
        )
        
        # Create filtered mask
        filtered_mask = np.zeros_like(binary, dtype=np.uint8)
        
        # Keep components larger than min_size (skip background label 0)
        for label in range(1, num_labels):
            component_size = stats[label, cv2.CC_STAT_AREA]
            if component_size >= min_size:
                filtered_mask[labels == label] = 1
        
        return filtered_mask
    
    def apply_wolf_threshold(self, myelin_concentration, bg_mask, h, w):
        """Apply Wolf's adaptive threshold method with component filtering"""
        # Get parameters
        alpha = self.detection_params['alpha']
        k = self.detection_params['k']
        window_size = self.detection_params['window_size']
        min_component_size = self.detection_params['min_component_size']
        
        # Reshape concentration
        myelin_reshaped = myelin_concentration.reshape(h, w)
        bg_mask_reshaped = bg_mask.reshape(h, w)
        
        # Create output binary mask
        binary = np.zeros((h, w), dtype=np.uint8)
        
        # Get global standard deviation
        if self.global_stats and 'global_myelin_std' in self.global_stats:
            R = self.global_stats['global_myelin_std']
        else:
            non_bg = ~bg_mask_reshaped
            if np.any(non_bg):
                R = np.std(myelin_reshaped[non_bg])
            else:
                R = 1.0
        
        R = max(R, 0.01)
        
        # Process non-background pixels
        non_bg = ~bg_mask_reshaped
        if np.any(non_bg):
            # Calculate local statistics
            local_mean, local_std = self.compute_local_statistics(
                myelin_reshaped, window_size, bg_mask_reshaped
            )
            
            # Apply Wolf's formula
            threshold = (1-alpha) * local_mean + alpha * (local_mean + k * (local_std/R - 1))
            
            # Apply the threshold
            binary[non_bg & (myelin_reshaped > threshold)] = 1
            binary[bg_mask_reshaped] = 0
            
            # Apply component filtering if min_component_size > 0
            if min_component_size > 0:
                binary = self.filter_small_components(binary, min_component_size)
        
        return binary
    
    def extract_tile_position(self, tile_path):
        """Extract row and column position from tile filename"""
        try:
            if isinstance(tile_path, str):
                tile_path = Path(tile_path)
                
            filename = tile_path.stem
            
            # Try different patterns
            patterns = [
                r'tile_r(\d+)_c(\d+)',
                r'r(\d+)c(\d+)',
                r'_(\d+)_(\d+)$'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, filename)
                if match:
                    row = int(match.group(1))
                    col = int(match.group(2))
                    return (row, col)
            
            return None
            
        except Exception as e:
            return None
    
    def process_single_tile(self, args):
        """Process a single tile"""
        tile_path, output_dirs = args
        
        try:
            start_time = time.time()
            overlay_dir, results_dir = output_dirs
            
            tile_path = Path(tile_path)
            
            # Get parameters
            min_component_size = self.detection_params['min_component_size']
            
            # Load image
            original_img = tifffile.imread(str(tile_path))
            
            # Ensure proper format
            if len(original_img.shape) == 2:
                original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
            elif len(original_img.shape) == 3 and original_img.shape[2] > 3:
                original_img = original_img[:, :, :3]
            
            # Extract myelin channel
            myelin_concentration, extended_bg_mask, h, w = self.extract_myelin_channel(original_img)
            
            # Calculate mean myelin intensity
            bg_mask_reshaped = extended_bg_mask.reshape(h, w)
            myelin_reshaped = myelin_concentration.reshape(h, w)
            non_bg_mask = ~bg_mask_reshaped
            
            mean_myelin_intensity = np.mean(myelin_reshaped[non_bg_mask]) if np.any(non_bg_mask) else 0
            
            # Extract position
            position = self.extract_tile_position(tile_path)
            
            # Apply Wolf's threshold (now includes component filtering)
            binary_result = self.apply_wolf_threshold(myelin_concentration, extended_bg_mask, h, w)
            
            # Count positive pixels
            positive_pixels = np.sum(binary_result > 0)
            
            # Create overlay for visualization
            overlay = original_img.copy()
            overlay[binary_result > 0, 0] = 255   # Red channel
            overlay[binary_result > 0, 1] = 0     # Green channel
            overlay[binary_result > 0, 2] = 0     # Blue channel
            
            # Create blended overlay
            alpha_blend = 0.3
            blended = cv2.addWeighted(original_img, 1-alpha_blend, overlay, alpha_blend, 0)
            
            # Save overlay
            overlay_path = overlay_dir / f"{tile_path.stem}_overlay.tif"
            tifffile.imwrite(str(overlay_path), blended)
            
            processing_time = time.time() - start_time
            
            return {
                'tile_name': tile_path.name,
                'success': True,
                'positive_pixels': int(positive_pixels),
                'mean_myelin': mean_myelin_intensity,
                'position': position,
                'min_component_size_used': int(min_component_size),
                'processing_time': processing_time
            }
            
        except Exception as e:
            error_trace = traceback.format_exc()
            return {
                'tile_name': Path(tile_path).name,
                'success': False,
                'error': str(e),
                'trace': error_trace
            }
    
    def save_results_csv(self, results_dir, results_list):
        """Save myelin detection results to CSV file"""
        csv_path = results_dir / "myelin_detection_results.csv"
        
        # Filter successful results
        successful_results = [res for res in results_list if res.get('success', False)]
        
        # Format results
        formatted_results = []
        for result in successful_results:
            tile_name = result['tile_name']
            if tile_name.lower().endswith('.tif'):
                tile_name = tile_name[:-4]
            
            position = result.get('position')
            row, col = position if position is not None else (-1, -1)
            position_str = f"{row},{col}"
            
            formatted_results.append({
                'tile_name': tile_name,
                'myelin_count': result['positive_pixels'],
                'mean_myelin': result.get('mean_myelin', 0),
                'min_component_size': result.get('min_component_size_used', 0),
                'position': position_str,
                'row': row,
                'col': col
            })
        
        # Sort by position
        sorted_results = sorted(formatted_results, key=lambda x: (x['row'], x['col']))
        
        # Write CSV file
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['tile_name', 'myelin_count', 'mean_myelin', 'min_component_size', 'position'])
            
            for result in sorted_results:
                writer.writerow([
                    result['tile_name'],
                    result['myelin_count'],
                    f"{result['mean_myelin']:.6f}",
                    result['min_component_size'],
                    result['position']
                ])
        
        print(f"Results saved to: {csv_path}")
        sys.stdout.flush()
    
    def process_all_tiles(self, output_dir):
        """Process all tiles in the input directory"""
        # Setup output directories
        results_dir = Path(output_dir) / "Myelin-Detection"
        overlay_dir = results_dir / "overlays"
        
        # Create directories
        for dir_path in [results_dir, overlay_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Find all tiles
        tile_files = list(self.input_dir.glob("*.tif"))
        if not tile_files:
            print(f"No tiles found in {self.input_dir}")
            sys.stdout.flush()
            return
        
        # Sort tiles naturally
        def natural_sort_key(s):
            def try_int(text):
                try:
                    return int(text)
                except ValueError:
                    return text
            return [try_int(c) for c in re.split('([0-9]+)', str(s))]
        
        tile_files = sorted(tile_files, key=natural_sort_key)
        total_tiles = len(tile_files)
        
        print(f"Found {total_tiles} tiles to process")
        sys.stdout.flush()
        
        # Configure multiprocessing - following GUI-compatible pattern
        available_cpus = multiprocessing.cpu_count()
        if available_cpus >= 8:
            max_processes = available_cpus - 1  # Leave one core free for GUI
        else:
            max_processes = max(1, available_cpus // 2)  # Use half on smaller machines
        
        print(f"Processing using {max_processes} CPU cores.")
        sys.stdout.flush()
        
        # Prepare arguments
        output_dirs = (overlay_dir, results_dir)
        args_list = [(tile_path, output_dirs) for tile_path in tile_files]
        
        # Process tiles in parallel
        print("Starting myelin detection processing...")
        sys.stdout.flush()
        
        start_time = time.time()
        results = []
        
        # Use spawn method for cross-platform compatibility
        ctx = multiprocessing.get_context('spawn')
        
        with ctx.Pool(processes=max_processes) as pool:
            # Use smaller chunksize for better load balancing
            chunksize = max(1, total_tiles // (max_processes * 4))
            
            processed_count = 0
            
            # Process with progress reporting
            for result in pool.imap(self.process_single_tile, args_list, chunksize=chunksize):
                processed_count += 1
                results.append(result)
                
                # Report progress every 5% or at least every 50 tiles
                if processed_count % max(1, min(50, total_tiles // 20)) == 0 or processed_count == total_tiles:
                    progress_pct = (processed_count / total_tiles) * 100
                    print(f"Progress: {processed_count}/{total_tiles} tiles ({progress_pct:.1f}%)")
                    sys.stdout.flush()
        
        # Save results
        self.save_results_csv(results_dir, results)
        
        # Print summary
        successful = [res for res in results if res.get('success', False)]
        failed = [res for res in results if not res.get('success', False)]
        
        print(f"\nMyelin Detection Summary:")
        print(f"Total tiles processed: {total_tiles}")
        print(f"Successfully processed: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            total_positive = sum(res['positive_pixels'] for res in successful)
            print(f"Total myelin positive pixels: {total_positive}")
            print(f"Component filtering applied with min_size: {self.detection_params['min_component_size']}")
        
        if failed and len(failed) <= 10:
            print("\nFailed tiles:")
            for res in failed:
                print(f"- {res['tile_name']}: {res.get('error', 'Unknown error')}")
        
        total_time = time.time() - start_time
        print(f"\nTotal processing time: {total_time:.2f} seconds")
        print("Myelin detection completed successfully")
        sys.stdout.flush()

def main():
    """Main function for myelin detection"""
    parser = argparse.ArgumentParser(description='Myelin detection using Wolf adaptive thresholding')
    parser.add_argument('--data-dir', help='Base data directory')
    parser.add_argument('--output-dir', help='Output directory')
    parser.add_argument('--parameters-dir', help='Parameters directory')
    parser.add_argument('--base-dir', help='Base directory (for standalone use)')
    
    args = parser.parse_args()
    
    # Determine paths based on GUI vs standalone mode
    if args.data_dir and args.output_dir and args.parameters_dir:
        # Called from GUI
        base_dir = args.data_dir
        output_dir = args.output_dir
        parameters_dir = args.parameters_dir
        log_dir = os.path.join(os.path.dirname(args.data_dir), "Logs")
    elif args.base_dir:
        # Called standalone
        base_dir = args.base_dir
        output_dir = os.path.join(base_dir, "Results")
        parameters_dir = args.parameters_dir if args.parameters_dir else os.path.join(base_dir, "Parameters")
        log_dir = os.path.join(base_dir, "Logs")
    else:
        print("ERROR: Either provide --data-dir, --output-dir, and --parameters-dir OR provide --base-dir")
        sys.stdout.flush()
        return
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(parameters_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    log_file = os.path.join(log_dir, "myelin_detection.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    print("Starting Myelin Detection")
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Parameters directory: {parameters_dir}")
    print(f"Log file: {log_file}")
    sys.stdout.flush()
    
    logger.info("Starting Myelin Detection")
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Parameters directory: {parameters_dir}")
    
    try:
        # Create detector
        detector = MyelinDetector(base_dir, parameters_dir)
        
        # Process all tiles
        detector.process_all_tiles(output_dir)
        
    except Exception as e:
        error_msg = f"Error in myelin detection: {str(e)}"
        print(f"ERROR: {error_msg}")
        sys.stdout.flush()
        logger.error(error_msg)
        raise

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
