#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Microglia Parameter Estimation Script for IHC Pipeline
-----------------------------------------------------
Interactive tool for semi-automated parameter adjustment for microglia detection
with color deconvolution and multi-pass processing.

Compatible with IHC Pipeline Master GUI.
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tifffile
import cv2
from pathlib import Path
import random
import time
import traceback
from scipy import stats, ndimage
from scipy.ndimage import binary_fill_holes
from scipy.linalg import pinv
from skimage.measure import label as ski_label
from scipy.stats import median_abs_deviation
from iaf.morph.watershed import separate_neighboring_objects, label_to_eroded_bw_mask
import multiprocessing

def setup_logging(log_dir):
    """Set up logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "microglia_parameter_estimation.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("MicrogliaParameterEstimation")

class MicrogliaParameterEstimation:
    """Main class for microglia detection parameter estimation with validation workflow"""
    
    def __init__(self, data_dir, parameters_dir, output_dir):
        """Initialize the parameter estimation tool"""
        self.data_dir = Path(data_dir)
        self.parameters_dir = Path(parameters_dir)
        self.output_dir = Path(output_dir)
        
        # Input paths
        self.input_tiles_dir = self.data_dir / "Microglia-Detection-Test-Tiles"
        
        # Tile management for validation workflow
        self.all_tiles = []
        self.adjustment_tiles = []  # 5 tiles for parameter adjustment
        self.validation_tiles = []  # Up to 25 tiles for validation
        self.previously_used_tiles = set()  # Track used adjustment tiles
        self.current_sample_tiles = []  # Current tiles being visualized
        
        # Parameters for microglia detection processing
        self.processing_params = {
            'median_mad_k': 4.0,                    # Median-MAD k parameter
            'first_pass_min_size': 110,             # Minimum component size for first pass
            'first_pass_percentile': 25,            # Percentile threshold for first pass
            'second_pass_min_size': 1300,           # Minimum component size for second pass
            'second_pass_max_size': 3500,           # Maximum component size for second pass
            'second_pass_threshold_intensity': 1.1, # Intensity threshold for second pass
            'second_pass_min_high_pixels': 15,      # Minimum high-intensity pixels for second pass
            'third_pass_min_size': 100,             # Minimum component size for third pass
            'third_pass_percentile': 75,            # Percentile for third pass centroid filtering
            'watershed_min_size': 50,               # Minimum size for watershed segmentation
            'watershed_max_size': 2000,             # Maximum size for watershed segmentation
        }
        
        # Background detection parameters
        self.background_value = 255
        self.background_threshold = 253
        self.background = np.array([255, 255, 255])
        
        # Filtering range for stain concentrations
        self.min_value = -3
        self.max_value = 3
        
        # Processing channel (microglia channel, typically index 2)
        self.processing_channel_index = 2
        self.processing_channel_name = "Microglia"
        
        # Global statistics for visualization normalization
        self.global_microglia_min = None
        self.global_microglia_max = None
        self.global_stats = None
        
        # Store the stain matrix and names
        self.stain_matrix = None
        self.stain_names = None
        
        # Parameter definitions for the interface
        self.param_definitions = None
        self.current_param_index = 0
        self.sections = None
        
        # Setup logging
        log_dir = self.data_dir.parent / "Logs"
        self.logger = setup_logging(log_dir)
    
    def load_stain_matrix(self):
        """Load stain matrix from reference_stain_vectors.txt"""
        stain_vectors_path = self.parameters_dir / "reference_stain_vectors.txt"
        
        if not stain_vectors_path.exists():
            print("ERROR: reference_stain_vectors.txt not found")
            sys.stdout.flush()
            self.logger.error(f"Stain vectors file not found: {stain_vectors_path}")
            sys.exit(1)
        
        try:
            stain_matrix = []
            stain_names = []
            
            print("Loading stain matrix...")
            sys.stdout.flush()
            
            with open(stain_vectors_path, 'r') as f:
                lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    # Skip empty lines, comments, and header lines
                    if not line or line.startswith('#') or ':' not in line:
                        continue
                    
                    # Skip lines that are just headers
                    if line.endswith(':'):
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
                            except ValueError:
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
                            except ValueError:
                                continue
            
            if not stain_matrix:
                raise ValueError("No valid stain vectors found in file")
                
            self.stain_matrix = np.array(stain_matrix)
            self.stain_names = stain_names
            
            # Auto-detect microglia channel
            for i, name in enumerate(self.stain_names):
                if 'microglia' in name.lower() or 'brown' in name.lower() or 'dab' in name.lower():
                    self.processing_channel_index = i
                    self.processing_channel_name = name
                    break
            
            print(f"Loaded stain matrix: {self.stain_matrix.shape}")
            print(f"Stain names: {', '.join(self.stain_names)}")
            print(f"Using {self.processing_channel_name} channel (index {self.processing_channel_index}) for processing")
            sys.stdout.flush()
            self.logger.info(f"Loaded stain matrix: {self.stain_matrix.shape}")
            self.logger.info(f"Using {self.processing_channel_name} channel for processing")
            
        except Exception as e:
            print(f"ERROR: Error loading stain matrix: {e}")
            sys.stdout.flush()
            self.logger.error(f"Error loading stain matrix: {e}")
            sys.exit(1)
    
    def load_global_stats(self):
        """Load global statistics for normalization"""
        global_stats_path = self.parameters_dir / "microglia_global_stats.json"
        
        if not global_stats_path.exists():
            print(f"Note: Global statistics file not found: {global_stats_path}")
            print("Will use per-tile normalization for visualization.")
            sys.stdout.flush()
            return
        
        try:
            with open(global_stats_path, 'r') as f:
                self.global_stats = json.load(f)
            
            print(f"Loaded global statistics:")
            for key, value in self.global_stats.items():
                if key != "timestamp":
                    print(f"  {key}: {value}")
            sys.stdout.flush()
                    
            # Set visualization min/max
            self.global_microglia_min = self.global_stats.get('global_microglia_min', -3.0)
            self.global_microglia_max = self.global_stats.get('global_microglia_max', 3.0)
            
        except Exception as e:
            print(f"Error loading global stats: {e}")
            sys.stdout.flush()
    
    def find_tiles(self):
        """Find all tiles and split into adjustment and validation sets"""
        self.all_tiles = list(self.input_tiles_dir.glob("*.tif")) + list(self.input_tiles_dir.glob("*.tiff"))
        
        if not self.all_tiles:
            print(f"ERROR: No tiles found in {self.input_tiles_dir}")
            sys.stdout.flush()
            return False
        
        # Shuffle tiles
        random.shuffle(self.all_tiles)
        
        # Select 5 tiles for adjustment
        self.adjustment_tiles = self.all_tiles[:5]
        
        # Select up to 25 different tiles for validation
        remaining_tiles = self.all_tiles[5:]
        self.validation_tiles = remaining_tiles[:25] if len(remaining_tiles) >= 25 else remaining_tiles
        
        # Track used tiles
        self.previously_used_tiles = set(self.adjustment_tiles)
        
        print(f"Found {len(self.all_tiles)} total tiles")
        print(f"Selected {len(self.adjustment_tiles)} tiles for parameter adjustment")
        print(f"Selected {len(self.validation_tiles)} tiles for validation")
        sys.stdout.flush()
        
        return True
    
    def select_new_adjustment_tiles(self):
        """Select new tiles for parameter adjustment that weren't used before"""
        print("Selecting new tiles for parameter adjustment...")
        sys.stdout.flush()
        
        # Get unused tiles
        unused_tiles = [tile for tile in self.all_tiles if tile not in self.previously_used_tiles]
        
        if len(unused_tiles) >= 5:
            random.shuffle(unused_tiles)
            self.adjustment_tiles = unused_tiles[:5]
        else:
            # If not enough unused tiles, select from all but avoid most recent
            potential_tiles = [tile for tile in self.all_tiles if tile not in self.adjustment_tiles]
            if not potential_tiles:
                potential_tiles = self.all_tiles
            
            random.shuffle(potential_tiles)
            self.adjustment_tiles = potential_tiles[:5]
        
        # Update current sample tiles for visualization
        self.current_sample_tiles = self.adjustment_tiles
        
        # Update previously used set
        self.previously_used_tiles.update(self.adjustment_tiles)
        
        print(f"Selected {len(self.adjustment_tiles)} new tiles for parameter adjustment")
        sys.stdout.flush()
    
    def rgb_to_od(self, img):
        """Convert RGB image to optical density (OD) space"""
        h, w, c = img.shape
        img_flat = img.reshape(h*w, c).T
        img_flat = img_flat.astype(float)
        
        eps = 1e-6
        img_flat = np.maximum(img_flat, eps)
        
        od = -np.log10(img_flat / self.background[:, np.newaxis])
        
        return od
    
    def od_to_rgb(self, od):
        """Convert back from OD space to RGB"""
        rgb_flat = self.background[:, np.newaxis] * np.power(10, -od)
        rgb_flat = np.clip(rgb_flat, 0, 255)
        
        return rgb_flat
    
    def identify_background(self, img):
        """Identify background pixels in preprocessed tiles"""
        h, w, _ = img.shape
        img_flat = img.reshape(h*w, 3)
        
        bg_mask = np.all(img_flat >= self.background_threshold, axis=1)
        
        print(f"Background pixels: {np.sum(bg_mask)} ({np.sum(bg_mask)/len(bg_mask)*100:.2f}%)")
        sys.stdout.flush()
        
        return bg_mask
    
    def extend_background_mask_with_negative_values(self, concentration, bg_mask, channel_name="Microglia"):
        """Add pixels with negative concentration to the background mask"""
        negative_mask = concentration < 0
        extended_bg_mask = bg_mask | negative_mask
        
        additional_pixels = np.sum(negative_mask & ~bg_mask)
        total_bg_pixels = np.sum(extended_bg_mask)
        
        print(f"Extended background mask with {additional_pixels} negative {channel_name} pixels")
        print(f"Total background pixels: {total_bg_pixels} ({total_bg_pixels/len(bg_mask)*100:.2f}%)")
        sys.stdout.flush()
        
        return extended_bg_mask
    
    def separate_stains(self, img):
        """Separate stains using the provided stain matrix"""
        h, w, _ = img.shape
        
        od = self.rgb_to_od(img)
        bg_mask = self.identify_background(img)
        
        stain_matrix_T = self.stain_matrix.T
        concentrations = np.dot(pinv(stain_matrix_T), od)
        
        filtered_concentrations = np.clip(concentrations, self.min_value, self.max_value)
        filtered_concentrations[:, bg_mask] = 0
        
        n_stains = self.stain_matrix.shape[0]
        separated_stains = []
        
        for s in range(n_stains):
            C_single = np.zeros_like(filtered_concentrations)
            C_single[s, :] = filtered_concentrations[s, :]
            
            od_single = np.dot(stain_matrix_T, C_single)
            rgb_single_flat = self.od_to_rgb(od_single)
            rgb_single = rgb_single_flat.T.reshape(h, w, 3).astype(np.uint8)
            
            bg_mask_reshaped = bg_mask.reshape(h, w)
            rgb_single[bg_mask_reshaped] = [255, 255, 255]
            
            separated_stains.append(rgb_single)
        
        return separated_stains, filtered_concentrations, bg_mask
    
    def extract_microglia_channel(self, img):
        """Extract microglia channel from an image"""
        separated_stains, concentrations, bg_mask = self.separate_stains(img)
        
        microglia_rgb = separated_stains[self.processing_channel_index]
        microglia_concentration = concentrations[self.processing_channel_index, :]
        
        extended_bg_mask = self.extend_background_mask_with_negative_values(
            microglia_concentration, bg_mask, self.processing_channel_name
        )
        
        microglia_concentration[extended_bg_mask] = 0
        
        return microglia_rgb, microglia_concentration, extended_bg_mask
    
    def create_microglia_intensity_image(self, microglia_concentration, bg_mask, h, w):
        """Create a visualization of microglia intensity"""
        microglia_reshaped = microglia_concentration.reshape(h, w)
        bg_mask_reshaped = bg_mask.reshape(h, w)
        
        heatmap = np.zeros((h, w, 3), dtype=np.uint8)
        non_bg = ~bg_mask_reshaped
        
        if np.any(non_bg):
            if self.global_microglia_min is not None and self.global_microglia_max is not None:
                min_val = self.global_microglia_min
                max_val = self.global_microglia_max
            else:
                min_val = np.min(microglia_reshaped[non_bg])
                max_val = np.max(microglia_reshaped[non_bg])
            
            if max_val > min_val:
                normalized = np.clip((microglia_reshaped - min_val) / (max_val - min_val), 0, 1)
            else:
                normalized = np.zeros_like(microglia_reshaped)
            
            # Brown colormap for microglia
            heatmap[:,:,0] = (normalized * 139).astype(np.uint8)  # Red component for brown
            heatmap[:,:,1] = (normalized * 69).astype(np.uint8)   # Green component for brown
            heatmap[:,:,2] = (normalized * 19).astype(np.uint8)   # Blue component for brown
            
            heatmap[bg_mask_reshaped] = [255, 255, 255]
        
        return heatmap
    
    def median_mad_threshold(self, channel, k=3):
        """Median-MAD thresholding with background handling"""
        if len(channel.shape) == 3:
            background_mask = np.all(channel == 0, axis=2)
        else:
            background_mask = channel == 0
            
        valid_mask = ~background_mask & (channel > 0)
        
        if not np.any(valid_mask):
            return np.zeros_like(channel)
        
        valid_values = channel[valid_mask]
        median = float(np.median(valid_values))
        mad = float(stats.median_abs_deviation(valid_values, scale='normal'))
        
        thresh_value = median + k * mad
        
        thresholded = np.zeros_like(channel)
        thresholded[valid_mask & (channel >= thresh_value)] = channel[valid_mask & (channel >= thresh_value)]
        
        return thresholded
    
    def first_pass_processing(self, image, min_size, percentile_value):
        """First pass: Size threshold and quartile filtering"""
        result = np.zeros_like(image)
        binary = image > 0
        
        # Use ndimage.label
        labeled_array, num_features = ndimage.label(binary)
        
        for label_num in range(1, num_features + 1):
            # Create binary mask for this component
            component_mask = labeled_array == label_num
            component_size = np.sum(component_mask)
            
            if component_size >= min_size:
                # Get intensity values
                component_intensities = image[component_mask]
                if len(component_intensities) > 0 and np.any(component_intensities > 0):
                    # Calculate threshold percentile
                    q1 = np.percentile(component_intensities[component_intensities > 0], percentile_value)
                    # Only keep pixels above threshold
                    keep_mask = component_mask & (image > q1)
                    result[keep_mask] = image[keep_mask]
        
        return result

    def second_pass_processing(self, image, min_size=1000, max_size=3500, threshold_intensity=1.0, min_high_pixels=200):
        """Second-pass processing with both minimum and maximum size thresholds"""
        result = np.zeros_like(image)
        binary = image > 0
        
        if not np.any(binary):
            return result
            
        # Use ndimage.label
        labeled_array, num_features = ndimage.label(binary)
        
        if num_features == 0:
            return result
        
        # Process each component
        for label_num in range(1, num_features + 1):
            component_mask = labeled_array == label_num
            component_size = np.sum(component_mask)
            
            # Skip components that are too large
            if component_size > max_size:
                continue
                
            # Check the size criterion for medium-sized components
            if component_size >= min_size:
                # Keep this component based on size alone
                result[component_mask] = image[component_mask]
                continue
            
            # If smaller than min_size, check intensity criterion
            high_intensity_pixels = np.sum((image > threshold_intensity) & component_mask)
            
            # Keep the component if it has enough high-intensity pixels
            if high_intensity_pixels >= min_high_pixels:
                result[component_mask] = image[component_mask]
        
        return result

    def third_pass_processing(self, image, min_size, percentile_value):
        """Process image using intensity-weighted centroid filtering"""
        # Initialize result image
        result = np.zeros_like(image)
        
        # Create binary mask and label components
        binary = (image > 0)
        if not np.any(binary):
            return result
            
        # Use ndimage.label
        labels, num_labels = ndimage.label(binary)
        
        # Process each component
        for label in range(1, num_labels + 1):
            # Create mask for current component
            mask = (labels == label)
            
            # Check size threshold
            if np.sum(mask) < min_size:
                continue
                
            # Get row and column coordinates
            rows, cols = np.where(mask)
            if len(rows) == 0:
                continue
                
            # Get pixel values
            pixel_values = image[rows, cols]
            
            # Calculate the intensity-weighted centroid
            total_intensity = np.sum(pixel_values)
            if total_intensity <= 0:
                continue
                
            # Calculate weighted centroid
            centroid_row = np.sum(rows * pixel_values) / total_intensity
            centroid_col = np.sum(cols * pixel_values) / total_intensity
            
            # Calculate distances from centroid
            distances = np.sqrt((rows - centroid_row)**2 + (cols - centroid_col)**2)
            
            # Apply percentile-based distance threshold
            threshold = np.percentile(distances, percentile_value)
            
            # Keep only pixels within threshold distance
            keep_indices = distances <= threshold
            keep_rows = rows[keep_indices]
            keep_cols = cols[keep_indices]
            
            # Copy pixel values to result
            if len(keep_rows) > 0:
                result[keep_rows, keep_cols] = image[keep_rows, keep_cols]
        
        return result

    def post_processing(self, image, min_size=50, max_size=2000):
        """Modified post-processing without large component filtering"""
        # Create binary mask
        binary = (image > 0).astype(np.uint8) * 255
        if not np.any(binary):
            # Return empty result if no foreground pixels
            return np.zeros_like(binary), 0
            
        # Fill holes in binary mask
        filled = binary_fill_holes(binary).astype(np.uint8) * 255
        
        # Label connected components
        labeled_img = ski_label(filled)
        
        # Create eroded mask for watershed
        bw = label_to_eroded_bw_mask(labeled_img)
        
        # Apply watershed to separate touching objects
        try:
            labels_ws, num_ws, _ = separate_neighboring_objects(
                bw, labeled_img, min_size=min_size, max_size=max_size
            )
            num_objects = len(np.unique(labels_ws)) - 1  # Subtract 1 for background
        except Exception as e:
            print(f"Error in watershed segmentation: {e}")
            sys.stdout.flush()
            traceback.print_exc()
            # Fall back to simple labeling if watershed fails
            labels_ws = labeled_img
            num_objects = len(np.unique(labeled_img)) - 1
        
        return labels_ws, num_objects

    def create_colored_labels(self, labels):
        """Create colored visualization of labeled image"""
        if labels is None or not np.any(labels):
            if labels is None:
                return np.zeros((10, 10, 3), dtype=np.uint8)
            return np.zeros((*labels.shape, 3), dtype=np.uint8)
            
        colored = np.zeros((*labels.shape, 3), dtype=np.uint8)
        unique_labels = np.unique(labels)
        
        if len(unique_labels) <= 1:
            return colored
            
        np.random.seed(42)
        colors = np.random.randint(1, 255, size=(len(unique_labels), 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]
        
        for i, label_val in enumerate(unique_labels):
            label_mask = (labels == label_val)
            for c in range(3):
                colored[:,:,c][label_mask] = colors[i][c]
        
        return colored
    
    def process_tile(self, tile_path, params=None):
        """Process a single tile with the given parameters"""
        if params is None:
            params = self.processing_params
            
        try:
            # Load image
            original_img = tifffile.imread(str(tile_path))
            
            # Ensure proper format
            if len(original_img.shape) == 2:
                original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
            elif len(original_img.shape) == 3 and original_img.shape[2] > 3:
                original_img = original_img[:, :, :3]
            
            # Apply color deconvolution and extract microglia channel
            microglia_rgb, microglia_concentration, bg_mask = self.extract_microglia_channel(original_img)
            
            # Reshape concentrations to 2D array
            h, w, _ = original_img.shape
            microglia_channel = microglia_concentration.reshape(h, w)
            
            # Create visualization of microglia intensity
            microglia_heatmap = self.create_microglia_intensity_image(
                microglia_concentration, bg_mask, h, w
            )
            
            # Apply median-MAD threshold
            thresholded = self.median_mad_threshold(microglia_channel, params['median_mad_k'])
            
            # First pass processing
            first_pass = self.first_pass_processing(
                thresholded, 
                int(params['first_pass_min_size']),
                int(params['first_pass_percentile'])
            )
            
            # Second pass processing
            second_pass = self.second_pass_processing(
                first_pass,
                min_size=params['second_pass_min_size'],
                max_size=params['second_pass_max_size'],
                threshold_intensity=params['second_pass_threshold_intensity'],
                min_high_pixels=params['second_pass_min_high_pixels']
            )
            
            # Third pass processing
            third_pass = self.third_pass_processing(
                second_pass, 
                int(params['third_pass_min_size']),
                int(params['third_pass_percentile'])
            )
            
            # Watershed segmentation
            labels_ws, num_objects = self.post_processing(
                third_pass,
                min_size=int(params['watershed_min_size']),
                max_size=int(params['watershed_max_size'])
            )
            
            # Create visualization
            colored_watershed = self.create_colored_labels(labels_ws)
            
            # Create overlay
            full_overlay = original_img.copy()
            full_mask = (labels_ws > 0).astype(np.uint8)
            
            # Use red color for microglia overlay (same as cell detection)
            for i in range(3):
                channel = full_overlay[:,:,i].copy()
                if i == 0:  # Red channel
                    channel[full_mask == 1] = 255
                else:  # Green and Blue channels
                    channel[full_mask == 1] = 0
                full_overlay[:,:,i] = channel
            
            # Create blended overlay
            alpha = 0.3
            full_blended = cv2.addWeighted(original_img, 1-alpha, full_overlay, alpha, 0)
            
            return {
                'original_img': original_img,
                'microglia_channel': microglia_channel,
                'microglia_heatmap': microglia_heatmap,
                'thresholded': thresholded,
                'first_pass': first_pass,
                'second_pass': second_pass,
                'third_pass': third_pass,
                'colored_watershed': colored_watershed,
                'full_blended': full_blended,
                'labels_ws': labels_ws,
                'num_objects': num_objects
            }
        except Exception as e:
            print(f"Error processing tile {tile_path}: {str(e)}")
            sys.stdout.flush()
            traceback.print_exc()
            return None
    
    def visualize_processing_steps(self, tile_paths, params=None):
        """Visualize processing steps for selected tiles"""
        if params is None:
            params = self.processing_params
            
        # Process each tile
        results = []
        for tile_path in tile_paths:
            print(f"Processing {tile_path.name}...")
            sys.stdout.flush()
            result = self.process_tile(tile_path, params)
            if result is not None:
                results.append((tile_path.name, result))
        
        if not results:
            print("No successful processing results to visualize")
            sys.stdout.flush()
            return None
            
        # Create a figure to show all processing steps
        fig = plt.figure(figsize=(24, 4.5 * len(results)))
        gs = gridspec.GridSpec(len(results), 8, figure=fig, 
                              wspace=0.15, hspace=0.3)
        
        plt.suptitle("Microglia Detection Processing Steps", fontsize=18, y=0.98)
        
        column_titles = [
            "Original Image", 
            f"{self.processing_channel_name}", 
            "Thresholded", 
            "First Pass", 
            "Second Pass", 
            "Third Pass",
            "Watershed",
            "Final Overlay"
        ]
        
        for row, (tile_name, result) in enumerate(results):
            for col in range(8):
                ax = fig.add_subplot(gs[row, col])
                
                if row == 0:
                    ax.set_title(column_titles[col], fontsize=14, pad=10)
                
                if col == 0:  # Original Image
                    ax.imshow(result['original_img'])
                    ax.set_ylabel(tile_name, fontsize=12, rotation=0, labelpad=80, ha='right')
                elif col == 1:  # Microglia Channel
                    ax.imshow(result['microglia_heatmap'])
                elif col == 2:  # Thresholded
                    ax.imshow(result['thresholded'], cmap='gray')
                elif col == 3:  # First Pass
                    ax.imshow(result['first_pass'], cmap='gray')
                elif col == 4:  # Second Pass
                    ax.imshow(result['second_pass'], cmap='gray')
                elif col == 5:  # Third Pass
                    ax.imshow(result['third_pass'], cmap='gray')
                elif col == 6:  # Watershed Result
                    ax.imshow(result['colored_watershed'])
                    ax.text(0.5, 0.02, f"n={result['num_objects']}", 
                           transform=ax.transAxes, ha='center', 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                elif col == 7:  # Final Overlay
                    ax.imshow(result['full_blended'])
                
                ax.set_xticks([])
                ax.set_yticks([])
                
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color('black')
                    spine.set_linewidth(1)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig
    
    def parameter_adjustment_interface(self):
        """Interactive keyboard-based interface for parameter adjustment"""
        plt.close('all')
        
        print(f"\n=== Parameter Adjustment Interface ({self.processing_channel_name} Channel) ===")
        print("Loading sample tiles and creating initial visualization...")
        sys.stdout.flush()
        
        self.current_sample_tiles = self.adjustment_tiles
        
        # Create parameter control window
        fig_params = plt.figure(figsize=(8, 18))
        fig_params.canvas.manager.set_window_title("Parameter Controls")
        
        # Create initial visualization
        self.update_visualization_figure(self.processing_params)
        
        # Parameter definitions
        self.param_definitions = [
            {'name': 'median_mad_k', 'description': 'Median-MAD Threshold k', 
             'min': 0.5, 'max': 15.0, 'step': 0.1},
            
            {'name': 'first_pass_min_size', 'description': 'First Pass Min Size', 
             'min': 10, 'max': 500, 'step': 5},
            {'name': 'first_pass_percentile', 'description': 'First Pass Percentile', 
             'min': 5, 'max': 75, 'step': 1},
            
            {'name': 'second_pass_min_size', 'description': 'Second Pass Min Size', 
             'min': 100, 'max': 3000, 'step': 50},
            {'name': 'second_pass_max_size', 'description': 'Second Pass Max Size', 
             'min': 1500, 'max': 7000, 'step': 100},
            {'name': 'second_pass_threshold_intensity', 'description': 'Second Pass Intensity Threshold', 
             'min': 0.1, 'max': 4.0, 'step': 0.1},
            {'name': 'second_pass_min_high_pixels', 'description': 'Second Pass Min High Pixels', 
             'min': 0, 'max': 1000, 'step': 5},
            
            {'name': 'third_pass_min_size', 'description': 'Third Pass Min Size', 
             'min': 25, 'max': 400, 'step': 5},
            {'name': 'third_pass_percentile', 'description': 'Third Pass Percentile', 
             'min': 25, 'max': 99, 'step': 1},
            
            {'name': 'watershed_min_size', 'description': 'Watershed Min Size', 
             'min': 5, 'max': 150, 'step': 5},
            {'name': 'watershed_max_size', 'description': 'Watershed Max Size', 
             'min': 500, 'max': 5000, 'step': 100},
        ]
        
        self.current_param_index = 0
        
        # Create sections
        self.sections = [
            {'title': 'Initial Thresholding', 'params': [0]},
            {'title': 'First Pass Processing', 'params': [1, 2]},
            {'title': 'Second Pass Processing', 'params': [3, 4, 5, 6]},
            {'title': 'Third Pass Processing', 'params': [7, 8]},
            {'title': 'Watershed Segmentation', 'params': [9, 10]},
        ]
        
        # Update parameter display
        self.update_parameter_display(fig_params)
        
        # Keyboard event handler
        def handle_key_press(event):
            if event.key == 'up' or event.key == 'k':
                self.current_param_index = max(0, self.current_param_index - 1)
                self.update_parameter_display(fig_params)
            elif event.key == 'down' or event.key == 'j':
                self.current_param_index = min(len(self.param_definitions) - 1, self.current_param_index + 1)
                self.update_parameter_display(fig_params)
            elif event.key == 'left' or event.key == 'h':
                self.adjust_current_parameter(-1)
                self.update_parameter_display(fig_params)
            elif event.key == 'right' or event.key == 'l':
                self.adjust_current_parameter(1)
                self.update_parameter_display(fig_params)
            elif event.key == 'u':
                print("Updating visualization...")
                sys.stdout.flush()
                self.update_visualization_figure(self.processing_params)
            elif event.key == 'v':
                print("Parameter adjustment complete. Moving to validation...")
                sys.stdout.flush()
                plt.close('all')
                self.show_validation_interface()
        
        fig_params.canvas.mpl_connect('key_press_event', handle_key_press)
        
        print("\nKEYBOARD CONTROLS:")
        print("  ↑ (Up Arrow) / ↓ (Down Arrow): Select parameter")
        print("  ← (Left Arrow) / → (Right Arrow): Adjust value")
        print("  u: Update visualization with current parameters")
        print("  v: Go to validation interface")
        sys.stdout.flush()
        
        plt.show()
        
        return self.processing_params
    
    def update_parameter_display(self, fig):
        """Update the parameter display with current values"""
        fig.clear()
        
        fig.suptitle("Parameter Adjustment", fontsize=16)
        
        # Create subplot with room for instruction box at bottom
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        
        for section_idx, section in enumerate(self.sections):
            ax.text(0.5, y_pos, section['title'], fontsize=14, fontweight='bold', 
                   ha='center', transform=fig.transFigure)
            y_pos -= 0.025  
            
            for param_idx in section['params']:
                param = self.param_definitions[param_idx]
                value = self.processing_params[param['name']]
                
                if isinstance(value, int):
                    value_text = f"{value}"
                else:
                    value_text = f"{value:.2f}"
                
                is_selected = param_idx == self.current_param_index
                text_color = 'red' if is_selected else 'black'
                bbox_props = dict(boxstyle="round,pad=0.3", 
                                 fc="lightyellow" if is_selected else "white", 
                                 ec="red" if is_selected else "gray",
                                 alpha=0.8)
                
                ax.text(0.05, y_pos, f"{param['description']}:", fontsize=12, 
                       color=text_color,
                       transform=fig.transFigure)
                
                range_text = f"({param['min']} - {param['max']}, step: {param['step']})"
                ax.text(0.95, y_pos, f"{value_text} {range_text}", fontsize=12, 
                       ha='right', color=text_color, bbox=bbox_props,
                       transform=fig.transFigure)
                
                y_pos -= 0.04  
            
            y_pos -= 0.02  
        
        instructions = (
            "KEYBOARD CONTROLS:\n"
            "↑ (Up) / ↓ (Down): Select parameter\n"
            "← (Left) / → (Right): Adjust value\n"
            "u: Update visualization\n"
            "v: View validation results"
        )
        fig.text(0.5, 0.03, instructions, fontsize=11, ha='center',
                bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.8),
                transform=fig.transFigure)
        
        fig.canvas.draw()
    
    def adjust_current_parameter(self, direction):
        """Adjust the current parameter value by one step"""
        param = self.param_definitions[self.current_param_index]
        
        current_value = self.processing_params[param['name']]
        step = param['step'] * direction
        
        new_value = current_value + step
        new_value = max(param['min'], min(param['max'], new_value))
        
        self.processing_params[param['name']] = new_value
        
        print(f"Changed {param['description']} from {current_value} to {new_value}")
        sys.stdout.flush()
    
    def update_visualization_figure(self, params):
        """Update or create the visualization figure"""
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            if hasattr(fig, '_is_visualization_figure'):
                plt.close(fig)
        
        fig = self.visualize_processing_steps(self.current_sample_tiles, params)
        if fig:
            fig._is_visualization_figure = True
            plt.figure(fig.number)
            plt.show(block=False)
    
    def show_validation_interface(self):
        """Show validation interface with keyboard controls"""
        print("Processing validation tiles...")
        sys.stdout.flush()
        
        # Process validation tiles
        validation_results = []
        for tile_path in self.validation_tiles:
            print(f"Processing validation tile {tile_path.name}...")
            sys.stdout.flush()
            result = self.process_tile(tile_path, self.processing_params)
            if result is not None:
                validation_results.append((tile_path.name, result))
        
        # Calculate grid size
        grid_size = min(5, max(1, int(np.ceil(np.sqrt(len(validation_results))))))
        
        # Create figure with reduced height to make room for bottom instruction box
        fig = plt.figure(figsize=(15, 14))
        
        fig.suptitle("Validation Results with Current Parameters", fontsize=16, y=0.98)
        
        # Create grid with adjusted spacing for title and bottom instruction box
        gs = gridspec.GridSpec(grid_size, grid_size, figure=fig,
                              top=0.90, bottom=0.12, left=0.05, right=0.95)
        
        for i in range(grid_size * grid_size):
            if i < len(validation_results):
                row = i // grid_size
                col = i % grid_size
                
                ax = fig.add_subplot(gs[row, col])
                
                tile_name, result = validation_results[i]
                
                ax.imshow(result['full_blended'])
                ax.set_title(f"{tile_name}\n{result['num_objects']} cells")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                row = i // grid_size
                col = i % grid_size
                
                ax = fig.add_subplot(gs[row, col])
                ax.axis('off')
        
        # Add instruction box at the bottom
        instruction_text = "KEYBOARD CONTROLS: Press 'r' to refine parameters, 'm' to save and exit"
        plt.figtext(0.5, 0.05, instruction_text, 
                   ha="center", fontsize=14,
                   bbox=dict(boxstyle="round,pad=0.5", fc="yellow", ec="orange", alpha=0.9))
        
        def key_press_handler(event):
            if event.key == 'r':
                print("Key pressed: r (Refine Parameters)")
                sys.stdout.flush()
                plt.close(fig)
                self.select_new_adjustment_tiles()
                self.parameter_adjustment_interface()
            elif event.key == 'm':
                print("Key pressed: m (Save Parameters)")
                sys.stdout.flush()
                plt.close(fig)
                self.save_parameters()
                print("All Microglia Parameter Estimation steps complete!")
                sys.stdout.flush()
        
        fig.canvas.mpl_connect('key_press_event', key_press_handler)
        
        print("\nValidation interface ready. Use keyboard to navigate:")
        print("  Press 'r' to return to parameter adjustment")
        print("  Press 'm' to save parameters and exit")
        sys.stdout.flush()
        
        plt.show()
    
    def save_parameters(self):
        """Save the optimized parameters to the Parameters directory"""
        print("\n" + "="*50)
        print("SAVING PARAMETERS")
        print("="*50)
        sys.stdout.flush()
        
        try:
            self.parameters_dir.mkdir(exist_ok=True, parents=True)
            
            json_path = self.parameters_dir / 'microglia_detection_parameters.json'
            py_path = self.parameters_dir / 'microglia_detection_parameters.py'
            
            print(f"\nSaving to Parameters directory: {self.parameters_dir.absolute()}")
            sys.stdout.flush()
            
            # Save JSON with pipeline determination
            parameters = {
                "pipeline_determination": {
                    "processing_channel_index": self.processing_channel_index,
                    "processing_channel_name": self.processing_channel_name
                },
                "processing": self.processing_params
            }
            
            with open(json_path, 'w') as f:
                json.dump(parameters, f, indent=2)
            
            # Save Python module
            with open(py_path, 'w') as f:
                f.write(f"# Microglia detection parameters optimized on {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Using {self.processing_channel_name} Channel (index {self.processing_channel_index}) for processing\n\n")
                f.write("MICROGLIA_DETECTION_PARAMS = {\n")
                f.write("    'pipeline_determination': {\n")
                f.write(f"        'processing_channel_index': {self.processing_channel_index},  # {self.processing_channel_name} Channel\n")
                f.write("    },\n")
                f.write("    'processing': {\n")
                for param, value in self.processing_params.items():
                    f.write(f"        '{param}': {value},\n")
                f.write("    }\n")
                f.write("}\n")
            
            print(f"✓ Successfully saved to Parameters directory:")
            print(f"  - {json_path.absolute()}")
            print(f"  - {py_path.absolute()}")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"× Error saving to Parameters directory: {e}")
            sys.stdout.flush()
            traceback.print_exc()
        
        print(f"\nFinal Microglia Detection Parameters ({self.processing_channel_name} Channel):")
        for param, value in self.processing_params.items():
            print(f"  {param}: {value}")
        sys.stdout.flush()
        
        print("\nParameter saving complete!")
        print("="*50)
        sys.stdout.flush()
    
    def run_workflow(self):
        """Run the complete microglia parameter adjustment workflow"""
        print("Starting Microglia Parameter Estimation workflow...")
        sys.stdout.flush()
        
        # Load required files
        self.load_stain_matrix()
        self.load_global_stats()
        
        # Find all tiles and split into adjustment and validation sets
        if not self.find_tiles():
            print("ERROR: No tiles found. Please ensure tiles are in the appropriate directory.")
            sys.stdout.flush()
            return False
        
        # Parameter adjustment interface
        print(f"\nAdjusting parameters for microglia detection using {self.processing_channel_name} channel...")
        print("Launching interactive parameter adjustment interface...")
        sys.stdout.flush()
        self.parameter_adjustment_interface()
        
        print("\nWorkflow complete!")
        sys.stdout.flush()
        return True

def main():
    """Main function"""
    # Windows compatibility
    if __name__ == "__main__":
        multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description="Interactive microglia parameter estimation with validation")
    parser.add_argument("--data-dir", required=True, help="Base data directory")
    parser.add_argument("--output-dir", help="Output directory (not used)")
    parser.add_argument("--parameters-dir", required=True, help="Parameters directory")
    parser.add_argument("--base-dir", help="Base directory (for standalone use)")
    
    args = parser.parse_args()
    
    # Determine paths
    if args.data_dir and args.parameters_dir:
        # Called from GUI
        data_dir = args.data_dir
        parameters_dir = args.parameters_dir
        output_dir = args.parameters_dir
    elif args.base_dir:
        # Called standalone
        data_dir = os.path.join(args.base_dir, "Data")
        parameters_dir = args.parameters_dir if args.parameters_dir else os.path.join(args.base_dir, "Parameters")
        output_dir = parameters_dir
    else:
        print("ERROR: Either provide --data-dir and --parameters-dir OR provide --base-dir")
        sys.stdout.flush()
        return
    
    try:
        # Create and run the tool
        estimation_tool = MicrogliaParameterEstimation(data_dir, parameters_dir, output_dir)
        estimation_tool.run_workflow()
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.stdout.flush()
        raise

if __name__ == "__main__":
    main()
