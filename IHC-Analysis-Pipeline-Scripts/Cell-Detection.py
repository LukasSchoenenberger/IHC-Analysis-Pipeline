#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cell Detection Script
--------------------
This script detects cell nuclei in RGB IHC images based on the hematoxylin channel.
The process includes stain separation, dynamic thresholding, and 
multi-pass processing to refine the detection of cells.

Part of the IHC Pipeline GUI application.
"""

import os
import sys
import tifffile
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_fill_holes, uniform_filter
from scipy.linalg import pinv
import matplotlib.pyplot as plt
import cv2
from scipy import stats
from skimage.measure import label as ski_label
from iaf.morph.watershed import separate_neighboring_objects, label_to_eroded_bw_mask
from scipy.stats import median_abs_deviation
from pathlib import Path
import re
import multiprocessing
from functools import partial
import time
import csv
import json
import argparse
import logging

# Configure logging
logger = logging.getLogger("Cell-Detection")

class CellDetector:
    """
    Cell Detection Pipeline
    
    This class detects cell nuclei in RGB IHC images based on the hematoxylin channel.
    The process includes stain separation, dynamic thresholding, and 
    multi-pass processing to refine the detection of cells.
    
    The key steps are:
    1. Stain separation using color deconvolution with simplified background handling
    2. Thresholding and noise reduction through multi-pass processing
    3. Watershed segmentation for separating touching nuclei
    4. Parallel processing support for handling multiple tiles efficiently
    5. Output of overlay images and cell count data
    """
    
    # Default parameters that will be overridden from the loaded file
    DEFAULT_PARAMS = {
        # Thresholding parameters
        'median_mad_k': 3.0,              # Median-MAD threshold factor
        
        # First pass parameters
        'first_pass_min_size': 100,       # Minimum component size for first pass
        'first_pass_percentile': 25,      # Percentile threshold for first pass filtering
        
        # Second pass parameters
        'second_pass_intensity_threshold': 0.5,  # Intensity threshold for second pass
        'second_pass_max_size': 1000,     # Maximum component size for second pass
        
        # Third pass parameters
        'third_pass_min_size': 80,        # Minimum component size for third pass
        'third_pass_percentile': 80,      # Percentile for distance threshold in third pass
        
        # Component filtering
        'connected_component_size': 1300, # Size threshold for filter_connected_components
        
        # Watershed parameters
        'watershed_min_size': 13,         # Minimum size for watershed segmentation
        'watershed_max_size': 150,        # Maximum size for watershed segmentation
        'small_component_threshold': 20   # Pixel threshold for small component filtering
    }
    
    def __init__(self, parameters_dir, create_density_map=False, create_overview=False):
        """
        Initialize the cell detector with parameters and stain matrix
        
        Args:
            parameters_dir (str): Directory containing parameter files
            create_density_map (bool): Whether to create density maps
            create_overview (bool): Whether to create overview visualizations
        """
        # Set parameters directory
        self.parameters_dir = Path(parameters_dir)
        
        # Set output options
        self.create_density_map = create_density_map
        self.create_overview = create_overview
        
        # Define paths for parameter and reference files
        self.stain_vectors_path = self.parameters_dir / "reference_stain_vectors.txt"
        self.hematoxylin_global_stats_path = self.parameters_dir / "hematoxylin_global_stats.json"
        self.cell_detection_params_path = self.parameters_dir / "cell_detection_parameters.json"
        
        # Load parameters
        self.params = self.DEFAULT_PARAMS.copy()
        self._load_parameters()
        
        # Load stain matrix
        self.stain_matrix = None
        self.stain_names = None
        self._load_stain_matrix()
        
        # Background detection parameters
        self.background_value = 255  # Background is 255 in preprocessed images
        self.background_threshold = 253  # Allow slight tolerance for near-white pixels
        
        # Background RGB values for OD calculation
        self.background = np.array([255, 255, 255])
        
        # Filtering range for stain concentrations
        self.min_value = -3
        self.max_value = 3
        
        # Global statistics for visualization normalization
        self.global_hema_min = None
        self.global_hema_max = None
        
        # Load global statistics if available
        self._load_global_stats()
    
    def _load_parameters(self):
        """Load processing parameters from JSON file"""
        if not self.cell_detection_params_path.exists():
            print(f"ERROR: Cell detection parameters file not found: {self.cell_detection_params_path}")
            sys.stdout.flush()
            sys.exit(1)
            
        try:
            with open(self.cell_detection_params_path, 'r') as f:
                loaded_params = json.load(f)
            
            # Update parameters
            self.params.update(loaded_params)
            
            print(f"Successfully loaded cell detection parameters from: {self.cell_detection_params_path}")
            print(f"Cell detection will run with the following settings:")
            print(f"  - Median-MAD threshold: k={self.params['median_mad_k']}")
            print(f"  - Component size limits: {self.params['first_pass_min_size']}-{self.params['second_pass_max_size']} pixels")
            print(f"  - Watershed parameters: min={self.params['watershed_min_size']}, max={self.params['watershed_max_size']}")
            print(f"  - Small component filter threshold: {self.params['small_component_threshold']} pixels")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"ERROR: Failed to load cell detection parameters: {str(e)}")
            sys.stdout.flush()
            sys.exit(1)
    
    def _load_stain_matrix(self):
        """Load stain matrix from reference_stain_vectors.txt"""
        if not self.stain_vectors_path.exists():
            print(f"ERROR: No stain-vectors found. Expected file: {self.stain_vectors_path}")
            sys.stdout.flush()
            sys.exit(1)
            
        try:
            # Parse the stain vectors file
            stain_matrix = []
            stain_names = []
            
            with open(self.stain_vectors_path, 'r') as f:
                lines = f.readlines()
                
                # Skip the first line (title)
                for line in lines[1:]:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Parse line format: "Stain_Name: [r, g, b]"
                    if ':' in line and '[' in line and ']' in line:
                        parts = line.split(': [')
                        if len(parts) == 2:
                            name = parts[0].strip()
                            stain_names.append(name)
                            
                            # Extract vector values
                            vector_str = parts[1].rstrip(']')
                            vector = np.array([float(x.strip()) for x in vector_str.split(',')])
                            stain_matrix.append(vector)
            
            if not stain_matrix:
                raise ValueError("No valid stain vectors found in file")
                
            self.stain_matrix = np.array(stain_matrix)
            self.stain_names = stain_names
            
            print(f"Loaded stain matrix with shape: {self.stain_matrix.shape}")
            print("Stain names:", self.stain_names)
            sys.stdout.flush()
            
        except Exception as e:
            print(f"ERROR: Failed to load stain matrix: {str(e)}")
            sys.stdout.flush()
            sys.exit(1)
    
    def _load_global_stats(self):
        """Load global statistics for normalization, if available"""
        try:
            if self.hematoxylin_global_stats_path.exists():
                with open(self.hematoxylin_global_stats_path, 'r') as f:
                    self.global_stats = json.load(f)
                
                print(f"Loaded global statistics:")
                for key, value in self.global_stats.items():
                    if key != "timestamp":
                        print(f"  {key}: {value}")
                        
                # Set visualization min/max
                self.global_hema_min = self.global_stats.get('global_hema_min', -3.0)
                self.global_hema_max = self.global_stats.get('global_hema_max', 3.0)
                
                sys.stdout.flush()
                return True
            else:
                print(f"Note: Global statistics file not found: {self.hematoxylin_global_stats_path}")
                print("Will use per-tile normalization for visualization.")
                sys.stdout.flush()
                return False
        except Exception as e:
            print(f"Error loading global stats: {e}")
            sys.stdout.flush()
            return False
    
    def rgb_to_od(self, img):
        """
        Convert RGB image to optical density (OD) space
        
        Args:
            img (numpy.ndarray): RGB image
            
        Returns:
            numpy.ndarray: Optical density representation
        """
        h, w, c = img.shape
        img_flat = img.reshape(h*w, c).T
        img_flat = img_flat.astype(float)
        
        eps = 1e-6
        img_flat = np.maximum(img_flat, eps)
        
        od = -np.log10(img_flat / self.background[:, np.newaxis])
        
        return od
    
    def od_to_rgb(self, od):
        """
        Convert back from OD space to RGB
        
        Args:
            od (numpy.ndarray): Optical density values
            
        Returns:
            numpy.ndarray: RGB values
        """
        rgb_flat = self.background[:, np.newaxis] * np.power(10, -od)
        rgb_flat = np.clip(rgb_flat, 0, 255)
        
        return rgb_flat
    
    def identify_background(self, img):
        """
        Identify background pixels in preprocessed tiles with background value of 255
        
        Args:
            img: Original RGB image
            
        Returns:
            numpy.ndarray: Binary mask where True = background, False = foreground
        """
        h, w, _ = img.shape
        img_flat = img.reshape(h*w, 3)
        
        # Pixels where all channels are very close to or equal to 255
        bg_mask = np.all(img_flat >= self.background_threshold, axis=1)
        
        return bg_mask
    
    def extend_background_mask_with_negative_values(self, concentration, bg_mask, channel_name="Hematoxylin"):
        """
        Add pixels with negative concentration to the background mask
        
        Args:
            concentration: Flattened concentration values
            bg_mask: Initial background mask from identify_background
            channel_name: Name of the channel for reporting purposes
            
        Returns:
            numpy.ndarray: Extended background mask including negative concentration areas
        """
        # Identify pixels with negative concentration
        negative_mask = concentration < 0
        
        # Combine with existing background mask
        extended_bg_mask = bg_mask | negative_mask
        
        return extended_bg_mask
    
    def separate_stains(self, img):
        """
        Separate stains using the provided stain matrix with simplified background detection
        for preprocessed tiles where background = 255
        
        Args:
            img: Input RGB image
            
        Returns:
            tuple: (separated_stains, concentrations, background_mask)
        """
        h, w, _ = img.shape
        
        # Convert to optical density
        od = self.rgb_to_od(img)
        
        # Identify background pixels with simplified method
        bg_mask = self.identify_background(img)
        
        # Transpose stain matrix for calculation
        stain_matrix_T = self.stain_matrix.T
        
        # Calculate stain concentrations using the provided stain matrix
        concentrations = np.dot(pinv(stain_matrix_T), od)
        
        # Filter concentrations to the specified range
        filtered_concentrations = np.clip(concentrations, self.min_value, self.max_value)
        
        # Zero out concentration values for background pixels
        filtered_concentrations[:, bg_mask] = 0
        
        # Separate individual stains
        n_stains = self.stain_matrix.shape[0]
        separated_stains = []
        
        for s in range(n_stains):
            # Create concentration matrix with only one stain
            C_single = np.zeros_like(filtered_concentrations)
            C_single[s, :] = filtered_concentrations[s, :]
            
            # Reconstruct OD image with only this stain
            od_single = np.dot(stain_matrix_T, C_single)
            
            # Convert back to RGB
            rgb_single_flat = self.od_to_rgb(od_single)
            
            # Reshape to image
            rgb_single = rgb_single_flat.T.reshape(h, w, 3).astype(np.uint8)
            
            # Make background white
            bg_mask_reshaped = bg_mask.reshape(h, w)
            rgb_single[bg_mask_reshaped] = [255, 255, 255]
            
            separated_stains.append(rgb_single)
        
        return separated_stains, filtered_concentrations, bg_mask
    
    def extract_hematoxylin_channel(self, img):
        """
        Extract hematoxylin channel from an image using the stain matrix
        
        Args:
            img: Input RGB image
            
        Returns:
            tuple: (hematoxylin_rgb, hematoxylin_concentration, extended_bg_mask)
        """
        # Separate stains
        separated_stains, concentrations, bg_mask = self.separate_stains(img)
        
        # Find the index of the hematoxylin stain based on name
        hema_idx = None
        for i, name in enumerate(self.stain_names):
            if 'hematoxylin' in name.lower() or 'h&e' in name.lower() or 'he' in name.lower() or 'nuclei' in name.lower():
                hema_idx = i
                break
        
        if hema_idx is None:
            hema_idx = 0
        
        # Extract hematoxylin channel
        hema_rgb = separated_stains[hema_idx]
        hema_concentration = concentrations[hema_idx, :]
        
        # Extend background mask to include negative hematoxylin intensity values
        extended_bg_mask = self.extend_background_mask_with_negative_values(
            hema_concentration, bg_mask, "Hematoxylin"
        )
        
        # Ensure background has zero concentration
        hema_concentration[extended_bg_mask] = 0
        
        return hema_rgb, hema_concentration, extended_bg_mask
    
    def median_mad_threshold(self, he_channel, k=None):
        """
        Thresholds image using median and MAD statistics
        
        Args:
            he_channel (numpy.ndarray): Hematoxylin channel to threshold
            k (float, optional): Threshold factor, defaults to value in params
            
        Returns:
            numpy.ndarray: Thresholded image
        """
        if k is None:
            k = self.params['median_mad_k']
            
        # Create mask for valid pixels
        if len(he_channel.shape) == 3:
            background_mask = np.all(he_channel == 0, axis=2)
        else:
            background_mask = he_channel == 0
            
        valid_mask = ~background_mask & (he_channel > 0)
        
        if not np.any(valid_mask):
            return np.zeros_like(he_channel)
        
        # Calculate statistics using only valid pixels
        valid_values = he_channel[valid_mask]
        median = float(np.median(valid_values))
        mad = float(stats.median_abs_deviation(valid_values, scale='normal'))
        
        # Apply threshold value
        thresh_value = median + k * mad
        
        # Apply threshold while preserving original values
        thresholded = np.zeros_like(he_channel)
        thresholded[valid_mask & (he_channel >= thresh_value)] = he_channel[valid_mask & (he_channel >= thresh_value)]
        
        return thresholded

    def first_pass_processing(self, image, min_size=None, percentile=None):
        """
        First pass filtering based on size and intensity
        
        Args:
            image (numpy.ndarray): Image to process
            min_size (int, optional): Minimum component size
            percentile (float, optional): Percentile threshold
            
        Returns:
            numpy.ndarray: Filtered image
        """
        if min_size is None:
            min_size = self.params['first_pass_min_size']
        if percentile is None:
            percentile = self.params['first_pass_percentile']
        
        # Initialize output image
        result = np.zeros_like(image)
        
        # Create binary mask and label connected components
        binary = image > 0
        labeled_array, num_features = ndimage.label(binary)
        
        # Process each connected component
        for label_num in range(1, num_features + 1):
            # Create mask for current component
            component_mask = labeled_array == label_num
            component_size = np.sum(component_mask)
            
            # Only process components above minimum size
            if component_size >= min_size:
                # Get intensities of pixels in this component
                component_intensities = image[component_mask]
                
                # Calculate percentile threshold of non-zero intensities
                percentile_thresh = np.percentile(component_intensities[component_intensities > 0], percentile)
                
                # Keep only pixels above percentile threshold, maintaining their original intensities
                keep_mask = component_mask & (image > percentile_thresh)
                result[keep_mask] = image[keep_mask]
        
        return result

    def second_pass_processing(self, image, intensity_threshold=None, max_size=None):
        """
        Second pass filtering based on intensity threshold and size
        
        Args:
            image (numpy.ndarray): Image to process
            intensity_threshold (float, optional): Threshold for component mean intensity
            max_size (int, optional): Maximum component size in pixels
            
        Returns:
            numpy.ndarray: Filtered image
        """
        if intensity_threshold is None:
            intensity_threshold = self.params['second_pass_intensity_threshold']
        if max_size is None:
            max_size = self.params['second_pass_max_size']
        
        # Initialize output image
        result = np.zeros_like(image)
        
        # Create binary mask and label connected components
        binary = image > 0
        labeled_array, num_features = ndimage.label(binary)
        
        # Handle empty images
        if num_features == 0:
            return result
        
        # Process each component
        for label_num in range(1, num_features + 1):
            # Get component mask
            component_mask = labeled_array == label_num
            component_size = np.sum(component_mask)
            
            # Check size threshold
            if component_size > max_size:
                continue  # Skip component that's too large
            
            # Get mean intensity
            mean_intensity = np.mean(image[component_mask])
            
            # Keep components below the intensity threshold
            if mean_intensity <= intensity_threshold:
                result[component_mask] = image[component_mask]
        
        return result

    def third_pass_processing(self, image, min_size=None, percentile=None):
        """
        Third pass refinement using intensity-weighted centroids
        
        Args:
            image (numpy.ndarray): Image to process
            min_size (int, optional): Minimum component size
            percentile (float, optional): Percentile for distance threshold
            
        Returns:
            numpy.ndarray: Filtered image
        """
        if min_size is None:
            min_size = self.params['third_pass_min_size']
        if percentile is None:
            percentile = self.params['third_pass_percentile']
        
        # Initialize output image
        result = np.zeros_like(image)
        
        # Create binary mask and label components
        binary = image > 0
        labeled_array, num_features = ndimage.label(binary)
        
        # Process each component
        for label_num in range(1, num_features + 1):
            # Get component mask
            component_mask = labeled_array == label_num
            component_size = np.sum(component_mask)
            
            # Skip components smaller than minimum size
            if component_size >= min_size:
                # Get pixel positions and intensities for this component
                component_positions = np.where(component_mask)
                component_intensities = image[component_positions]
                
                # Calculate intensity-weighted centroid
                total_intensity = np.sum(component_intensities)
                if total_intensity <= 0:
                    continue
                    
                centroid_y = np.sum(component_positions[0] * component_intensities) / total_intensity
                centroid_x = np.sum(component_positions[1] * component_intensities) / total_intensity
                
                # Calculate distances from centroid
                distances = np.sqrt(
                    (component_positions[0] - centroid_y)**2 + 
                    (component_positions[1] - centroid_x)**2
                )
                
                # Keep pixels within specified percentile of distances
                distance_threshold = np.percentile(distances, percentile)
                keep_indices = distances <= distance_threshold
                
                # Get positions of pixels to keep
                keep_positions = (
                    component_positions[0][keep_indices],
                    component_positions[1][keep_indices]
                )
                
                # Copy original intensities for kept pixels
                if len(keep_positions[0]) > 0:
                    result[keep_positions] = image[keep_positions]
        
        return result

    def filter_small_watershed_components(self, labels, size_threshold=None):
        """
        Remove watershed components smaller than threshold
        
        Args:
            labels (numpy.ndarray): Labeled image from watershed
            size_threshold (int, optional): Size threshold for filtering
            
        Returns:
            numpy.ndarray: Filtered labels
        """
        if size_threshold is None:
            size_threshold = self.params['small_component_threshold']
        
        # Get unique labels from watershed segmentation
        unique_labels = np.unique(labels)
        
        # Create output array
        filtered_labels = labels.copy()
        
        # Handle empty images
        if len(unique_labels) <= 1:  # Only background present
            return filtered_labels
        
        # Remove components smaller than threshold
        for label in unique_labels[1:]:  
            component_mask = labels == label
            component_size = np.sum(component_mask)
            
            if component_size < size_threshold:
                filtered_labels[component_mask] = 0
        
        return filtered_labels

    def filter_connected_components(self, binary_img, size_threshold=None):
        """
        Remove components larger than the threshold
        
        Args:
            binary_img (numpy.ndarray): Binary image to filter
            size_threshold (int, optional): Maximum component size
            
        Returns:
            numpy.ndarray: Filtered binary image
        """
        if size_threshold is None:
            size_threshold = self.params['connected_component_size']
        
        # Perform connected component labeling with 8-connectivity
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
        
        # Get sizes of all components (excluding background)
        component_sizes = stats[1:, cv2.CC_STAT_AREA]
        
        # Handle empty images
        if len(component_sizes) == 0:
            return np.zeros_like(binary_img)
        
        # Initialize output image
        output = np.zeros_like(binary_img)
        
        # Keep only components below size threshold
        for label in range(1, num_labels):  # Start from 1 to skip background
            if stats[label, cv2.CC_STAT_AREA] <= size_threshold:
                output[labels == label] = 255
                
        return output

    def post_processing(self, image, min_size=None, max_size=None, small_threshold=None):
        """
        Perform watershed segmentation to separate touching nuclei
        
        Args:
            image (numpy.ndarray): Image to process
            min_size (int, optional): Minimum size for watershed
            max_size (int, optional): Maximum size for watershed
            small_threshold (int, optional): Threshold for small component filtering
            
        Returns:
            tuple: (watershed_labels, object_count)
        """
        if min_size is None:
            min_size = self.params['watershed_min_size']
        if max_size is None:
            max_size = self.params['watershed_max_size']
        if small_threshold is None:
            small_threshold = self.params['small_component_threshold']
        
        # Convert to binary and scale to 8-bit range
        binary = (image > 0).astype(np.uint8) * 255
        if not np.any(binary):
            # Return empty result if no foreground pixels
            return np.zeros_like(binary), 0
        
        # Fill holes 
        filled = binary_fill_holes(binary).astype(np.uint8) * 255
        
        # Filter connected components to remove artifacts
        filtered = self.filter_connected_components(filled)
        
        # Add dilation step with 3x3 kernel and one iteration
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(filtered, kernel, iterations=1)
        
        # Fill holes after dilation
        dilated_filled = binary_fill_holes(dilated).astype(np.uint8) * 255
        
        # Label connected components for watershed
        labeled_img = ski_label(dilated_filled)
        
        # Create eroded binary mask for watershed markers
        bw = label_to_eroded_bw_mask(labeled_img)
        
        # Perform watershed segmentation
        try:
            labels_ws, num_ws, _ = separate_neighboring_objects(
                bw, labeled_img, min_size=min_size, max_size=max_size
            )
            
            # Apply small component filtering
            labels_ws = self.filter_small_watershed_components(labels_ws, small_threshold)
            
            # Count objects after filtering
            num_objects = len(np.unique(labels_ws)) - 1  # Subtract 1 for background
            
        except Exception as e:
            print(f"Error in watershed segmentation: {e}")
            sys.stdout.flush()
            # Fall back to simple labeling if watershed fails
            labels_ws = labeled_img
            num_objects = len(np.unique(labeled_img)) - 1
        
        return labels_ws, num_objects
    
    @staticmethod
    def natural_sort_key(s):
        """
        Key function for natural sorting of strings with numbers
        
        Args:
            s: String to process
            
        Returns:
            list: Key elements for sorting
        """
        def try_int(text):
            try:
                return int(text)
            except ValueError:
                return text
        return [try_int(c) for c in re.split('([0-9]+)', s)]

    def process_single_tile(self, tile_path, output_dir):
        """
        Process a single tile and generate outputs
        
        Args:
            tile_path (Path): Path to the tile image
            output_dir (Path): Output directory for results
            
        Returns:
            dict: Processing results
        """
        try:
            start_time = time.time()
            
            # Load image
            original_img = tifffile.imread(str(tile_path))
            
            # Ensure proper format
            if len(original_img.shape) == 2:
                # Convert grayscale to RGB
                original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
            elif len(original_img.shape) == 3 and original_img.shape[2] > 3:
                # Handle multi-channel images by using only RGB channels
                original_img = original_img[:, :, :3]
            
            # Apply color deconvolution and extract hematoxylin channel
            hema_rgb, hema_concentration, bg_mask = self.extract_hematoxylin_channel(original_img)
            
            # Reshape concentrations to 2D array for further processing
            h, w, _ = original_img.shape
            he_channel = hema_concentration.reshape(h, w)
            
            # Apply median-MAD threshold
            thresholded = self.median_mad_threshold(he_channel)
            
            # First pass processing
            first_pass = self.first_pass_processing(thresholded)
            
            # Second pass processing
            second_pass = self.second_pass_processing(first_pass)
            
            # Third pass processing
            third_pass = self.third_pass_processing(second_pass)
            
            # Watershed segmentation
            labels_ws, num_objects = self.post_processing(third_pass)
            
            # Create overlay
            full_overlay = original_img.copy()
            full_mask = (labels_ws > 0).astype(np.uint8)
            
            # Apply red overlay for detected cells
            for i in range(3):
                channel = full_overlay[:,:,i].copy()
                if i == 0:  # Red channel
                    channel[full_mask == 1] = 255
                else:  # Green and Blue channels
                    channel[full_mask == 1] = 0
                full_overlay[:,:,i] = channel
            
            # Create blended overlay
            alpha = 0.4
            full_blended = cv2.addWeighted(original_img, 1-alpha, full_overlay, alpha, 0)
            
            # Save overlay
            overlay_path = output_dir / f"{tile_path.stem}_overlay.tif"
            tifffile.imwrite(str(overlay_path), full_blended)
            
            processing_time = time.time() - start_time
            
            # Extract row and column from filename if possible
            position = None
            match = re.search(r'tile_r(\d+)_c(\d+)', tile_path.name)
            if match:
                position = (int(match.group(1)), int(match.group(2)))
            
            return {
                'tile_name': tile_path.name,
                'success': True,
                'total_cells': num_objects,
                'processing_time': processing_time,
                'position': position
            }
            
        except Exception as e:
            print(f"Error processing tile {tile_path.name}: {str(e)}")
            sys.stdout.flush()
            import traceback
            traceback.print_exc()
            return {
                'tile_name': tile_path.name,
                'success': False,
                'error': str(e)
            }

    def save_cell_counts_csv(self, output_dir, results_list):
        """
        Save cell counts for all tiles in a CSV file
        
        Args:
            output_dir (Path): Output directory to save CSV
            results_list (list): List of processing results
        """
        csv_path = output_dir / "cell_count.csv"
        
        # Filter for only successful results
        successful_results = [res for res in results_list if res['success']]
        
        # Create a list of results with proper structure
        formatted_results = []
        for result in successful_results:
            # Remove .tif extension if present
            tile_name = result['tile_name']
            if tile_name.lower().endswith('.tif'):
                tile_name = tile_name[:-4]  # Remove last 4 characters (.tif)
            
            # Get position data (default to -1,-1 if not available)
            position = result.get('position')
            row, col = position if position is not None else (-1, -1)
            
            # Format position consistently with comma
            position_str = f"{row},{col}" if position else ""
            
            # Add to formatted results
            formatted_results.append({
                'tile_name': tile_name,
                'cell_count': result['total_cells'],
                'position': position_str,
                'row': row,
                'col': col
            })
        
        # Sort by row then column if position is available
        has_position = any(res['position'] for res in formatted_results)
        if has_position:
            sorted_results = sorted(formatted_results, key=lambda x: (x['row'], x['col']))
        else:
            # Otherwise, sort by natural sort order of tile name
            sorted_results = sorted(formatted_results, key=lambda x: self.natural_sort_key(x['tile_name']))
        
        # Write the CSV file
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header with correct column names
            writer.writerow(['tile_name', 'cell_count', 'position'])
            
            # Write data for each tile
            for result in sorted_results:
                writer.writerow([
                    result['tile_name'],
                    result['cell_count'],
                    result['position']
                ])
        
        print(f"Cell counts saved to: {csv_path}")
        sys.stdout.flush()

def process_single_tile_wrapper(args):
    """
    Wrapper function for multiprocessing
    
    Args:
        args: Tuple containing (tile_path, output_dir, detector_config)
        
    Returns:
        Processing result dictionary
    """
    tile_path, output_dir, detector_config = args
    
    # Create a new detector instance for this process
    detector = CellDetector(
        parameters_dir=detector_config['parameters_dir'],
        create_density_map=detector_config['create_density_map'],
        create_overview=detector_config['create_overview']
    )
    
    # Process the tile
    return detector.process_single_tile(tile_path, output_dir)

def main():
    """Main function for cell detection processing"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Cell Detection Pipeline')
    parser.add_argument('--data-dir', help='Base data directory')
    parser.add_argument('--output-dir', help='Base output directory') 
    parser.add_argument('--parameters-dir', help='Parameters directory')
    parser.add_argument('--base-dir', help='Base directory (for standalone use)')
    parser.add_argument('--density-map', action='store_true', help='Create density maps')
    parser.add_argument('--overview', action='store_true', help='Create overview visualizations')
    
    args = parser.parse_args()
    
    # Determine paths based on GUI vs standalone mode
    if args.data_dir and args.output_dir and args.parameters_dir:
        # Called from GUI
        base_data_dir = args.data_dir
        base_output_dir = args.output_dir
        parameters_dir = args.parameters_dir
        
        # Set up log directory
        log_dir = os.path.join(os.path.dirname(args.data_dir), "Logs")
        os.makedirs(log_dir, exist_ok=True)
    elif args.base_dir:
        # Called standalone
        base_data_dir = args.base_dir
        base_output_dir = os.path.join(args.base_dir, "Results")
        parameters_dir = args.parameters_dir if args.parameters_dir else os.path.join(args.base_dir, "Parameters")
        
        # Set up log directory for standalone mode
        log_dir = os.path.join(args.base_dir, "Logs")
        os.makedirs(log_dir, exist_ok=True)
    else:
        print("ERROR: Either provide --data-dir, --output-dir, and --parameters-dir OR provide --base-dir")
        sys.stdout.flush()
        return
    
    # Set up specific input and output directories
    input_dir = os.path.join(base_data_dir, "Tiles-Medium-L-Channel-Normalized-BG-Removed-Illumination-Corrected-Stain-Normalized-Small-Tiles")
    output_dir = os.path.join(base_output_dir, "Cell-Detection")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure logging to file
    log_file = os.path.join(log_dir, "cell_detection.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    print("Starting Cell Detection")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Parameters directory: {parameters_dir}")
    print(f"Log file: {log_file}")
    sys.stdout.flush()
    
    logger.info("Starting Cell Detection")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Parameters directory: {parameters_dir}")
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.stdout.flush()
        logger.error(f"Input directory not found: {input_dir}")
        return
    
    # Initialize detector
    try:
        detector = CellDetector(
            parameters_dir=parameters_dir,
            create_density_map=args.density_map,
            create_overview=args.overview
        )
    except SystemExit:
        # CellDetector already printed error and called sys.exit()
        return
    
    # Find all tiles to process
    input_path = Path(input_dir)
    tile_files = sorted(input_path.glob("tile_r*_c*.tif"), key=lambda x: CellDetector.natural_sort_key(x.name))
    
    if not tile_files:
        # If no tiles match the specific pattern, try to find any .tif files
        tile_files = sorted(input_path.glob("*.tif"), key=lambda x: CellDetector.natural_sort_key(x.name))
    
    total_tiles = len(tile_files)
    
    if total_tiles == 0:
        print(f"No tiles found in {input_dir}")
        sys.stdout.flush()
        logger.warning(f"No tiles found in {input_dir}")
        return
    
    print(f"Found {total_tiles} tiles to process")
    sys.stdout.flush()
    
    # Smart CPU core allocation
    available_cpus = multiprocessing.cpu_count()
    if available_cpus >= 8:
        max_processes = available_cpus - 1 
    else:
        max_processes = max(1, available_cpus // 2)  # Use half on smaller machines
    
    print(f"Processing using {max_processes} CPU cores.")
    sys.stdout.flush()
    
    # Prepare detector configuration for worker processes
    detector_config = {
        'parameters_dir': parameters_dir,
        'create_density_map': args.density_map,
        'create_overview': args.overview
    }
    
    # Prepare arguments for multiprocessing
    output_path = Path(output_dir)
    args_list = [(tile_path, output_path, detector_config) for tile_path in tile_files]
    
    # Process tiles in parallel using spawn context
    ctx = multiprocessing.get_context('spawn')
    
    with ctx.Pool(processes=max_processes) as pool:
        # Use smaller chunksize for better load balancing
        chunksize = max(1, total_tiles // (max_processes * 4))
        
        # Custom progress tracking
        processed_count = 0
        results = []
        
        # Process tiles with explicit progress reporting
        for result in pool.imap(process_single_tile_wrapper, args_list, chunksize=chunksize):
            processed_count += 1
            results.append(result)
            
            # Log progress every 5% or at least every 50 files
            if processed_count % max(1, min(50, total_tiles // 20)) == 0 or processed_count == total_tiles:
                progress_pct = (processed_count / total_tiles) * 100
                print(f"Progress: {processed_count}/{total_tiles} tiles ({progress_pct:.1f}%)")
                sys.stdout.flush()
    
    # Save results to CSV
    detector.save_cell_counts_csv(output_path, results)
    
    # Count successes and failures
    successful = [res for res in results if res['success']]
    failed = [res for res in results if not res['success']]
    
    # Print summary
    print("\nCell Detection Summary:")
    print(f"Total tiles processed: {total_tiles}")
    print(f"Successfully processed: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    total_cells = 0
    if successful:
        total_cells = sum(res['total_cells'] for res in successful)
        print(f"Total cells detected: {total_cells}")
    
    if failed:
        print("\nFailed tiles:")
        for res in failed:
            print(f"- {res['tile_name']}: {res.get('error', 'Unknown error')}")
            logger.error(f"Failed tile {res['tile_name']}: {res.get('error', 'Unknown error')}")
    
    print("Cell Detection completed successfully")
    sys.stdout.flush()
    logger.info("Cell Detection completed successfully")

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
