#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Microglia Detection Script
-------------------------
This script detects and quantifies microglia in IHC images using stain separation
and morphological analysis. Uses improved stain separation and color deconvolution.

Part of the IHC Pipeline GUI application.
"""

import os
import re
import numpy as np
import tifffile
import cv2
import json
import argparse
import logging
import sys
import time
import traceback
import multiprocessing
from functools import partial
from pathlib import Path
import pandas as pd
from scipy import stats, ndimage
from scipy.ndimage import binary_fill_holes
from scipy.linalg import pinv
from skimage import measure, morphology, img_as_float
from skimage.measure import label as ski_label

# Try to import iaf.morph.watershed, provide fallback if not available
try:
    from iaf.morph.watershed import separate_neighboring_objects, label_to_eroded_bw_mask
    WATERSHED_AVAILABLE = True
except ImportError:
    print("Warning: iaf.morph.watershed not available. Using simplified watershed fallback.")
    WATERSHED_AVAILABLE = False

# Configure logging
logger = logging.getLogger("Microglia-Detection")

class MicrogliaDetector:
    """
    Class for detecting and quantifying microglia in IHC images.
    Uses improved stain separation and color deconvolution with simplified
    background handling that treats pixels with value 255 as background.
    """
    def __init__(self, data_dir, output_dir, parameters_dir, verbose=True):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.parameters_dir = Path(parameters_dir)
        self.verbose = verbose
        
        # Define directory paths
        self.input_tiles_dir = self.data_dir / "Tiles-Medium-L-Channel-Normalized-BG-Removed-Illumination-Corrected-Stain-Normalized-Small-Tiles"
        self.results_dir = self.output_dir / "Microglia-Detection"
        self.overlay_dir = self.results_dir / "overlays"
        
        # Create output directories
        for dir_path in [self.results_dir, self.overlay_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Paths to stain matrices and parameters
        self.reference_stain_vectors_path = self.parameters_dir / "reference_stain_vectors.txt"
        self.microglia_global_stats_path = self.parameters_dir / "microglia_global_stats.json"
        self.processing_params_json_path = self.parameters_dir / "microglia_detection_parameters.json"
        
        # Background detection parameters
        self.background_value = 255  # Background is 255 in preprocessed images
        self.background_threshold = 253  # Allow slight tolerance for near-white pixels
        
        # Background RGB values for OD calculation
        self.background = np.array([255, 255, 255])
        
        # Filtering range for stain concentrations
        self.min_value = -3
        self.max_value = 3
        
        # Global statistics for visualization normalization
        self.global_microglia_min = None
        self.global_microglia_max = None
        
        # Initialize matrices and parameters
        self.stain_matrix = None
        self.stain_names = None
        self.processing_params = None
        
        # Processing channel attributes - will be auto-detected
        self.processing_channel_index = 2  # Default fallback
        self.processing_channel_name = "Microglia"  # Default fallback
        
        # Load matrices and parameters at initialization
        self.load_stain_matrix()
        self.load_global_stats()
        self.load_processing_parameters()
        
    def load_stain_matrix(self):
        """Load stain matrix from reference_stain_vectors.txt - improved version"""
        if not self.reference_stain_vectors_path.exists():
            print(f"ERROR: reference_stain_vectors.txt not found at {self.reference_stain_vectors_path}")
            sys.stdout.flush()
            sys.exit(1)
        
        try:
            stain_matrix = []
            stain_names = []
            
            if self.verbose:
                print(f"Loading stain matrix from: {self.reference_stain_vectors_path.absolute()}")
                sys.stdout.flush()
            
            with open(self.reference_stain_vectors_path, 'r') as f:
                lines = f.readlines()
                
                if self.verbose:
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
                                    if self.verbose:
                                        print(f"  Parsed line {line_num}: {name} -> {vector}")
                                        sys.stdout.flush()
                            except ValueError as e:
                                if self.verbose:
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
                                    if self.verbose:
                                        print(f"  Parsed line {line_num}: {name} -> {vector}")
                                        sys.stdout.flush()
                            except ValueError as e:
                                if self.verbose:
                                    print(f"  Skipped line {line_num} (parsing error): {line}")
                                    sys.stdout.flush()
                                continue
            
            if not stain_matrix:
                raise ValueError("No valid stain vectors found in file")
                
            self.stain_matrix = np.array(stain_matrix)
            self.stain_names = stain_names
            
            # Auto-detect microglia channel
            original_index = self.processing_channel_index
            original_name = self.processing_channel_name
            
            for i, name in enumerate(self.stain_names):
                if 'microglia' in name.lower() or 'brown' in name.lower() or 'dab' in name.lower():
                    self.processing_channel_index = i
                    self.processing_channel_name = name
                    if self.verbose:
                        print(f"Auto-detected microglia channel: '{name}' at index {i}")
                        sys.stdout.flush()
                    break
            else:
                if self.verbose:
                    print(f"Warning: Could not auto-detect microglia channel, using default index {original_index}")
                    sys.stdout.flush()
            
            if self.verbose:
                print(f"Successfully loaded stain matrix with shape: {self.stain_matrix.shape}")
                print(f"Parsed {len(self.stain_names)} stain channels:")
                for i, (name, vector) in enumerate(zip(self.stain_names, self.stain_matrix)):
                    print(f"  [{i}] {name}: {vector}")
                print(f"Auto-detected microglia processing channel: {self.processing_channel_name} (index {self.processing_channel_index})")
                sys.stdout.flush()
            
            # Verify we have the processing channel
            if len(self.stain_matrix) <= self.processing_channel_index:
                print(f"ERROR: Processing channel (index {self.processing_channel_index}) not available in stain matrix")
                sys.stdout.flush()
                sys.exit(1)
                
            return True
            
        except Exception as e:
            print(f"ERROR: Error loading stain matrix: {e}")
            sys.stdout.flush()
            traceback.print_exc()
            sys.exit(1)
    
    def load_global_stats(self):
        """Load global statistics for normalization, if available"""
        try:
            if self.microglia_global_stats_path.exists():
                with open(self.microglia_global_stats_path, 'r') as f:
                    self.global_stats = json.load(f)
                
                if self.verbose:
                    print(f"Loaded global statistics:")
                    for key, value in self.global_stats.items():
                        if key != "timestamp":
                            print(f"  {key}: {value}")
                    sys.stdout.flush()
                        
                # Set visualization min/max
                self.global_microglia_min = self.global_stats.get('global_microglia_min', -3.0)
                self.global_microglia_max = self.global_stats.get('global_microglia_max', 3.0)
                
                return True
            else:
                if self.verbose:
                    print(f"Note: Global statistics file not found: {self.microglia_global_stats_path}")
                    print("Will use per-tile normalization for visualization.")
                    sys.stdout.flush()
                return False
        except Exception as e:
            if self.verbose:
                print(f"Warning: Error loading global stats: {e}")
                sys.stdout.flush()
            return False
    
    def load_processing_parameters(self):
        """Load processing parameters from JSON file"""
        try:
            if self.processing_params_json_path.exists():
                with open(self.processing_params_json_path, 'r') as f:
                    params_data = json.load(f)
                
                # Flatten the structure for easier access
                self.processing_params = {}
                if isinstance(params_data, dict):
                    for section_key, section_value in params_data.items():
                        if isinstance(section_value, dict):
                            for key, value in section_value.items():
                                self.processing_params[key] = value
                        else:
                            self.processing_params[section_key] = section_value
                
                # Update channel info from parameters if available
                if 'processing_channel_index' in self.processing_params:
                    old_index = self.processing_channel_index
                    old_name = self.processing_channel_name
                    self.processing_channel_index = self.processing_params['processing_channel_index']
                    if 'processing_channel_name' in self.processing_params:
                        self.processing_channel_name = self.processing_params['processing_channel_name']
                    else:
                        # Update name based on new index if available
                        if 0 <= self.processing_channel_index < len(self.stain_names):
                            self.processing_channel_name = self.stain_names[self.processing_channel_index]
                    
                    if self.verbose:
                        print(f"Parameters file overrode channel selection:")
                        print(f"  Changed from: '{old_name}' (index {old_index})")
                        print(f"  Changed to: '{self.processing_channel_name}' (index {self.processing_channel_index})")
                        sys.stdout.flush()
                
                if self.verbose:
                    print(f"Loaded processing parameters from {self.processing_params_json_path}")
                    print(f"Final processing channel: '{self.processing_channel_name}' (index {self.processing_channel_index})")
                    sys.stdout.flush()
                return True
                
            else:
                if self.verbose:
                    print(f"WARNING: Processing parameters not found at {self.processing_params_json_path}")
                    print("Using default processing parameters")
                    sys.stdout.flush()
                
                # Use default parameters
                self.processing_params = {
                    'median_mad_k': 4.0,
                    'first_pass_min_size': 110,
                    'first_pass_percentile': 25,
                    'second_pass_min_size': 1300,
                    'second_pass_max_size': 3500,
                    'second_pass_threshold_intensity': 1.1,
                    'second_pass_min_high_pixels': 15,
                    'third_pass_min_size': 100,
                    'third_pass_percentile': 75,
                    'watershed_min_size': 50,
                    'watershed_max_size': 2000
                }
                return False
                
        except Exception as e:
            if self.verbose:
                print(f"Error loading processing parameters: {e}")
                sys.stdout.flush()
                traceback.print_exc()
            
            # Use default parameters
            self.processing_params = {
                'median_mad_k': 4.0,
                'first_pass_min_size': 110,
                'first_pass_percentile': 25,
                'second_pass_min_size': 1300,
                'second_pass_max_size': 3500,
                'second_pass_threshold_intensity': 1.1,
                'second_pass_min_high_pixels': 15,
                'third_pass_min_size': 100,
                'third_pass_percentile': 75,
                'watershed_min_size': 50,
                'watershed_max_size': 2000
            }
            if self.verbose:
                print("Using default processing parameters (due to error)")
                sys.stdout.flush()
            return False
    
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
        
        return filtered_concentrations, bg_mask
    
    def extract_microglia_channel(self, img):
        """
        Extract microglia channel from an image using the stain matrix
        
        Args:
            img: Input RGB image
            
        Returns:
            tuple: (channel_concentration, extended_bg_mask)
        """
        # Separate stains
        concentrations, bg_mask = self.separate_stains(img)
        
        # Extract the microglia channel
        channel_concentration = concentrations[self.processing_channel_index, :]
        
        # Extend background mask to include negative channel intensity values
        negative_mask = channel_concentration < 0
        extended_bg_mask = bg_mask | negative_mask
        
        # Ensure background has zero concentration
        channel_concentration[extended_bg_mask] = 0
        
        return channel_concentration, extended_bg_mask
    
    def median_mad_threshold(self, channel, k=3):
        """Median-MAD thresholding with background handling"""
        # Create mask for valid pixels (non-background and non-zero)
        valid_mask = (channel > 0)
        
        if not np.any(valid_mask):
            return np.zeros_like(channel)
        
        # Calculate statistics using only valid pixels
        valid_values = channel[valid_mask]
        median = float(np.median(valid_values))
        mad = float(stats.median_abs_deviation(valid_values, scale='normal'))
        
        # Apply single threshold value
        thresh_value = median + k * mad
        
        # Apply threshold while preserving original values
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
    
    def watershed_fallback(self, binary):
        """Simple fallback watershed when iaf.morph.watershed is not available"""
        from scipy import ndimage
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_maxima
        
        # Distance transform
        distance = ndimage.distance_transform_edt(binary)
        
        # Find local maxima as seeds
        local_maxima = peak_local_maxima(distance, min_distance=5, threshold_abs=0.3*distance.max())
        
        # Create markers
        markers = np.zeros_like(binary, dtype=int)
        for i, (y, x) in enumerate(local_maxima):
            markers[y, x] = i + 1
        
        # Apply watershed
        labels = watershed(-distance, markers, mask=binary)
        
        return labels, len(local_maxima)
    
    def post_processing(self, image):
        """Post-processing with watershed segmentation"""
        # Create binary mask
        binary = (image > 0).astype(np.uint8) * 255
        if not np.any(binary):
            # Return empty result if no foreground pixels
            return np.zeros_like(binary), 0
            
        # Fill holes in binary mask
        filled = binary_fill_holes(binary).astype(np.uint8) * 255
        
        # Label connected components
        labeled_img = ski_label(filled)
        
        # Apply watershed to separate touching objects
        try:
            if WATERSHED_AVAILABLE:
                # Create eroded mask for watershed
                bw = label_to_eroded_bw_mask(labeled_img)
                min_size = self.processing_params.get('watershed_min_size', 50)
                max_size = self.processing_params.get('watershed_max_size', 2000)
                labels_ws, num_ws, _ = separate_neighboring_objects(
                    bw, labeled_img, min_size=min_size, max_size=max_size
                )
                num_objects = len(np.unique(labels_ws)) - 1  # Subtract 1 for background
            else:
                # Use fallback watershed
                labels_ws, num_objects = self.watershed_fallback(filled > 0)
                
        except Exception as e:
            print(f"Error in watershed segmentation: {e}")
            sys.stdout.flush()
            # Fall back to simple labeling if watershed fails
            labels_ws = labeled_img
            num_objects = len(np.unique(labeled_img)) - 1
        
        return labels_ws, num_objects
    
    def create_overlay(self, original_img, labels):
        """Create overlay visualization with segmented cells"""
        # Ensure we have RGB image
        if original_img.ndim == 2:
            original_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        else:
            original_rgb = original_img.copy()
        
        # Create overlay with red cells
        full_overlay = original_rgb.copy()
        full_mask = (labels > 0).astype(np.uint8)
        
        # Overlay cells in red
        for i in range(3):
            channel = full_overlay[:,:,i].copy()
            if i == 0:  # Red channel
                channel[full_mask == 1] = 255
            else:  # Green and Blue channels
                channel[full_mask == 1] = 0
            full_overlay[:,:,i] = channel
        
        # Blend original and overlay
        alpha = 0.3
        full_blended = cv2.addWeighted(original_rgb, 1-alpha, full_overlay, alpha, 0)
        
        return full_blended

# Standalone functions for multiprocessing
def extract_microglia_channel_standalone(img, stain_matrix, processing_channel_index):
    """Standalone version of extract_microglia_channel for multiprocessing"""
    background = np.array([255, 255, 255])
    background_threshold = 253
    min_value = -3
    max_value = 3
    
    h, w, _ = img.shape
    
    # Convert to optical density
    img_flat = img.reshape(h*w, 3).T.astype(float)
    eps = 1e-6
    img_flat = np.maximum(img_flat, eps)
    od = -np.log10(img_flat / background[:, np.newaxis])
    
    # Identify background pixels
    img_flat_check = img.reshape(h*w, 3)
    bg_mask = np.all(img_flat_check >= background_threshold, axis=1)
    
    # Calculate stain concentrations
    stain_matrix_T = stain_matrix.T
    concentrations = np.dot(pinv(stain_matrix_T), od)
    
    # Filter concentrations to the specified range
    filtered_concentrations = np.clip(concentrations, min_value, max_value)
    
    # Zero out concentration values for background pixels
    filtered_concentrations[:, bg_mask] = 0
    
    # Extract the microglia channel
    channel_concentration = filtered_concentrations[processing_channel_index, :]
    
    # Extend background mask to include negative channel intensity values
    negative_mask = channel_concentration < 0
    extended_bg_mask = bg_mask | negative_mask
    
    # Ensure background has zero concentration
    channel_concentration[extended_bg_mask] = 0
    
    return channel_concentration, extended_bg_mask

def median_mad_threshold_standalone(channel, k=3):
    """Standalone version of median_mad_threshold for multiprocessing"""
    valid_mask = (channel > 0)
    
    if not np.any(valid_mask):
        return np.zeros_like(channel)
    
    valid_values = channel[valid_mask]
    median = float(np.median(valid_values))
    mad = float(stats.median_abs_deviation(valid_values, scale='normal'))
    
    thresh_value = median + k * mad
    
    thresholded = np.zeros_like(channel)
    thresholded[valid_mask & (channel >= thresh_value)] = channel[valid_mask & (channel >= thresh_value)]
    
    return thresholded

def first_pass_processing_standalone(image, min_size, percentile_value):
    """Standalone version of first_pass_processing for multiprocessing"""
    result = np.zeros_like(image)
    binary = image > 0
    
    labeled_array, num_features = ndimage.label(binary)
    
    for label_num in range(1, num_features + 1):
        component_mask = labeled_array == label_num
        component_size = np.sum(component_mask)
        
        if component_size >= min_size:
            component_intensities = image[component_mask]
            if len(component_intensities) > 0 and np.any(component_intensities > 0):
                q1 = np.percentile(component_intensities[component_intensities > 0], percentile_value)
                keep_mask = component_mask & (image > q1)
                result[keep_mask] = image[keep_mask]
    
    return result

def second_pass_processing_standalone(image, min_size=1000, max_size=3500, threshold_intensity=1.0, min_high_pixels=200):
    """Standalone version of second_pass_processing for multiprocessing"""
    result = np.zeros_like(image)
    binary = image > 0
    
    if not np.any(binary):
        return result
        
    labeled_array, num_features = ndimage.label(binary)
    
    if num_features == 0:
        return result
    
    for label_num in range(1, num_features + 1):
        component_mask = labeled_array == label_num
        component_size = np.sum(component_mask)
        
        if component_size > max_size:
            continue
            
        if component_size >= min_size:
            result[component_mask] = image[component_mask]
            continue
        
        high_intensity_pixels = np.sum((image > threshold_intensity) & component_mask)
        
        if high_intensity_pixels >= min_high_pixels:
            result[component_mask] = image[component_mask]
    
    return result

def third_pass_processing_standalone(image, min_size, percentile_value):
    """Standalone version of third_pass_processing for multiprocessing"""
    result = np.zeros_like(image)
    
    binary = (image > 0)
    if not np.any(binary):
        return result
        
    labels, num_labels = ndimage.label(binary)
    
    for label in range(1, num_labels + 1):
        mask = (labels == label)
        
        if np.sum(mask) < min_size:
            continue
            
        rows, cols = np.where(mask)
        if len(rows) == 0:
            continue
            
        pixel_values = image[rows, cols]
        
        total_intensity = np.sum(pixel_values)
        if total_intensity <= 0:
            continue
            
        centroid_row = np.sum(rows * pixel_values) / total_intensity
        centroid_col = np.sum(cols * pixel_values) / total_intensity
        
        distances = np.sqrt((rows - centroid_row)**2 + (cols - centroid_col)**2)
        
        threshold = np.percentile(distances, percentile_value)
        
        keep_indices = distances <= threshold
        keep_rows = rows[keep_indices]
        keep_cols = cols[keep_indices]
        
        if len(keep_rows) > 0:
            result[keep_rows, keep_cols] = image[keep_rows, keep_cols]
    
    return result

def post_processing_standalone(image, watershed_params):
    """Standalone version of post_processing for multiprocessing"""
    binary = (image > 0).astype(np.uint8) * 255
    if not np.any(binary):
        return np.zeros_like(binary), 0
        
    filled = binary_fill_holes(binary).astype(np.uint8) * 255
    
    labeled_img = ski_label(filled)
    
    try:
        if WATERSHED_AVAILABLE:
            bw = label_to_eroded_bw_mask(labeled_img)
            min_size = watershed_params.get('watershed_min_size', 50)
            max_size = watershed_params.get('watershed_max_size', 2000)
            labels_ws, num_ws, _ = separate_neighboring_objects(
                bw, labeled_img, min_size=min_size, max_size=max_size
            )
            num_objects = len(np.unique(labels_ws)) - 1
        else:
            # Fallback watershed
            from skimage.segmentation import watershed
            from skimage.feature import peak_local_maxima
            
            distance = ndimage.distance_transform_edt(filled > 0)
            local_maxima = peak_local_maxima(distance, min_distance=5, threshold_abs=0.3*distance.max())
            
            markers = np.zeros_like(filled, dtype=int)
            for i, (y, x) in enumerate(local_maxima):
                markers[y, x] = i + 1
            
            labels_ws = watershed(-distance, markers, mask=filled > 0)
            num_objects = len(local_maxima)
            
    except Exception as e:
        labels_ws = labeled_img
        num_objects = len(np.unique(labeled_img)) - 1
    
    return labels_ws, num_objects

def create_overlay_standalone(original_img, labels):
    """Standalone version of create_overlay for multiprocessing"""
    if original_img.ndim == 2:
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    else:
        original_rgb = original_img.copy()
    
    full_overlay = original_rgb.copy()
    full_mask = (labels > 0).astype(np.uint8)
    
    for i in range(3):
        channel = full_overlay[:,:,i].copy()
        if i == 0:  # Red channel
            channel[full_mask == 1] = 255
        else:  # Green and Blue channels
            channel[full_mask == 1] = 0
        full_overlay[:,:,i] = channel
    
    alpha = 0.3
    full_blended = cv2.addWeighted(original_rgb, 1-alpha, full_overlay, alpha, 0)
    
    return full_blended

def process_single_tile(args):
    """
    Process a single tile - designed for multiprocessing
    
    Args:
        args: Tuple containing (tile_path, shared_config)
    
    Returns:
        dict: Processing results with tile name, success, cell count, etc.
    """
    tile_path, shared_config = args
    
    try:
        start_time = time.time()
        tile_name = tile_path.stem
        
        # Extract shared configuration data (no printing during multiprocessing)
        stain_matrix = shared_config['stain_matrix']
        processing_params = shared_config['processing_params']
        output_dir = shared_config['output_dir']
        processing_channel_index = shared_config['processing_channel_index']
        
        # Create overlay directory path
        overlay_dir = Path(output_dir) / "Microglia-Detection" / "overlays"
        
        # Load the original image
        original_img = tifffile.imread(str(tile_path))
        
        # Ensure RGB format
        if len(original_img.shape) == 2:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        elif len(original_img.shape) == 3 and original_img.shape[2] > 3:
            original_img = original_img[:, :, :3]  # Take first 3 channels
        
        # Check if image is entirely background
        background_mask = np.all(original_img == 0, axis=2)
        if np.all(background_mask):
            # Save the original image as overlay
            overlay_path = overlay_dir / f"{tile_name}_overlay.tif"
            tifffile.imwrite(str(overlay_path), original_img)
            
            # Return results with zero cell count
            return {
                'tile_name': tile_name,
                'success': True,
                'cell_count': 0,
                'processing_time': time.time() - start_time
            }
            
        # Extract microglia channel using shared stain matrix
        channel_concentration, bg_mask = extract_microglia_channel_standalone(
            original_img, stain_matrix, processing_channel_index
        )
        
        # Reshape concentrations to 2D array for further processing
        h, w, _ = original_img.shape
        channel_reshaped = channel_concentration.reshape(h, w)
        
        # Apply processing pipeline using the extracted channel
        thresholded = median_mad_threshold_standalone(
            channel_reshaped, processing_params['median_mad_k']
        )
        
        first_pass = first_pass_processing_standalone(
            thresholded, 
            int(processing_params['first_pass_min_size']),
            int(processing_params['first_pass_percentile'])
        )
        
        # Apply second pass processing
        second_pass = second_pass_processing_standalone(
            first_pass,
            min_size=processing_params['second_pass_min_size'],
            max_size=processing_params['second_pass_max_size'],
            threshold_intensity=processing_params['second_pass_threshold_intensity'],
            min_high_pixels=processing_params['second_pass_min_high_pixels']
        )
        
        # Apply third pass processing
        third_pass = third_pass_processing_standalone(
            second_pass, 
            int(processing_params['third_pass_min_size']),
            int(processing_params['third_pass_percentile'])
        )
        
        # Watershed segmentation
        watershed_params = {
            'watershed_min_size': processing_params.get('watershed_min_size', 50),
            'watershed_max_size': processing_params.get('watershed_max_size', 2000)
        }
        labels_ws, num_objects = post_processing_standalone(third_pass, watershed_params)
        
        # Create visualization overlay
        overlay_img = create_overlay_standalone(original_img, labels_ws)
        
        overlay_path = overlay_dir / f"{tile_name}_overlay.tif"
        tifffile.imwrite(str(overlay_path), overlay_img)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return results
        return {
            'tile_name': tile_name,
            'success': True,
            'cell_count': num_objects,
            'processing_time': processing_time
        }
        
    except Exception as e:
        return {
            'tile_name': tile_path.name,
            'success': False,
            'error': str(e),
            'cell_count': 0
        }

def natural_sort_key(s):
    """Create key for natural sorting of strings containing numbers."""
    def try_int(text):
        try:
            return int(text)
        except ValueError:
            return text
    return [try_int(c) for c in re.split('([0-9]+)', s)]

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Microglia Detection using stain separation')
    parser.add_argument('--data-dir', help='Base data directory')
    parser.add_argument('--output-dir', help='Base output directory')  
    parser.add_argument('--parameters-dir', help='Parameters directory')
    parser.add_argument('--density-map', action='store_true', help='Create density map (handled by separate script)')
    parser.add_argument('--overview', action='store_true', help='Create overview (handled by separate script)')
    
    args = parser.parse_args()
    
    # Determine paths based on whether called from GUI or standalone
    if args.data_dir and args.output_dir and args.parameters_dir:
        # Called from GUI - use provided paths
        data_dir = args.data_dir
        output_dir = args.output_dir
        parameters_dir = args.parameters_dir
        
        # Set up log directory - use Logs directory that is at the same level as Data, Results, Parameters
        base_dir = os.path.dirname(args.data_dir)  # Get parent directory of Data
        log_dir = os.path.join(base_dir, "Logs")
        os.makedirs(log_dir, exist_ok=True)
    else:
        print("ERROR: Either provide --data-dir, --output-dir, and --parameters-dir")
        sys.stdout.flush()
        return
    
    # Configure logging to file in the Logs directory
    log_file = os.path.join(log_dir, "microglia_detection.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    print("Starting Microglia Detection")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Parameters directory: {parameters_dir}")
    print(f"Log file: {log_file}")
    sys.stdout.flush()
    
    logger.info("Starting Microglia Detection")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Parameters directory: {parameters_dir}")
    
    try:
        # Create detector instance (with verbose output)
        detector = MicrogliaDetector(data_dir, output_dir, parameters_dir, verbose=True)
        
        # Check if input directory exists
        if not detector.input_tiles_dir.exists():
            print(f"ERROR: Input directory {detector.input_tiles_dir} does not exist!")
            sys.stdout.flush()
            logger.error(f"Input directory {detector.input_tiles_dir} does not exist!")
            return
        
        # Find all tile files and sort them by name
        tile_files = sorted(
            detector.input_tiles_dir.glob("*.tif"),
            key=lambda x: natural_sort_key(x.name)
        )
        
        if not tile_files:
            print(f"ERROR: No tile files found in {detector.input_tiles_dir}")
            sys.stdout.flush()
            logger.error(f"No tile files found in {detector.input_tiles_dir}")
            return
        
        total_tiles = len(tile_files)
        print(f"Found {total_tiles} tiles to process")
        print(f"Using {detector.processing_channel_name} channel (index {detector.processing_channel_index}) for processing")
        sys.stdout.flush()
        
        # Smart CPU core allocation
        available_cpus = multiprocessing.cpu_count()
        if available_cpus >= 8:
            max_processes = available_cpus - 1  # Leave one core free for GUI
        else:
            max_processes = max(1, available_cpus // 2)  # Use half on smaller machines
        
        print(f"Processing using {max_processes} CPU cores.")
        sys.stdout.flush()
        
        # Prepare shared configuration for multiprocessing (avoid repeated loading)
        shared_config = {
            'stain_matrix': detector.stain_matrix,
            'processing_params': detector.processing_params,
            'output_dir': output_dir,
            'processing_channel_index': detector.processing_channel_index
        }
        
        # Prepare arguments for each tile
        args_list = [(tile_path, shared_config) for tile_path in tile_files]
        
        # Use the 'spawn' method for better cross-platform compatibility
        ctx = multiprocessing.get_context('spawn')
        
        # Process tiles in parallel
        results = []
        with ctx.Pool(processes=max_processes) as pool:
            # Use a smaller chunksize to balance load better
            chunksize = max(1, total_tiles // (max_processes * 4))
            
            # Custom progress tracking
            processed_count = 0
            
            # Process files with explicit progress reporting
            for i, result in enumerate(pool.imap(process_single_tile, args_list, chunksize=chunksize)):
                processed_count += 1
                results.append(result)
                
                # Log progress every 5% or at least every 50 files
                if processed_count % max(1, min(50, total_tiles // 20)) == 0 or processed_count == total_tiles:
                    progress_pct = (processed_count / total_tiles) * 100
                    print(f"Progress: {processed_count}/{total_tiles} tiles ({progress_pct:.1f}%)")
                    sys.stdout.flush()
        
        # Collect results for CSV
        successful = [res for res in results if res['success']]
        failed = [res for res in results if not res['success']]
        
        # Create CSV with results
        results_data = []
        for res in results:
            if res['success']:
                results_data.append({
                    'tile_name': res['tile_name'], 
                    'cell_count': res['cell_count']
                })
            else:
                results_data.append({
                    'tile_name': res['tile_name'], 
                    'cell_count': 0,
                    'error': res.get('error', 'Unknown error')
                })
        
        # Create DataFrame
        results_df = pd.DataFrame(results_data)
        
        # Save to CSV
        csv_path = detector.results_dir / "microglia_cell_counts.csv"
        results_df.to_csv(csv_path, index=False)
        
        # Print summary
        print(f"\nMicroglia Detection Complete:")
        print(f"Successfully processed: {len(successful)}/{total_tiles} tiles")
        if failed:
            print(f"Failed: {len(failed)}/{total_tiles} tiles")
            # Print first few error messages for debugging
            for i, res in enumerate(failed[:3]):
                print(f"  Error {i+1}: {res.get('error', 'Unknown error')}")
        
        total_cells = sum(res['cell_count'] for res in successful)
        print(f"Total microglia cells detected: {total_cells}")
        print(f"Results saved to: {csv_path}")
        sys.stdout.flush()
        
        logger.info(f"Microglia detection completed successfully")
        logger.info(f"Total tiles: {total_tiles}, Successful: {len(successful)}, Failed: {len(failed)}")
        logger.info(f"Total microglia cells detected: {total_cells}")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.stdout.flush()
        logger.error(f"Error in microglia detection: {str(e)}")
        traceback.print_exc()

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
