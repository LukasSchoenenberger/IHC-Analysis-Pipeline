#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Myelin Parameter Estimation Script for IHC Pipeline
--------------------------------------------------
Interactive tool to adjust Wolf's adaptive threshold parameters for myelin detection.
Includes validation workflow with parameter refinement capability.

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
from scipy.linalg import pinv
from scipy.ndimage import uniform_filter
from matplotlib.widgets import Slider

def setup_logging(log_dir):
    """Set up logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "myelin_parameter_estimation.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("MyelinParameterEstimation")

class MyelinParameterEstimation:
    """Main class for myelin parameter estimation with validation workflow"""
    
    def __init__(self, data_dir, parameters_dir, output_dir):
        """Initialize the parameter estimation tool"""
        self.data_dir = Path(data_dir)
        self.parameters_dir = Path(parameters_dir)
        self.output_dir = Path(output_dir)
        
        # Input paths
        self.input_tiles_dir = self.data_dir / "Myelin-Detection-Test-Tiles"
        
        # Tile management for validation workflow
        self.all_tiles = []
        self.adjustment_tiles = []  # 5 tiles for parameter adjustment
        self.validation_tiles = []  # Up to 25 tiles for validation
        self.previously_used_tiles = set()  # Track used adjustment tiles
        
        # Wolf's Method Parameters
        self.current_alpha = 0.5  # Default alpha value
        self.current_k = 0.0  # Default k value
        self.window_size = 15  # Size of neighborhood for local statistics
        self.current_min_component_size = 20  # Default minimum component size
        
        # Background handling
        self.background_threshold = 253
        self.background_rgb = np.array([255, 255, 255])
        
        # Filtering range for stain concentrations
        self.min_value = -3
        self.max_value = 3
        
        # Data storage
        self.global_stats = None
        self.stain_matrix = None
        self.stain_names = None
        
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
                    
                    # Skip lines that are just headers (don't contain vector data)
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
                                if len(vector) == 3:  # Ensure it's a valid RGB vector
                                    stain_names.append(name)
                                    stain_matrix.append(vector)
                            except ValueError:
                                continue  # Skip invalid lines
                    else:
                        # Format: "Name: R G B" (space-separated)
                        parts = line.split(':')
                        if len(parts) == 2:
                            name = parts[0].strip()
                            values_str = parts[1].strip()
                            try:
                                values = values_str.split()
                                if len(values) == 3:  # Ensure we have exactly 3 values
                                    vector = np.array([float(x) for x in values])
                                    stain_names.append(name)
                                    stain_matrix.append(vector)
                            except ValueError:
                                continue  # Skip invalid lines
            
            if not stain_matrix:
                raise ValueError("No valid stain vectors found in file")
                
            self.stain_matrix = np.array(stain_matrix)
            self.stain_names = stain_names
            
            print(f"Loaded stain matrix: {self.stain_matrix.shape}")
            print(f"Stain names: {', '.join(self.stain_names)}")
            sys.stdout.flush()
            self.logger.info(f"Loaded stain matrix: {self.stain_matrix.shape}")
            self.logger.info(f"Stain names: {', '.join(self.stain_names)}")
            
        except Exception as e:
            print(f"ERROR: Error loading stain matrix: {e}")
            sys.stdout.flush()
            self.logger.error(f"Error loading stain matrix: {e}")
            sys.exit(1)
    
    def load_global_stats(self):
        """Load global myelin statistics if available"""
        global_stats_path = self.parameters_dir / "myelin_global_stats.json"
        
        if global_stats_path.exists():
            try:
                with open(global_stats_path, 'r') as f:
                    self.global_stats = json.load(f)
                print("Loaded global myelin statistics")
                sys.stdout.flush()
            except Exception as e:
                print(f"Warning: Error loading global stats: {e}")
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
        
        # Update previously used set
        self.previously_used_tiles.update(self.adjustment_tiles)
        
        print(f"Selected {len(self.adjustment_tiles)} new tiles for parameter adjustment")
        sys.stdout.flush()
    
    def rgb_to_od(self, img):
        """Convert RGB image to optical density"""
        h, w, c = img.shape
        img_flat = img.reshape(h*w, c).T
        img_flat = img_flat.astype(float)
        
        eps = 1e-6
        img_flat = np.maximum(img_flat, eps)
        
        od = -np.log10(img_flat / self.background_rgb[:, np.newaxis])
        return od
    
    def identify_background(self, img):
        """Identify background pixels"""
        h, w, _ = img.shape
        img_flat = img.reshape(h*w, 3)
        bg_mask = np.all(img_flat >= self.background_threshold, axis=1)
        return bg_mask
    
    def separate_stains(self, img):
        """Separate stains using the stain matrix"""
        od = self.rgb_to_od(img)
        bg_mask = self.identify_background(img)
        
        # Calculate stain concentrations
        stain_matrix_T = self.stain_matrix.T
        concentrations = np.dot(pinv(stain_matrix_T), od)
        
        # Filter concentrations
        filtered_concentrations = np.clip(concentrations, self.min_value, self.max_value)
        
        # Zero out background
        filtered_concentrations[:, bg_mask] = 0
        
        return filtered_concentrations, bg_mask
    
    def extract_myelin_channel(self, img):
        """Extract myelin channel from image"""
        h, w, _ = img.shape
        concentrations, bg_mask = self.separate_stains(img)
        
        # Find myelin stain index
        myelin_idx = 1  # Default to second stain
        for i, name in enumerate(self.stain_names):
            if 'myelin' in name.lower() or 'mbp' in name.lower() or 'dab' in name.lower():
                myelin_idx = i
                break
        
        # Extract myelin concentration
        myelin_concentration = concentrations[myelin_idx, :]
        
        # Extend background mask for negative values
        negative_mask = myelin_concentration < 0
        extended_bg_mask = bg_mask | negative_mask
        myelin_concentration[extended_bg_mask] = 0
        
        return myelin_concentration, extended_bg_mask, h, w
    
    def create_myelin_heatmap(self, myelin_concentration, bg_mask, h, w):
        """Create heatmap visualization of myelin intensity"""
        myelin_reshaped = myelin_concentration.reshape(h, w)
        bg_mask_reshaped = bg_mask.reshape(h, w)
        
        heatmap = np.zeros((h, w, 3), dtype=np.uint8)
        non_bg = ~bg_mask_reshaped
        
        if np.any(non_bg):
            # Use global stats if available, otherwise per-tile normalization
            if self.global_stats and 'global_myelin_min' in self.global_stats:
                min_val = self.global_stats['global_myelin_min']
                max_val = self.global_stats['global_myelin_max']
            else:
                min_val = np.min(myelin_reshaped[non_bg]) if np.any(non_bg) else -3
                max_val = np.max(myelin_reshaped[non_bg]) if np.any(non_bg) else 3
            
            # Normalize to 0-1
            if max_val > min_val:
                normalized = np.clip((myelin_reshaped - min_val) / (max_val - min_val), 0, 1)
            else:
                normalized = np.zeros_like(myelin_reshaped)
            
            # Hot colormap
            heatmap[:,:,0] = np.minimum(normalized * 510, 255).astype(np.uint8)
            heatmap[:,:,1] = np.maximum(np.minimum((normalized - 0.5) * 510, 255), 0).astype(np.uint8)
            heatmap[:,:,2] = np.maximum(np.minimum((normalized - 0.75) * 1020, 255), 0).astype(np.uint8)
            
            # Set background to white
            heatmap[bg_mask_reshaped] = [255, 255, 255]
        
        return heatmap
    
    def filter_small_components(self, binary_mask, min_size):
        """Filter out small connected components from binary mask"""
        if min_size <= 0:
            return binary_mask
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask.astype(np.uint8), connectivity=8
        )
        
        # Create filtered mask
        filtered_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        
        # Keep components larger than min_size (skip background label 0)
        for label in range(1, num_labels):
            component_size = stats[label, cv2.CC_STAT_AREA]
            if component_size >= min_size:
                filtered_mask[labels == label] = 1
        
        return filtered_mask
    
    def process_single_tile(self, tile_path):
        """Process a single tile with current parameters"""
        try:
            # Load image
            original_img = tifffile.imread(str(tile_path))
            
            # Ensure proper format
            if len(original_img.shape) == 2:
                original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
            elif len(original_img.shape) == 3 and original_img.shape[2] > 3:
                original_img = original_img[:, :, :3]
            
            # Extract myelin channel
            myelin_concentration, bg_mask, h, w = self.extract_myelin_channel(original_img)
            
            # Create heatmap
            myelin_heatmap = self.create_myelin_heatmap(myelin_concentration, bg_mask, h, w)
            
            # Apply Wolf's threshold
            thresholded = self.apply_wolf_threshold(myelin_concentration, bg_mask, h, w)
            
            # Create overlay
            overlay = self.create_overlay(original_img, thresholded)
            
            # Calculate percentage of positive pixels
            bg_mask_reshaped = bg_mask.reshape(h, w)
            non_bg_count = np.sum(~bg_mask_reshaped)
            if non_bg_count > 0:
                percent_positive = np.sum(thresholded > 0) / non_bg_count * 100
            else:
                percent_positive = 0.0
            
            return {
                'original_img': original_img,
                'myelin_heatmap': myelin_heatmap,
                'myelin_concentration': myelin_concentration,
                'background_mask': bg_mask,
                'thresholded': thresholded,
                'overlay': overlay,
                'percent_positive': percent_positive,
                'dimensions': (h, w)
            }
            
        except Exception as e:
            print(f"Error processing {tile_path}: {e}")
            sys.stdout.flush()
            return None
    
    def compute_local_statistics(self, image, window_size, bg_mask):
        """Compute local mean and standard deviation"""
        img_masked = image.copy()
        img_masked[bg_mask] = 0
        
        sum_img = uniform_filter(img_masked, size=window_size)
        mask = (~bg_mask).astype(float)
        count = uniform_filter(mask, size=window_size)
        count = np.maximum(count, 1e-6)
        
        mean = sum_img / count
        
        sum_sq_img = uniform_filter(img_masked**2, size=window_size)
        mean_sq = sum_sq_img / count
        variance = np.maximum(mean_sq - mean**2, 0.0)
        std = np.sqrt(variance)
        
        mean[bg_mask] = 0
        std[bg_mask] = 0
        
        return mean, std
    
    def apply_wolf_threshold(self, myelin_concentration, bg_mask, h, w):
        """Apply Wolf's adaptive threshold method with component filtering"""
        myelin_reshaped = myelin_concentration.reshape(h, w)
        bg_mask_reshaped = bg_mask.reshape(h, w)
        binary = np.zeros((h, w), dtype=np.uint8)
        
        # Get global standard deviation
        R = 1.0
        if self.global_stats and 'global_myelin_std' in self.global_stats:
            R = self.global_stats['global_myelin_std']
        else:
            non_bg = ~bg_mask_reshaped
            if np.any(non_bg):
                R = np.std(myelin_reshaped[non_bg])
        
        R = max(R, 0.01)
        
        non_bg = ~bg_mask_reshaped
        if np.any(non_bg):
            local_mean, local_std = self.compute_local_statistics(
                myelin_reshaped, self.window_size, bg_mask_reshaped
            )
            
            threshold = (1-self.current_alpha) * local_mean + self.current_alpha * (local_mean + self.current_k * (local_std/R - 1))
            binary[non_bg & (myelin_reshaped > threshold)] = 1
            binary[bg_mask_reshaped] = 0
            
            # Filter small components
            binary = self.filter_small_components(binary, self.current_min_component_size)
        
        return binary
    
    def create_overlay(self, original_img, binary_mask, alpha=0.3):
        """Create overlay visualization"""
        overlay = original_img.copy()
        overlay[binary_mask > 0, 0] = 255
        overlay[binary_mask > 0, 1] = 0
        overlay[binary_mask > 0, 2] = 0
        
        blended = cv2.addWeighted(original_img, 1-alpha, overlay, alpha, 0)
        return blended
    
    def load_sample_tiles(self):
        """Load and process sample tiles for parameter estimation"""
        print("Starting Myelin Parameter Estimation workflow...")
        sys.stdout.flush()
        
        # Load required files
        self.load_stain_matrix()
        self.load_global_stats()
        
        # Find and organize tiles
        if not self.find_tiles():
            sys.exit(1)
        
        print("Launching interactive parameter adjustment interface...")
        sys.stdout.flush()
        
        return True
    
    def save_parameters(self):
        """Save final parameters to JSON file"""
        try:
            self.parameters_dir.mkdir(parents=True, exist_ok=True)
            
            parameters = {
                "alpha": float(self.current_alpha),
                "k": float(self.current_k),
                "min_component_size": int(self.current_min_component_size),
                "window_size": int(self.window_size),
                "method": "wolf_threshold",
                "description": "Wolf's adaptive threshold parameters for myelin detection with component filtering"
            }
            
            output_file = self.parameters_dir / "myelin_detection_parameters.json"
            
            with open(output_file, 'w') as f:
                json.dump(parameters, f, indent=4)
            
            print(f"Parameters saved to: {output_file}")
            print(f"Final parameters: alpha={self.current_alpha:.3f}, k={self.current_k:.3f}, min_component_size={self.current_min_component_size}")
            sys.stdout.flush()
            
            self.logger.info(f"Parameters saved: alpha={self.current_alpha}, k={self.current_k}, min_component_size={self.current_min_component_size}")
            
        except Exception as e:
            print(f"Error saving parameters: {e}")
            sys.stdout.flush()
    
    def show_validation_interface(self):
        """Show validation interface with keyboard controls"""
        print("Processing validation tiles...")
        sys.stdout.flush()
        
        # Process validation tiles
        validation_results = []
        for tile_path in self.validation_tiles:
            result = self.process_single_tile(tile_path)
            if result is not None:
                validation_results.append((tile_path.name, result))
        
        if not validation_results:
            print("No validation results to display")
            sys.stdout.flush()
            return
        
        # Calculate grid size
        grid_size = min(5, max(1, int(np.ceil(np.sqrt(len(validation_results))))))
        
        # Create figure
        fig = plt.figure(figsize=(15, 16))
        fig.suptitle("Validation Results with Current Parameters", fontsize=16)
        
        # Instructions
        instruction_text = "KEYBOARD CONTROLS: Press 'r' to refine parameters, 'm' to save and exit"
        plt.figtext(0.5, 0.95, instruction_text, 
                   ha="center", fontsize=14,
                   bbox=dict(boxstyle="round,pad=0.5", fc="yellow", ec="orange", alpha=0.9))
        
        # Create subplot grid
        gs = gridspec.GridSpec(grid_size, grid_size, figure=fig)
        
        for i in range(grid_size * grid_size):
            if i < len(validation_results):
                row = i // grid_size
                col = i % grid_size
                ax = fig.add_subplot(gs[row, col])
                
                tile_name, result = validation_results[i]
                ax.imshow(result['overlay'])
                ax.set_title(f"{tile_name}\n{result['percent_positive']:.1f}% positive")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                row = i // grid_size
                col = i % grid_size
                ax = fig.add_subplot(gs[row, col])
                ax.axis('off')
        
        # Keyboard event handler
        def key_press_handler(event):
            if event.key == 'r':
                print("Refining parameters with new tiles...")
                sys.stdout.flush()
                plt.close(fig)
                self.select_new_adjustment_tiles()
                self.run_interactive_interface()
            elif event.key == 'm':
                print("Saving parameters and exiting...")
                sys.stdout.flush()
                plt.close(fig)
                self.save_parameters()
                print("All Myelin Parameter Estimation steps complete!")
                sys.stdout.flush()
        
        fig.canvas.mpl_connect('key_press_event', key_press_handler)
        
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        
        print("\nValidation interface ready:")
        print("  Press 'r' to refine parameters with new tiles")
        print("  Press 'm' to save parameters and exit")
        sys.stdout.flush()
        
        plt.show()
    
    def run_interactive_interface(self):
        """Run the interactive parameter adjustment interface"""
        # Process adjustment tiles
        tile_data = []
        for tile_path in self.adjustment_tiles:
            result = self.process_single_tile(tile_path)
            if result is not None:
                result['file_path'] = tile_path
                tile_data.append(result)
        
        # Fill remaining slots if less than 5
        while len(tile_data) < 5:
            tile_data.append({
                'file_path': Path(f"Empty_{len(tile_data) + 1}"),
                'original_img': np.zeros((100, 100, 3), dtype=np.uint8),
                'myelin_heatmap': np.zeros((100, 100, 3), dtype=np.uint8),
                'myelin_concentration': np.zeros(100*100, dtype=float),
                'background_mask': np.ones(100*100, dtype=bool),
                'thresholded': np.zeros((100, 100), dtype=np.uint8),
                'overlay': np.zeros((100, 100, 3), dtype=np.uint8),
                'percent_positive': 0.0,
                'dimensions': (100, 100)
            })
        
        # Launch interactive interface
        interactive_tool = InteractiveInterface(self, tile_data)
        interactive_tool.run()
    
    def run(self):
        """Run the parameter estimation workflow with validation"""
        # Load sample tiles and organize them
        self.load_sample_tiles()
        
        # Launch interactive interface
        self.run_interactive_interface()

class InteractiveInterface:
    """Interactive matplotlib interface for parameter adjustment"""
    
    def __init__(self, estimation_tool, tile_data):
        """Initialize the interface"""
        self.tool = estimation_tool
        self.tile_data = tile_data
        self.thresholded_results = []
        
    def setup_figure(self):
        """Set up the matplotlib figure"""
        self.fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(5, 5)
        
        # Create subplot arrays
        self.ax_originals = []
        self.ax_myelin = []
        self.ax_overlays = []
        
        # First row: original images
        for i in range(5):
            ax = self.fig.add_subplot(gs[0, i])
            ax.set_title(f"Original {i+1}")
            ax.set_xticks([])
            ax.set_yticks([])
            self.ax_originals.append(ax)
        
        # Second row: myelin intensity
        for i in range(5):
            ax = self.fig.add_subplot(gs[1, i])
            ax.set_title(f"Myelin Intensity {i+1}")
            ax.set_xticks([])
            ax.set_yticks([])
            self.ax_myelin.append(ax)
        
        # Third row: overlays
        for i in range(5):
            ax = self.fig.add_subplot(gs[2, i])
            ax.set_title(f"Overlay {i+1}")
            ax.set_xticks([])
            ax.set_yticks([])
            self.ax_overlays.append(ax)
        
        # Fourth and fifth rows: sliders
        self.ax_slider_alpha = self.fig.add_subplot(gs[3, 0:2])
        self.ax_slider_k = self.fig.add_subplot(gs[3, 2:4])
        self.ax_slider_min_size = self.fig.add_subplot(gs[4, 0:2])
        
        self.slider_alpha = Slider(
            self.ax_slider_alpha, 'Alpha (α)', 
            0.0, 1.0, 
            valinit=self.tool.current_alpha,
            valstep=0.05
        )
        
        self.slider_k = Slider(
            self.ax_slider_k, 'K parameter', 
            -0.5, 1.5, 
            valinit=self.tool.current_k,
            valstep=0.05
        )
        
        self.slider_min_size = Slider(
            self.ax_slider_min_size, 'Min Component Size', 
            0, 60, 
            valinit=self.tool.current_min_component_size,
            valstep=5
        )
        
        # Instructions and controls - smaller box in bottom right
        self.ax_info = self.fig.add_subplot(gs[4, 3:5])
        self.ax_info.axis('off')
        self.ax_info.text(
            0.05, 0.95, 
            "Parameter Controls:\n"
            "α: Local variance contribution (0.0-1.0)\n"
            "K: Threshold sensitivity (-0.5-1.5)\n"
            "Min Size: Filter components <N pixels (0-60)\n\n"
            "Close window to proceed to validation", 
            ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.8)
        )
        
        # Connect events
        self.slider_alpha.on_changed(self.update)
        self.slider_k.on_changed(self.update)
        self.slider_min_size.on_changed(self.update)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
        self.fig.suptitle("Wolf's Threshold Parameters for Myelin Detection", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    def update(self, val):
        """Update visualization when sliders change"""
        self.tool.current_alpha = self.slider_alpha.val
        self.tool.current_k = self.slider_k.val
        self.tool.current_min_component_size = int(self.slider_min_size.val)
        
        self.thresholded_results = []
        
        for i, tile_info in enumerate(self.tile_data):
            original = tile_info['original_img']
            myelin_heatmap = tile_info['myelin_heatmap']
            myelin_concentration = tile_info['myelin_concentration']
            bg_mask = tile_info['background_mask']
            h, w = tile_info['dimensions']
            tile_name = tile_info['file_path'].name
            
            # Apply thresholding
            thresholded = self.tool.apply_wolf_threshold(
                myelin_concentration, bg_mask, h, w
            )
            
            self.thresholded_results.append(thresholded)
            overlay = self.tool.create_overlay(original, thresholded)
            
            # Calculate percentage
            bg_mask_reshaped = bg_mask.reshape(h, w)
            non_bg_count = np.sum(~bg_mask_reshaped)
            if non_bg_count > 0:
                percent_positive = np.sum(thresholded > 0) / non_bg_count * 100
            else:
                percent_positive = 0.0
            
            # Update displays
            self.ax_originals[i].clear()
            self.ax_originals[i].imshow(original)
            self.ax_originals[i].set_title(f"Original: {tile_name}")
            self.ax_originals[i].set_xticks([])
            self.ax_originals[i].set_yticks([])
            
            self.ax_myelin[i].clear()
            self.ax_myelin[i].imshow(myelin_heatmap)
            self.ax_myelin[i].set_title(f"Myelin Intensity {i+1}")
            self.ax_myelin[i].set_xticks([])
            self.ax_myelin[i].set_yticks([])
            
            self.ax_overlays[i].clear()
            self.ax_overlays[i].imshow(overlay)
            self.ax_overlays[i].set_title(f"Overlay {i+1} - {percent_positive:.1f}%\n"
                                         f"α={self.tool.current_alpha:.2f}, k={self.tool.current_k:.2f}, "
                                         f"min_size={self.tool.current_min_component_size}")
            self.ax_overlays[i].set_xticks([])
            self.ax_overlays[i].set_yticks([])
        
        self.fig.canvas.draw()
    
    def on_close(self, event):
        """Handle window close - proceed to validation"""
        print("Parameter adjustment completed. Proceeding to validation...")
        sys.stdout.flush()
        # Show validation interface
        self.tool.show_validation_interface()
    
    def run(self):
        """Run the interface"""
        self.setup_figure()
        self.update(None)
        plt.show()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Interactive myelin parameter estimation with validation")
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
        estimation_tool = MyelinParameterEstimation(data_dir, parameters_dir, output_dir)
        estimation_tool.run()
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.stdout.flush()
        raise

if __name__ == "__main__":
    main()
