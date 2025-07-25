#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tile Selection Tool for IHC Pipeline
-----------------------------------
Tool for selecting tiles for parameter estimation in cell, myelin, and microglia detection.
This tool uses multiprocessing for overview creation and matplotlib for interactive selection.

Part of the IHC Pipeline GUI application.
"""

import os
import re
import numpy as np
import tifffile
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.gridspec as gridspec
import shutil
import multiprocessing
from functools import partial
import argparse
import logging
import sys

# Configure logging
logger = logging.getLogger("Tile-Selection-Tool")

class TileSelectionTool:
    """
    Tool for selecting tiles for various parameter adjustments in a single workflow.
    This tool centralizes the tile selection process for all parameter adjustment steps.
    """
    def __init__(self, base_dir, scale_factor=0.1):
        self.base_dir = Path(base_dir)
        self.scale_factor = scale_factor
        
        # Input directory - the long path with all preprocessing steps
        self.input_tiles_dir = self.base_dir / "Tiles-Medium-L-Channel-Normalized-BG-Removed-Illumination-Corrected-Stain-Normalized-Small-Tiles"
        
        # Define output directories in Data folder
        self.output_dirs = {
            "cell_detection": self.base_dir / "Cell-Detection-Test-Tiles",
            "myelin_detection": self.base_dir / "Myelin-Detection-Test-Tiles", 
            "microglia_detection": self.base_dir / "Microglia-Detection-Test-Tiles"
        }
        
        # Create output directories
        for dir_path in self.output_dirs.values():
            dir_path.mkdir(exist_ok=True, parents=True)
            print(f"Created output directory: {dir_path}")
            sys.stdout.flush()
        
        # Store all tile paths and their corresponding row/column info
        self.all_tiles = []
        self.tile_positions = {}  # Maps (row, col) to tile_path
        self.selected_tiles = set()  # Keep track of selected tiles
        
        # Initialize variables for the overview image
        self.overview_img = None
        self.grid_overlay = None
        self.max_row = 0
        self.max_col = 0
        self.tile_width = 0
        self.tile_height = 0
        
        # For selection tool
        self.selection_mode = True  # True for select, False for zoom/pan
        
        self.selection_steps = [
            {"name": "cell_detection", "title": "Cell Detection Parameter Estimation", "folder": "cell_detection"},
            {"name": "myelin_detection", "title": "Myelin Detection Parameter Estimation", "folder": "myelin_detection"},
            {"name": "microglia_detection", "title": "Microglia Detection Parameter Estimation", "folder": "microglia_detection"}
        ]
        self.current_step_index = 0
        self.current_step = self.selection_steps[self.current_step_index]
    
    def find_tiles(self):
        """Find all tile images and extract their row/column positions"""
        print("Scanning for tiles...")
        sys.stdout.flush()
        logger.info("Scanning for tiles...")
        
        tile_pattern = re.compile(r'tile_r(\d+)_c(\d+)\.tif')
        
        if not self.input_tiles_dir.exists():
            error_msg = f"Input tiles directory not found: {self.input_tiles_dir}"
            print(f"ERROR: {error_msg}")
            sys.stdout.flush()
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        all_files = list(self.input_tiles_dir.glob("*.tif"))
        print(f"Found {len(all_files)} .tif files in the directory")
        sys.stdout.flush()
        
        for tile_path in all_files:
            match = tile_pattern.match(tile_path.name)
            if match:
                row, col = int(match.group(1)), int(match.group(2))
                self.all_tiles.append(tile_path)
                self.tile_positions[(row, col)] = tile_path
                self.max_row = max(self.max_row, row)
                self.max_col = max(self.max_col, col)
        
        print(f"Found {len(self.all_tiles)} valid tiles. Grid size: {self.max_row+1}×{self.max_col+1}")
        sys.stdout.flush()
        logger.info(f"Found {len(self.all_tiles)} valid tiles. Grid size: {self.max_row+1}×{self.max_col+1}")
        
        if not self.all_tiles:
            error_msg = f"No valid tiles found in {self.input_tiles_dir}"
            print(f"ERROR: {error_msg}")
            sys.stdout.flush()
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _process_tile_for_overview(self, args):
        """Helper function to process a single tile for the overview image using multiprocessing"""
        try:
            row, col, tile_path, tile_height, tile_width = args
            
            # Read the image
            img = tifffile.imread(str(tile_path))
            
            # Convert grayscale to RGB if needed
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            # Use faster OpenCV resize with INTER_NEAREST for speed
            resized_img = cv2.resize(img, (tile_width, tile_height), 
                                   interpolation=cv2.INTER_NEAREST)
            
            # Calculate position
            y_start = row * tile_height
            x_start = col * tile_width
            
            # Return the processed tile with its position
            return (y_start, x_start, resized_img)
        except Exception as e:
            return f"Error processing tile {tile_path}: {str(e)}"
    
    def create_overview_image(self, parameters_dir):
        """Create a downsampled overview of all tiles arranged in their grid positions using multiprocessing"""
        # Check if a cached overview exists in Parameters directory
        cache_file = Path(parameters_dir) / "tile_selection_cached_overview.npz"
        
        # Try to load from cache if it exists
        if cache_file.exists():
            try:
                print("Loading overview from cache...")
                sys.stdout.flush()
                logger.info("Loading overview from cache...")
                
                cached_data = np.load(cache_file, allow_pickle=True)
                self.overview_img = cached_data['overview']
                self.tile_height = int(cached_data['tile_height'])
                self.tile_width = int(cached_data['tile_width'])
                
                # Create grid overlay
                self.create_grid_overlay()
                print("Overview image loaded from cache successfully")
                sys.stdout.flush()
                logger.info("Overview image loaded from cache successfully")
                return
            except Exception as e:
                print(f"Error loading cached overview: {e}")
                print("Proceeding with full overview creation...")
                sys.stdout.flush()
                logger.warning(f"Error loading cached overview: {e}")
        
        print("Creating overview image using multiprocessing...")
        sys.stdout.flush()
        logger.info("Creating overview image using multiprocessing...")
        
        # Get dimensions from the first tile
        sample_tile = tifffile.imread(str(self.all_tiles[0]))
        tile_h, tile_w = sample_tile.shape[:2]
        
        # Calculate dimensions for the downsampled tiles
        self.tile_height = int(tile_h * self.scale_factor)
        self.tile_width = int(tile_w * self.scale_factor)
        
        # Create an empty canvas for the overview image
        canvas_height = (self.max_row + 1) * self.tile_height
        canvas_width = (self.max_col + 1) * self.tile_width
        
        # Initialize with black background
        self.overview_img = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Smart CPU core allocation following guidelines
        available_cpus = multiprocessing.cpu_count()
        if available_cpus >= 8:
            max_processes = available_cpus - 1  # Leave one core free for GUI
        else:
            max_processes = max(1, available_cpus // 2)  # Use half on smaller machines
        
        print(f"Processing using {max_processes} CPU cores.")
        sys.stdout.flush()
        logger.info(f"Processing using {max_processes} CPU cores.")
        
        # Prepare arguments for multiprocessing
        total_tiles = len(self.tile_positions)
        process_args = []
        for (row, col), tile_path in self.tile_positions.items():
            process_args.append((row, col, tile_path, self.tile_height, self.tile_width))
        
        # Use spawn method for better cross-platform compatibility
        ctx = multiprocessing.get_context('spawn')
        
        # Process tiles in parallel using multiprocessing.Pool
        with ctx.Pool(processes=max_processes) as pool:
            # Calculate optimal chunk size for balanced load
            chunksize = max(1, total_tiles // (max_processes * 4))
            
            # Process with explicit progress reporting
            processed_count = 0
            errors = []
            
            # Use imap for progress tracking
            for result in pool.imap(self._process_tile_for_overview, process_args, chunksize=chunksize):
                processed_count += 1
                
                # Log progress at regular intervals
                if processed_count % max(1, min(50, total_tiles // 20)) == 0 or processed_count == total_tiles:
                    progress_pct = (processed_count / total_tiles) * 100
                    print(f"Progress: {processed_count}/{total_tiles} tiles ({progress_pct:.1f}%)")
                    sys.stdout.flush()
                
                # Handle results
                if isinstance(result, str):  # Error message
                    errors.append(result)
                    continue
                    
                # Unpack the result
                y_start, x_start, resized_img = result
                y_end = y_start + self.tile_height
                x_end = x_start + self.tile_width
                
                # Place on canvas
                self.overview_img[y_start:y_end, x_start:x_end] = resized_img
        
        # Report any errors
        if errors:
            print(f"{len(errors)} errors occurred during processing:")
            for error in errors[:10]:  # Show first 10 errors
                print(f"  {error}")
            sys.stdout.flush()
            logger.warning(f"{len(errors)} errors occurred during processing")
        
        # Create grid overlay
        self.create_grid_overlay()
        print("Overview image created successfully")
        sys.stdout.flush()
        logger.info("Overview image created successfully")
        
        # Save to cache for future use
        try:
            print("Saving overview to cache...")
            sys.stdout.flush()
            os.makedirs(parameters_dir, exist_ok=True)
            np.savez(cache_file, 
                     overview=self.overview_img, 
                     tile_height=self.tile_height,
                     tile_width=self.tile_width)
            print("Overview cached successfully")
            sys.stdout.flush()
            logger.info("Overview cached successfully")
        except Exception as e:
            print(f"Error saving overview cache: {e}")
            print("Continuing without caching...")
            sys.stdout.flush()
            logger.warning(f"Error saving overview cache: {e}")
    
    def create_grid_overlay(self):
        """Prepare for grid display using matplotlib's grid capabilities"""
        # Store the grid line positions for later use in matplotlib
        self.grid_lines = {
            'horizontal': [row * self.tile_height for row in range(self.max_row + 2) 
                          if row * self.tile_height < self.overview_img.shape[0]],
            'vertical': [col * self.tile_width for col in range(self.max_col + 2)
                        if col * self.tile_width < self.overview_img.shape[1]]
        }
    
    def display_selection_interface(self):
        """Display the overview image and allow users to select tiles for the current step"""
        print("Launching interactive tile selection interface...")
        sys.stdout.flush()
        logger.info("Launching interactive tile selection interface...")
        
        # Reset selection for the new step
        self.selected_tiles = set()
        
        # Get current step info
        step = self.selection_steps[self.current_step_index]
        title = f"Step {self.current_step_index + 1}/{len(self.selection_steps)}: Select tiles for {step['title']}"
        
        # Create figure with proper title
        fig = plt.figure(figsize=(12, 10))
        plt.title(title)
        
        # Display the base image
        plt.imshow(self.overview_img)
        
        # Draw grid lines directly on the plot
        for y in self.grid_lines['horizontal']:
            plt.axhline(y=y, color='white', linewidth=0.8, alpha=0.8)
        
        for x in self.grid_lines['vertical']:
            plt.axvline(x=x, color='white', linewidth=0.8, alpha=0.8)
        
        # Connect event handlers
        cid_click = fig.canvas.mpl_connect('button_press_event', self._on_click)
        cid_key = fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # Add a counter for selected tiles
        self.selected_counter = plt.annotate(
            f"Selected: 0 tiles | Mode: Selection", xy=(0.01, 0.01), xycoords='figure fraction',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8)
        )
        
        # Add mode instructions
        plt.figtext(0.5, 0.01, 
                   "Press 'm' to toggle between Selection/Navigation modes | Press Enter when finished",
                   ha="center", 
                   bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="orange", alpha=0.8))
        
        # Create rectangle patches for selected tiles
        self.selection_patches = []
        
        # Start in selection mode
        self.selection_mode = True
        
        plt.tight_layout()
        plt.show()
    
    def _on_key_press(self, event):
        """Handle key presses to finalize selection or change mode"""
        if event.key == 'enter':
            plt.close()
            step_title = self.current_step['title']
            print(f"Selection complete. {len(self.selected_tiles)} tiles selected for {step_title}.")
            sys.stdout.flush()
            logger.info(f"Selection complete. {len(self.selected_tiles)} tiles selected for {step_title}.")
            
            # Copy the selected tiles to the appropriate directory
            self._copy_selected_tiles()
            
            # Move to the next step or finish
            self._next_step()
        elif event.key == 'm':  # Toggle mode
            # Toggle selection mode
            self.selection_mode = not self.selection_mode
            mode_name = "Selection" if self.selection_mode else "Navigation"
            self.selected_counter.set_text(f"Selected: {len(self.selected_tiles)} tiles | Mode: {mode_name}")
            plt.draw()
            print(f"Switched to {mode_name} mode")
            sys.stdout.flush()
    
    def _on_click(self, event):
        """Handle mouse clicks for tile selection"""
        if event.xdata is None or event.ydata is None:
            return
        
        # Only process clicks in selection mode
        if not self.selection_mode:
            return
        
        # Calculate which tile was clicked
        col = int(event.xdata // self.tile_width)
        row = int(event.ydata // self.tile_height)
        
        # Check if the tile exists
        if (row, col) not in self.tile_positions:
            print(f"No tile at position r{row}_c{col}")
            sys.stdout.flush()
            return
        
        # Toggle selection
        if (row, col) in self.selected_tiles:
            self.selected_tiles.remove((row, col))
            
            # Remove the highlight rectangle
            for patch in self.selection_patches:
                if patch.get_xy() == (col * self.tile_width, row * self.tile_height):
                    patch.remove()
                    self.selection_patches.remove(patch)
                    break
        else:
            self.selected_tiles.add((row, col))
            
            # Add a highlight rectangle
            rect = plt.Rectangle(
                (col * self.tile_width, row * self.tile_height),
                self.tile_width, self.tile_height,
                edgecolor='lime', facecolor='none', linewidth=2
            )
            plt.gca().add_patch(rect)
            self.selection_patches.append(rect)
        
        # Update the counter
        self.selected_counter.set_text(f"Selected: {len(self.selected_tiles)} tiles | Mode: Selection")
        plt.draw()
        print(f"Tile r{row}_c{col} {'deselected' if (row, col) not in self.selected_tiles else 'selected'}")
        sys.stdout.flush()
    
    def _copy_selected_tiles(self):
        """Copy selected tiles to the appropriate directory for the current step"""
        current_folder = self.current_step["folder"]
        dest_dir = self.output_dirs[current_folder]
        
        # Clear the directory
        for file in dest_dir.glob("*.tif"):
            file.unlink()
        
        # Copy selected files
        copied_count = 0
        for row, col in self.selected_tiles:
            src_path = self.tile_positions[(row, col)]
            dest_path = dest_dir / src_path.name
            try:
                shutil.copy(src_path, dest_path)
                copied_count += 1
            except Exception as e:
                print(f"Error copying {src_path.name}: {str(e)}")
                sys.stdout.flush()
                logger.error(f"Error copying {src_path.name}: {str(e)}")
        
        print(f"Copied {copied_count} tiles to {dest_dir}")
        sys.stdout.flush()
        logger.info(f"Copied {copied_count} tiles to {dest_dir}")
    
    def _next_step(self):
        """Move to the next selection step or finish if all steps are complete"""
        self.current_step_index += 1
        
        # Check if we've completed all steps
        if self.current_step_index >= len(self.selection_steps):
            print("All tile selection steps complete!")
            sys.stdout.flush()
            logger.info("All tile selection steps complete!")
            # Show final completion message
            self._show_completion_dialog()
            return
        
        # Update the current step and show the selection interface
        self.current_step = self.selection_steps[self.current_step_index]
        print(f"\nMoving to step {self.current_step_index + 1}: {self.current_step['title']}")
        sys.stdout.flush()
        logger.info(f"Moving to step {self.current_step_index + 1}: {self.current_step['title']}")
        self.display_selection_interface()
    
    def _show_completion_dialog(self):
        """Show a completion dialog with summary information"""
        fig = plt.figure(figsize=(8, 6))
        plt.axis('off')  # Hide axes
        
        # Add completion message
        plt.figtext(0.5, 0.8, "Tile Selection Complete!", 
                   ha="center", fontsize=16, fontweight='bold')
        
        # Show where tiles were saved
        plt.figtext(0.5, 0.6, "Selected tiles have been saved to:",
                   ha="center", fontsize=12)
        
        # List directories
        y_pos = 0.5
        for step in self.selection_steps:
            folder_name = self.output_dirs[step['folder']].name
            plt.figtext(0.5, y_pos, f"• {step['title']}: {folder_name}",
                       ha="center", fontsize=10)
            y_pos -= 0.08
        
        # Add close button
        plt.figtext(0.5, 0.2, "Click anywhere to close",
                   ha="center", fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="blue", alpha=0.7))
        
        # Add click handler to close
        def close_on_click(event):
            plt.close()
            print("Tile selection tool completed successfully")
            sys.stdout.flush()
            logger.info("Tile selection tool completed successfully")
        
        fig.canvas.mpl_connect('button_press_event', close_on_click)
        
        plt.tight_layout()
        plt.show()
    
    def run_workflow(self, parameters_dir):
        """Run the complete tile selection workflow"""
        print("Starting tile selection workflow...")
        sys.stdout.flush()
        logger.info("Starting tile selection workflow...")
        
        try:
            # Step 1: Find all tiles
            self.find_tiles()
            
            # Step 2: Create overview image with multiprocessing
            self.create_overview_image(parameters_dir)
            
            # Step 3: Start the sequential selection process
            print("Overview creation completed. Starting interactive selection...")
            sys.stdout.flush()
            self.display_selection_interface()
            
        except Exception as e:
            error_msg = f"Error in tile selection workflow: {str(e)}"
            print(f"ERROR: {error_msg}")
            sys.stdout.flush()
            logger.error(error_msg)
            raise

def main():
    """Main function with proper argument parsing following GUI guidelines"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Tile Selection Tool for IHC Pipeline")
    parser.add_argument("--data-dir", help="Base data directory")
    parser.add_argument("--output-dir", help="Output directory (not used, tiles go to data-dir)")
    parser.add_argument("--parameters-dir", help="Parameters directory")
    parser.add_argument("--base-dir", help="Base directory containing the input tiles folder (for standalone use)")
    parser.add_argument("--scale", type=float, default=0.1, help="Scale factor for overview image (0.0-1.0)")
    
    args = parser.parse_args()
    
    # Determine base directory based on whether called from GUI or standalone
    if args.data_dir and args.parameters_dir:
        # Called from GUI - use data directory as base
        base_dir = args.data_dir
        parameters_dir = args.parameters_dir
        
        # Set up log directory - use Logs directory at the same level as Data
        log_dir = os.path.join(os.path.dirname(args.data_dir), "Logs")
        os.makedirs(log_dir, exist_ok=True)
    elif args.base_dir:
        # Called standalone with custom base directory
        base_dir = args.base_dir
        parameters_dir = args.parameters_dir if args.parameters_dir else os.path.join(base_dir, "Parameters")
        
        # Set up log directory for standalone mode
        log_dir = os.path.join(base_dir, "Logs")
        os.makedirs(log_dir, exist_ok=True)
    else:
        print("ERROR: Either provide --data-dir and --parameters-dir OR provide --base-dir")
        sys.stdout.flush()
        return
    
    # Create parameters directory if it doesn't exist
    os.makedirs(parameters_dir, exist_ok=True)
    
    # Configure logging to file
    log_file = os.path.join(log_dir, "tile_selection_tool.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    print("Starting Tile Selection Tool")
    print(f"Base directory: {base_dir}")
    print(f"Parameters directory: {parameters_dir}")
    print(f"Log file: {log_file}")
    sys.stdout.flush()
    
    logger.info("Starting Tile Selection Tool")
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Parameters directory: {parameters_dir}")
    
    try:
        # Create and run the tool
        selector = TileSelectionTool(base_dir, scale_factor=args.scale)
        selector.run_workflow(parameters_dir)
        
        print("Tile Selection Tool completed successfully")
        sys.stdout.flush()
        logger.info("Tile Selection Tool completed successfully")
        
    except Exception as e:
        error_msg = f"Fatal error in Tile Selection Tool: {str(e)}"
        print(f"ERROR: {error_msg}")
        sys.stdout.flush()
        logger.error(error_msg)
        return 1
    
    return 0

if __name__ == "__main__":
    # For Windows compatibility
    multiprocessing.freeze_support()
    
    # Configure basic console logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    exit_code = main()
    sys.exit(exit_code)
