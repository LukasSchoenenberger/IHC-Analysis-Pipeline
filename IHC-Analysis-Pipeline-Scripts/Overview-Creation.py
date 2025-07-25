#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overview Creation Script (Fixed)
--------------------------------
This script creates downsampled overview images from processed image tiles.
It supports different processing steps (background removal, normalization, etc.)
and creates appropriately named overview files.

FIXED: Auto-detects grid dimensions from actual tiles instead of relying on metadata.
This solves issues with detection overlay tiles having different grid structures.

Part of the IHC Pipeline GUI application.
"""

import numpy as np
from PIL import Image
import os
from pathlib import Path
import time
import multiprocessing
from multiprocessing import Pool
from functools import partial
import traceback
import argparse
import csv
import sys
import logging
import re

# Configure logging
logger = logging.getLogger("Overview-Creation")

class OverviewCreator:
    """Class for creating downsampled overview from image tiles"""
    
    def __init__(self, input_folder, output_folder, step_type, scale_factor=0.1, 
                 parameters_dir=None):
        """
        Initialize the Overview Creator
        
        Args:
            input_folder: Directory containing processed tiles
            output_folder: Directory where overview will be saved
            step_type: Type of processing step (background, l-channel, illumination, normalization, etc.)
            scale_factor: Scale factor for downsampling
            parameters_dir: Directory containing metadata file (optional for detection steps)
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.step_type = step_type
        self.scale_factor = scale_factor
        self.parameters_dir = parameters_dir
        
        # These will be auto-detected from tiles
        self.min_row = None
        self.max_row = None
        self.min_col = None
        self.max_col = None
        self.row_span = None
        self.col_span = None
        self.canvas_rows = None
        self.canvas_cols = None
        self.tile_coordinates = None
        self.tile_width = None
        self.tile_height = None
        
        # Smart CPU core allocation following GUI guidelines
        available_cpus = multiprocessing.cpu_count()
        if available_cpus >= 8:
            self.num_cores = available_cpus - 1  # Leave one core free for GUI
        else:
            self.num_cores = max(1, available_cpus // 2)  # Use half on smaller machines
        
        # Define output file names based on step type
        self.output_file_names = {
            'background': 'Background-Removal-Overview.jpg',
            'l-channel': 'Luminance-Normalization-Overview.jpg', 
            'illumination': 'Illumination-Correction-Overview.jpg',
            'normalization': 'Stain-Normalization-Overview.jpg',
            'cell': 'Cell-Detection-Overview.jpg',
            'microglia': 'Microglia-Detection-Overview.jpg',
            'myelin': 'Myelin-Detection-Overview.jpg'
        }
    
    def load_metadata_fallback(self):
        """Load tile grid dimensions from metadata CSV file (fallback for non-detection steps)"""
        if not self.parameters_dir:
            print("INFO: No parameters directory specified, will auto-detect grid dimensions")
            sys.stdout.flush()
            return False
        
        metadata_file = os.path.join(self.parameters_dir, "Metadata.csv")
        
        if not os.path.exists(metadata_file):
            print(f"INFO: Metadata file not found: {metadata_file}, will auto-detect grid dimensions")
            sys.stdout.flush()
            return False
        
        try:
            with open(metadata_file, 'r', newline='', encoding='utf-8-sig') as csvfile:
                # Try different possible column name variations
                possible_row_names = ['#Rows', 'Rows', 'rows', 'num_rows', 'nrows']
                possible_col_names = ['#Columns', 'Columns', 'columns', 'cols', 'num_cols', 'ncols']
                
                reader = csv.DictReader(csvfile)
                actual_columns = reader.fieldnames
                
                # Find the correct column names
                row_col_name = None
                col_col_name = None
                
                for col_name in actual_columns:
                    if col_name in possible_row_names:
                        row_col_name = col_name
                    if col_name in possible_col_names:
                        col_col_name = col_name
                
                if not row_col_name or not col_col_name:
                    print(f"INFO: Could not find row/column information in CSV, will auto-detect")
                    sys.stdout.flush()
                    return False
                
                # Read the first row
                row = next(reader)
                
                try:
                    metadata_rows = int(float(row[row_col_name]))
                    metadata_cols = int(float(row[col_col_name]))
                    
                    print(f"Found metadata: {metadata_rows} rows, {metadata_cols} columns")
                    sys.stdout.flush()
                    
                    # Store as fallback values, but we'll verify against actual tiles
                    self.metadata_rows = metadata_rows
                    self.metadata_cols = metadata_cols
                    
                    logger.info(f"Loaded metadata grid dimensions: {metadata_rows}x{metadata_cols}")
                    return True
                except (ValueError, KeyError) as e:
                    print(f"INFO: Could not parse metadata values, will auto-detect: {e}")
                    sys.stdout.flush()
                    return False
                    
        except Exception as e:
            print(f"INFO: Error reading metadata file, will auto-detect: {str(e)}")
            sys.stdout.flush()
            return False

    def extract_row_col(self, filename):
        """Extract row and column from tile filename using regex"""
        try:
            # Match patterns like tile_r156_c159_overlay.tif
            match = re.search(r"r(\d+)_c(\d+)", filename.lower())
            if match:
                row = int(match.group(1))
                col = int(match.group(2))
                return row, col
            else:
                logger.warning(f"Could not extract row/col from filename: {filename}")
                return None, None
        except Exception as e:
            logger.error(f"Error extracting row/col from {filename}: {e}")
            return None, None

    def analyze_tile_grid(self, tile_paths):
        """Analyze all tiles to determine actual coordinate system and create coordinate mapping"""
        print("Analyzing tile coordinate system...")
        sys.stdout.flush()
        
        # Create a set of all (row, col) coordinates that actually exist
        self.tile_coordinates = set()
        rows = []
        cols = []
        
        # Extract all row and column indices
        for tile_path in tile_paths:
            row, col = self.extract_row_col(tile_path.name)
            if row is not None and col is not None:
                self.tile_coordinates.add((row, col))
                rows.append(row)
                cols.append(col)
        
        if not rows or not cols:
            print("ERROR: Could not extract row/col information from any tiles")
            sys.stdout.flush()
            return False
        
        # Determine actual coordinate bounds
        self.min_row = min(rows)
        self.max_row = max(rows)
        self.min_col = min(cols)
        self.max_col = max(cols)
        
        # Calculate span (not assuming dense grid)
        self.row_span = self.max_row - self.min_row + 1
        self.col_span = self.max_col - self.min_col + 1
        
        # Calculate actual grid density
        total_possible_positions = self.row_span * self.col_span
        actual_tiles = len(self.tile_coordinates)
        grid_density = (actual_tiles / total_possible_positions) * 100
        
        print(f"Auto-detected coordinate system:")
        print(f"  Row range: {self.min_row} to {self.max_row} (span: {self.row_span})")
        print(f"  Col range: {self.min_col} to {self.max_col} (span: {self.col_span})")
        print(f"  Actual tile positions: {actual_tiles}")
        print(f"  Possible positions in range: {total_possible_positions}")
        print(f"  Grid density: {grid_density:.1f}%")
        
        if grid_density < 90:
            print(f"  WARNING: Sparse grid detected ({grid_density:.1f}% density)")
            print("  Will create canvas for full coordinate range with empty spaces")
        
        sys.stdout.flush()
        
        # Create row and column mapping for consistent positioning
        unique_rows = sorted(set(rows))
        unique_cols = sorted(set(cols))
        
        print(f"  Unique row positions: {len(unique_rows)} rows")
        print(f"  Unique col positions: {len(unique_cols)} columns")
        
        if len(unique_rows) * len(unique_cols) != actual_tiles:
            print(f"  Note: Not all row/col combinations exist (sparse/irregular grid)")
        
        sys.stdout.flush()
        
        # Store the dimensions we'll actually use for the canvas
        # We need to accommodate the full coordinate range, not just unique positions
        self.canvas_rows = self.row_span
        self.canvas_cols = self.col_span
        
        # Check if we have metadata and compare
        if hasattr(self, 'metadata_rows') and hasattr(self, 'metadata_cols'):
            if (self.canvas_rows != self.metadata_rows or self.canvas_cols != self.metadata_cols):
                print(f"WARNING: Metadata grid ({self.metadata_rows}x{self.metadata_cols}) != detected span ({self.canvas_rows}x{self.canvas_cols})")
                print("Using detected coordinate range for accurate overview creation")
                sys.stdout.flush()
        
        logger.info(f"Coordinate system: span {self.canvas_rows}x{self.canvas_cols}, range r{self.min_row}-{self.max_row}, c{self.min_col}-{self.max_col}, density {grid_density:.1f}%")
        return True
    
    def debug_tile_distribution(self, max_display_size=50):
        """Print a visual representation of tile distribution for debugging"""
        if not self.tile_coordinates:
            return
        
        print("Tile distribution visualization:")
        
        # Only show visualization if the grid isn't too large
        if self.canvas_rows <= max_display_size and self.canvas_cols <= max_display_size:
            for r in range(self.min_row, self.max_row + 1):
                row_str = ""
                for c in range(self.min_col, self.max_col + 1):
                    if (r, c) in self.tile_coordinates:
                        row_str += "█"
                    else:
                        row_str += "·"
                print(f"  {row_str}")
        else:
            print(f"  Grid too large to display ({self.canvas_rows}x{self.canvas_cols})")
        
        sys.stdout.flush()
    
    def find_tiles(self):
        """Find all tile files in the input folder with comprehensive search"""
        folder = Path(self.input_folder)
        
        if not folder.exists():
            print(f"ERROR: Input folder {self.input_folder} does not exist")
            sys.stdout.flush()
            logger.error(f"Input folder {self.input_folder} does not exist")
            return []
        
        # Look for common image file extensions (both upper and lower case)
        extensions = ['tif', 'tiff', 'png', 'jpg', 'jpeg', 'TIF', 'TIFF', 'PNG', 'JPG', 'JPEG']
        tiles = []
        
        # Try different naming patterns
        patterns = [
            "*r*_c*",  # Basic pattern
            "*_r*_c*", # With underscore prefix
            "tile_r*_c*", # With tile prefix
            "corrected_tile_r*_c*", # With corrected tile prefix
            "tile_r*_c*_overlay*" # With overlay suffix
        ]
        
        for pattern in patterns:
            for ext in extensions:
                search_pattern = f"{pattern}.{ext}"
                found_tiles = list(folder.glob(search_pattern))
                tiles.extend(found_tiles)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tiles = []
        for tile in tiles:
            if tile not in seen:
                seen.add(tile)
                unique_tiles.append(tile)
        
        tiles = unique_tiles
        
        if not tiles:
            print(f"ERROR: No tiles found in {self.input_folder}")
            print(f"Searched for patterns: {patterns} with extensions: {extensions}")
            
            # List what files are actually in the directory for debugging
            all_files = list(folder.glob("*"))
            print(f"Files in directory: {[f.name for f in all_files[:10]]}...")  # Show first 10
            sys.stdout.flush()
            logger.warning(f"No tiles found in {self.input_folder}")
            return []
        
        print(f"Found {len(tiles)} tiles in {self.input_folder}")
        sys.stdout.flush()
        logger.info(f"Found {len(tiles)} tiles")
        
        return tiles
    
    def determine_tile_size(self, tile_paths):
        """Determine tile size by reading the first tile"""
        if not tile_paths:
            return False
        
        first_tile_path = tile_paths[0]
        
        try:
            with Image.open(first_tile_path) as img:
                self.tile_width, self.tile_height = img.size
                
                print(f"Determined tile size: {self.tile_width}x{self.tile_height}")
                sys.stdout.flush()
                logger.info(f"Tile size: {self.tile_width}x{self.tile_height}")
                return True
        except Exception as e:
            print(f"ERROR: Error reading tile {first_tile_path}: {e}")
            sys.stdout.flush()
            logger.error(f"Error reading tile {first_tile_path}: {e}")
            return False
    
    def process_tile(self, tile_path, scaled_tile_width, scaled_tile_height):
        """Process a single tile and return row, col, and array"""
        try:
            row, col = self.extract_row_col(tile_path.name)
            if row is None or col is None:
                return None
            
            # Open and resize image
            with Image.open(tile_path) as img:
                # Use appropriate resampling method based on PIL version
                try:
                    resample_method = Image.Resampling.LANCZOS
                except AttributeError:
                    # Fallback for older PIL versions
                    resample_method = Image.LANCZOS
                
                img_small = img.resize((scaled_tile_width, scaled_tile_height), resample_method)
                # Convert to array (makes copy)
                array = np.array(img_small)
            
            return row, col, array
        except Exception as e:
            logger.error(f"Error preprocessing {tile_path.name}: {e}")
            return None
    
    def batch_process_images(self, tile_paths, scaled_tile_width, scaled_tile_height):
        """Preprocess images in parallel and return as a dictionary"""
        # Create partial function with fixed parameters
        process_fn = partial(self.process_tile, 
                           scaled_tile_width=scaled_tile_width, 
                           scaled_tile_height=scaled_tile_height)
        
        total_tiles = len(tile_paths)
        print(f"Processing {total_tiles} tiles using {self.num_cores} CPU cores.")
        sys.stdout.flush()
        
        # Use spawn method for better cross-platform compatibility
        ctx = multiprocessing.get_context('spawn')
        
        # Process in parallel
        try:
            with ctx.Pool(processes=self.num_cores) as pool:
                # Calculate optimal chunk size for balanced load
                chunksize = max(1, total_tiles // (self.num_cores * 4))
                
                # Process with explicit progress reporting
                processed_tiles = {}
                processed_count = 0
                
                for result in pool.imap(process_fn, tile_paths, chunksize=chunksize):
                    processed_count += 1
                    
                    # Log progress at regular intervals
                    if processed_count % max(1, min(50, total_tiles // 20)) == 0 or processed_count == total_tiles:
                        progress_pct = (processed_count / total_tiles) * 100
                        print(f"Progress: {processed_count}/{total_tiles} tiles ({progress_pct:.1f}%)")
                        sys.stdout.flush()
                    
                    if result is not None:
                        row, col, array = result
                        if row not in processed_tiles:
                            processed_tiles[row] = {}
                        processed_tiles[row][col] = array
        
        except Exception as e:
            print(f"ERROR: Error during parallel processing: {e}")
            sys.stdout.flush()
            logger.error(f"Error during parallel processing: {e}")
            return {}
        
        return processed_tiles
    
    def create_overview(self):
        """Create downsampled overview from tiles"""
        start_time = time.time()
        
        print(f"Starting overview creation for {self.step_type}")
        sys.stdout.flush()
        logger.info(f"Starting overview creation for {self.step_type}")
        
        # Find all tile files
        tile_paths = self.find_tiles()
        if not tile_paths:
            return False
        
        # Determine tile size
        if not self.determine_tile_size(tile_paths):
            print("ERROR: Failed to determine tile size")
            sys.stdout.flush()
            return False
        
        # Auto-detect grid dimensions from actual tiles
        if not self.analyze_tile_grid(tile_paths):
            print("ERROR: Failed to analyze tile coordinate system")
            sys.stdout.flush()
            return False
        
        # Show tile distribution for debugging if sparse grid
        actual_tiles = len(self.tile_coordinates)
        total_possible = self.canvas_rows * self.canvas_cols
        if actual_tiles < total_possible * 0.9:  # Less than 90% density
            self.debug_tile_distribution()
        
        # Try to load metadata as fallback/comparison (but don't fail if not available)
        self.load_metadata_fallback()
        
        # Calculate scaled tile dimensions
        scaled_tile_width = int(self.tile_width * self.scale_factor)
        scaled_tile_height = int(self.tile_height * self.scale_factor)
        
        # Ensure minimum size
        scaled_tile_width = max(1, scaled_tile_width)
        scaled_tile_height = max(1, scaled_tile_height)
        
        # Calculate output dimensions based on coordinate span (not dense grid assumption)
        final_width = scaled_tile_width * self.canvas_cols
        final_height = scaled_tile_height * self.canvas_rows
        print(f"Output canvas dimensions: {final_width}x{final_height}")
        print(f"  (Accommodates coordinate range: r{self.min_row}-{self.max_row}, c{self.min_col}-{self.max_col})")
        sys.stdout.flush()
        
        # Create empty canvas (white background)
        output_array = np.ones((final_height, final_width, 3), dtype=np.uint8) * 255
        
        # Sort tiles by row and column
        def sort_key(x):
            row, col = self.extract_row_col(x.name)
            if row is None or col is None:
                return (999999, 999999)
            return (row, col)
        
        tile_paths = sorted(tile_paths, key=sort_key)
        
        # Preprocess all images in parallel
        processed_tiles = self.batch_process_images(tile_paths, scaled_tile_width, scaled_tile_height)
        
        if not processed_tiles:
            print("ERROR: No tiles were processed successfully")
            sys.stdout.flush()
            return False
        
        # Place tiles in the output array using exact coordinate mapping
        print("Placing tiles according to their coordinate specifications...")
        sys.stdout.flush()
        placed_count = 0
        skipped_count = 0
        
        # Process all tiles
        total_tiles = sum(len(row_dict) for row_dict in processed_tiles.values())
        current_tile = 0
        
        for row in sorted(processed_tiles.keys()):
            row_dict = processed_tiles[row]
            
            for col in sorted(row_dict.keys()):
                current_tile += 1
                tile_array = row_dict[col]
                
                # Show progress periodically
                if current_tile % max(1, min(100, total_tiles // 20)) == 0 or current_tile == total_tiles:
                    progress_pct = (current_tile / total_tiles) * 100
                    print(f"Placing progress: {current_tile}/{total_tiles} tiles ({progress_pct:.1f}%)")
                    sys.stdout.flush()
                
                # Verify this coordinate should exist (redundant check, but safe)
                if (row, col) not in self.tile_coordinates:
                    logger.warning(f"Tile at ({row}, {col}) not in expected coordinates set")
                    skipped_count += 1
                    continue
                
                # Calculate position using exact coordinate mapping
                # Map actual row/col to canvas position
                canvas_row = row - self.min_row
                canvas_col = col - self.min_col
                
                pixel_x = canvas_col * scaled_tile_width
                pixel_y = canvas_row * scaled_tile_height
                
                # Validate canvas position
                if canvas_row < 0 or canvas_col < 0:
                    logger.error(f"Tile at ({row}, {col}) maps to negative canvas position ({canvas_row}, {canvas_col})")
                    skipped_count += 1
                    continue
                
                if canvas_row >= self.canvas_rows or canvas_col >= self.canvas_cols:
                    logger.error(f"Tile at ({row}, {col}) maps to canvas position ({canvas_row}, {canvas_col}) beyond canvas size ({self.canvas_rows}, {self.canvas_cols})")
                    skipped_count += 1
                    continue
                
                # Final bounds check against actual array
                if pixel_y >= output_array.shape[0] or pixel_x >= output_array.shape[1]:
                    logger.error(f"Tile at ({row}, {col}) -> canvas pos ({canvas_row}, {canvas_col}) -> pixel pos ({pixel_y}, {pixel_x}) exceeds output array shape {output_array.shape}")
                    skipped_count += 1
                    continue
                
                # Calculate actual placement dimensions
                height = min(tile_array.shape[0], output_array.shape[0] - pixel_y)
                width = min(tile_array.shape[1], output_array.shape[1] - pixel_x)
                
                if height > 0 and width > 0:
                    try:
                        # Ensure tile array has the right number of dimensions
                        if len(tile_array.shape) == 2:
                            # Grayscale - convert to RGB
                            tile_array = np.stack([tile_array] * 3, axis=-1)
                        elif len(tile_array.shape) == 3 and tile_array.shape[2] == 4:
                            # RGBA - convert to RGB by dropping alpha
                            tile_array = tile_array[:, :, :3]
                        
                        output_array[pixel_y:pixel_y+height, pixel_x:pixel_x+width] = tile_array[:height, :width]
                        placed_count += 1
                    except Exception as e:
                        logger.warning(f"Error placing tile at ({row}, {col}): {e}")
                        skipped_count += 1
                else:
                    logger.warning(f"Tile at ({row}, {col}) has zero dimensions after clipping")
                    skipped_count += 1
        
        print(f"Tile placement results:")
        print(f"  Successfully placed: {placed_count}")
        print(f"  Skipped/failed: {skipped_count}")
        print(f"  Total processed: {placed_count + skipped_count}")
        print(f"  Success rate: {(placed_count / (placed_count + skipped_count) * 100):.1f}%")
        sys.stdout.flush()
        
        # Create output directory if it doesn't exist
        try:
            os.makedirs(self.output_folder, exist_ok=True)
        except Exception as e:
            print(f"ERROR: Could not create output directory {self.output_folder}: {e}")
            sys.stdout.flush()
            return False
        
        # Get output filename based on step type
        output_filename = self.output_file_names.get(self.step_type, f"{self.step_type}-Overview.jpg")
        output_path = os.path.join(self.output_folder, output_filename)
        
        # Create image from array and save
        try:
            output_image = Image.fromarray(output_array)
            output_image.save(output_path, 'JPEG', quality=95)
        except Exception as e:
            print(f"ERROR: Could not save output image to {output_path}: {e}")
            sys.stdout.flush()
            logger.error(f"Could not save output image: {e}")
            return False
        
        total_time = (time.time() - start_time) / 60
        print(f"Overview creation completed in {total_time:.2f} minutes")
        print(f"Saved to {output_path}")
        sys.stdout.flush()
        
        logger.info(f"Overview creation completed successfully: {output_path}")
        return True
    
    def run(self):
        """Main method to run the overview creation process"""
        try:
            print("=" * 60)
            print(f"Creating overview for {self.step_type} processing step")
            print("=" * 60)
            sys.stdout.flush()
            
            success = self.create_overview()
            
            if success:
                print("Process completed successfully")
                sys.stdout.flush()
                logger.info("Process completed successfully")
            else:
                print("ERROR: Process failed")
                sys.stdout.flush()
                logger.error("Process failed")
            
            return success
        except Exception as e:
            print(f"ERROR: Error creating overview: {e}")
            sys.stdout.flush()
            logger.error(f"Error creating overview: {e}")
            traceback.print_exc()
            return False


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Create a downsampled overview from processed tiles.')
    
    # Standard arguments for GUI compatibility
    parser.add_argument('--data-dir', help='Base data directory')
    parser.add_argument('--output-dir', help='Base output directory')  
    parser.add_argument('--parameters-dir', help='Parameters directory')
    
    # Specific arguments for overview creation
    parser.add_argument('--step-type', type=str, required=True,
                       choices=['background', 'l-channel', 'illumination', 'normalization', 
                               'cell', 'microglia', 'myelin'],
                       help='Type of processing step for which to create overview')
    
    parser.add_argument('--input-folder', type=str,
                       help='Input folder containing processed tiles (for standalone use)')
    
    parser.add_argument('--scale-factor', type=float, default=0.1,
                       help='Scale factor for downsampling (default: 0.1)')
    
    return parser.parse_args()


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Determine paths based on whether called from GUI or standalone
    if args.data_dir and args.output_dir and args.parameters_dir:
        # Called from GUI - determine input folder based on step type
        step_type = args.step_type
        
        # Map step types to their expected output folder names
        step_to_folder = {
            'background': 'Tiles-Medium-L-Channel-Normalized-BG-Removed',
            'l-channel': 'Tiles-Medium-L-Channel-Normalized', 
            'illumination': 'Tiles-Medium-L-Channel-Normalized-BG-Removed-Illumination-Corrected',
            'normalization': 'Tiles-Medium-L-Channel-Normalized-BG-Removed-Illumination-Corrected-Stain-Normalized',
            'cell': 'Cell-Detection/overlays',
            'microglia': 'Microglia-Detection/overlays', 
            'myelin': 'Myelin-Detection/overlays'
        }
        
        folder_name = step_to_folder.get(step_type)
        if not folder_name:
            print(f"ERROR: Unknown step type: {step_type}")
            sys.stdout.flush()
            return
        
        # For detection steps, look in Results directory, otherwise in Data directory
        if step_type in ['cell', 'microglia', 'myelin']:
            input_folder = os.path.join(args.output_dir, folder_name)
        else:
            input_folder = os.path.join(args.data_dir, folder_name)
        
        output_folder = os.path.join(args.output_dir, "Overviews")
        parameters_dir = args.parameters_dir
        
        # Set up log directory
        base_dir = os.path.dirname(args.data_dir)
        log_dir = os.path.join(base_dir, "Logs")
        os.makedirs(log_dir, exist_ok=True)
        
    elif args.input_folder:
        # Called standalone with custom paths
        input_folder = args.input_folder
        output_folder = os.path.join(os.path.dirname(args.input_folder), "Overviews")
        parameters_dir = args.parameters_dir if args.parameters_dir else os.path.dirname(args.input_folder)
        
        # Set up log directory for standalone mode
        log_dir = "Logs"
        os.makedirs(log_dir, exist_ok=True)
    else:
        print("ERROR: Either provide --data-dir, --output-dir, and --parameters-dir OR provide --input-folder")
        sys.stdout.flush()
        return
    
    # Configure logging to file
    log_file = os.path.join(log_dir, "overview_creation.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Parameters directory: {parameters_dir}")
    print(f"Log file: {log_file}")
    sys.stdout.flush()
    
    # Create and run the overview creator
    creator = OverviewCreator(
        input_folder=input_folder,
        output_folder=output_folder,
        step_type=args.step_type,
        scale_factor=args.scale_factor,
        parameters_dir=parameters_dir
    )
    
    success = creator.run()
    
    if not success:
        sys.exit(1)


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
