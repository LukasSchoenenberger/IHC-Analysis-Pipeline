#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Microglia Density Map Script
---------------------------
This script generates density maps from microglia detection results.
It reads microglia count data from CSV and creates two TIFF density maps:
1. Raw microglia counts per tile
2. Normalized microglia counts (0-255 range)

Part of the IHC Pipeline GUI application.
"""

import os
import sys
import argparse
import logging
import multiprocessing
import pandas as pd
import numpy as np
from PIL import Image
import re
from pathlib import Path

# Configure logging
logger = logging.getLogger("Microglia-Density-Map")

class MicrogliaDensityMapGenerator:
    """
    Class for generating density maps from microglia detection results
    """
    
    def __init__(self, data_dir, output_dir, parameters_dir):
        """
        Initialize the MicrogliaDensityMapGenerator
        
        Args:
            data_dir (str): Base data directory
            output_dir (str): Base output directory
            parameters_dir (str): Parameters directory
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.parameters_dir = Path(parameters_dir)
        
        # Set input CSV path
        self.csv_path = self.output_dir / "Microglia-Detection" / "microglia_cell_counts.csv"
        
        # Set output directory for density maps
        self.density_output_dir = self.output_dir / "Density-Maps"
        self.density_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default tile size in pixels
        self.tile_size = 36  # Standard tile size for IHC pipeline
        
        logger.info(f"Initialized Microglia Density Map Generator")
        logger.info(f"Input CSV: {self.csv_path}")
        logger.info(f"Output directory: {self.density_output_dir}")
    
    def extract_row_col(self, filename):
        """
        Extract row and column numbers from filenames using different possible patterns
        
        Args:
            filename (str): Tile filename
            
        Returns:
            tuple: (row, column) or None if pattern doesn't match
        """
        # Try different patterns to match filenames
        patterns = [
            r'tile_r(\d+)_c(\d+)\.tif',  # Standard pattern with .tif extension
            r'tile_r(\d+)_c(\d+)',       # Without extension
            r'r(\d+)c(\d+)\.tif',        # Alternative pattern
            r'r(\d+)c(\d+)',             # Alternative without extension
            r'tile_(\d+)_(\d+)',         # Simple tile_row_col pattern
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return int(match.group(1)), int(match.group(2))
        
        return None
    
    def extract_position_from_field(self, position_str):
        """
        Extract row and column from a position field in the format "row,col"
        
        Args:
            position_str (str): Position string in format "row,col"
            
        Returns:
            tuple: (row, column) or None if parsing fails
        """
        try:
            if isinstance(position_str, str) and ',' in position_str:
                row, col = map(int, position_str.split(','))
                return row, col
        except:
            pass
        return None
    
    def calculate_dimensions(self, df):
        """
        Calculate dimensions for the density map based on row and column values
        
        Args:
            df (pandas.DataFrame): DataFrame containing tile data
            
        Returns:
            tuple: (height, width) in pixels
        """
        max_row = -1
        max_col = -1
        
        print("Analyzing tile positions to determine map dimensions...")
        sys.stdout.flush()
        
        # First try to use position column if available
        if 'position' in df.columns:
            for pos_str in df['position']:
                position = self.extract_position_from_field(pos_str)
                if position:
                    row, col = position
                    max_row = max(max_row, row)
                    max_col = max(max_col, col)
        
        # If no position information found, try to extract from tile_name
        if max_row < 0 or max_col < 0:
            if 'tile_name' in df.columns:
                for tile_name in df['tile_name']:
                    position = self.extract_row_col(tile_name)
                    if position:
                        row, col = position
                        max_row = max(max_row, row)
                        max_col = max(max_col, col)
        
        if max_row < 0 or max_col < 0:
            logger.warning("Could not determine dimensions from position or tile names")
            # Default to a reasonable size if we can't determine dimensions
            return 1000, 1000
        
        # Add 1 to account for zero-indexing and multiply by tile size
        height = (max_row + 1) * self.tile_size
        width = (max_col + 1) * self.tile_size
        
        print(f"Calculated density map dimensions: {height}x{width} pixels")
        sys.stdout.flush()
        logger.info(f"Calculated dimensions: {height}x{width} pixels")
        
        return height, width
    
    def find_count_column(self, df):
        """
        Find the correct microglia count column name in the DataFrame
        
        Args:
            df (pandas.DataFrame): DataFrame containing tile data
            
        Returns:
            str: Column name for microglia counts or None if not found
        """
        # Check for different case variants of microglia count column
        potential_columns = [
            'cell_count',           # Primary column name from Microglia-Detection.py
            'microglia_count',      
            'Microglia_Count',
            'count',
            'total_microglia',
            'detection_count',
            'positive_pixels'
        ]
        
        for col in potential_columns:
            if col in df.columns:
                print(f"Using microglia count column: {col}")
                sys.stdout.flush()
                logger.info(f"Using microglia count column: {col}")
                return col
        
        logger.error(f"Could not find microglia count column in DataFrame")
        logger.error(f"Available columns: {df.columns.tolist()}")
        return None
    
    def generate_density_maps(self):
        """
        Generate both raw and normalized density maps for microglia counts
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print("Starting microglia density map generation...")
            sys.stdout.flush()
            logger.info(f"Generating microglia density maps from {self.csv_path}")
            
            # Check if the CSV file exists
            if not self.csv_path.exists():
                print(f"ERROR: Microglia detection results CSV file not found at {self.csv_path}")
                sys.stdout.flush()
                logger.error(f"CSV file not found at {self.csv_path}")
                return False
            
            print("Reading microglia detection results...")
            sys.stdout.flush()
            
            # Read the CSV file
            df = pd.read_csv(self.csv_path)
            
            print(f"Loaded {len(df)} microglia detection results")
            sys.stdout.flush()
            
            # Find the correct count column
            count_col = self.find_count_column(df)
            if not count_col:
                print("ERROR: Could not find microglia count column in CSV")
                sys.stdout.flush()
                return False
            
            # Convert count to int for calculations
            df[count_col] = df[count_col].astype(int)
            
            # Get min and max for reporting
            min_count = df[count_col].min()
            max_count = df[count_col].max()
            
            print(f"Microglia count range: {min_count} to {max_count}")
            sys.stdout.flush()
            
            # Calculate dimensions
            height, width = self.calculate_dimensions(df)
            
            print("Creating density map arrays...")
            sys.stdout.flush()
            
            # Create empty output arrays
            raw_output = np.zeros((height, width), dtype=np.uint16)  # Use uint16 for raw counts
            normalized_output = np.zeros((height, width), dtype=np.uint8)  # Use uint8 for normalized
            
            # Normalize counts to 0-255 range for the normalized map
            if max_count == min_count:
                print("WARNING: All microglia counts are identical, creating uniform normalized density map")
                sys.stdout.flush()
                df['Count_Normalized'] = np.full(len(df), 128, dtype=np.uint8)  # Middle gray
            else:
                # Normalize to 0-255 range for grayscale image
                df['Count_Normalized'] = ((df[count_col] - min_count) / 
                                         (max_count - min_count) * 255).astype(np.uint8)
            
            # Count of positioned tiles
            positioned_tiles = 0
            total_tiles = len(df)
            
            print("Processing tiles and populating density maps...")
            sys.stdout.flush()
            
            # Process each tile
            for idx, row in df.iterrows():
                # Progress reporting
                if (idx + 1) % max(1, min(50, total_tiles // 20)) == 0 or (idx + 1) == total_tiles:
                    progress_pct = ((idx + 1) / total_tiles) * 100
                    print(f"Progress: {idx + 1}/{total_tiles} tiles processed ({progress_pct:.1f}%)")
                    sys.stdout.flush()
                
                # First try to use position field if available
                position = None
                if 'position' in df.columns:
                    position = self.extract_position_from_field(row['position'])
                
                # If position field failed, try to extract from tile_name
                if position is None and 'tile_name' in df.columns:
                    position = self.extract_row_col(row['tile_name'])
                
                if position:
                    r, c = position
                    raw_count_value = row[count_col]
                    normalized_count_value = row['Count_Normalized']
                    
                    # Calculate pixel positions
                    r_start = r * self.tile_size
                    r_end = (r + 1) * self.tile_size
                    c_start = c * self.tile_size
                    c_end = (c + 1) * self.tile_size
                    
                    # Make sure we don't go out of bounds
                    if r_start < height and c_start < width:
                        # Adjust end points if needed
                        r_end = min(r_end, height)
                        c_end = min(c_end, width)
                        
                        # Fill the tile area with the count values
                        raw_output[r_start:r_end, c_start:c_end] = raw_count_value
                        normalized_output[r_start:r_end, c_start:c_end] = normalized_count_value
                        positioned_tiles += 1
            
            print("Saving density maps...")
            sys.stdout.flush()
            
            # Save raw density map as 16-bit TIFF
            raw_output_path = self.density_output_dir / "Microglia-Density-Map.tif"
            raw_img = Image.fromarray(raw_output)
            raw_img.save(raw_output_path)
            
            # Save normalized density map as 8-bit TIFF
            normalized_output_path = self.density_output_dir / "Microglia-Density-Map-Normalized.tif"
            normalized_img = Image.fromarray(normalized_output)
            normalized_img.save(normalized_output_path)
            
            print(f"Microglia density map generation completed successfully!")
            print(f"Positioned {positioned_tiles} of {total_tiles} tiles on the density maps")
            print(f"Raw density map saved to: {raw_output_path}")
            print(f"Normalized density map saved to: {normalized_output_path}")
            sys.stdout.flush()
            
            logger.info(f"Positioned {positioned_tiles} of {total_tiles} tiles on the density maps")
            logger.info(f"Raw density map saved to: {raw_output_path}")
            logger.info(f"Normalized density map saved to: {normalized_output_path}")
            logger.info(f"Original count range: {min_count} to {max_count}")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to generate microglia density maps: {str(e)}")
            sys.stdout.flush()
            logger.error(f"Error generating density maps: {e}", exc_info=True)
            return False

def main():
    """
    Main function to run the script from command line
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate microglia density maps from detection results')
    parser.add_argument('--data-dir', help='Base data directory')
    parser.add_argument('--output-dir', help='Base output directory')  
    parser.add_argument('--parameters-dir', help='Parameters directory')
    parser.add_argument('--tile-size', type=int, default=36,
                      help='Size of each tile in pixels (default: 36)')
    
    args = parser.parse_args()
    
    # Validate required arguments
    if not all([args.data_dir, args.output_dir, args.parameters_dir]):
        print("ERROR: --data-dir, --output-dir, and --parameters-dir are required")
        sys.stdout.flush()
        return 1
    
    # Set up log directory
    base_dir = os.path.dirname(args.data_dir)
    log_dir = os.path.join(base_dir, "Logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging to file
    log_file = os.path.join(log_dir, "microglia_density_map.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    print("Starting Microglia Density Map Generation")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Parameters directory: {args.parameters_dir}")
    print(f"Log file: {log_file}")
    sys.stdout.flush()
    
    logger.info("Starting Microglia Density Map Generation")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Parameters directory: {args.parameters_dir}")
    
    # Create generator instance
    generator = MicrogliaDensityMapGenerator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        parameters_dir=args.parameters_dir
    )
    
    # Set tile size if specified
    if args.tile_size:
        generator.tile_size = args.tile_size
        print(f"Using tile size: {args.tile_size} pixels")
        sys.stdout.flush()
    
    # Generate density maps
    success = generator.generate_density_maps()
    
    if success:
        print("Microglia density map generation completed successfully")
        sys.stdout.flush()
        logger.info("Microglia density map generation completed successfully")
        return 0
    else:
        print("ERROR: Failed to generate microglia density maps")
        sys.stdout.flush()
        logger.error("Failed to generate microglia density maps")
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
    
    exit(main())
