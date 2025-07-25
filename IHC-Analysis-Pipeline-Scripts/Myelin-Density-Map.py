#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Myelin Density Map Script
-------------------------
This script generates a density map from myelin detection results.
It reads myelin count data from CSV and creates a grayscale TIFF density map.

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
logger = logging.getLogger("Myelin-Density-Map")

class MyelinDensityMapGenerator:
    """
    Class for generating density maps from myelin detection results
    """
    
    def __init__(self, data_dir, output_dir, parameters_dir):
        """
        Initialize the MyelinDensityMapGenerator
        
        Args:
            data_dir (str): Base data directory
            output_dir (str): Base output directory
            parameters_dir (str): Parameters directory
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.parameters_dir = Path(parameters_dir)
        
        # Set input CSV path
        self.csv_path = self.output_dir / "Myelin-Detection" / "myelin_detection_results.csv"
        
        # Set output directory for density maps
        self.density_output_dir = self.output_dir / "Density-Maps"
        self.density_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default tile size in pixels
        self.tile_size = 36  # Standard tile size for IHC pipeline
        
        logger.info(f"Initialized Myelin Density Map Generator")
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
        Find the correct myelin count column name in the DataFrame
        
        Args:
            df (pandas.DataFrame): DataFrame containing tile data
            
        Returns:
            str: Column name for myelin counts or None if not found
        """
        # Check for different case variants of myelin count column
        potential_columns = [
            'Myelin_Count',
            'myelin_count', 
            'count',
            'positive_pixels',
            'detection_count'
        ]
        
        for col in potential_columns:
            if col in df.columns:
                print(f"Using myelin count column: {col}")
                sys.stdout.flush()
                logger.info(f"Using count column: {col}")
                return col
        
        logger.error(f"Could not find myelin count column in DataFrame")
        logger.error(f"Available columns: {df.columns.tolist()}")
        return None
    
    def generate_density_map(self):
        """
        Generate a grayscale density map for myelin counts
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print("Starting myelin density map generation...")
            sys.stdout.flush()
            logger.info(f"Generating myelin density map from {self.csv_path}")
            
            # Check if the CSV file exists
            if not self.csv_path.exists():
                print(f"ERROR: Myelin detection results CSV file not found at {self.csv_path}")
                sys.stdout.flush()
                logger.error(f"CSV file not found at {self.csv_path}")
                return False
            
            print("Reading myelin detection results...")
            sys.stdout.flush()
            
            # Read the CSV file
            df = pd.read_csv(self.csv_path)
            
            print(f"Loaded {len(df)} myelin detection results")
            sys.stdout.flush()
            
            # Find the correct count column
            count_col = self.find_count_column(df)
            if not count_col:
                print("ERROR: Could not find myelin count column in CSV")
                sys.stdout.flush()
                return False
            
            # Convert count to float64 for calculations
            df[count_col] = df[count_col].astype(np.float64)
            
            # Get min and max for normalization
            min_count = df[count_col].min()
            max_count = df[count_col].max()
            
            print(f"Myelin count range: {min_count} to {max_count}")
            sys.stdout.flush()
            
            # Avoid division by zero if min and max are the same
            if max_count == min_count:
                print("WARNING: All myelin counts are identical, creating uniform density map")
                sys.stdout.flush()
                df['Count_Normalized'] = np.full(len(df), 128, dtype=np.uint8)  # Middle gray
            else:
                # Normalize to 0-255 range for grayscale image
                df['Count_Normalized'] = ((df[count_col] - min_count) / 
                                         (max_count - min_count) * 255).astype(np.uint8)
            
            # Calculate dimensions
            height, width = self.calculate_dimensions(df)
            
            print("Creating density map array...")
            sys.stdout.flush()
            
            # Create empty output array
            output = np.zeros((height, width), dtype=np.uint8)
            
            # Count of positioned tiles
            positioned_tiles = 0
            total_tiles = len(df)
            
            print("Processing tiles and populating density map...")
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
                    count_value = row['Count_Normalized']
                    
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
                        
                        # Fill the tile area with the normalized count
                        output[r_start:r_end, c_start:c_end] = count_value
                        positioned_tiles += 1
            
            print("Saving density map...")
            sys.stdout.flush()
            
            # Save as grayscale TIFF
            output_path = self.density_output_dir / "Myelin-Density-Map.tif"
            img = Image.fromarray(output)
            img.save(output_path)
            
            print(f"Myelin density map generation completed successfully!")
            print(f"Positioned {positioned_tiles} of {total_tiles} tiles on the density map")
            print(f"Output saved to: {output_path}")
            sys.stdout.flush()
            
            logger.info(f"Positioned {positioned_tiles} of {total_tiles} tiles on the density map")
            logger.info(f"Density map saved to: {output_path}")
            logger.info(f"Original count range: {min_count} to {max_count}")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to generate myelin density map: {str(e)}")
            sys.stdout.flush()
            logger.error(f"Error generating density map: {e}", exc_info=True)
            return False

def main():
    """
    Main function to run the script from command line
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate myelin density map from detection results')
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
    log_file = os.path.join(log_dir, "myelin_density_map.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    print("Starting Myelin Density Map Generation")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Parameters directory: {args.parameters_dir}")
    print(f"Log file: {log_file}")
    sys.stdout.flush()
    
    logger.info("Starting Myelin Density Map Generation")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Parameters directory: {args.parameters_dir}")
    
    # Create generator instance
    generator = MyelinDensityMapGenerator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        parameters_dir=args.parameters_dir
    )
    
    # Set tile size if specified
    if args.tile_size:
        generator.tile_size = args.tile_size
        print(f"Using tile size: {args.tile_size} pixels")
        sys.stdout.flush()
    
    # Generate density map
    success = generator.generate_density_map()
    
    if success:
        print("Myelin density map generation completed successfully")
        sys.stdout.flush()
        logger.info("Myelin density map generation completed successfully")
        return 0
    else:
        print("ERROR: Failed to generate myelin density map")
        sys.stdout.flush()
        logger.error("Failed to generate myelin density map")
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
