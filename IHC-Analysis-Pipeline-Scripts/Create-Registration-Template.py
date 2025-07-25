#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Black Background Overview Creation Script
----------------------------------------
This script creates a downsampled overview from background-removed image tiles
with a black background instead of white. Specifically designed to process
tiles from the "Tiles-Medium-L-Channel-Normalized-BG-Removed" folder.

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

# Import for connected component analysis
try:
    from skimage.morphology import remove_small_objects
    from skimage.measure import label
    SKIMAGE_AVAILABLE = True
except ImportError:
    print("WARNING: scikit-image not available. Small component removal will be skipped.")
    SKIMAGE_AVAILABLE = False

# Configure logging
logger = logging.getLogger("Black-Background-Overview")

class BlackBackgroundOverviewCreator:
    """Class for creating downsampled overview with black background from background-removed tiles"""
    
    def __init__(self, input_folder, output_folder, scale_factor=0.1, parameters_dir=None, 
                 min_component_size=10000):
        """
        Initialize the Black Background Overview Creator
        
        Args:
            input_folder: Directory containing background-removed tiles
            output_folder: Directory where overview will be saved
            scale_factor: Scale factor for downsampling
            parameters_dir: Directory containing metadata file
            min_component_size: Minimum size (in pixels) for connected components to keep
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.scale_factor = scale_factor
        self.parameters_dir = parameters_dir
        self.min_component_size = min_component_size
        
        # These will be loaded from metadata
        self.rows = None
        self.cols = None
        self.tile_width = None
        self.tile_height = None
        self.background_threshold = None
        
        # Smart CPU core allocation following GUI guidelines
        available_cpus = multiprocessing.cpu_count()
        if available_cpus >= 8:
            self.num_cores = available_cpus - 1  # Leave one core free for GUI
        else:
            self.num_cores = max(1, available_cpus // 2)  # Use half on smaller machines
        
        # Fixed output filename for registration template
        self.output_filename = 'Registration-Template.jpg'
    
    def load_metadata(self):
        """Load tile grid dimensions from metadata CSV file"""
        if not self.parameters_dir:
            print("ERROR: Parameters directory not specified")
            sys.stdout.flush()
            logger.error("Parameters directory not specified")
            return False
        
        metadata_file = os.path.join(self.parameters_dir, "Metadata.csv")
        
        if not os.path.exists(metadata_file):
            print(f"ERROR: Metadata file not found: {metadata_file}")
            sys.stdout.flush()
            logger.error(f"Metadata file not found: {metadata_file}")
            return False
        
        try:
            with open(metadata_file, 'r', newline='', encoding='utf-8-sig') as csvfile:
                # First, try to detect the actual column names
                sample = csvfile.read(1024)
                csvfile.seek(0)
                
                # Try different possible column name variations
                possible_row_names = ['#Rows', 'Rows', 'rows', 'num_rows', 'nrows']
                possible_col_names = ['#Columns', 'Columns', 'columns', 'cols', 'num_cols', 'ncols']
                possible_bg_names = ['Background-Value', 'Background_Value', 'background_value', 'bg_value']
                
                reader = csv.DictReader(csvfile)
                
                # Get the actual column names from the file
                actual_columns = reader.fieldnames
                print(f"Found CSV columns: {actual_columns}")
                sys.stdout.flush()
                
                # Find the correct column names
                row_col_name = None
                col_col_name = None
                bg_col_name = None
                
                for col_name in actual_columns:
                    if col_name in possible_row_names:
                        row_col_name = col_name
                    if col_name in possible_col_names:
                        col_col_name = col_name
                    if col_name in possible_bg_names:
                        bg_col_name = col_name
                
                if not row_col_name or not col_col_name:
                    print(f"ERROR: Could not find row/column information in CSV. Available columns: {actual_columns}")
                    print(f"Looking for columns like: {possible_row_names} and {possible_col_names}")
                    sys.stdout.flush()
                    logger.error(f"Required columns not found in metadata. Available: {actual_columns}")
                    return False
                
                if not bg_col_name:
                    print(f"WARNING: Could not find background value column in CSV. Available columns: {actual_columns}")
                    print(f"Looking for columns like: {possible_bg_names}")
                    print("Will use default threshold of 240")
                    sys.stdout.flush()
                    logger.warning(f"Background value column not found. Using default threshold.")
                    self.background_threshold = 240  # fallback
                
                # Read the first row (assuming single image metadata)
                row = next(reader)
                
                # Extract rows and columns
                try:
                    self.rows = int(float(row[row_col_name]))  # float first to handle decimal strings
                    self.cols = int(float(row[col_col_name]))
                    
                    print(f"Loaded metadata: {self.rows} rows, {self.cols} columns")
                    
                    # Parse background values if available
                    if bg_col_name and bg_col_name in row:
                        bg_value_str = row[bg_col_name].strip()
                        self.background_threshold = self.parse_background_value(bg_value_str)
                        if self.background_threshold is not None:
                            print(f"Using background threshold from metadata: {self.background_threshold:.1f}")
                        else:
                            print("Could not parse background value, using default threshold: 240")
                            self.background_threshold = 240
                    else:
                        print("No background value found, using default threshold: 240")
                        self.background_threshold = 240
                    
                    sys.stdout.flush()
                    
                    logger.info(f"Loaded tile grid dimensions: {self.rows}x{self.cols}")
                    logger.info(f"Background threshold: {self.background_threshold}")
                    return True
                except (ValueError, KeyError) as e:
                    print(f"ERROR: Could not parse row/column values: {e}")
                    print(f"Row value: '{row.get(row_col_name, 'N/A')}', Column value: '{row.get(col_col_name, 'N/A')}'")
                    sys.stdout.flush()
                    logger.error(f"Error parsing row/column values: {e}")
                    return False
                    
        except Exception as e:
            print(f"ERROR: Error reading metadata file: {str(e)}")
            sys.stdout.flush()
            logger.error(f"Error reading metadata file: {str(e)}")
            return False
    
    def parse_background_value(self, bg_value_str):
        """Parse background value string and return average of RGB values"""
        try:
            # Remove any brackets, parentheses, and whitespace
            cleaned = bg_value_str.replace('(', '').replace(')', '').replace('[', '').replace(']', '').strip()
            
            # Try different possible separators
            separators = [',', ';', ' ', '\t']
            values = None
            
            for sep in separators:
                if sep in cleaned:
                    parts = [part.strip() for part in cleaned.split(sep) if part.strip()]
                    if len(parts) >= 3:
                        try:
                            values = [float(part) for part in parts[:3]]  # Take first 3 values
                            break
                        except ValueError:
                            continue
            
            # If no separator worked, try to parse as a single number (grayscale)
            if values is None:
                try:
                    single_value = float(cleaned)
                    values = [single_value, single_value, single_value]  # Use same value for RGB
                except ValueError:
                    return None
            
            if len(values) == 3:
                average = sum(values) / 3.0
                print(f"Parsed background values: R={values[0]:.1f}, G={values[1]:.1f}, B={values[2]:.1f}")
                print(f"Calculated average threshold: {average:.1f}")
                return average
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Error parsing background value '{bg_value_str}': {e}")
            return None
    
    def remove_small_components(self, image_array, min_size=10000):
        """Remove connected components smaller than min_size pixels from the image"""
        if not SKIMAGE_AVAILABLE:
            print("WARNING: Skipping small component removal - scikit-image not available")
            sys.stdout.flush()
            return image_array
        
        try:
            print(f"Removing connected components smaller than {min_size} pixels...")
            sys.stdout.flush()
            
            # Convert to grayscale for component analysis (create binary mask of non-black pixels)
            if len(image_array.shape) == 3:
                # For RGB images, find non-black pixels (any channel > 0)
                gray = np.max(image_array, axis=2)
            else:
                # Already grayscale
                gray = image_array.copy()
            
            # Create binary mask (True for non-black pixels)
            binary_mask = gray > 0
            
            # Label connected components
            labeled = label(binary_mask)
            
            # Remove small objects
            cleaned_mask = remove_small_objects(labeled, min_size=min_size, connectivity=2)
            
            # Create final mask (True where we want to keep pixels)
            final_mask = cleaned_mask > 0
            
            # Apply mask to original image
            if len(image_array.shape) == 3:
                # RGB image
                cleaned_array = image_array.copy()
                # Set pixels not in mask to black
                cleaned_array[~final_mask] = [0, 0, 0]
            else:
                # Grayscale image
                cleaned_array = image_array.copy()
                cleaned_array[~final_mask] = 0
            
            # Count removed components
            original_components = np.max(labeled)
            remaining_components = np.max(cleaned_mask)
            removed_components = original_components - remaining_components
            
            print(f"Component analysis: {original_components} total, {remaining_components} kept, {removed_components} removed")
            sys.stdout.flush()
            logger.info(f"Removed {removed_components} small components (< {min_size} pixels)")
            
            return cleaned_array
            
        except Exception as e:
            print(f"WARNING: Error during component removal: {e}")
            print("Continuing with original image...")
            sys.stdout.flush()
            logger.warning(f"Error during component removal: {e}")
            return image_array
    
    def extract_row_col(self, filename):
        """Extract row and column from tile filename with multiple pattern support"""
        try:
            filename_lower = filename.lower()
            
            # Try multiple possible naming patterns
            patterns = [
                # Pattern 1: corrected_tile_r{row}_c{column}.ext
                ('_r', '_c'),
                # Pattern 2: tile_r{row}_c{column}.ext  
                ('r', 'c'),
                # Pattern 3: other patterns might be added here
            ]
            
            for r_pattern, c_pattern in patterns:
                try:
                    if r_pattern in filename_lower and c_pattern in filename_lower:
                        # Extract row
                        r_parts = filename_lower.split(r_pattern)
                        if len(r_parts) < 2:
                            continue
                            
                        r_part = r_parts[1]
                        if '_' in r_part:
                            row_str = r_part.split('_')[0]
                        else:
                            # Take digits only
                            row_str = ''.join(filter(str.isdigit, r_part))
                        
                        if not row_str:
                            continue
                            
                        row = int(row_str)
                        
                        # Extract column
                        c_parts = filename_lower.split(c_pattern)
                        if len(c_parts) < 2:
                            continue
                            
                        c_part = c_parts[1]
                        if '.' in c_part:
                            col_str = c_part.split('.')[0]
                        elif '_' in c_part:
                            col_str = c_part.split('_')[0]
                        else:
                            # Take digits only
                            col_str = ''.join(filter(str.isdigit, c_part))
                        
                        if not col_str:
                            continue
                            
                        col = int(col_str)
                        
                        return row, col
                        
                except (ValueError, IndexError):
                    continue
            
            # If no pattern worked, log it
            logger.warning(f"Could not extract row/col from filename: {filename}")
            return None, None
                
        except Exception as e:
            logger.error(f"Error extracting row/col from {filename}: {e}")
            return None, None
    
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
        
        print(f"Found {len(tiles)} background-removed tiles in {self.input_folder}")
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
        """Process a single tile and return row, col, and array with white-to-black background conversion"""
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
                
                # Convert white background pixels to black
                # Use background threshold from metadata (or default if not available)
                white_threshold = self.background_threshold if self.background_threshold is not None else 240
                
                # Create mask for background pixels (all RGB channels >= threshold)
                if len(array.shape) == 3 and array.shape[2] == 3:  # RGB image
                    white_mask = (array[:, :, 0] >= white_threshold) & \
                                (array[:, :, 1] >= white_threshold) & \
                                (array[:, :, 2] >= white_threshold)
                    
                    # Set background pixels to black
                    array[white_mask] = [0, 0, 0]
                elif len(array.shape) == 2:  # Grayscale image
                    white_mask = array >= white_threshold
                    array[white_mask] = 0
                    # Convert to RGB if it was grayscale
                    array = np.stack([array, array, array], axis=2)
            
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
        print(f"Processing tiles with background-to-black conversion (threshold: {self.background_threshold:.1f})...")
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
    
    def create_black_background_overview(self):
        """Create downsampled overview with black background from background-removed tiles"""
        start_time = time.time()
        
        print("Starting registration template creation from background-removed tiles")
        sys.stdout.flush()
        logger.info("Starting registration template creation")
        
        # Load metadata first
        if not self.load_metadata():
            print("ERROR: Failed to load metadata. Cannot create overview.")
            sys.stdout.flush()
            return False
        
        # Find all tile files
        tile_paths = self.find_tiles()
        if not tile_paths:
            return False
        
        # Determine tile size
        if not self.determine_tile_size(tile_paths):
            print("ERROR: Failed to determine tile size")
            sys.stdout.flush()
            return False
        
        # Calculate scaled tile dimensions
        scaled_tile_width = int(self.tile_width * self.scale_factor)
        scaled_tile_height = int(self.tile_height * self.scale_factor)
        
        # Ensure minimum size
        scaled_tile_width = max(1, scaled_tile_width)
        scaled_tile_height = max(1, scaled_tile_height)
        
        # Calculate output dimensions
        final_width = scaled_tile_width * self.cols
        final_height = scaled_tile_height * self.rows
        print(f"Output dimensions: {final_width}x{final_height}")
        sys.stdout.flush()
        
        # Create empty canvas with BLACK background (this is the key difference!)
        output_array = np.zeros((final_height, final_width, 3), dtype=np.uint8)
        print("Created black background canvas")
        sys.stdout.flush()
        
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
        
        # Place tiles in the output array
        print("Placing tiles in black background output image...")
        sys.stdout.flush()
        placed_count = 0
        
        # Process row by row, column by column
        total_expected = len(processed_tiles)
        current_row = 0
        
        for row in sorted(processed_tiles.keys()):
            current_row += 1
            row_dict = processed_tiles[row]
            
            # Show progress every 10% of rows or at least every 5 rows
            if current_row % max(1, min(5, total_expected // 10)) == 0 or current_row == total_expected:
                progress_pct = (current_row / total_expected) * 100
                print(f"Placing progress: {current_row}/{total_expected} rows ({progress_pct:.1f}%)")
                sys.stdout.flush()
            
            for col in sorted(row_dict.keys()):
                tile_array = row_dict[col]
                
                # Calculate position
                pixel_x = col * scaled_tile_width
                pixel_y = row * scaled_tile_height
                
                # Make sure we don't exceed array bounds
                if pixel_y >= output_array.shape[0] or pixel_x >= output_array.shape[1]:
                    logger.warning(f"Tile at row {row}, col {col} exceeds output bounds")
                    continue
                
                height = min(tile_array.shape[0], output_array.shape[0] - pixel_y)
                width = min(tile_array.shape[1], output_array.shape[1] - pixel_x)
                
                if height > 0 and width > 0:
                    try:
                        output_array[pixel_y:pixel_y+height, pixel_x:pixel_x+width] = tile_array[:height, :width]
                        placed_count += 1
                    except Exception as e:
                        logger.warning(f"Error placing tile at row {row}, col {col}: {e}")
        
        print(f"Successfully placed {placed_count}/{len(tile_paths)} tiles on black background")
        sys.stdout.flush()
        
        # Remove small connected components before saving
        print("Cleaning up small artifacts...")
        sys.stdout.flush()
        output_array = self.remove_small_components(output_array, min_size=self.min_component_size)
        
        # Create output directory if it doesn't exist
        try:
            os.makedirs(self.output_folder, exist_ok=True)
        except Exception as e:
            print(f"ERROR: Could not create output directory {self.output_folder}: {e}")
            sys.stdout.flush()
            return False
        
        # Save with fixed filename for black background overview
        output_path = os.path.join(self.output_folder, self.output_filename)
        
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
        print(f"Registration template creation completed in {total_time:.2f} minutes")
        print(f"Saved to {output_path}")
        sys.stdout.flush()
        
        logger.info(f"Registration template creation completed successfully: {output_path}")
        return True
    
    def run(self):
        """Main method to run the black background overview creation process"""
        try:
            print("=" * 70)
            print("Creating REGISTRATION TEMPLATE from background-removed tiles")
            if SKIMAGE_AVAILABLE:
                print(f"Will remove connected components < {self.min_component_size} pixels")
            else:
                print("Note: Small component removal disabled (scikit-image not available)")
            print("=" * 70)
            sys.stdout.flush()
            
            success = self.create_black_background_overview()
            
            if success:
                print("Registration template creation completed successfully")
                sys.stdout.flush()
                logger.info("Process completed successfully")
            else:
                print("ERROR: Registration template creation failed")
                sys.stdout.flush()
                logger.error("Process failed")
            
            return success
        except Exception as e:
            print(f"ERROR: Error creating registration template: {e}")
            sys.stdout.flush()
            logger.error(f"Error creating registration template: {e}")
            traceback.print_exc()
            return False


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Create a registration template from background-removed tiles.')
    
    # Standard arguments for GUI compatibility
    parser.add_argument('--data-dir', help='Base data directory')
    parser.add_argument('--output-dir', help='Base output directory')  
    parser.add_argument('--parameters-dir', help='Parameters directory')
    
    # Specific arguments for registration template creation
    parser.add_argument('--input-folder', type=str,
                       help='Input folder containing background-removed tiles (for standalone use)')
    
    parser.add_argument('--scale-factor', type=float, default=0.1,
                       help='Scale factor for downsampling (default: 0.1)')
    
    parser.add_argument('--min-component-size', type=int, default=10000,
                       help='Minimum size (pixels) for connected components to keep (default: 1000)')
    
    return parser.parse_args()


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Determine paths based on whether called from GUI or standalone
    if args.data_dir and args.output_dir and args.parameters_dir:
        # Called from GUI - use fixed input folder for background-removed tiles
        input_folder = os.path.join(args.data_dir, 'Tiles-Medium-L-Channel-Normalized-BG-Removed')
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
    log_file = os.path.join(log_dir, "registration_template.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Parameters directory: {parameters_dir}")
    print(f"Log file: {log_file}")
    sys.stdout.flush()
    
    # Verify that we're targeting the correct input folder
    expected_folder_name = 'Tiles-Medium-L-Channel-Normalized-BG-Removed'
    if expected_folder_name not in input_folder:
        print(f"WARNING: Input folder does not contain expected folder name '{expected_folder_name}'")
        print(f"This script is specifically designed for background-removed tiles")
        sys.stdout.flush()
    
    # Create and run the black background overview creator
    creator = BlackBackgroundOverviewCreator(
        input_folder=input_folder,
        output_folder=output_folder,
        scale_factor=args.scale_factor,
        parameters_dir=parameters_dir,
        min_component_size=args.min_component_size
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
