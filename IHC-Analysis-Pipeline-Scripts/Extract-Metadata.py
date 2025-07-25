#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract-Metadata.py
------------------
This script automatically extracts metadata from the Original_Metadata.txt file
and creates a structured Metadata.csv file. It also generates a QuPath groovy
script for tile export.
"""

import os
import csv
import re
import argparse
import logging
import sys

# Configure logging
logger = logging.getLogger("IHC_Pipeline.Extract_Metadata")

def extract_metadata(parameters_dir, data_dir, output_dir):
    """
    Extract metadata from Original_Metadata.txt and create Metadata.csv
    
    Args:
        parameters_dir (str): Path to the parameters directory
        data_dir (str): Path to the data directory
        output_dir (str): Path to the output directory
    
    Returns:
        bool: True if metadata extraction was successful, False otherwise
    """
    # Define the columns for the CSV file
    columns = [
        "Block-ID", 
        "Pixel-Width", 
        "Pixel-Height", 
        "WSI-Width_Micrometer", 
        "WSI-Width_Pixel", 
        "WSI-Height_Micrometer", 
        "WSI-Height_Pixel", 
        "Tile-Width_Micrometer", 
        "Tile-Width_Pixel", 
        "Tile-Height_Micrometer", 
        "Tile-Height_Pixel", 
        "#Columns", 
        "#Rows", 
        "Background-Value"
    ]
    
    # Define the path to the Parameters folder and files
    metadata_file = os.path.join(parameters_dir, "Original_Metadata.txt")
    output_csv_file = os.path.join(parameters_dir, "Metadata.csv")
    
    # Get the base directory (parent of the Parameters directory)
    base_dir = os.path.dirname(parameters_dir)
    
    # Define Scripts directory in the same level as the Parameters directory
    scripts_dir = os.path.join(base_dir, "Scripts")
    
    # Save the groovy script to the Scripts folder
    output_groovy_file = os.path.join(scripts_dir, "Tile-Export-QuPath.groovy")
    
    # Check if the original metadata file exists
    if not os.path.exists(metadata_file):
        logger.error(f"Original metadata file not found: {metadata_file}")
        return False
    
    # Check if Scripts directory exists
    if not os.path.exists(scripts_dir):
        logger.error(f"Scripts directory not found: {scripts_dir}")
        return False
    
    logger.info(f"Extracting metadata from: {metadata_file}")
    
    # Initialize variables for extracted values
    block_id = ""
    pixel_width = None
    pixel_height = None
    wsi_width_pixel = None
    wsi_height_pixel = None
    wsi_width_micrometer = None
    wsi_height_micrometer = None
    tile_width_micrometer = None
    tile_height_micrometer = None
    tile_width_pixel = None
    tile_height_pixel = None
    num_columns = None
    num_rows = None
    background_value = ""
    
    # Extract and calculate values from Original_Metadata.txt
    with open(metadata_file, 'r') as file:
        content = file.read()
    
    # Extract Block-ID
    document_name_match = re.search(r'Document Name #1=.*?_(\d+_\d+_\d+)', content)
    if document_name_match:
        block_id = document_name_match.group(1)
        logger.info(f"Extracted Block-ID: {block_id}")
    else:
        logger.warning("Block-ID not found in metadata file")
    
    # Extract Pixel-Width and Pixel-Height
    calibration1_match = re.search(r'Calibration #1=\((.*?), (.*?)\)', content)
    if calibration1_match:
        try:
            pixel_width = float(calibration1_match.group(1))
            pixel_height = float(calibration1_match.group(2))
            logger.info(f"Extracted Pixel dimensions: {pixel_width} x {pixel_height}")
        except (ValueError, TypeError) as e:
            logger.warning(f"Error parsing pixel dimensions: {e}")
    else:
        logger.warning("Pixel dimensions not found in metadata file")
    
    # Extract WSI-Width_Pixel and WSI-Height_Pixel
    image_size_match = re.search(r'Image size=\((\d+), (\d+), (\d+), (\d+)\)', content)
    if image_size_match:
        try:
            wsi_width_pixel = int(image_size_match.group(3))
            wsi_height_pixel = int(image_size_match.group(4))
            logger.info(f"Extracted Image size: {wsi_width_pixel} x {wsi_height_pixel} pixels")
            
            # Calculate WSI-Width_Micrometer and WSI-Height_Micrometer
            if pixel_width is not None and wsi_width_pixel is not None:
                wsi_width_micrometer = pixel_width * wsi_width_pixel
                logger.info(f"Calculated WSI width: {wsi_width_micrometer} micrometers")
            
            if pixel_height is not None and wsi_height_pixel is not None:
                wsi_height_micrometer = pixel_height * wsi_height_pixel
                logger.info(f"Calculated WSI height: {wsi_height_micrometer} micrometers")
        except (ValueError, TypeError) as e:
            logger.warning(f"Error parsing image size: {e}")
    else:
        logger.warning("Image size not found in metadata file")
    
    # Extract Tile-Width_Micrometer and Tile-Height_Micrometer
    calibration2_match = re.search(r'Calibration #2=\((.*?), (.*?)\)', content)
    if calibration2_match:
        try:
            tile_width_micrometer = float(calibration2_match.group(1))
            tile_height_micrometer = float(calibration2_match.group(2))
            logger.info(f"Extracted Tile dimensions: {tile_width_micrometer} x {tile_height_micrometer} micrometers")
            
            # Calculate Tile-Width_Pixel and Tile-Height_Pixel
            # Using round() to get the closest integer
            if pixel_width is not None and tile_width_micrometer is not None:
                tile_width_pixel = round(tile_width_micrometer / pixel_width)
                logger.info(f"Calculated tile width: {tile_width_pixel} pixels")
            
            if pixel_height is not None and tile_height_micrometer is not None:
                tile_height_pixel = round(tile_height_micrometer / pixel_height)
                logger.info(f"Calculated tile height: {tile_height_pixel} pixels")
            
            # Calculate #Columns and #Rows
            if wsi_width_pixel is not None and tile_width_pixel is not None:
                num_columns = round(wsi_width_pixel / tile_width_pixel)
                logger.info(f"Calculated columns: {num_columns}")
            
            if wsi_height_pixel is not None and tile_height_pixel is not None:
                num_rows = round(wsi_height_pixel / tile_height_pixel)
                logger.info(f"Calculated rows: {num_rows}")
        except (ValueError, TypeError) as e:
            logger.warning(f"Error parsing tile dimensions: {e}")
    else:
        logger.warning("Tile dimensions not found in metadata file")
    
    # Initialize the metadata dictionary
    metadata = {
        "Block-ID": block_id,
        "Pixel-Width": pixel_width if pixel_width is not None else "",
        "Pixel-Height": pixel_height if pixel_height is not None else "",
        "WSI-Width_Micrometer": wsi_width_micrometer if wsi_width_micrometer is not None else "",
        "WSI-Width_Pixel": wsi_width_pixel if wsi_width_pixel is not None else "",
        "WSI-Height_Micrometer": wsi_height_micrometer if wsi_height_micrometer is not None else "",
        "WSI-Height_Pixel": wsi_height_pixel if wsi_height_pixel is not None else "",
        "Tile-Width_Micrometer": tile_width_micrometer if tile_width_micrometer is not None else "",
        "Tile-Width_Pixel": tile_width_pixel if tile_width_pixel is not None else "",
        "Tile-Height_Micrometer": tile_height_micrometer if tile_height_micrometer is not None else "",
        "Tile-Height_Pixel": tile_height_pixel if tile_height_pixel is not None else "",
        "#Columns": num_columns if num_columns is not None else "",
        "#Rows": num_rows if num_rows is not None else "",
        "Background-Value": background_value
    }
    
    # Write the metadata to the CSV file
    with open(output_csv_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerow(metadata)
    
    logger.info(f"Metadata CSV file has been created at: {output_csv_file}")
    
    # Create the modified groovy script - Now pointing to Data directory instead of PROJECT_BASE_DIR
    groovy_template = """// Get the current image from QuPath
def imageData = getCurrentImageData()
def server = imageData.getServer()

// Image dimensions from server
def width = server.getWidth()
def height = server.getHeight()

// *** CONFIGURE ROWS AND COLUMNS HERE ***
def rows = {rows}  // Automatically set from Metadata.csv
def cols = {cols}  // Automatically set from Metadata.csv

// Calculate tile dimensions based on image size and grid
def tileWidth = Math.ceil(width / cols)   // Width for each column
def tileHeight = Math.ceil(height / rows)  // Height for each row

// Set path to tiles directory in the Data folder
def pathTiles = "{data_path}/Tiles-Medium"  // Updated to use Data directory
mkdirs(pathTiles)

// Export all tiles
for (int row = 0; row < rows; row++) {{
    for (int col = 0; col < cols; col++) {{
        // Calculate tile coordinates (without overlap)
        double x = col * tileWidth
        double y = row * tileHeight
        double w = tileWidth
        double h = tileHeight
        
        // Create request for this tile
        def request = RegionRequest.createInstance(
            server.getPath(),
            1,  // Full resolution (downsample = 1)
            x as int,
            y as int,
            w as int,
            h as int
        )
        
        // Export tile
        def tileOutputPath = buildFilePath(pathTiles, String.format('tile_r%d_c%d.tif', row, col))
        writeImageRegion(server, request, tileOutputPath)
        
        print String.format('Exported tile r%d_c%d (%d/%d)\\n', 
            row, col, (row * cols + col + 1), rows * cols)
    }}
}}

print 'Export of all tiles complete!'
print String.format('Exported %d tiles (%d rows x %d columns) to folder: %s', rows * cols, rows, cols, pathTiles)
"""
    
    # Convert any Windows backslashes to forward slashes for groovy
    data_path = data_dir.replace('\\', '/')
    
    # Only create the groovy script if we have calculated the rows and columns
    if num_rows is not None and num_columns is not None:
        groovy_content = groovy_template.format(rows=num_rows, cols=num_columns, data_path=data_path)
        
        with open(output_groovy_file, 'w') as file:
            file.write(groovy_content)
        
        logger.info(f"QuPath export script has been created at: {output_groovy_file}")
    else:
        # Create a default groovy script with placeholder values if metadata was not available
        groovy_content = groovy_template.format(
            rows="/* ROWS NOT AVAILABLE */", 
            cols="/* COLUMNS NOT AVAILABLE */",
            data_path=data_path
        )
        
        with open(output_groovy_file, 'w') as file:
            file.write(groovy_content)
        
        logger.warning(f"QuPath export script has been created at: {output_groovy_file} (without rows/columns values)")
    
    return True

def main():
    """Main function for the script."""
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Extract metadata from Original_Metadata.txt")
    parser.add_argument("--data-dir", required=True, help="Path to the data directory")
    parser.add_argument("--output-dir", required=True, help="Path to the output directory")
    parser.add_argument("--parameters-dir", required=True, help="Path to the parameters directory")
    
    # If no arguments were provided, show help and exit
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Extract metadata
    success = extract_metadata(args.parameters_dir, args.data_dir, args.output_dir)
    
    if success:
        logger.info("Metadata extraction completed successfully.")
        return 0
    else:
        logger.error("Metadata extraction failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
