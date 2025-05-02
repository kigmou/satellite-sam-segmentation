import os
import sys
import glob
from pathlib import Path
import time
from datetime import datetime
import geopandas as gpd
import logging
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.polygon_merger import merge_overlapping_segments

def rerun_merging(base_dir, year, color_type='nrg', grid_size=10, min_pixels=20):
    """
    Rerun the merging process with a new threshold and concatenate all polygons.
    
    Args:
        base_dir: Base directory containing tile folders
        year: Year of the Sentinel data (e.g., 2023)
        color_type: Either 'rgb' for true color or 'nrg' for NIR-Red-Green
        grid_size: Grid size used for tiling (default: 10)
        min_pixels: Minimum number of pixels for a segment to be kept (default: 20)
    """
    # Get all tile directories
    tile_dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]
    
    # Process each tile
    for tile_dir in tile_dirs:
        tile_id = os.path.basename(tile_dir)
        logging.info(f"\nProcessing tile: {tile_id}")
        
        # Run merging with new threshold
        merge_overlapping_segments(tile_dir, list(range(1, 5)), year, color_type, grid_size, min_pixels)
    
    # Concatenate all polygons
    concat_quarter_polygons(tile_dirs, year, color_type, grid_size)
    concat_intersection_polygons(tile_dirs, color_type, grid_size)

def concat_quarter_polygons(tile_dirs, year, color_type='nrg', grid_size=10):
    """Concatenate polygons for each quarter."""
    logging.info("\nConcatenating quarter polygons...")
    
    # Process each quarter
    for quarter in range(1, 5):
        logging.info(f"\nProcessing Q{quarter} polygons...")
        quarter_gdfs = []
        
        # Load polygons from each tile
        for tile_dir in tile_dirs:
            tile_id = os.path.basename(tile_dir)
            parquet_path = os.path.join(
                tile_dir,
                f"Sentinel-2_mosaic_{year}_Q{quarter}_{tile_id}_0_0",
                color_type,
                f"polygons_{grid_size}x{grid_size}.parquet"
            )
            if os.path.exists(parquet_path):
                try:
                    gdf = gpd.read_parquet(parquet_path)
                    quarter_gdfs.append(gdf)
                    logging.info(f"Loaded {parquet_path}: {len(gdf)} polygons")
                except Exception as e:
                    logging.error(f"Error loading {parquet_path}: {str(e)}")
        
        if quarter_gdfs:
            # Combine all polygons for this quarter
            combined_gdf = gpd.GeoDataFrame(pd.concat(quarter_gdfs, ignore_index=True))
            logging.info(f"\nTotal polygons for Q{quarter}: {len(combined_gdf)}")
            
            # Save combined results
            output_dir = os.path.join(
                os.path.dirname(tile_dirs[0]),  # Use the parent directory of the first tile
                f"Q{quarter}_polygons_{color_type}_{grid_size}x{grid_size}"
            )
            os.makedirs(output_dir, exist_ok=True)
            
            combined_gdf.to_parquet(os.path.join(output_dir, f"Q{quarter}_polygons.parquet"))
            combined_gdf.to_file(os.path.join(output_dir, f"Q{quarter}_polygons.shp"))
            logging.info(f"Saved Q{quarter} polygons to {output_dir}")
        else:
            logging.warning(f"No data found for Q{quarter}")

def concat_intersection_polygons(tile_dirs, color_type='nrg', grid_size=10):
    """Concatenate intersection polygons."""
    logging.info("\nConcatenating intersection polygons...")
    intersection_gdfs = []
    
    # Load intersection polygons from each tile
    for tile_dir in tile_dirs:
        tile_id = os.path.basename(tile_dir)
        parquet_path = os.path.join(
            tile_dir,
            color_type,
            "intersection_polygons",
            f"{tile_id}_intersection.parquet"
        )
        if os.path.exists(parquet_path):
            try:
                gdf = gpd.read_parquet(parquet_path)
                intersection_gdfs.append(gdf)
                logging.info(f"Loaded {parquet_path}: {len(gdf)} polygons")
            except Exception as e:
                logging.error(f"Error loading {parquet_path}: {str(e)}")
    
    if intersection_gdfs:
        # Combine all intersection polygons
        combined_gdf = gpd.GeoDataFrame(pd.concat(intersection_gdfs, ignore_index=True))
        logging.info(f"\nTotal intersection polygons: {len(combined_gdf)}")
        
        # Save combined results
        output_dir = os.path.join(
            os.path.dirname(tile_dirs[0]),  # Use the parent directory of the first tile
            f"intersection_polygons_{color_type}_{grid_size}x{grid_size}"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        combined_gdf.to_parquet(os.path.join(output_dir, "intersection_polygons.parquet"))
        combined_gdf.to_file(os.path.join(output_dir, "intersection_polygons.shp"))
        logging.info(f"Saved intersection polygons to {output_dir}")
    else:
        logging.warning("No intersection polygons found to merge")

def main():
    base_dir = "/home/teulade/dataset_download/shapefiles_copy/"
    year = 2023
    
    # Rerun merging with 20-pixel threshold
    rerun_merging(base_dir, year, min_pixels=20)

if __name__ == "__main__":
    main() 