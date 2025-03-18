import os
import sys
import zipfile
import glob
from pathlib import Path

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

try:
    from src.sentinel_preprocessing import preprocess_imagery
    from src.sam_satellite_processor import segment_satellite_imagery
    from src.polygon_merger import merge_overlapping_segments, concat_polygons
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print(f"Current Python path: {sys.path}")
    print(f"Project root: {project_root}")
    print("\nPlease make sure you have:")
    print("1. Installed all requirements from requirements.txt")
    print("2. The project structure is correct with a 'src' directory containing the required modules")
    print("3. You're running the script from the project root directory")
    sys.exit(1)

def unzip_sentinel_products(base_dir):
    """Unzip all Sentinel product zip files in the given directory."""
    print(f"Checking Sentinel products in {base_dir}")
    zip_files = glob.glob(os.path.join(base_dir, "**/*.zip"), recursive=True)
    
    for zip_path in zip_files:
        # Check if already unzipped
        output_dir = os.path.splitext(zip_path)[0]
        if os.path.exists(output_dir) and os.path.isdir(output_dir):
            print(f"Directory {output_dir} already exists, skipping unzip...")
            continue
            
        print(f"Unzipping {zip_path} to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            first_dir = zip_ref.namelist()[0].split('/')[0]
            zip_ref.extractall(output_dir)
            nested_dir = os.path.join(output_dir, first_dir)
            if os.path.exists(nested_dir) and os.path.isdir(nested_dir):
                for item in os.listdir(nested_dir):
                    os.rename(os.path.join(nested_dir, item), os.path.join(output_dir, item))
                os.rmdir(nested_dir)
    
    return output_dir

def is_preprocessing_done(quarter_dir):
    """Check if preprocessing has already been done."""
    # Check for color composite and tiles
    color_composite = os.path.join(quarter_dir, "nrg", "sentinel_composite.tif")
    tiles_dir = os.path.join(quarter_dir, "nrg", "tiles_10x10")
    return os.path.exists(color_composite) and os.path.exists(tiles_dir)

def is_sam_done(quarter_dir):
    """Check if SAM segmentation has already been done."""
    # Check for SAM output files
    parquet_file = os.path.join(quarter_dir, "nrg", "polygons_10x10.parquet")
    shapefile_dir = os.path.join(quarter_dir, "nrg", "shapefiles_10x10")
    return os.path.exists(parquet_file) and os.path.exists(shapefile_dir)

def is_merging_done(tile_dir, quarter):
    """Check if polygon merging has already been done."""
    # Check for merged polygon files
    tile_id = os.path.basename(tile_dir)
    parquet_file = os.path.join(tile_dir, "nrg", "intersection_polygons", f"{tile_id}_intersection.parquet")
    shapefile = os.path.join(tile_dir, "nrg", "intersection_polygons", f"{tile_id}_intersection.shp")
    return os.path.exists(parquet_file) and os.path.exists(shapefile)

def setup_sam_model():
    """Initialize and return the SAM model and mask generator."""
    # Get the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sam_checkpoint = os.path.join(project_root, "models", "sam_vit_h_4b8939.pth")
    model_type = "vit_h"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=7,
        pred_iou_thresh=0.6,
        stability_score_thresh=0.6,
        crop_nms_thresh=0,
        crop_overlap_ratio=1,
        crop_n_layers=1,
        min_mask_region_area=50,
    )
    
    return mask_generator

def process_sentinel_products(base_dir, year, n_samples=None):
    """Process all Sentinel products in the given directory."""
    # Get all tile directories
    tile_dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]
    
    # Setup SAM model
    mask_generator = setup_sam_model()
    
    for tile_dir in tile_dirs:
        tile_id = os.path.basename(tile_dir)
        print(f"\nProcessing tile: {tile_id}")
        
        # Process each quarter
        for quarter in range(1, 5):
            # Construct the correct quarter directory path
            quarter_dir = os.path.join(tile_dir, f"Sentinel-2_mosaic_{year}_Q{quarter}_{tile_id}_0_0")
            if not os.path.exists(quarter_dir):
                print(f"Quarter {quarter} not found for tile {tile_id}, skipping...")
                continue
                
            print(f"\nProcessing quarter {quarter}")
            
            # Step 1: Preprocess
            if is_preprocessing_done(quarter_dir):
                print("Step 1: Preprocessing already done, skipping...")
            else:
                print("Step 1: Preprocessing imagery...")
                preprocess_imagery(quarter_dir)
            
            # Step 2: SAM Segmentation
            if is_sam_done(quarter_dir):
                print("Step 2: SAM segmentation already done, skipping...")
            else:
                print("Step 2: Running SAM segmentation...")
                segment_satellite_imagery(quarter_dir, mask_generator, n_samples=n_samples)
            
            # Step 3: Merge polygons
            if is_merging_done(tile_dir, quarter):
                print("Step 3: Polygon merging already done, skipping...")
            else:
                print("Step 3: Merging polygons...")
                merge_overlapping_segments(tile_dir, [quarter], year)
    
    # Final step: Concatenate all polygons
    print("\nConcatenating all polygons...")
    concat_polygons(tile_dirs, color_type='nrg', grid_size=10, polygons_name="all_polygons")

if __name__ == "__main__":
    base_dir = "/home/jules/Projects/dataset_download/downloads/test_run/2023/"
    year = 2023  # Extract year from base_dir or specify it explicitly
    
    # First unzip all products (if needed)
    unzip_sentinel_products(base_dir)
    
    # Then process all products with n_samples=5
    process_sentinel_products(base_dir, year, n_samples=5) 