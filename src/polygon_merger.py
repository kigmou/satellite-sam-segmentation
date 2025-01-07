import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
import rasterio
from tqdm import tqdm
import os
from shapely.validation import make_valid  # Add this import

def get_pixel_area(tile_path, quarters):
      # Get pixel area from any available quarter's B02.tif
    pixel_area = None
    for quarter in quarters:
        b02_path = f"{tile_path}/Sentinel-2_mosaic_2022_Q{quarter}_{os.path.basename(tile_path)}_0_0/B02.tif"
        if os.path.exists(b02_path):
            with rasterio.open(b02_path) as src:
                transform = src.transform
                return abs(transform[0] * transform[4])
            
    return None


def create_intersection_gdf(filtered_gdf):
    """
    Create a GeoDataFrame of merged geometries based on intersections.

    Args:
        filtered_gdf: GeoDataFrame containing filtered polygons.

    Returns:
        result_gdf: GeoDataFrame of merged geometries.
    """
    spatial_index = filtered_gdf.sindex
    final_geometries = []
    processed_indices = set()
    
    # Counters
    count_no_intersection = 0
    count_intersection = 0
    count_skipped = 0
    
    # Process polygons
    for i, row in tqdm(enumerate(filtered_gdf.iterrows()), total=len(filtered_gdf), desc="Processing Polygons"):
        if i in processed_indices:
            count_skipped += 1
            continue
            
        geom = row[1].geometry
        confidence = row[1].confidence
        processed_indices.add(i)
        
        possible_matches_index = list(spatial_index.intersection(geom.bounds))
        found_intersection = False
        
        for j in possible_matches_index:
            if j in processed_indices:
                continue
            
            other_row = filtered_gdf.iloc[j]
            other_geom = other_row.geometry
            other_confidence = other_row.confidence
            
            intersection = geom.intersection(other_geom)
            if not intersection.is_empty and intersection.area > 0.25 * geom.area:
                if not found_intersection:
                    final_geometries.append({
                        'geometry': make_valid(intersection),
                        'confidence': max(confidence, other_confidence)
                    })
                found_intersection = True
                processed_indices.add(j)
                count_intersection += 1
        
        if not found_intersection:
            final_geometries.append({
                'geometry': geom,
                'confidence': confidence
            })
            count_no_intersection += 1
    
    # Create final GeoDataFrame
    intersection_gdf = gpd.GeoDataFrame(
        geometry=[item['geometry'] for item in final_geometries if isinstance(item['geometry'], Polygon)],
        data={'confidence': [item['confidence'] for item in final_geometries if isinstance(item['geometry'], Polygon)]},
        crs=filtered_gdf.crs
    )
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"- Polygons without significant intersection: {count_no_intersection}")
    print(f"- Polygons reduced to intersection (>25%): {count_intersection}")
    print(f"- Skipped polygons (already processed): {count_skipped}")
    print(f"- Remaining polygons: {len(intersection_gdf)}")
    
    return intersection_gdf

def merge_overlapping_segments(tile_path, quarters):
    """
    Merge quarterly polygons for a single tile from a list of quarters.
    
    Args:
        tile_path: Full path to the tile directory
        quarters: List of quarters to process
    """
    tile_id = os.path.basename(tile_path)
    print(f"\nProcessing tile {tile_id} for quarters: {quarters}")
    

    
    # Get pixel area from any available quarter's B02.tif
    pixel_area = get_pixel_area(tile_path, quarters)
    
    if pixel_area is None:
        print(f"Could not find B02.tif for tile {tile_id}")
        return
    
    # Load quarterly parquet files
    geodfs = [gpd.read_parquet(f"{tile_path}/Sentinel-2_mosaic_2022_Q{quarter}_{tile_id}_0_0/polygons.parquet").to_crs("EPSG:32632") for quarter in quarters]
    
    if not geodfs:
        print(f"No parquet files found for tile {tile_id}")
        return
    
    # Concatenate GeoDataFrames
    combined_gdf = gpd.GeoDataFrame(pd.concat(geodfs, ignore_index=True), crs="EPSG:32632")
    
    # Fix invalid geometries
    combined_gdf['geometry'] = combined_gdf['geometry'].apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)
    print(f"After merging, number of segments: {len(combined_gdf)}")
    
    # Filter and process polygons
    filtered_gdf = combined_gdf[combined_gdf.geometry.area / pixel_area >= 100]
    print(f"After removing segments smaller than 100 pixels, number of segments: {len(filtered_gdf)}")
    
    filtered_gdf["area"] = filtered_gdf.geometry.area
    filtered_gdf = filtered_gdf.sort_values(by="area").reset_index(drop=True)
    
    # Create result GeoDataFrame using the new function
    intersection_gdf = create_intersection_gdf(filtered_gdf)
    
        # Create merge directory
    merge_path = os.path.join(tile_path, "intersection_polygons")
    os.makedirs(merge_path, exist_ok=True)
    
    intersection_gdf.to_file(os.path.join(merge_path, f"{tile_id}_intersection.shp"))
    intersection_gdf.to_parquet(os.path.join(merge_path, f"{tile_id}_intersection.parquet"))

def concat_polygons(tile_paths, polygons_name = "all_polygons"):
    # Initialize list to store GeoDataFrames
    gdfs = []

    # Load all parquet files
    for tile_path in tqdm(tile_paths, desc="Loading tiles"):
        parquet_path = os.path.join(tile_path, "intersection_polygons", f"{os.path.basename(tile_path)}_intersection.shp")
        if os.path.exists(parquet_path):
            try:
                gdf = gpd.read_parquet(parquet_path)
                gdfs.append(gdf)
                print(f"Loaded {parquet_path}: {len(gdf)} polygons")
            except Exception as e:
                print(f"Error loading {parquet_path}: {str(e)}")
        else:
            print(f"No parquet file found for {parquet_path}")

    if gdfs:
        # Combine all GeoDataFrames
        combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
        print(f"\nTotal number of polygons: {len(combined_gdf)}")
        
        # Create output directory if it doesn't exist
        output_dir = f"/home/teulade/images/Sentinel-2_mosaic_2022/{polygons_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save combined results
        combined_gdf.to_parquet(f"{output_dir}/{polygons_name}.parquet")
        combined_gdf.to_file(f"{output_dir}/{polygons_name}.shp")
        
        print(f"\nSaved merged files to {output_dir}")
    else:
        print("No data found to merge")