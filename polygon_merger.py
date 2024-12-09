import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
import rasterio
from tqdm import tqdm
import os
from shapely.validation import make_valid  # Add this import

def merge_quarterly_polygons(tile_id):
    print(f"\nProcessing tile {tile_id}")
    
    # Create merge directory
    base_path = f"/home/teulade/images/Sentinel-2_mosaic_2022/{tile_id}"
    merge_path = os.path.join(base_path, "merge")
    os.makedirs(merge_path, exist_ok=True)
    
    # Get pixel area from any available quarter's B02.tif
    pixel_area = None
    for quarter in range(1, 5):
        b02_path = f"{base_path}/Sentinel-2_mosaic_2022_Q{quarter}_{tile_id}_0_0/B02.tif"
        if os.path.exists(b02_path):
            with rasterio.open(b02_path) as src:
                transform = src.transform
                pixel_area = abs(transform[0] * transform[4])
            break
    
    if pixel_area is None:
        print(f"Could not find B02.tif for tile {tile_id}")
        return
    
    # Load quarterly parquet files
    file_paths = []
    for quarter in range(1, 5):
        parquet_path = f"{base_path}/Sentinel-2_mosaic_2022_Q{quarter}_{tile_id}_0_0/polygons.parquet"
        if os.path.exists(parquet_path):
            file_paths.append(parquet_path)
    
    if not file_paths:
        print(f"No parquet files found for tile {tile_id}")
        return
    
    # Read and merge GeoDataFrames
    geodfs = [gpd.read_parquet(file).to_crs("EPSG:32632") for file in file_paths]
    combined_gdf = gpd.GeoDataFrame(pd.concat(geodfs, ignore_index=True), crs=geodfs[0].crs)
    
    # Fix invalid geometries
    filtered_gdf = combined_gdf[combined_gdf.geometry.area / pixel_area >= 100]
    combined_gdf['geometry'] = combined_gdf['geometry'].apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)
    print(f"After merging, number of segments: {len(combined_gdf)}")
    
    # Filter and process polygons
    filtered_gdf = combined_gdf[combined_gdf.geometry.area / pixel_area >= 100]
    print(f"After removing segments smaller than 100 pixels, number of segments: {len(filtered_gdf)}")
    
    filtered_gdf["area"] = filtered_gdf.geometry.area
    filtered_gdf = filtered_gdf.sort_values(by="area").reset_index(drop=True)
    
    # Process intersections
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
    result_gdf = gpd.GeoDataFrame(
        geometry=[item['geometry'] for item in final_geometries if isinstance(item['geometry'], Polygon)],
        data={'confidence': [item['confidence'] for item in final_geometries if isinstance(item['geometry'], Polygon)]},
        crs=filtered_gdf.crs
    )
    
    # Save results
    print(f"\nStatistics for tile {tile_id}:")
    print(f"- Polygons without significant intersection: {count_no_intersection}")
    print(f"- Polygons reduced to intersection (>25%): {count_intersection}")
    print(f"- Skipped polygons (already processed): {count_skipped}")
    print(f"- Remaining polygons: {len(result_gdf)}")
    
    result_gdf.to_file(os.path.join(merge_path, "merge.shp"))
    result_gdf.to_parquet(os.path.join(merge_path, "polygons.parquet"))

# List of tile IDs
tile_ids = [
    "30UVU", "30TXT", "30TYR", "31TCJ", 
    "31TFJ", "31TGL", "31TDM", "31UGP", 
    "31UDP", "31UDR"
]

# Process all tiles
for tile_id in tile_ids:
    merge_quarterly_polygons(tile_id)