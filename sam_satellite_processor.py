import matplotlib.pyplot as plt
import rasterio
import numpy as np
from segment_anything import SamPredictor, sam_model_registry

import cv2
import os
import geopandas as gpd
from tqdm import tqdm


import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

from shapely.geometry import Polygon
from rasterio.transform import Affine
from pyproj import Transformer


mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=150,
    pred_iou_thresh=0.6,
    stability_score_thresh=0.6,

    crop_nms_thresh = 0,
    crop_overlap_ratio=1,
    crop_n_layers=1,
    min_mask_region_area=50,  # Requires open-cv to run post-processing
)


def get_masks(image, mask_generator: SamAutomaticMaskGenerator):
    return mask_generator.generate(image)



def get_georeferenced_polygons(masks, transformer: Transformer, transform, crs):
    # Extraire les polygones géoréférencés
    georeferenced_data = []
    # Extraire et afficher les points du polygone pour chaque masque
    for i, mask_data in enumerate(masks):
        mask = mask_data['segmentation']  # Accéder au masque binaire
        confidence = mask_data.get('predicted_iou', 1.0)  # Get confidence, default to 1.0 if missing
        mask = mask.astype(np.uint8) * 255  # Convertir le masque en 0 et 255 pour OpenCV

        # Trouver les contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Afficher les coordonnées des points du polygone pour chaque contour
        for contour in contours:
            if len(contour) >= 3:  # Check if there are at least 3 points

                polygon_points = contour.reshape(-1, 2)  # Transformer en format (N, 2) pour obtenir les points x, y

                # Convertir chaque point en coordonnées géographiques
                geo_points = []
                for x, y in polygon_points:
                    # Transformer les coordonnées de pixel en coordonnées géoréférencées
                    lon, lat = rasterio.transform.xy(transform, y, x, offset='center')
                    geo_points.append((lon, lat))

                # Transformer le polygone en EPSG:4326 si ce n'est pas le CRS d'origine
                geo_polygon = Polygon(geo_points)
                if crs != "EPSG:4326":
                    geo_polygon = Polygon([transformer.transform(x, y) for x, y in geo_polygon.exterior.coords])
                
                georeferenced_data.append({
                    'geometry': geo_polygon,
                    'confidence': confidence
                })

    return georeferenced_data

def get_georeferenced_polygons_from_image(path, mask_generator: SamAutomaticMaskGenerator):
    # Charger l'image satellite GeoTIFF
    with rasterio.open(path) as src:
        image = src.read()  # Lire toutes les bandes de l'image
        crs = src.crs  # Système de coordonnées de l'image
        transform = src.transform  # Transform matrix for pixel to CRS
            # Initialiser une liste pour les bandes normalisées
        #normalized_bands = [cumulative_count_cut(band) for band in image]

        # Si nécessaire, réduire à une image RGB ou choisir des bandes spécifiques
        #image = np.stack([image[1], image[2], image[0]], axis=-1)  # Exemple avec bandes RGB

    # Normaliser l'image pour qu'elle soit compatible avec les modèles d'apprentissage profond
    image = image.transpose(1, 2, 0)
    image = (image / 255.0).astype(np.float32)
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return get_georeferenced_polygons(get_masks(image, mask_generator), transformer, transform, crs)





def process_tile_all_quarters(tile_id, n_samples=20, random_seed=42):
    # Generate all possible quadrant combinations
    all_quadrants = [(row + 1, col + 1) for row in range(10) for col in range(10)]
    
    # Randomly select n_samples quadrants - same selection for all quarters
    np.random.seed(random_seed)
    selected_quadrants = np.random.choice(len(all_quadrants), size=n_samples, replace=False)
    selected_quadrants = [all_quadrants[i] for i in selected_quadrants]
        
    # Create log file for the tile
    base_tile_path = f"/home/teulade/images/Sentinel-2_mosaic_2022/{tile_id}"
    os.makedirs(base_tile_path, exist_ok=True)
    log_path = f"{base_tile_path}/selected_quadrants.txt"
    
    # Write only the paths to log file
    with open(log_path, 'w') as log_file:
        for quarter in range(1, 5):
            for row, col in selected_quadrants:
                path = f"Sentinel-2_mosaic_2022_Q{quarter}_{tile_id}_0_0/split_images_100/output_nrg_q{row}_{col}.tif"
                log_file.write(f"{path}\n")

    print(f"\nProcessing tile {tile_id}")
    print(f"Selected quadrants: {selected_quadrants}")
    
    # Process each quarter with the same selected quadrants
    for quarter in range(1, 5):
        base_path = f"/home/teulade/images/Sentinel-2_mosaic_2022/{tile_id}/Sentinel-2_mosaic_2022_Q{quarter}_{tile_id}_0_0"
        
        # Skip if directory doesn't exist
        if not os.path.exists(base_path):
            print(f"Skipping {tile_id} Q{quarter} - directory not found")
            continue
        
        print(f"Processing Q{quarter}")
        georeferenced_polygons = []
        
        # Process selected quadrants
        with tqdm(total=n_samples, desc=f"Processing {tile_id} Q{quarter}", unit="quadrant") as pbar:
            for row, col in selected_quadrants:
                path = f"{base_path}/split_images_100/output_nrg_q{row}_{col}.tif"
                if os.path.exists(path):
                    try:
                        georeferenced_polygons.extend(get_georeferenced_polygons_from_image(path, mask_generator_2))
                    except Exception as e:
                        print(f"Error processing {path}: {str(e)}")
                pbar.update(1)
        
        print(f"Found {len(georeferenced_polygons)} polygons for Q{quarter}")
        
        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame(georeferenced_polygons, crs="EPSG:4326")
        
        # Create output directories if they don't exist
        os.makedirs(f"{base_path}/shapefiles", exist_ok=True)
        
        # Save to Parquet
        gdf.to_parquet(f"{base_path}/polygons.parquet")
        
        # Save to Shapefile
        output_shapefile = f"{base_path}/shapefiles/polygons.shp"
        gdf.to_file(output_shapefile, driver='ESRI Shapefile')


# List of tile IDs
tile_ids = [
    "30UVU", "30TXT", "30TYR", "31TCJ", 
    "31TFJ", "31TGL", "31TDM", "31UGP", 
    "31UDP", "31UDR"
]

# Process all tiles and quarters
for tile_id in tile_ids:
    process_tile_all_quarters(tile_id, n_samples=20, random_seed=42)