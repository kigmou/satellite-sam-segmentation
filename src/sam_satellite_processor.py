import rasterio
import numpy as np

import cv2
import os
import geopandas as gpd
from tqdm import tqdm

from segment_anything import SamAutomaticMaskGenerator

from shapely.geometry import Polygon
from pyproj import Transformer

    

def get_georeferenced_polygons_from_image(path, mask_generator: SamAutomaticMaskGenerator):
    """
    Extract georeferenced polygons from a satellite GeoTIFF image using a mask generator.
    Args:
        path (str): The file path to the satellite GeoTIFF image.
        mask_generator (SamAutomaticMaskGenerator): An instance of SamAutomaticMaskGenerator used to generate masks.
    Returns:
        list: A list of dictionaries containing georeferenced polygons and their confidence scores. 
              Each dictionary has the following keys:
              - 'geometry' (shapely.geometry.Polygon): The georeferenced polygon in EPSG:4326.
              - 'confidence' (float): The confidence score of the mask.
    """
    # Charger l'image satellite GeoTIFF
    with rasterio.open(path) as src:
        image = src.read()  # Lire toutes les bandes de l'image
        crs = src.crs  # Système de coordonnées de l'image
        transform = src.transform  # Transform matrix for pixel to CRS


    # Normaliser l'image pour qu'elle soit compatible avec le modèle
    image = image.transpose(1, 2, 0)
    image = (image / 255.0).astype(np.float32)
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    masks = mask_generator.generate(image)

    # Extraire les polygones géoréférencés
    georeferenced_data = []
    # Extraire et afficher les points du polygone pour chaque masque
    for i, mask_data in enumerate(masks):
        mask = mask_data['segmentation'].astype(np.uint8) * 255  # Accéder au masque binaire et le convertir en 0 et 255 pour OpenCV
        confidence = mask_data.get('predicted_iou', 1.0)  # Get confidence, default to 1.0 if missing

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





def segment_satellite_imagery(sentinel_path, mask_generator: SamAutomaticMaskGenerator, n_samples=None, random_seed=42):
    """
    Process a tile using the provided mask generator.
    
    Args:
        sentinel_path: Full path to the tile directory
        mask_generator: Initialized SamAutomaticMaskGenerator instance
        n_samples: Number of quadrants to randomly sample. If None, process all quadrants.
        random_seed: Random seed for reproducibility (only used if n_samples is not None)
    """
    # Skip if directory doesn't exist
    if not os.path.exists(sentinel_path):
        print(f"Skipping {sentinel_path} - directory not found")
        return

    # Generate all possible quadrant combinations
    all_quadrants = [(row + 1, col + 1) for row in range(10) for col in range(10)]
    
    # Select quadrants based on n_samples
    if n_samples is not None:
        np.random.seed(random_seed)
        selected_quadrants = np.random.choice(len(all_quadrants), size=n_samples, replace=False)
        selected_quadrants = [all_quadrants[i] for i in selected_quadrants]
        print(f"Selected {n_samples} random quadrants: {selected_quadrants}")
    else:
        selected_quadrants = all_quadrants
        print(f"Processing all {len(selected_quadrants)} quadrants")
   
    print(f"Processing {sentinel_path}")
    georeferenced_polygons = []
    
    # Process selected quadrants
    with tqdm(total=len(selected_quadrants), desc=f"Processing {sentinel_path}", unit="quadrant") as pbar:
        for row, col in selected_quadrants:
            path = f"{sentinel_path}/split_images_100/output_nrg_q{row}_{col}.tif"
            if os.path.exists(path):
                try:
                    georeferenced_polygons.extend(get_georeferenced_polygons_from_image(path, mask_generator))
                except Exception as e:
                    print(f"Error processing {path}: {str(e)}")
            pbar.update(1)
    
    print(f"Found {len(georeferenced_polygons)} polygons for {sentinel_path}")
    
    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(georeferenced_polygons, crs="EPSG:4326")
    
    # Create output directories if they don't exist
    os.makedirs(f"{sentinel_path}/shapefiles", exist_ok=True)
    
    # Save to Parquet
    gdf.to_parquet(f"{sentinel_path}/polygons.parquet")
    
    # Save to Shapefile
    output_shapefile = f"{sentinel_path}/shapefiles/polygons.shp"
    gdf.to_file(output_shapefile, driver='ESRI Shapefile')