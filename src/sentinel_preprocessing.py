import rasterio
from rasterio.windows import Window
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
import os


def cumulative_count_cut(band, min_percentile=2, max_percentile=98):
    """Apply contrast enhancement stretch similar to QGIS"""
    new_band = band[band != 0]
    min_val = np.nanpercentile(new_band, min_percentile)
    max_val = np.nanpercentile(new_band, max_percentile)
    #print(min_val,max_val)
    return (band - min_val) / (max_val - min_val) * 255
    
# Définir les chemins vers les fichiers de bandes

def build_rgb_from_bands(sentinel_path):
    b3_path = sentinel_path + "B02.tif"   # Chemin vers la bande verte
    b4_path = sentinel_path + "B03.tif"    # Chemin vers la bande rouge
    b8_path = sentinel_path + "B08.tif"  # Chemin vers la bande proche infrarouge
    output_path = sentinel_path + f"{os.path.basename(sentinel_path)}_nrg.tif"    # Chemin vers l'image RGB
    # Charger l'image satellite GeoTIFF
    with rasterio.open(b3_path) as b3, rasterio.open(b4_path) as b4, rasterio.open(b8_path) as b8:
            # Lire les données de chaque bande
        green = b3.read(1)  # B03 pour le canal vert
        red = b4.read(1)    # B04 pour le canal rouge
        nir = b8.read(1)  # B02 pour le canal proche infrarouge

        transform = b3.transform  # Matrice de transformation affine
        crs = b3.crs  # Système de coordonnées de l'image

            # Initialiser une liste pour les bandes normalisées
        # Si nécessaire, réduire à une image RGB ou choisir des bandes spécifiques
        image = np.stack([cumulative_count_cut(nir), cumulative_count_cut(red), cumulative_count_cut(green)], axis=0)  # Exemple avec bandes RGB
            # Définir les métadonnées pour le nouveau fichier
        profile = b3.profile  # Utiliser les métadonnées de l'une des bandes comme référence
        profile.update(count=3)  # Mettre à jour le nombre de canaux à 3

        # Enregistrer l'image RGB
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(image)

    print("Image RGB créée et enregistrée sous " + output_path) 


def split_rgb_image_in_100(sentinel_path, output_rgb_path=None):
    if output_rgb_path is None:
        output_rgb_path = sentinel_path + f"{os.path.basename(sentinel_path)}_nrg.tif"

    # Ouvrir l'image et obtenir ses dimensions
    with rasterio.open(output_rgb_path) as src:
        width = src.width
        height = src.height

        # Définir les dimensions pour chaque sous-image
        quarter_width = width // 10
        quarter_height = height // 10

        # Préparer la liste des fichiers de sortie
        output_files = []

        # Boucle pour chaque quadrant (4x4 = 16 quadrants)
        for row in range(10):
            for col in range(10):
                # Calculer les coordonnées de chaque fenêtre
                x_offset = col * quarter_width
                y_offset = row * quarter_height
                window = Window(x_offset, y_offset, quarter_width, quarter_height)

                # Créer un nom de fichier pour chaque sous-image
                output_file = sentinel_path + f"split_images_100/output_nrg_q{row + 1}_{col + 1}.tif"
                output_files.append(output_file)

                # Mise à jour du profil pour la sous-image
                profile = src.profile
                profile.update({
                    "width": quarter_width,
                    "height": quarter_height,
                    "transform": rasterio.windows.transform(window, src.transform)
                })

                # Ensure the directory exists
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                # Lire les données de la fenêtre et sauvegarder
                with rasterio.open(output_file, "w", **profile) as dst:
                    dst.write(src.read(window=window))

    print("Découpage terminé. Les sous-images sont enregistrées sous les noms suivants :")
    print(output_files)


def preprocess_sentinel_imagery(sentinel_bands_path):
    """
    Process a single tile by building RGB images and splitting them for a specific quarter.
    
    Args:
        sentinel_path: Full path to the sentinel directory with bands
    """   
    if os.path.exists(sentinel_bands_path):
        print(f"Processing {sentinel_bands_path}")
        
        try:
            build_rgb_from_bands(sentinel_bands_path)
            split_rgb_image_in_100(sentinel_bands_path)
            print(f"Successfully processed {sentinel_bands_path}")
        except Exception as e:
            print(f"Error processing {sentinel_bands_path}: {str(e)}")
    else:
        print(f"Directory not found: {sentinel_bands_path}")