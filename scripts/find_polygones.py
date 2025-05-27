import os
import math
from itertools import product
from typing import Any, Generator, List, Tuple

import numpy as np
import torch
import geopandas as gpd
import rasterio
from matplotlib import pyplot as plt
import cv2
from shapely.geometry import Polygon
from pycocotools import mask as mask_utils
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.amg import (
    MaskData,
    generate_crop_boxes,
    mask_to_rle_pytorch,
    uncrop_masks,
    is_box_near_crop_edge,
    uncrop_boxes_xyxy,
    uncrop_points,
    generate_crop_boxes,
    batched_mask_to_box,
    calculate_stability_score,
    batch_iterator
)
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def sam_segment_from_centroids_no_overlap(raster_path,centroid_shp_path,predictor: SamPredictor,min_area=10,output_path=None,):
    gdf = gpd.read_file(centroid_shp_path)

    with rasterio.open(raster_path) as src:
        transform = src.transform
        crs = src.crs
        image = src.read()
        image = np.transpose(image, (1, 2, 0))
        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            image = 255 * (image - image.min()) / (image.max() - image.min())
            image = image.astype(np.uint8)

    # conversion des points shapefile en coordonnées pixel
    point_grids = []
    for pt in gdf.geometry:
        row, col = rasterio.transform.rowcol(transform, pt.x, pt.y)
        point_grids.append([col, row])  # x = col, y = row
    point_grids = np.array(point_grids)
    print(f"{len(point_grids)} centroïdes détectés")

    orig_size = image.shape[:2]
    crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, 0, 512 / 2000
        )
    
    data = MaskData()
    for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
        crop_data = _process_crop(image, crop_box, layer_idx, orig_size, point_grids, predictor)
        data.cat(crop_data)
    
    if len(crop_boxes) > 1:
        scores = 1 / box_area(data["crop_boxes"])
        scores = scores.to(data["boxes"].device)
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            scores,
            torch.zeros_like(data["boxes"][:, 0]),
            iou_threshold=0.7,
        )
        data.filter(keep_by_nms)

    data.to_numpy()
    return data,crs,transform

def masks_to_polygons(masks, transform, crs, output_path=None, scores=None, ious=None, stabilities=None):
    all_polygons = []
    for i, mask in enumerate(masks):
        mask_bin = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) >= 3:
            
                geo_points = []
                for x, y in contour.reshape(-1, 2):
                    lon, lat = rasterio.transform.xy(transform, y, x, offset='center')
                    geo_points.append((lon, lat))
                poly = Polygon(geo_points)
                if poly.is_valid:
                    props = {"geometry": poly}
                    all_polygons.append(props)
    if not all_polygons:
        print("Aucun polygone valide généré.")
        return gpd.GeoDataFrame(columns=["geometry"], crs=crs)
    gdf = gpd.GeoDataFrame(all_polygons, crs=crs)
    if output_path:
        gdf.to_file(output_path)
    return gdf
 
def setup_sam_model():
    sam_checkpoint = os.path.join("models", "sam_vit_h_4b8939.pth")
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return SamPredictor(sam)

def _process_crop(
    image: np.ndarray,
    crop_box: List[int],
    crop_layer_idx: int,
    orig_size: Tuple[int, ...],
    point_grids: List[np.ndarray],
    predictor: SamPredictor
) -> MaskData:
    x0, y0, x1, y1 = crop_box
    cropped_im = image[y0:y1, x0:x1, :]
    cropped_im_size = cropped_im.shape[:2]
    predictor.set_image(cropped_im)

    # Get points for this crop
    # points_scale = np.array(cropped_im_size)[None, ::-1]
    # points_for_image = point_grids[crop_layer_idx] * points_scale
    points_for_image = point_grids[
        (point_grids[:, 0] >= x0) & (point_grids[:, 0] < x1) &
        (point_grids[:, 1] >= y0) & (point_grids[:, 1] < y1)
    ]

    points_for_image = points_for_image - np.array([x0, y0])
    # Generate masks for this crop in batches
    data = MaskData()
    for (points,) in batch_iterator(64, points_for_image):
        batch_data = _process_batch(predictor,points, cropped_im_size, crop_box, orig_size)
        data.cat(batch_data)
        del batch_data
    predictor.reset_image()

    # Remove duplicates within this crop.
    keep_by_nms = batched_nms(
        data["boxes"].float(),
        data["iou_preds"],
        torch.zeros_like(data["boxes"][:, 0]),  # categories
        iou_threshold=0.7,
    )
    data.filter(keep_by_nms)

    # Return to the original image frame
    data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
    data["points"] = uncrop_points(data["points"], crop_box)
    data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

    return data



def _process_batch(
        predictor: SamPredictor,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        orig_h, orig_w = orig_size

        # Run model on this batch
        transformed_points = predictor.transform.apply_coords(points, im_size)
        in_points = torch.as_tensor(transformed_points, device=predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        masks, iou_preds, _ = predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=True,
            return_logits=True,
        )

        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
        )
        del masks
        pred_iou_thresh = 0.88

        keep_mask = data["iou_preds"] > pred_iou_thresh
        data.filter(keep_mask)

        data["stability_score"] = calculate_stability_score(
            data["masks"], predictor.model.mask_threshold, 1.0
        )
        keep_mask = data["stability_score"] >= 0.9
        data.filter(keep_mask)

        data["masks"] = data["masks"] > predictor.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        return data

        

def view(gdf: gpd.GeoDataFrame):
    print("Nombre de polygones :", len(gdf))
    gdf = gdf.copy()
    gdf["color_id"] = range(len(gdf))
    ax = gdf.plot(column="color_id", cmap='tab20', edgecolor='k', linewidth=0.8, figsize=(8, 8))
    plt.title("Polygones segmentés")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

if __name__ == "__main__":
    predictor = setup_sam_model()
    output_path = "results/output_true.shp"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data, crs, trans = sam_segment_from_centroids_no_overlap(
        r"2023\31UDQ\Sentinel-2_mosaic_2023_Q1_31UDQ_0_0\nrg\tiles_10x10\tile_1_1.tif",
        r"2023\31UDQ\Sentinel-2_mosaic_2023_Q1_31UDQ_0_0\nrg\tiles_10x10\tile_1_1_centroids.shp",
        predictor,
        output_path=output_path
    )
    print("Nombre de masques :", len(data["rles"]))
    
    rles = [mask_utils.frPyObjects(rle, *data["rles"][0]["size"]) for rle in data["rles"]]

    rles_flat = [r[0] if isinstance(r, list) else r for r in rles]

    binary_masks = mask_utils.decode(rles_flat)

    binary_masks_list = [binary_masks[:, :, i] for i in range(binary_masks.shape[2])]

    # Création du shapefile
    gdf = masks_to_polygons(
        binary_masks_list,
        transform=trans,
        crs=crs,
        output_path=output_path
    )

    print("Shapefile écrit avec", len(gdf), "polygones.")
