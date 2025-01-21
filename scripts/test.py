import os
import sys


sys.path.append("..")
from src.sentinel_preprocessing import preprocess_imagery
from src.sentinel_preprocessing import build_rgb_from_JP2
from src.sentinel_preprocessing import split_image_in_tiles
from src.sam_satellite_processor import segment_satellite_imagery


from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

sam_checkpoint = "/home/teulade/satellite-sam-segmentation/models/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.6,
    stability_score_thresh=0.6,
    crop_nms_thresh=0,
    crop_overlap_ratio=1,
    crop_n_layers=1,
    min_mask_region_area=50,
)

segment_satellite_imagery("/home/teulade/images/images_pleiades/20160609T104841_PLD_SEN_PMS_1816164101-001/", mask_generator)
