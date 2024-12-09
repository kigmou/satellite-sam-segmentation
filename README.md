# Satellite Image Segmentation with SAM

## Overview
This project implements an automated pipeline for segmenting Sentinel-2 satellite imagery using Meta's Segment Anything Model (SAM). It processes multiple satellite tiles across quarterly periods, generates segmentation masks, and merges the results into clean, georeferenced polygons.

## Features
- Automated processing of Sentinel-2 satellite imagery
- Multi-quarter image segmentation using SAM
- Georeferenced polygon generation
- Polygon merging with intersection handling
- Support for batch processing multiple tiles

## Prerequisites
- CUDA-capable GPU
- Python 3.8+
- SAM checkpoint file (`sam_vit_h_4b8939.pth`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/teulade/satellite_segmentation.git
cd satellite_segmentation
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the SAM checkpoint file and place it in the project root:
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Usage

### 1. Image Preparation
Processes raw Sentinel-2 bands and prepares them for segmentation:
```bash
python sentinel_preprocessing.py
```
This step:
- Reads bands B02, B03, B08
- Creates RGB composites
- Splits images into 10x10 grids

### 2. SAM Segmentation
Runs the SAM model on prepared images:
```bash
python sam_satellite_processor.py
```
This step:
- Processes each tile with SAM
- Converts masks to georeferenced polygons
- Saves results as quarterly parquet/shapefile

### 3. Polygon Merging
Merges results from all quarters:
```bash
python polygon_merger.py
```
This step:
- Combines quarterly results
- Filters small polygons (<100 pixels)
- Handles overlapping segments
- Generates final clean polygons

## Project Structure

```
satellite-sam-segmentation/
│
├── sam_satellite_processor.py    # Main SAM processing
├── polygon_merger.py            # Merging quarterly results
├── sentinel_preprocessing.py    # Image preparation
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

## Configuration

### Supported Tile IDs

```python
tile_ids = [
"30UVU", "30TXT", "30TYR", "31TCJ",
"31TFJ", "31TGL", "31TDM", "31UGP",
"31UDP", "31UDR"
]
```

### SAM Parameters
- Model: `vit_h`
- Points per side: 150
- IoU threshold: 0.6
- Stability score threshold: 0.6
- Minimum mask region area: 50 pixels

## Output Structure
```
/home/teulade/images/Sentinel-2_mosaic_2022/
└── {tile_id}/
├── Sentinel-2_mosaic_2022_Q{1,2,3,4}{tile_id}_0_0/
│ ├── polygons.parquet
│ └── shapefiles/
│ └── polygons.shp
└── merge/
├── merge.shp
└── polygons.parquet
```


## Processing Pipeline

1. **Quarterly Processing**
   - Each quarter (Q1-Q4) is processed independently
   - Images are split into 100 tiles (10x10 grid)
   - SAM generates segmentation masks
   - Masks are converted to georeferenced polygons

2. **Merging Process**
   - Combines all quarterly results
   - Filters out segments smaller than 100 pixels
   - Handles overlapping polygons
   - Produces final clean polygon set

## Performance

Typical processing metrics:
- Input: 4 quarters × 100 tiles per quarter
- Initial segments: ~192,328 total
- Final output: ~60,607 clean polygons after merging

## Notes
- Ensure sufficient disk space for intermediate files
- GPU memory requirements depend on image size
- Processing time varies based on number of tiles and hardware

## License
[Your chosen license]

## Contributors
[Your name/organization]