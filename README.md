# Satellite SAM Segmentation

This project uses the Segment Anything Model (SAM) to segment satellite imagery from Sentinel products.

## Setup

### 1. Environment Setup

Create and activate a conda environment:

```bash
conda create -n sentinel python=3.11
conda activate sentinel
```

### 2. Install Dependencies

Install the required packages using conda and pip:

```bash
# Install core dependencies via conda
conda install -c conda-forge rasterio geopandas shapely tqdm matplotlib
conda install pytorch torchvision -c pytorch

# Install remaining dependencies via pip
pip install segment-anything opencv-python pillow
```

Or install all dependencies at once using pip:

```bash
pip install -r requirements.txt
```

### 3. Download SAM Model

The project requires the SAM model checkpoint. Download it using:

```bash
# Create models directory
mkdir -p models

# Download the SAM model checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O models/sam_vit_h_4b8939.pth
```

The model file is approximately 2.4GB.

## Usage

The project provides scripts to process Sentinel satellite imagery:

1. Place your Sentinel product zip files in a directory (e.g., `/path/to/sentinel/products/`)
2. Run the processing script:

```bash
python scripts/process_sentinel_products.py
```

#### Key Arguments

- `--base_dir`: Path to the directory containing tiles (required). This is the base directory where the Sentinel product tiles are located.
- `--overwrite`: Optional flag. If provided, existing files will be overwritten during processing. If not provided, the script will skip processing for files that already exist.

The script will:
1. Unzip all Sentinel product zip files
2. Preprocess the imagery
3. Run SAM segmentation
4. Merge and concatenate the resulting polygons

## Project Structure

```
satellite-sam-segmentation/
├── models/
│   └── sam_vit_h_4b8939.pth    # SAM model checkpoint
├── scripts/
│   └── process_sentinel_products.py
├── src/
│   ├── sentinel_preprocessing.py
│   ├── sam_satellite_processor.py
│   └── polygon_merger.py
└── requirements.txt
```

## Notes

- The SAM model checkpoint is required and must be placed in the `models/` directory
- The script expects Sentinel product zip files in the specified input directory
- Processing is done per tile and quarter
- Results are saved in the same directory structure as the input

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

## License
[Your chosen license]

## Contributors
[Your name/organization]