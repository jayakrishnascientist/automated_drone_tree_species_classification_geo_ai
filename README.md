# Forest-tree-crown-species-classification-geo-AI



Automated **tree species mapping from high-resolution drone orthomosaics** using **Deep learning foundation models** and translation of species labels to Google Earth for geospatial investigation. 


<img width="612" height="916" alt="overview" src="https://github.com/user-attachments/assets/9b63f18e-4f02-4ae5-a12a-5f8ea9c32bfa" />


This repository provides an automated pipeline aimed at classifying multiple forest tree species using drone data.

The pipeline supports any type of drone data in the orthomosaic.tif extension. In our application, we identified tree species (Acacia or Non-Acacia) in the Tropical Forest using crown polygons extracted from drone imagery. in both version **supervised** and **unsupervised** **classification**

Only two inputs are required (Drone RGB Orthomosaic + Tree crown polygon) automatically processes orthomosaic images, extracts tree crowns, predicts species classes, and generates Google Earth compatible KML maps showing tree species distribution across forest landscapes

The framework integrates **computer vision, geospatial processing, and deep learning** to generate spatial outputs that can be visualized directly in **QGIS or Google Earth**.

---
# Pipeline Overview

1-Orthomosaic imagery and crown polygons are processed through the following workflow:

2-Tree crown cropping from orthomosaic images

3-Deep feature extraction using DINOv2 Vision Transformer

4-Feature clustering and evaluation

5-Asssigning the cluster labels of tree species

6-Automated prediction of species for all crowns

7-Export of labeled polygons to GeoJSON and KML

8-The final output is a species distribution map visualized in Google Earth.

# Dataset Collection and required inputs illustration

<img width="1446" height="810" alt="sanjayvan" src="https://github.com/user-attachments/assets/fa89e2e8-cfa1-4cb9-9c80-95435cf65595" />

# Self-Developed Dataset

Drone orthomosaic images were collected from four survey locations.

Site Number of Crowns

S1-656

S2-717

S3-628

S4-155

Total tree crowns extracted: 2156 crowns

Manual species labeling was performed for: 400 crowns

ClassCount

Acacia= 193

Non-Acacia= 207


# GeoAI Tree Crown Species Mapping Pipeline

## Overview

This pipeline performs tree crown species mapping using drone imagery and crown polygon data. It combines self-supervised feature extraction, unsupervised clustering, and human-in-the-loop labeling to generate a geospatial species distribution map.

The workflow does not require ground truth labels initially and is scalable to large datasets.

---

## Inputs

The pipeline requires the following inputs:

- **Drone Orthomosaic**
  - High-resolution GeoTIFF image generated from drone imagery

- **Crown Polygon GeoJSON**
  - Vector file containing individual tree crown boundaries

---

## Pipeline Steps

### Step 1A: Crown Cropping

- Each crown polygon is used to crop the orthomosaic
- Produces individual tree crown images (GeoTIFFs)

**Output:**
- Cropped crown images (~2,000+ depending on dataset)

---

### Step 1B: Feature Extraction (DINOv2)

- Uses DINOv2 (ViT-B/14, frozen) model
- Extracts a 768-dimensional feature vector for each crown
- Feature vectors are L2-normalized

**Output:**
- Feature matrix (N x 768)

---

### Step 1C: Multi-k Clustering

- Applies K-Means clustering with multiple values of k:
  - k = 2, 4, 6, 8, 10, 12

**Purpose:**
- Identify natural groupings without prior knowledge of species count

**Output:**
- Cluster assignments for each k

---

### Step 1D: K Selection

Clustering results are evaluated using:

- Silhouette Score
- Davies-Bouldin Index
- Elbow Method

These metrics are combined into a ranked table to select the optimal k.

---

### Step 1E: t-SNE Visualization

- Reduces feature space from 768D to 2D
- Helps visualize cluster separability and overlap

**Output:**
- 2D visualization of clusters

---

## Scenario: Unsupervised (Human-in-the-loop)

### Visual Inspection

- Clustered images are organized into folders
- Each folder corresponds to a cluster
- User manually inspects images and assigns a species label

**Output:**
- CSV file mapping:


---

### Step 2A: Load Species Mapping

- Loads the user-defined cluster-to-species mapping
- Assigns species labels to all crowns

---

### Step 2C: Generate polygon_species.csv

- Creates final mapping file containing:
- polygon_id
- species_name

---

### Step 3: KMZ Export

- Exports results as KMZ for visualization
- Includes crown polygons with species labels

**Compatible with:**
- Google Earth
- QGIS

---

## Outputs

- KMZ file (species visualization)
- GeoJSON (optional)
- CSV (polygon to species mapping)
- Species distribution charts (optional)

---

## Learning Paradigm

This pipeline integrates three approaches:

- **Self-supervised learning**
- DINOv2 extracts meaningful visual features without labels

- **Unsupervised learning**
- K-Means groups similar tree crowns

- **Human-in-the-loop**
- Manual labeling assigns semantic meaning to clusters


#  Repository Structure

```
tree-species-mapping-dinov2/
│
├── docs/                      # Documentation and figures
│
├── examples/                  # Example inputs and outputs
│   ├── google_earth_view/
│   │   └── species_advanced_map_fixed.kml
│   ├── predictions.csv
│   ├── s3_tree.geojson
│   └── s3_tree.tif
│
├── models/
│   └── best_dinov2_linear.pth
│
├── notebooks/
│   ├── supervised/
│   │   ├── 01_crop_tree_crowns.ipynb
│   │   ├── 02_species_prediction_dinov2.ipynb
│   │   ├── 03_model_evaluation.ipynb
│   │   ├── 04_export_geojson_kml.ipynb
│   │   └── tree_species_full_pipeline.ipynb
│   │
│   └── unsupervised/
│       └── tree_species_unsupervised_pipeline.ipynb
│
├── outputs/                   # Generated outputs
│
├── pipelines/
│   ├── run_supervised_pipeline_full.py
│   └── run_unsupervised_pipeline_full.py
│
├── src/
│   ├── crown_cropping.py
│   ├── feature_extraction_dinov2.py
│   ├── feature_extraction_resnet.py
│   ├── feature_extraction_vit.py
│   ├── clustering.py
│   ├── classifier_training.py
│   ├── prediction.py
│   ├── geojson_to_kml.py
│   ├── labeling_sheet_creation.py
│   └── utils.py
│
├── requirements.txt
└── README.md
```

---

#  Workflow
```
Orthomosaic + Crown GeoJSONs
        │
        ▼
Step 1A: Crown cropping (one GeoTIFF per crown)
        │
        ▼
Step 1B: DINOv2 feature extraction (ViT-B/14, frozen)
        │
        ▼
Step 1C: K-Means clustering (multi-k: 2–12)
        │  → writes k{n}_cluster_species_map.csv  (blank species column)
        ▼
Step 1D: k-selection — Elbow + Silhouette + Davies-Bouldin
Step 1E: t-SNE visualisation per k
        │
 ── PAUSE: browse cluster image folders, fill species CSV ──
        │
        ▼
Step 2A: Load filled cluster→species CSV
Step 2B: Apply labels to all crowns
Step 2C: Build crown_master.csv + polygon_species.csv
Step 2D: Copy TIFFs into per-species folders
        │
        ├──── Step 3 (optional) ────────────────────────────┐
        │     Load ground truth CSV (image_name + label)    │
        │     Match on image_name → confusion matrix + F1   │
        │     Outputs: confusion_matrix.png, metrics.csv    │
        └───────────────────────────────────────────────────┘
        │
        ▼
Step 4A: Load crown_master.csv + merge with geometries
Step 4B: Build KMZ (colored species polygons)
Step 4D: Species distribution chart
        │
        ▼
Google Earth — species_map.kmz
```

---


---

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- macOS, Linux, or Windows

##  Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/automated_drone_tree_species_classification_geo_ai.git

cd automated_drone_tree_species_classification_geo_ai
```

### 2. Set Up Environment

#### Option A: Using conda (Recommended)

```bash
# Create conda environment
conda create -n tree-crown python=3.9
conda activate tree-crown

# Install PyTorch (choose based on your system)
# For Mac (CPU only)
conda install pytorch torchvision -c pytorch

# For Linux/Windows with CUDA 11.8
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

#### Option B: Using venv

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Your Paths

Edit the `Config` class in `tree_crown_pipeline.py`:

```python
class Config:
    # STEP 1 - Data paths
    ORTHO_FOLDER = '/path/to/your/orthomosaics'  # Folder with .tif files
    POLY_FOLDER = '/path/to/your/geojson_polygons'  # Folder with crown .geojson
    STEP1_OUTPUT = '/path/to/step1_output'  # Output directory
    
    # STEP 1 - Parameters
    K_LIST = [2, 4, 6, 8, 10, 12]  # k values to test
    CHOSEN_K = 2  # Update after Step 1
    
    # ... (see file for all options)
```

### 4. Run the Pipeline

#### Run All Steps

```bash
python tree_crown_pipeline.py --step all --config config.py
```

#### Run Individual Steps

```bash
cp config_example.py config.py
# edit config.py — set your paths, K_LIST, CHOSEN_K etc.

python tree_crown_pipeline.py --step 1 --config config.py
# browse cluster folders, fill k{n}_cluster_species_map.csv
# update CHOSEN_K in config.py

python tree_crown_pipeline.py --step 2 --config config.py
python tree_crown_pipeline.py --step 3 --config config.py   # optional
python tree_crown_pipeline.py --step 4 --config config.py
## 📁 Input Data Structure

```
your_data/
├── orthomosaics/
│   ├── site1.tif
│   ├── site2.tif
│   └── site3.tif
├── polygons/
│   ├── site1.geojson
│   ├── site2.geojson
│   └── site3.geojson
└── ground_truth.csv (optional)
    ├── image_name
    └── species
```

### Polygon GeoJSON Format

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "id": 1,
        "crown_id": 1
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [...]
      }
    }
  ]
}
```

##  Workflow

### Step 1: Clustering & Analysis

1. **Run Step 1**:
   ```bash
   python tree_crown_pipeline.py --step 1
   ```

2. **Review Outputs**:
   - Browse cluster folders: `step1_output/clustering/k{k}/cluster_{i}/`
   - Check k-selection plot: `clustering/k_selection.png`
   - Review t-SNE visualizations: `clustering/tsne_k{k}.png`

3. **Choose k**: Based on:
   - Silhouette score (higher is better)
   - Davies-Bouldin index (lower is better)
   - Visual inspection of cluster folders

4. **Assign Species**:
   - Open: `clustering/k{chosen_k}_cluster_species_map.csv`
   - Fill in the `species` column for each cluster
   - Example:
     ```csv
     cluster,cluster_folder,species,notes
     0,cluster_0,acacia,Clear acacia morphology
     1,cluster_1,non-acacia,Broad leaves visible
     
     ```

5. **Update Config**:
   ```python
   Config.CHOSEN_K = 6  # Your chosen k
   ```

### Step 2: Species Assignment

```bash
python tree_crown_pipeline.py --step 2
```

**Outputs**:
- `crown_master.csv`: Complete mapping (image_name → species)
- `polygon_species.csv`: Polygon IDs with species
- `species_folders/`: Crowns organized by species

### Step 3: Validation (Optional)

If you have ground truth labels:

1. **Prepare ground truth CSV**:
   ```csv
   image_name,label
   site1_000.tif,acacia
   site1_001.tif,eucalyptus
   ```

2. **Update config**:
   ```python
   Config.GROUND_TRUTH_CSV = '/path/to/ground_truth.csv'
   ```

3. **Run validation**:
   ```bash
   python tree_crown_pipeline.py --step 3
   ```

**Outputs**:
- Confusion matrix
- Classification metrics (accuracy, F1, kappa)
- Per-sample validation results

### Step 4: Google Earth Export

```bash
python tree_crown_pipeline.py --step 4
```

**Outputs**:
- `species_map.kmz`: Import into Google Earth

To view:
1. Open [earth.google.com](https://earth.google.com)
2. Click **Projects** → **Import KML file**
3. Upload `species_map.kmz`

## ⚙️ Configuration Options

### Model Selection

```python
# Fast (lower memory)
MODEL_NAME = 'vit_small_patch14_dinov2.lvd142m'

# Recommended (balanced)
MODEL_NAME = 'vit_base_patch14_dinov2.lvd142m'

# Best quality (slower, high memory)
MODEL_NAME = 'vit_large_patch14_dinov2.lvd142m'
```

### Performance Tuning

```python
BATCH_SIZE = 32  # Reduce if out of memory (16, 8, 4)
PCA_COMPONENTS = 50  # Reduce dimensionality (None to skip)
COPY_TO_CLUSTER_FOLDERS = True  # Set False to save disk space
```

### Color Palette (KMZ)

```python
COLOR_PALETTE = [
    '990000ff',  # blue
    '9900ff00',  # green
    '99ff0000',  # red
    # Add more colors (AABBGGRR format)
]
```

##  Running on Mac (Visual Studio Code)

### Setup in VS Code

1. **Open project**:
   ```bash
   cd tree-crown-pipeline
   code .
   ```

2. **Select Python interpreter**:
   - Press `Cmd+Shift+P`
   - Type "Python: Select Interpreter"
   - Choose your conda/venv environment

3. **Run in terminal**:
   - Press `` Ctrl+` `` to open terminal
   - Run: `python tree_crown_pipeline.py --step 1`

4. **Debug configuration** (`.vscode/launch.json`):
   ```json
   {
     "version": "0.2.0",
     "configurations": [
       {
         "name": "Python: Pipeline",
         "type": "python",
         "request": "launch",
         "program": "${workspaceFolder}/tree_crown_pipeline.py",
         "args": ["--step", "1"],
         "console": "integratedTerminal"
       }
     ]
   }
   ```

##  Output Files

### Step 1 Outputs

```
step1_output/
├── crowns/                    # Individual crown TIFFs
│   ├── site1_000.tif
│   ├── site1_001.tif
│   └── ...
├── features/
│   ├── dinov2_features.npy    # Feature matrix
│   ├── X_reduced.npy          # PCA-reduced features
│   └── dinov2_features.csv    # Image name index
└── clustering/
    ├── k_selection.png        # k-selection analysis
    ├── k_recommendation_table.csv
    ├── tsne_coordinates.csv
    ├── tsne_k{k}.png          # t-SNE plots
    ├── k{k}_assignments.csv   # Cluster assignments
    ├── k{k}_cluster_species_map.csv  # Fill this!
    └── k{k}/                  # Cluster folders
        ├── cluster_0/
        ├── cluster_1/
        └── ...
```

### Step 2 Outputs

```
step2_output/
├── crown_master.csv           # Complete mapping
├── polygon_species.csv        # Polygon → species
├── pseudo_label_assignments.csv
└── species_folders/           # Organized by species
    ├── acacia/
    ├── eucalyptus/
    └── ...
```

### Step 3 Outputs

```
step2_output/step3_validation/
├── confusion_matrix.png
├── validation_metrics.csv
└── validation_detail.csv
```

### Step 4 Outputs

```
step4_output/
├── species_map.kmz           # Google Earth file
└── species_distribution.png
```

##  Troubleshooting

### Out of Memory Error

```bash
# Reduce batch size
BATCH_SIZE = 8  # or 4, 2

# Use smaller model
MODEL_NAME = 'vit_small_patch14_dinov2.lvd142m'

# Reduce PCA components
PCA_COMPONENTS = 20
```

---

## Key Advantages

- No labeled dataset required initially
- Scalable to large geographic areas
- Flexible across different forest types
- Combines AI automation with expert validation

---

## Summary

The pipeline extracts tree crowns from drone imagery, encodes them into feature vectors using DINOv2, clusters them using K-Means, and assigns species labels through human inspection to generate a geospatial species distribution map.

---

## Usage (High-Level)

1. Provide orthomosaic and crown polygons
2. Run cropping and feature extraction
3. Perform clustering and select optimal k
4. Inspect clusters and assign species labels
5. Generate final outputs and visualize in GIS tools

---
