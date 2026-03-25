# Tree Crown Species Classification Pipeline

Automated pipeline for classifying tree species from aerial imagery using unsupervised clustering and DINOv2 features.

##  Overview

This pipeline performs:
1. **Crown Cropping**: Extracts individual tree crowns from orthomosaic imagery
2. **Feature Extraction**: Generates DINOv2 embeddings for each crown
3. **Clustering**: Groups similar crowns using K-means with multiple k values
4. **Species Assignment**: Maps clusters to species based on visual inspection
5. **Validation**: (Optional) Evaluates predictions against ground truth
6. **KMZ Export**: Creates interactive Google Earth visualization

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- macOS, Linux, or Windows

##  Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/tree-crown-pipeline.git
cd tree-crown-pipeline
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
python tree_crown_pipeline.py --step all
```

#### Run Individual Steps

```bash
# Step 1: Crop crowns, extract features, cluster
python tree_crown_pipeline.py --step 1

# Step 2: Assign species labels
python tree_crown_pipeline.py --step 2

# Step 3: Validate (optional)
python tree_crown_pipeline.py --step 3

# Step 4: Export to KMZ
python tree_crown_pipeline.py --step 4
```

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

### CUDA Not Available on Mac

Mac doesn't support CUDA. The pipeline will automatically use CPU:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# On Mac: device = 'cpu'
```

CPU processing is slower but works fine for small datasets (<1000 crowns).

### Missing GeoJSON Columns

The pipeline auto-detects `crown_id` or `id` columns. If your GeoJSON uses different names, it falls back to index numbers.

### File Path Issues

Use absolute paths in the config:
```python
ORTHO_FOLDER = '/Users/yourname/data/orthomosaics'
```

se open an issue or PR.


