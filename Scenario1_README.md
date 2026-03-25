# Forest Tree Crown Species Classification — Scenario 1
## Semi-Supervised Pipeline (With Ground Truth Labels)

Automated **tree species mapping from drone orthomosaics** using **self-supervised DINOv2 embeddings and semi-supervised clustering**, validated with a manually labeled subset and exported as a Google Earth KMZ with orthomosaic overlay.

---

> **Which scenario should I use?**
>
> | | Scenario 1 (this repo) | Scenario 2 |
> |---|---|---|
> | Ground truth images available? | ✅ Yes (n ≥ 100 recommended) | ❌ No |
> | Label assignment method | Majority vote from labeled subset | Visual inspection of cluster folders |
> | Output | Pseudo-labels + confusion matrix + ortho overlay | Pseudo-labels + polygon species CSV |
> | Learning paradigm | Semi-supervised | Unsupervised + human-in-the-loop |

---

## Quick Start

```bash
git clone https://github.com/jayakrishnascientist/Forest-tree-crown-species-classification-geo-AI.git
cd Forest-tree-crown-species-classification-geo-AI
pip install -r requirements.txt
python pipelines/scenario1_pipeline_full.py
```

Edit the `USER CONFIG` block at the top of the script before running.

---

## Overview

This pipeline is designed for users who have **a small set of manually labeled crown images** alongside their drone data. The system uses those labels to assign species names to unsupervised clusters, then propagates those labels across all detected crowns.

```
Drone Orthomosaic + Crown Polygons
            │
            ▼
  Step 1A: Crown cropping (GeoTIFF per crown)
            │
            ▼
  Step 1B: DINOv2 feature extraction (frozen ViT-B/14)
            │
            ▼
  Step 1C: K-Means clustering (multi-k: 2–12)
            │
            ▼
  Step 1D: k-selection — Elbow + Silhouette + Davies-Bouldin
            │
            ▼
  Step 1E: t-SNE visualisation per k
            │
            ▼
  Step 2: Load manual labels → cluster purity → confusion matrix
            │
            ▼
  Step 2D: Pseudo-label all 2,156+ crowns → per-species folders
            │
            ▼
  Step 3: Ortho → PNG GroundOverlay + KMZ export (polygons + ortho)
            │
            ▼
  Google Earth — color-coded species map with orthomosaic background
```

---

## Learning Paradigm

| Component | Paradigm |
|-----------|----------|
| DINOv2 feature extraction | Self-supervised (pretrained, frozen) |
| K-Means clustering | Unsupervised |
| Species label assignment | Semi-supervised (majority vote from n=400) |
| Label propagation to all crowns | Label propagation |
| No end-to-end model training | ✗ Not fully supervised |

Best description for a paper: *"an unsupervised clustering pipeline built on self-supervised visual representations, with semi-supervised label propagation guided by a manually annotated validation subset."*

---

## Dataset — Sanjay Van, New Delhi

| Site | Number of Crowns |
|------|-----------------|
| S1   | 656             |
| S2   | 717             |
| S3   | 628             |
| S4   | 155             |
| **Total** | **2,156** |

Manual species labeling: **400 crowns** (193 Acacia, 207 Non-Acacia)

Drone platform: DJI Mini 4 Pro, flight altitude 80 m, GSD ≈ 2 cm/px, EPSG:32643.

---

## Features

- Crown extraction from orthomosaics using Detectree2 polygon annotations
- Deep feature extraction using **DINOv2 ViT-B/14** (self-supervised, frozen)
- **Multi-k K-Means** clustering with ranked k-selection table
- k-selection: **Silhouette + Davies-Bouldin + Elbow** — all three signals plotted
- **t-SNE visualisation** per k
- Auto-detection of CSV column names for any label format
- Cluster purity, alignment accuracy, and **confusion matrix** per k
- **Pseudo-labeling** of all crowns via cluster-to-class propagation
- Orthomosaic exported as **transparent PNG GroundOverlay** (georeferenced, WGS84)
- KMZ with species polygons + ortho overlay — one file, opens directly in Google Earth
- Works with **any species, any number of classes, any k range**

---

## Repository Structure

```
tree-species-mapping-dinov2/
│
├── pipelines/
│   ├── scenario1_pipeline_full.py    ← run this for Scenario 1
│   └── scenario2_pipeline_full.py    ← run this for Scenario 2
│
├── notebooks/
│   ├── tree_crown_pipeline_3step.ipynb
│   └── tree_crown_pipeline_scenario2.ipynb
│
├── outputs/
│   ├── step1_output/
│   │   ├── crowns/
│   │   ├── features/
│   │   │   ├── dinov2_features.npy
│   │   │   ├── dinov2_features.csv
│   │   │   └── X_reduced.npy
│   │   └── clustering/
│   │       ├── k2/, k4/, ...         ← cluster folders with TIFFs
│   │       ├── k{n}_assignments.csv
│   │       ├── k_selection.png
│   │       ├── tsne_k{n}.png
│   │       └── k_recommendation_table.csv
│   │
│   ├── step2_output/
│   │   ├── pseudo_label_assignments.csv
│   │   ├── pseudo_labels/
│   │   │   ├── acacia/
│   │   │   └── non_acacia/
│   │   └── validation/
│   │       ├── cluster_purity.csv
│   │       └── confusion_matrix_k{n}.png
│   │
│   └── step3_output/
│       ├── ortho_overlays/           ← georeferenced PNG per site
│       ├── species_map.kmz           ← open in Google Earth
│       └── species_distribution.png
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/jayakrishnascientist/Forest-tree-crown-species-classification-geo-AI.git
cd Forest-tree-crown-species-classification-geo-AI
pip install -r requirements.txt
```

---

## Running — Script (Recommended)

Edit the `USER CONFIG` block at the top of `pipelines/scenario1_pipeline_full.py`:

```python
ORTHO_FOLDER      = '/path/to/orthomosaics'
POLY_FOLDER       = '/path/to/geojson_polygons'
STEP1_OUTPUT_ROOT = '/path/to/step1_output'

K_LIST            = [2, 4, 6, 8, 10, 12]
MODEL_NAME        = 'vit_base_patch14_dinov2.lvd142m'
PCA_COMPONENTS    = 50

CHOSEN_K          = 2              # set after inspecting Step 1 k-selection plot
LABEL_CSV         = '/path/to/labels.csv'
LABELED_FOLDERS   = {
    'acacia':     '/path/to/Acacia',
    'non_acacia': '/path/to/Non-Acacia',
}
STEP2_OUTPUT_ROOT = '/path/to/step2_output'

STEP3_OUTPUT_ROOT = '/path/to/step3_output'
SOURCE_EPSG       = 32643
ORTHO_MAX_PX      = 4096
```

Then run:

```bash
python pipelines/scenario1_pipeline_full.py
```

The script runs all steps end-to-end in a single command. Features and t-SNE coordinates are cached — re-runs skip already-computed steps automatically.

---

## Running — Notebook (Interactive)

```bash
jupyter notebook notebooks/tree_crown_pipeline_3step.ipynb
```

---

## Label CSV Format

The script auto-detects column names. Any of these formats work:

```csv
image_name,label
s1_tree_006.tif,acacia
s1_tree_007.tif,non_acacia
```

```csv
filename,species
s1_tree_006.tif,Acacia
s1_tree_007.tif,Non-Acacia
```

Label values are normalised automatically (lowercase, spaces → underscore).

---

## Outputs

| File | Description |
|------|-------------|
| `clustering/k_selection.png` | Elbow + Silhouette + Davies-Bouldin panel |
| `clustering/k_recommendation_table.csv` | Ranked k table — rank 1 = recommended |
| `clustering/tsne_k{n}.png` | t-SNE plot per k |
| `clustering/k{n}_assignments.csv` | image_name → cluster for every crown |
| `validation/confusion_matrix_k{n}.png` | Cluster vs label confusion matrix |
| `validation/cluster_purity.csv` | Per-cluster purity and majority class |
| `pseudo_label_assignments.csv` | Species label for every crown |
| `pseudo_labels/acacia/` | Crown TIFFs classified as Acacia |
| `ortho_overlays/*.png` | Georeferenced transparent PNG per site |
| `species_map.kmz` | **Final output — open in Google Earth** |
| `species_distribution.png` | Species count bar chart |

---

## k-Selection Guidance

| Signal | What to look for |
|--------|-----------------|
| Silhouette score | Peak value — higher is better |
| Davies-Bouldin index | Lowest value — lower is better |
| Elbow (inertia) | Kink where adding k stops helping |
| Combined rank | Rank 1 = best combined signal |

Set `CHOSEN_K` to your preferred value in the config. The system recommends one automatically but ecological judgement takes priority.

---

## KMZ Output — Google Earth

The KMZ contains:
- **Orthomosaics folder** — one `<GroundOverlay>` per site, transparent PNG, georeferenced
- **Species folders** — one colored polygon layer per species
- Popup per crown: image name, species, cluster, confidence score

Open in Google Earth Pro or [earth.google.com](https://earth.google.com).

---

## Color Scheme (default)

KML colors are `AABBGGRR` format. Customise via `COLOR_PALETTE` in the config block.

---

## Applications

- Forest biodiversity monitoring
- Invasive species detection (Prosopis Juliflora / Acacia mapping)
- Ecological mapping and canopy structure analysis
- Urban tree inventories
- Large-scale GeoAI forest analysis

---

## Comparison with Scenario 2

| Feature | Scenario 1 | Scenario 2 |
|---------|-----------|-----------|
| Requires labeled images | Yes (n ≥ 100) | No |
| Label assignment | Majority vote (automatic) | Visual inspection (manual) |
| Cluster validation | Confusion matrix, F1 | Visual only |
| Ortho in KMZ | Yes (GroundOverlay) | No (polygons only) |
| Script runs in | Single command | Two commands (visual step between) |
| Best for | When ground truth is available | New sites, no prior labels |
