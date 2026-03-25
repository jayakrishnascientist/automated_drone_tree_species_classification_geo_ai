# Forest Tree Crown Species Classification — Scenario 2
## Unsupervised Pipeline (No Ground Truth Required)

## **Note: To deploy this task go to (automated_drone_tree_species_classification_geo_ai/pipeline/scenario_2) and acess the readme file**

Automated **tree species mapping from drone orthomosaics** using **self-supervised DINOv2 embeddings and unsupervised clustering**, with species labels assigned through **human visual inspection** of cluster image folders — no pre-labeled training data needed.

---

> **Which scenario should I use?**
>
> | | Scenario 1 | Scenario 2 (this repo) |
> |---|---|---|
> | Ground truth images available? | ✅ Yes (n ≥ 100 recommended) | ❌ No |
> | Label assignment method | Majority vote from labeled subset | Visual inspection of cluster folders |
> | Cluster validation | Confusion matrix + F1 (automatic) | Optional — Step 3 (user-provided GT) |
> | Ortho in KMZ | Yes (GroundOverlay) | No (polygons only) |
> | Learning paradigm | Semi-supervised | Unsupervised + human-in-the-loop |

---

## Quick Start

```bash
git clone https://github.com/jayakrishnascientist/Forest-tree-crown-species-classification-geo-AI.git
cd Forest-tree-crown-species-classification-geo-AI
pip install -r requirements.txt

# Run Step 1 (set STEP = '1' in config)
python pipelines/scenario2_pipeline_full.py

# Browse cluster folders → fill species CSV → run Steps 2–4
# (set STEP = '2' in config)
python pipelines/scenario2_pipeline_full.py
```

---

## Pipeline Overview

Four steps. Steps 1 and 2 must run sequentially. **Step 3 is optional and independent** — you can go directly from Step 2 to Step 4.

| Step | Input | Output | Required? |
|------|-------|--------|-----------|
| **Step 1** | Orthomosaics + Crown GeoJSONs | Crop TIFFs, DINOv2 embeddings, cluster folders, k-selection plots | ✅ Yes |
| **Step 2** | Filled `k{n}_cluster_species_map.csv` | `crown_master.csv`, `polygon_species.csv`, species folders | ✅ Yes |
| **Step 3** | Ground truth CSV (`image_name` + label) | Confusion matrix, accuracy, kappa, F1 | ⚪ Optional |
| **Step 4** | `crown_master.csv` + Crown GeoJSONs | `species_map.kmz` for Google Earth | ✅ Yes |

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

## Learning Paradigm

| Component | Paradigm |
|-----------|----------|
| DINOv2 feature extraction | Self-supervised (pretrained frozen model) |
| K-Means clustering | Unsupervised |
| Species label assignment | Human-in-the-loop (visual inspection) |
| Label propagation to all crowns | Unsupervised label propagation |
| Step 3 validation | Optional cross-check — not part of training |

Best one-line description for a paper: *"an unsupervised clustering pipeline built on self-supervised visual representations, with human-in-the-loop label propagation for species assignment."*

---

## Key Design: crown_master.csv

Step 2C produces a **single master file** that links every identifier in one place. All downstream steps (Step 3 and Step 4) use this as their source of truth — no joins across multiple files needed.

```
image_name,       image_stem,  polygon_id, site, cluster, species
s1_tree_006.tif,  s1_tree_006, 6,          s1,   5,       acacia
s1_tree_007.tif,  s1_tree_007, 7,          s1,   4,       non_acacia
s1_tree_008.tif,  s1_tree_008, 8,          s1,   0,       non_acacia
```

| Column | Description |
|--------|-------------|
| `image_name` | Crown TIFF filename — universal join key, matches Step 1 output and any user GT CSV |
| `image_stem` | Filename without extension (for display) |
| `polygon_id` | Original integer ID from the input GeoJSON feature |
| `site` | GeoJSON file prefix (e.g. `s1`, `s2`) |
| `cluster` | K-Means cluster index assigned in Step 1 |
| `species` | Predicted species from visual labelling |

`polygon_species.csv` (clean output for external use) contains only `polygon_id` and `species`.

---

## Dataset — Sanjay Van, New Delhi

| Site | Number of Crowns |
|------|-----------------|
| S1   | 656             |
| S2   | 717             |
| S3   | 628             |
| S4   | 155             |
| **Total** | **2,156** |

Drone platform: DJI Mini 4 Pro, flight altitude 80 m, GSD ≈ 2 cm/px, EPSG:32643.
No pre-labeled training data required.

---

## Features

- Crown extraction from orthomosaics using Detectree2 polygon annotations
- **DINOv2 ViT-B/14** feature extraction (self-supervised, frozen — no fine-tuning)
- **Multi-k K-Means** with ranked k-selection table (Silhouette + Davies-Bouldin + Elbow)
- Blank `k{n}_cluster_species_map.csv` auto-generated for human labelling
- **t-SNE visualisation** per k for cluster separability inspection
- `crown_master.csv` — single file with all identifiers: `image_name`, `polygon_id`, `site`, `cluster`, `species`
- Optional **Step 3 validation** — accepts any GT CSV format with `image_name + label`, auto-detects columns, produces confusion matrix + accuracy + kappa + F1
- Per-species crown TIFF folders for downstream analysis
- KMZ export with colored polygon overlays for Google Earth (no ortho overlay)
- Works with any species, any number of classes, any k range

---

## Repository Structure

```
tree-species-mapping-dinov2/
│
├── pipelines/
│   ├── scenario1_pipeline_full.py
│   └── scenario2_pipeline_full.py    ← Scenario 2
│
├── notebooks/
│   └── tree_crown_pipeline_scenario2.ipynb
│
├── outputs/
│   ├── step1_output/
│   │   ├── crowns/                         ← individual crown TIFFs
│   │   ├── features/
│   │   │   ├── dinov2_features.npy
│   │   │   ├── dinov2_features.csv
│   │   │   └── X_reduced.npy
│   │   └── clustering/
│   │       ├── k2/
│   │       │   ├── cluster_0/              ← browse these visually
│   │       │   └── cluster_1/
│   │       ├── k4/, k6/, ...
│   │       ├── k{n}_assignments.csv
│   │       ├── k{n}_cluster_species_map.csv  ← fill this between runs
│   │       ├── k_selection.png
│   │       ├── tsne_k{n}.png
│   │       └── k_recommendation_table.csv
│   │
│   ├── step2_output/
│   │   ├── crown_master.csv               ← all identifiers in one file
│   │   ├── polygon_species.csv            ← polygon_id + species (clean output)
│   │   ├── pseudo_label_assignments.csv   ← image_name + cluster + species
│   │   ├── species_folders/
│   │   │   ├── acacia/                    ← crown TIFFs per species
│   │   │   └── non_acacia/
│   │   └── step3_validation/              ← created only if Step 3 is run
│   │       ├── confusion_matrix.png
│   │       ├── validation_metrics.csv
│   │       └── validation_detail.csv
│   │
│   └── step4_output/
│       ├── species_map.kmz                ← open in Google Earth
│       └── species_distribution.png
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone (https://github.com/jayakrishnascientist/automated_drone_tree_species_classification_geo_ai.git)
cd automated_drone_tree_species_classification_geo_ai/pipeline/scenario2
pip install -r requirements.txt
```

---

## Running — Script

### Run 1 — Step 1  (set `STEP = '1'` in config)

```python
STEP = '1'

ORTHO_FOLDER      = '/path/to/orthomosaics'
POLY_FOLDER       = '/path/to/geojson_polygons'
STEP1_OUTPUT_ROOT = '/path/to/step1_output'
K_LIST            = [2, 4, 6, 8, 10, 12]
MODEL_NAME        = 'vit_base_patch14_dinov2.lvd142m'
PCA_COMPONENTS    = 50
```

```bash
python pipelines/scenario2_pipeline_full.py
```

Step 1 is fully cached — re-runs skip already-computed features and t-SNE automatically.

---

### Between runs — Visual labelling (you do this)

1. Check `clustering/k_selection.png` and `k_recommendation_table.csv` to choose your k
2. Browse `clustering/k{n}/cluster_0/`, `cluster_1/`, etc. — inspect the crown images
3. Open `clustering/k{n}_cluster_species_map.csv` and fill the `species` column:

```csv
cluster,cluster_folder,species,notes
0,cluster_0,acacia,dense round canopy — Prosopis Juliflora
1,cluster_1,non_acacia,broad irregular crowns
2,cluster_2,acacia,smaller acacia crowns
3,cluster_3,non_acacia,sparse mixed canopy
```

4. Save the CSV

---

### Run 2 — Steps 2, 3 (optional), and 4  (set `STEP = '2'` in config)

```python
STEP = '2'

CHOSEN_K            = 4
CLUSTER_SPECIES_CSV = '/path/to/step1_output/clustering/k4_cluster_species_map.csv'
STEP2_OUTPUT_ROOT   = '/path/to/step2_output'
STEP3_OUTPUT_ROOT   = '/path/to/step4_output'
SOURCE_EPSG         = 32643
```

```bash
python pipelines/scenario2_pipeline_full.py
```

---

## Running — Notebook (Interactive)

```bash
jupyter notebook notebooks/tree_crown_pipeline_scenario2.ipynb
```

---

## Step 3 — Validation (Optional)

Set `S3_GROUND_TRUTH_CSV` to your ground truth file and run the Step 3 cells.

### Ground truth CSV format

Your CSV only needs two columns — **filename and label**. Column names are auto-detected.

```csv
image_name,label
s1_tree_006.tif,acacia
s1_tree_009.tif,acacia
s1_tree_013.tif,non_acacia
```

Any of these formats work (column names do not matter):

```csv
filename,species          →  auto-detected
file,class                →  auto-detected
image_name,label          →  auto-detected
```

The system matches on `image_name` against `crown_master.csv` — no polygon ID needed.

### Step 3 outputs

| File | Description |
|------|-------------|
| `confusion_matrix.png` | Counts + row-normalised heatmap side by side |
| `validation_metrics.csv` | Accuracy, Cohen's kappa, F1 macro, F1 weighted |
| `validation_detail.csv` | Per-polygon comparison: true vs predicted, correct flag |

Step 3 also prints a per-site accuracy breakdown when site information is available.

---

## Outputs Summary

| File | Description |
|------|-------------|
| `clustering/k_selection.png` | Elbow + Silhouette + Davies-Bouldin — 3-panel |
| `clustering/k_recommendation_table.csv` | Ranked k table — rank 1 = recommended |
| `clustering/tsne_k{n}.png` | t-SNE plot per k |
| `clustering/k{n}_assignments.csv` | image_name → cluster for every crown |
| `clustering/k{n}_cluster_species_map.csv` | **Fill this between Step 1 and Step 2** |
| `crown_master.csv` | **All identifiers in one file** — used by Steps 3 and 4 |
| `polygon_species.csv` | `polygon_id` + `species` — clean output for external tools |
| `pseudo_label_assignments.csv` | `image_name` + `cluster` + `species` for every crown |
| `species_folders/acacia/` | Crown TIFFs predicted as Acacia |
| `species_folders/non_acacia/` | Crown TIFFs predicted as Non-Acacia |
| `step3_validation/confusion_matrix.png` | Validation heatmap (if Step 3 run) |
| `species_map.kmz` | **Final output — open in Google Earth** |
| `species_distribution.png` | Species count bar chart |

---

## k-Selection Guidance

| Signal | What to look for |
|--------|-----------------|
| Silhouette score | Peak value — higher is better |
| Davies-Bouldin index | Lowest value — lower is better |
| Elbow (inertia) | The kink where adding k stops helping |
| Combined rank | Rank 1 = best combined signal |

The system recommends a k automatically. For forest species mapping, ecologically meaningful k values are usually small (2–6). Always visually inspect the cluster folders before committing to a k — the statistics are a guide, not a decision.

---

## Troubleshooting

**Step 3 — zero matches:**
Confirm your ground truth filenames match the Step 1 crop naming convention exactly: `s1_tree_006.tif`. The system prints sample names from both files to help diagnose mismatches.

**Step 4 — unlabelled polygons:**
If polygons show as unlabelled in the KMZ, check that `CHOSEN_K` in Step 2 config matches the k you filled in the species map CSV.

**Feature extraction OOM:**
Reduce `BATCH_SIZE` from 32 to 8 or 16 in Step 1 config. Use `vit_small_patch14_dinov2.lvd142m` instead of the base model for lower memory usage.

**Re-running Step 1:**
Features and t-SNE are cached. Delete `dinov2_features.npy` and `tsne_coordinates.csv` to force a full re-run.

---

## Comparison with Scenario 1

| Feature | Scenario 1 | Scenario 2 |
|---------|-----------|-----------|
| Requires labeled images | Yes (n ≥ 100) | No |
| Label assignment | Majority vote (automatic) | Visual inspection (manual) |
| Cluster validation | Confusion matrix + F1 (built-in) | Optional Step 3 (user GT CSV) |
| Ground truth CSV format | `image_name` + label | `image_name` + label (same) |
| Master output file | `pseudo_label_assignments.csv` | `crown_master.csv` |
| Ortho in KMZ | Yes (GroundOverlay) | No (polygons only) |
| Script runs in | One command | Two commands (visual step between) |
| Best for | When ground truth exists | New sites, no prior labels |

---

## Applications

- Forest biodiversity monitoring without labeled training data
- Invasive species detection in remote or data-scarce regions
- Ecological mapping for new survey sites with no prior labels
- Urban tree inventories
- Rapid first-pass species grouping before targeted ground truth collection
