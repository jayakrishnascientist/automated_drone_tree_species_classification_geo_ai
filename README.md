# Forest-tree-crown-species-classification-geo-AI



Automated **tree species mapping from high-resolution drone orthomosaics** using **Deep learning foundation models** and translation of species labels to Google Earth for geospatial investigation. 

<img width="800" height="1300" alt="pipeline_diagram" src="https://github.com/user-attachments/assets/02200020-8138-4e7c-ad44-faaa104c40ea" />



This repository provides two dedicated pipelines
**Scenario 1**
**Scenario 2**

Both scenarios aim to classify multiple forest species using drone data. Scenario 1 requires ground-truth labels to predict species from the validation set, while Scenario 2 supports classification without ground-truth labels.

The pipeline supports any type of drone data in the orthomosaic.tif extension. In our application, we identified tree species (Acacia or Non-Acacia) in the Tropical Forest using crown polygons extracted from drone imagery. in both version **supervised** and **unsupervised** **classification**

Only two inputs are required (Drone RGB Orthomosaic + Tree crown polygon) automatically processes orthomosaic images, extracts tree crowns, predicts species classes, and generates Google Earth compatible KML maps showing tree species distribution across forest landscapes

The framework integrates **computer vision, geospatial processing, and deep learning** to generate spatial outputs that can be visualized directly in **QGIS or Google Earth**.

---
# Pipeline Overview

1-Orthomosaic imagery and crown polygons are processed through the following workflow:

2-Tree crown cropping from orthomosaic images

3-Deep feature extraction using DINOv2 Vision Transformer

4-Feature clustering and evaluation

5-Training of a species classification model

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



# Features

Crown extraction from orthomosaics using polygon annotations

Tree species prediction using **DINOv2 Vision Transformer**

Unsupervised species discovery via **feature clustering**

Spatial outputs as **GeoJSON and KML**

Visualization-ready results for **Google Earth**

Modular **GeoAI research pipeline**


# Unsupervised Pipeline

The unsupervised pipeline is used to explore the structure of crown imagery without labels.

Crop crown images from orthomosaic

Extract deep features using pretrained models

Apply dimensionality reduction (PCA)

Perform K-Means clustering

Evaluate clusters using labeled subset

Compare feature representations (ResNet, ViT, DINOv2)

DINOv2 produced the best cluster separability.

# Supervised Pipeline

The supervised pipeline trains a classifier using labeled crowns.

Architecture:

Image → DINOv2 Backbone → Feature Vector → Linear Classifier → Species Prediction
Model performance: Accuracy ≈ 95%

The trained model is applied to all crowns to generate spatial species predictions.


---

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

The pipeline follows a **GeoAI processing workflow**:

```
Drone Orthomosaic
       │
       ▼
Tree Crown Polygons
       │
       ▼
Crown Cropping
       │
       ▼
DINOv2 Feature Extraction
       │
       ├── Supervised Classification
       │         │
       │         ▼
       │   Tree Species Prediction
       │
       └── Unsupervised Clustering
                 │
                 ▼
         Species Group Discovery
       
       ▼
Attach Labels to Polygons
       │
       ▼
GeoJSON + KML Export
       │
       ▼
Visualization in QGIS / Google Earth
```

---

#  Installation

Clone the repository:

```bash
git clone https://github.com/jayakrishnascientist/Forest-tree-crown-species-classification-geo-AI.git

cd Forest-tree-crown-species-classification-geo-AI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

#  Running the Pipelines

## Supervised Species Prediction

Runs the **trained DINOv2 classifier**.

```bash
python pipelines/run_supervised_pipeline_full.py
```

Outputs:

```
outputs/supervised/
├── predictions.csv
├── species_labeled.geojson
├── species_map.kml
├── confusion_matrix.png
└── metrics.txt
```

---

## Unsupervised Species Clustering

Discovers species groups using **feature clustering**.

```bash
python pipelines/run_unsupervised_pipeline_full.py
```

Outputs:

```
outputs/unsupervised/
├── crowns/
├── features/
├── clusters/
└── clusters_map.kml
```

---

#  Example Output


Tree species predictions visualized in **Google Earth**.

```
examples/google_earth_view/species_advanced_map_fixed.kml
```

This file can be opened directly in **Google Earth Pro**.

# Color scheme:

Green → Acacia

Red → Non-Acacia

The final result is a species distribution map that can be opened directly in Google Earth.

---

#  Model

The supervised model uses:

**Backbone**

```
DINOv2 ViT-B/14
```

**Classifier**

```
Linear layer
```

---

#  Data Requirements

Inputs required:

```
Orthomosaic (GeoTIFF)
Tree crown polygons (GeoJSON)
```

Optional:

```
Ground truth labels for evaluation
```

---

# Applications

This framework can support:

• Forest biodiversity monitoring

• Invasive species detection

• Ecological mapping

• Urban tree inventories

• Large-scale GeoAI forest analysis

---


```

```

