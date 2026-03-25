
# run_supervised_pipeline_full.py
# Full supervised tree species pipeline

import os
import torch
import timm
import rasterio
import geopandas as gpd
import numpy as np
import pandas as pd
import simplekml
from shapely.geometry import mapping
from rasterio.mask import mask
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# CONFIG
# ============================

ORTHO_FOLDER = "data/ortho"
POLYGON_FOLDER = "data/polygons"
OUTPUT_DIR = "outputs/supervised"
MODEL_PATH = "models/best_dinov2_linear.pth"
GROUND_TRUTH_CSV = "data/labels/labeling_sheet.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224

os.makedirs(OUTPUT_DIR, exist_ok=True)
CROP_DIR = os.path.join(OUTPUT_DIR, "cropped_crowns")
os.makedirs(CROP_DIR, exist_ok=True)

# ============================
# LOAD MODEL
# ============================

print("Loading trained model...")

backbone = timm.create_model(
    "vit_base_patch14_dinov2",
    pretrained=False,
    num_classes=0,
    img_size=224
)

classifier = torch.nn.Linear(768, 2)
model = torch.nn.Sequential(backbone, classifier)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE).eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Normalize(
        mean=(0.485,0.456,0.406),
        std=(0.229,0.224,0.225)
    )
])

# ============================
# STEP 1 — CROWN CROPPING
# ============================

print("Cropping crowns")

for gj in os.listdir(POLYGON_FOLDER):

    if not gj.endswith(".geojson"):
        continue

    prefix = os.path.splitext(gj)[0]
    gdf = gpd.read_file(os.path.join(POLYGON_FOLDER, gj))

    for ortho in os.listdir(ORTHO_FOLDER):

        if not ortho.endswith(".tif"):
            continue

        with rasterio.open(os.path.join(ORTHO_FOLDER, ortho)) as src:

            if gdf.crs != src.crs:
                gdf_proj = gdf.to_crs(src.crs)
            else:
                gdf_proj = gdf

            for idx, row in tqdm(gdf_proj.iterrows(), total=len(gdf_proj)):

                crown_id = row["id"] if "id" in row else idx
                filename = f"{prefix}_{int(crown_id):03d}.tif"
                out_path = os.path.join(CROP_DIR, filename)

                try:
                    out_img, out_transform = mask(
                        src,
                        [mapping(row.geometry)],
                        crop=True
                    )

                    meta = src.meta.copy()
                    meta.update({
                        "height": out_img.shape[1],
                        "width": out_img.shape[2],
                        "transform": out_transform
                    })

                    with rasterio.open(out_path, "w", **meta) as dst:
                        dst.write(out_img)

                except:
                    continue

# ============================
# STEP 2 — PREDICTION
# ============================

print("Running predictions")

results=[]

for img_name in tqdm(os.listdir(CROP_DIR)):

    img_path=os.path.join(CROP_DIR,img_name)

    try:
        with rasterio.open(img_path) as src:
            img = src.read([1,2,3]).transpose(1,2,0)

        img=np.clip(img,0,255).astype(np.uint8)

        tensor=transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits=model(tensor)
            prob=torch.softmax(logits,dim=1)

        pred=torch.argmax(prob,dim=1).item()
        conf=prob[0][pred].item()

        label="acacia" if pred==0 else "non_acacia"

        results.append({
            "image_name":img_name,
            "label":label,
            "confidence":conf
        })

    except:
        continue

df_pred=pd.DataFrame(results)
df_pred.to_csv(os.path.join(OUTPUT_DIR,"predictions.csv"),index=False)

# ============================
# STEP 3 — MODEL EVALUATION
# ============================

if os.path.exists(GROUND_TRUTH_CSV):

    df_true=pd.read_csv(GROUND_TRUTH_CSV)
    df_eval=df_pred.merge(df_true,on="image_name",how="inner")

    label_map={"acacia":0,"non_acacia":1}

    y_true=df_eval["label_y"].map(label_map)
    y_pred=df_eval["label_x"].map(label_map)
    y_prob=df_eval["confidence"]

    cm=confusion_matrix(y_true,y_pred)

    plt.figure()
    sns.heatmap(cm,annot=True,fmt="d")
    plt.savefig(os.path.join(OUTPUT_DIR,"confusion_matrix.png"))
    plt.close()

    report=classification_report(y_true,y_pred)

    with open(os.path.join(OUTPUT_DIR,"metrics.txt"),"w") as f:
        f.write(report)

# ============================
# STEP 4 — ATTACH TO POLYGONS
# ============================

all_polys=[]

for gj in os.listdir(POLYGON_FOLDER):

    if not gj.endswith(".geojson"):
        continue

    prefix=os.path.splitext(gj)[0]
    g=gpd.read_file(os.path.join(POLYGON_FOLDER,gj))

    g["image_name"]=g["id"].apply(
        lambda x: f"{prefix}_{int(x):03d}.tif"
    )

    all_polys.append(g)

gdf_all=pd.concat(all_polys,ignore_index=True)
gdf_all=gdf_all.merge(df_pred,on="image_name",how="left")

geojson_path=os.path.join(OUTPUT_DIR,"species_labeled.geojson")
gdf_all.to_file(geojson_path,driver="GeoJSON")

# ============================
# STEP 5 — EXPORT KML
# ============================

if gdf_all.crs is None:
    gdf_all.set_crs(epsg=32643,inplace=True)

gdf_wgs=gdf_all.to_crs(epsg=4326)

kml=simplekml.Kml()

for _,row in gdf_wgs.iterrows():

    if pd.isna(row["label"]):
        continue

    geom=row.geometry

    if geom.geom_type=="Polygon":
        polygons=[geom]
    else:
        polygons=list(geom.geoms)

    for poly in polygons:

        pol=kml.newpolygon(
            name=row["label"],
            outerboundaryis=list(poly.exterior.coords)
        )

kml.save(os.path.join(OUTPUT_DIR,"species_map.kml"))

print("Supervised pipeline complete")
