
# run_unsupervised_pipeline_full.py
# Full unsupervised tree species pipeline

import os
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import simplekml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ============================
# CONFIG
# ============================

ORTHO_FOLDER="data/ortho"
POLY_FOLDER="data/polygons"
OUTPUT_ROOT="outputs/unsupervised"

IMG_SIZE=224
BATCH_SIZE=32
K=4

DIR_CROWNS=os.path.join(OUTPUT_ROOT,"crowns")
DIR_FEATURES=os.path.join(OUTPUT_ROOT,"features")
DIR_CLUSTER=os.path.join(OUTPUT_ROOT,"clusters")
DIR_GEO=os.path.join(OUTPUT_ROOT,"geo")

for d in [DIR_CROWNS,DIR_FEATURES,DIR_CLUSTER,DIR_GEO]:
    os.makedirs(d,exist_ok=True)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# STEP 0 — CROWN CROPPING
# ============================

orthos=[
    rasterio.open(os.path.join(ORTHO_FOLDER,f))
    for f in os.listdir(ORTHO_FOLDER)
    if f.endswith(".tif")
]

for gj in os.listdir(POLY_FOLDER):

    if not gj.endswith(".geojson"):
        continue

    prefix=os.path.splitext(gj)[0]
    gdf=gpd.read_file(os.path.join(POLY_FOLDER,gj))

    gdf["cid"]=gdf.index

    for _,row in tqdm(gdf.iterrows(),total=len(gdf)):

        geom=[row.geometry]
        cid=int(row["cid"])

        out_name=f"{prefix}_{cid:03d}.tif"
        out_path=os.path.join(DIR_CROWNS,out_name)

        for src in orthos:

            try:
                out_img,out_transform=mask(src,geom,crop=True)

                meta=src.meta.copy()
                meta.update({
                    "height":out_img.shape[1],
                    "width":out_img.shape[2],
                    "transform":out_transform
                })

                with rasterio.open(out_path,"w",**meta) as dst:
                    dst.write(out_img)

                break
            except:
                continue

# ============================
# STEP 1 — DINOv2 FEATURES
# ============================

transform=transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485,0.456,0.406),
        std=(0.229,0.224,0.225)
    )
])

model=timm.create_model(
    "vit_base_patch14_dinov2.lvd142m",
    pretrained=True,
    num_classes=0
).to(device).eval()

img_paths=[
    os.path.join(DIR_CROWNS,f)
    for f in os.listdir(DIR_CROWNS)
    if f.endswith(".tif")
]

features=[]
names=[]

for i in tqdm(range(0,len(img_paths),BATCH_SIZE)):

    batch=img_paths[i:i+BATCH_SIZE]
    imgs=[]

    for p in batch:
        imgs.append(transform(Image.open(p).convert("RGB")))
        names.append(os.path.basename(p))

    batch_tensor=torch.stack(imgs).to(device)

    with torch.no_grad():
        feat=model(batch_tensor)

    feat=F.normalize(feat,p=2,dim=1)
    features.append(feat.cpu().numpy())

features=np.vstack(features)

np.save(os.path.join(DIR_FEATURES,"features.npy"),features)

# ============================
# STEP 2 — PCA + CLUSTERING
# ============================

scaler=StandardScaler()
X=scaler.fit_transform(features)

X=PCA(n_components=50).fit_transform(X)

km=KMeans(n_clusters=K,random_state=42)
clusters=km.fit_predict(X)

df=pd.DataFrame({
    "image_name":names,
    "cluster":clusters
})

df.to_csv(os.path.join(DIR_CLUSTER,"cluster_assignments.csv"),index=False)

# ============================
# STEP 3 — ATTACH TO POLYGONS
# ============================

all_polys=[]

for gj in os.listdir(POLY_FOLDER):

    if not gj.endswith(".geojson"):
        continue

    prefix=os.path.splitext(gj)[0]
    g=gpd.read_file(os.path.join(POLY_FOLDER,gj))

    g["image_name"]=g.index.map(
        lambda x: f"{prefix}_{int(x):03d}.tif"
    )

    all_polys.append(g)

gdf=pd.concat(all_polys,ignore_index=True)

gdf=gdf.merge(df,on="image_name",how="left")

# ============================
# STEP 4 — EXPORT KML
# ============================

if gdf.crs is None:
    gdf.set_crs(epsg=32643,inplace=True)

gdf_wgs=gdf.to_crs(epsg=4326)

kml=simplekml.Kml()

for _,r in gdf_wgs.iterrows():

    if pd.isna(r["cluster"]):
        continue

    geom=r.geometry

    if geom.geom_type=="Polygon":
        geoms=[geom]
    else:
        geoms=list(geom.geoms)

    for g in geoms:

        pol=kml.newpolygon(
            name=f"Cluster {r['cluster']}",
            outerboundaryis=list(g.exterior.coords)
        )

kml.save(os.path.join(DIR_GEO,"clusters_map.kml"))

print("Unsupervised pipeline complete")
