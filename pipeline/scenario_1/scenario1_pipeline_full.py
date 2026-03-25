#!/usr/bin/env python3
"""
Forest Tree Crown Species Classification — Scenario 1
Semi-supervised pipeline: crop → DINOv2 embed → multi-k cluster →
label validation (n=400) → pseudo-label all crowns → KMZ with ortho overlay.

Usage:
    python pipelines/scenario1_pipeline_full.py
"""

# ══════════════════════════════════════════════════════════════════
#   USER CONFIG  ←  EDIT THIS BLOCK ONLY
# ══════════════════════════════════════════════════════════════════

ORTHO_FOLDER      = '/path/to/orthomosaics'
POLY_FOLDER       = '/path/to/geojson_polygons'
STEP1_OUTPUT_ROOT = '/path/to/step1_output'

K_LIST            = [2, 4, 6, 8, 10, 12]
MODEL_NAME        = 'vit_base_patch14_dinov2.lvd142m'
IMG_SIZE          = 224
BATCH_SIZE        = 32
PCA_COMPONENTS    = 50
COPY_TO_CLUSTER_FOLDERS = True

CHOSEN_K          = 2
LABEL_CSV         = '/path/to/labels.csv'
LABELED_FOLDERS   = {
    'acacia':     '/path/to/Acacia',
    'non_acacia': '/path/to/Non-Acacia',
}
STEP2_OUTPUT_ROOT = '/path/to/step2_output'

STEP3_OUTPUT_ROOT = '/path/to/step3_output'
SOURCE_EPSG       = 32643
ORTHO_MAX_PX      = 4096
ORTHO_DRAW_ORDER  = 10
POLY_DRAW_ORDER   = 50
COLOR_PALETTE = [
    '990000ff','9900ff00','99ff0000','9900ffff',
    '99ff00ff','99ff8800','9900ffff','99ffffff',
]

# ══════════════════════════════════════════════════════════════════
import os, re, shutil, math, zipfile, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from affine import Affine
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import simplekml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    confusion_matrix, classification_report,
    silhouette_score, davies_bouldin_score,
)
from sklearn.manifold import TSNE
warnings.filterwarnings('ignore')

# ── Helpers ──────────────────────────────────────────────────────
def make_dirs(*paths):
    for p in paths: os.makedirs(p, exist_ok=True)

def crown_id_from_gdf(gdf):
    for col in ['crown_id','id']:
        if col in gdf.columns: return gdf[col].astype(int)
    return pd.Series(gdf.index.astype(int))

def auto_detect_csv_columns(df):
    filename_col = label_col = None
    for col in df.columns:
        s = df[col].dropna().astype(str)
        if s.str.contains(r'\.(tif|tiff|jpg|png)', case=False).mean() > 0.5:
            filename_col = col; break
    if filename_col is None:
        for col in df.columns:
            if df[col].dropna().astype(str).str.contains(r'_tree_', case=False).mean() > 0.3:
                filename_col = col; break
    remaining = [c for c in df.columns if c != filename_col]
    if remaining: label_col = min(remaining, key=lambda c: df[c].nunique())
    return filename_col, label_col

def normalize_label(val):
    return str(val).strip().lower().replace(' ','_').replace('-','_')

plt.rcParams.update({'font.family':'DejaVu Sans','figure.dpi':150,
                     'axes.spines.top':False,'axes.spines.right':False})

# ── Paths ────────────────────────────────────────────────────────
DIR_CROWNS   = os.path.join(STEP1_OUTPUT_ROOT,'crowns')
DIR_FEATURES = os.path.join(STEP1_OUTPUT_ROOT,'features')
DIR_CLUSTER  = os.path.join(STEP1_OUTPUT_ROOT,'clustering')
DIR_VALID    = os.path.join(STEP2_OUTPUT_ROOT,'validation')
DIR_PSEUDO   = os.path.join(STEP2_OUTPUT_ROOT,'pseudo_labels')
make_dirs(DIR_CROWNS,DIR_FEATURES,DIR_CLUSTER,DIR_VALID,DIR_PSEUDO,STEP3_OUTPUT_ROOT)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice: {device}')

# ══════════════════════════════════════════════════════════════════
# STEP 1A — Crown cropping
# ══════════════════════════════════════════════════════════════════
print('\n=== STEP 1A: Crown cropping ===')
ortho_srcs = [rasterio.open(os.path.join(ORTHO_FOLDER,f))
              for f in os.listdir(ORTHO_FOLDER) if f.lower().endswith('.tif')]
print(f'Orthomosaics: {len(ortho_srcs)}')
total_saved = total_failed = 0
for gj_file in sorted(os.listdir(POLY_FOLDER)):
    if not gj_file.endswith('.geojson'): continue
    prefix = os.path.splitext(gj_file)[0]
    gdf = gpd.read_file(os.path.join(POLY_FOLDER,gj_file))
    gdf['_cid'] = crown_id_from_gdf(gdf)
    gdf = gdf.sort_values('_cid').reset_index(drop=True)
    print(f'\n  {gj_file} ({len(gdf)} crowns)')
    for _,row in tqdm(gdf.iterrows(),total=len(gdf),desc=prefix):
        cid=int(row['_cid']); out_name=f'{prefix}_{cid:03d}.tif'
        out_path=os.path.join(DIR_CROWNS,out_name)
        if os.path.exists(out_path): total_saved+=1; continue
        geom=[row.geometry]; saved=False
        for src in ortho_srcs:
            try:
                out_img,out_tf=rio_mask(src,geom,crop=True)
                meta=src.meta.copy()
                meta.update(height=out_img.shape[1],width=out_img.shape[2],transform=out_tf)
                with rasterio.open(out_path,'w',**meta) as dst: dst.write(out_img)
                saved=True; break
            except Exception: continue
        if saved: total_saved+=1
        else: total_failed+=1
for src in ortho_srcs: src.close()
print(f'\n✅ Cropping done — saved:{total_saved} failed:{total_failed}')

# ══════════════════════════════════════════════════════════════════
# STEP 1B — DINOv2 feature extraction
# ══════════════════════════════════════════════════════════════════
print('\n=== STEP 1B: DINOv2 feature extraction ===')
feat_npy=os.path.join(DIR_FEATURES,'dinov2_features.npy')
feat_csv=os.path.join(DIR_FEATURES,'dinov2_features.csv')
if os.path.exists(feat_npy):
    print('Cached features found — loading.')
    features=np.load(feat_npy)
    names=pd.read_csv(feat_csv)['image_name'].tolist()
else:
    tf=transforms.Compose([
        transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))])
    model=timm.create_model(MODEL_NAME,pretrained=True,num_classes=0,img_size=IMG_SIZE)
    model.eval().to(device)
    img_paths=sorted([os.path.join(DIR_CROWNS,f) for f in os.listdir(DIR_CROWNS) if f.lower().endswith('.tif')])
    print(f'Images to process: {len(img_paths)}')
    features,names=[],[]
    for i in tqdm(range(0,len(img_paths),BATCH_SIZE),desc='Extracting'):
        batch_paths=img_paths[i:i+BATCH_SIZE]; imgs,nms=[],[]
        for p in batch_paths:
            try: imgs.append(tf(Image.open(p).convert('RGB'))); nms.append(os.path.basename(p))
            except Exception as e: print(f'  ⚠️ Skipped {p}: {e}')
        if not imgs: continue
        batch=torch.stack(imgs).to(device)
        with torch.no_grad(): feat=model(batch)
        feat=F.normalize(feat,p=2,dim=1)
        features.append(feat.cpu().numpy()); names.extend(nms)
    features=np.vstack(features)
    np.save(feat_npy,features)
    pd.DataFrame({'image_name':names}).to_csv(feat_csv,index=False)
print(f'Feature matrix: {features.shape}')
X=StandardScaler().fit_transform(features)
if PCA_COMPONENTS and PCA_COMPONENTS<X.shape[1]:
    X=PCA(n_components=PCA_COMPONENTS,random_state=42).fit_transform(X)
    print(f'PCA applied → {X.shape}')
names_df=pd.DataFrame({'image_name':names})
np.save(os.path.join(DIR_FEATURES,'X_reduced.npy'),X)
print('✅ Features ready.')

# ══════════════════════════════════════════════════════════════════
# STEP 1C — Multi-k clustering
# ══════════════════════════════════════════════════════════════════
print('\n=== STEP 1C: Multi-k clustering ===')
inertia_vals,silhouette_vals,db_vals=[],[],[]
all_cluster_labels={}
for k in K_LIST:
    print(f'  k={k} ...',end=' ',flush=True)
    km=KMeans(n_clusters=k,random_state=42,n_init=10)
    cl=km.fit_predict(X); all_cluster_labels[k]=cl
    inertia_vals.append(km.inertia_)
    sil=silhouette_score(X,cl,sample_size=min(5000,len(X)),random_state=42)
    db=davies_bouldin_score(X,cl)
    silhouette_vals.append(sil); db_vals.append(db)
    print(f'inertia={km.inertia_:.0f}  sil={sil:.4f}  db={db:.4f}')
    cl_df=names_df.copy(); cl_df['cluster']=cl
    cl_df['cluster_label']=cl_df['cluster'].apply(lambda x:f'cluster_{x}')
    cl_df.to_csv(os.path.join(DIR_CLUSTER,f'k{k}_assignments.csv'),index=False)
    k_dir=os.path.join(DIR_CLUSTER,f'k{k}')
    for ci in range(k): os.makedirs(os.path.join(k_dir,f'cluster_{ci}'),exist_ok=True)
    if COPY_TO_CLUSTER_FOLDERS:
        for _,row in cl_df.iterrows():
            src=os.path.join(DIR_CROWNS,row['image_name'])
            dst=os.path.join(k_dir,f'cluster_{row["cluster"]}',row['image_name'])
            if os.path.exists(src) and not os.path.exists(dst): shutil.copy2(src,dst)
print('\n✅ Clustering done.')

# ══════════════════════════════════════════════════════════════════
# STEP 1D — k-selection plots + recommendation table
# ══════════════════════════════════════════════════════════════════
print('\n=== STEP 1D: k-selection analysis ===')
sil_arr=np.array(silhouette_vals); db_arr=np.array(db_vals); ine_arr=np.array(inertia_vals)
sil_norm=(sil_arr-sil_arr.min())/(sil_arr.ptp()+1e-9)
db_norm=1-(db_arr-db_arr.min())/(db_arr.ptp()+1e-9)
ine_diff=np.abs(np.diff(ine_arr,prepend=ine_arr[0]))
elbow_norm=(ine_diff-ine_diff.min())/(ine_diff.ptp()+1e-9)
combined=(sil_norm+db_norm+elbow_norm)/3
rec_df=pd.DataFrame({'k':K_LIST,'inertia':[round(v,1) for v in inertia_vals],
    'silhouette':[round(v,4) for v in silhouette_vals],
    'davies_bouldin':[round(v,4) for v in db_vals],
    'combined_score':[round(v,4) for v in combined]})
rec_df['rank']=rec_df['combined_score'].rank(ascending=False).astype(int)
rec_df=rec_df.sort_values('rank').reset_index(drop=True)
rec_df.to_csv(os.path.join(DIR_CLUSTER,'k_recommendation_table.csv'),index=False)
best_k_auto=int(rec_df.iloc[0]['k'])
print(rec_df.to_string(index=False))
print(f'\n⭐ Auto-recommended k = {best_k_auto}')
fig,axes=plt.subplots(1,3,figsize=(15,4.5))
axes[0].plot(K_LIST,inertia_vals,'o-',color='#1B4332',lw=2)
axes[0].set_title('Elbow Curve',fontweight='bold'); axes[0].set_xlabel('k'); axes[0].grid(alpha=0.3)
axes[1].plot(K_LIST,silhouette_vals,'o-',color='#40916C',lw=2)
axes[1].axvline(best_k_auto,color='red',linestyle='--',lw=1.2,label=f'Rec. k={best_k_auto}')
axes[1].set_title('Silhouette (higher=better)',fontweight='bold'); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
axes[2].plot(K_LIST,db_vals,'o-',color='#D97706',lw=2)
axes[2].axvline(best_k_auto,color='red',linestyle='--',lw=1.2)
axes[2].set_title('Davies-Bouldin (lower=better)',fontweight='bold'); axes[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DIR_CLUSTER,'k_selection.png'),dpi=150,bbox_inches='tight'); plt.close()
print('Saved: clustering/k_selection.png')

# ══════════════════════════════════════════════════════════════════
# STEP 1E — t-SNE
# ══════════════════════════════════════════════════════════════════
print('\n=== STEP 1E: t-SNE visualisation ===')
tsne_path=os.path.join(DIR_CLUSTER,'tsne_coordinates.csv')
if os.path.exists(tsne_path):
    tsne_df=pd.read_csv(tsne_path); print('Cached t-SNE loaded.')
else:
    print('Running t-SNE...')
    tsne=TSNE(n_components=2,perplexity=min(30,len(X)-1),random_state=42,init='pca',learning_rate='auto')
    X_tsne=tsne.fit_transform(X)
    tsne_df=pd.DataFrame({'x':X_tsne[:,0],'y':X_tsne[:,1],'image_name':names})
    tsne_df.to_csv(tsne_path,index=False)
for k in K_LIST:
    tsne_df['cluster']=all_cluster_labels[k]
    fig,ax=plt.subplots(figsize=(8,6))
    ax.scatter(tsne_df['x'],tsne_df['y'],c=tsne_df['cluster'],cmap='tab10',s=20,alpha=0.7,linewidths=0)
    handles=[mpatches.Patch(color=plt.cm.tab10(i/10),label=f'Cluster {i}') for i in range(k)]
    ax.legend(handles=handles,bbox_to_anchor=(1.05,1),loc='upper left',fontsize=9)
    ax.set_title(f't-SNE k={k} (n={len(tsne_df)})',fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_CLUSTER,f'tsne_k{k}.png'),dpi=150,bbox_inches='tight'); plt.close()
    print(f'Saved: tsne_k{k}.png')
print('\n✅ Step 1 complete.')

# ══════════════════════════════════════════════════════════════════
# STEP 2A — Load manual labels
# ══════════════════════════════════════════════════════════════════
print('\n=== STEP 2A: Loading manual labels ===')
raw_csv=pd.read_csv(LABEL_CSV)
filename_col,label_col=auto_detect_csv_columns(raw_csv)
print(f'Auto-detected: filename_col="{filename_col}"  label_col="{label_col}"')
labels_df=raw_csv[[filename_col,label_col]].copy()
labels_df.columns=['image_name','label_raw']
labels_df['image_name']=labels_df['image_name'].astype(str).str.strip()
labels_df['label']=labels_df['label_raw'].apply(normalize_label)
folder_rows=[]
if isinstance(LABELED_FOLDERS,dict):
    for sp_name,fp in LABELED_FOLDERS.items():
        if not os.path.isdir(fp): print(f'  ⚠️ Folder not found: {fp}'); continue
        for fname in os.listdir(fp):
            if fname.lower().endswith(('.tif','.tiff')):
                folder_rows.append({'image_name':fname,'folder_label':normalize_label(sp_name)})
    if folder_rows:
        fd=pd.DataFrame(folder_rows)
        labels_df=pd.merge(labels_df,fd,on='image_name',how='outer')
        labels_df['label']=labels_df['label'].fillna(labels_df.get('folder_label',pd.Series(dtype=str)))
labels_df=labels_df[['image_name','label']].dropna(subset=['label']).drop_duplicates('image_name')
unique_classes=sorted(labels_df['label'].unique())
label_to_int={cls:i for i,cls in enumerate(unique_classes)}
print(f'Classes: {unique_classes}  Total labeled: {len(labels_df)}')
for cls in unique_classes: print(f'  {cls:<20}: {(labels_df["label"]==cls).sum()}')

# ══════════════════════════════════════════════════════════════════
# STEP 2B — Match labels to cluster assignments
# ══════════════════════════════════════════════════════════════════
print('\n=== STEP 2B: Matching labels to clusters ===')
cl_csv=os.path.join(DIR_CLUSTER,f'k{CHOSEN_K}_assignments.csv')
if not os.path.exists(cl_csv):
    raise FileNotFoundError(f'k{CHOSEN_K}_assignments.csv not found. Run Step 1 first.')
cl_df2=pd.read_csv(cl_csv)
merged=pd.merge(labels_df,cl_df2,on='image_name',how='inner')
merged['true_int']=merged['label'].map(label_to_int)
print(f'Matched: {len(merged)} / {len(labels_df)}')
if len(merged)==0:
    raise RuntimeError('Zero matches — check image_name values match between CSV and Step 1.')
print(merged.groupby(['cluster','label']).size().unstack(fill_value=0).to_string())

# ══════════════════════════════════════════════════════════════════
# STEP 2C — Cluster purity + confusion matrix
# ══════════════════════════════════════════════════════════════════
print('\n=== STEP 2C: Cluster purity + confusion matrix ===')
cluster_to_class={}; purity_records=[]
for c in sorted(merged['cluster'].unique()):
    sub=merged[merged['cluster']==c]; vote=sub['label'].value_counts()
    winner=vote.index[0]; purity=vote.iloc[0]/len(sub)
    cluster_to_class[c]=winner
    purity_records.append({'cluster':c,'mapped_to':winner,'purity':round(purity,4),'n_samples':len(sub)})
    print(f'  Cluster {c:2d} → {winner:<20} purity={purity:.3f} n={len(sub)}')
purity_df=pd.DataFrame(purity_records)
purity_df.to_csv(os.path.join(DIR_VALID,'cluster_purity.csv'),index=False)
merged['predicted']=merged['cluster'].map(cluster_to_class)
overall_acc=(merged['label']==merged['predicted']).mean()
mean_purity=purity_df['purity'].mean()
print(f'\nOverall accuracy: {overall_acc*100:.1f}%   Mean purity: {mean_purity:.4f}')
cm=confusion_matrix(merged['label'],merged['predicted'],labels=unique_classes)
cm_norm=cm.astype(float)/cm.sum(axis=1,keepdims=True)
fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,data,title,fmt in [(axes[0],cm,f'Counts (k={CHOSEN_K})','d'),
                           (axes[1],cm_norm,f'Normalised (k={CHOSEN_K})','0.2f')]:
    sns.heatmap(data,annot=True,fmt=fmt,xticklabels=unique_classes,yticklabels=unique_classes,
                cmap='YlGn',ax=ax,linewidths=1,linecolor='white',annot_kws={'size':12,'weight':'bold'})
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title,fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(DIR_VALID,f'confusion_matrix_k{CHOSEN_K}.png'),dpi=150,bbox_inches='tight'); plt.close()
print(f'Saved: validation/confusion_matrix_k{CHOSEN_K}.png')
print(classification_report(merged['label'],merged['predicted'],target_names=unique_classes))

# ══════════════════════════════════════════════════════════════════
# STEP 2D — Pseudo-label all crowns
# ══════════════════════════════════════════════════════════════════
print('\n=== STEP 2D: Pseudo-labeling all crowns ===')
full_assign=cl_df2.copy()
full_assign['species']=full_assign['cluster'].map(cluster_to_class)
for sp in unique_classes: os.makedirs(os.path.join(DIR_PSEUDO,sp),exist_ok=True)
copied=skipped=0
for _,row in tqdm(full_assign.iterrows(),total=len(full_assign),desc='Copying'):
    sp=row.get('species')
    if pd.isna(sp): skipped+=1; continue
    src=os.path.join(DIR_CROWNS,row['image_name'])
    dst=os.path.join(DIR_PSEUDO,sp,row['image_name'])
    if os.path.exists(src) and not os.path.exists(dst): shutil.copy2(src,dst)
    copied+=1
pseudo_csv_path=os.path.join(STEP2_OUTPUT_ROOT,'pseudo_label_assignments.csv')
full_assign.to_csv(pseudo_csv_path,index=False)
print(f'Copied:{copied} Skipped:{skipped}')
for sp in unique_classes: print(f'  {sp:<20}: {len(os.listdir(os.path.join(DIR_PSEUDO,sp)))} images')
print(f'Pseudo-label CSV: {pseudo_csv_path}')
print('\n✅ Step 2 complete.')

# ══════════════════════════════════════════════════════════════════
# STEP 3A — Load + merge pseudo labels with geometries
# ══════════════════════════════════════════════════════════════════
print('\n=== STEP 3A: Loading and merging ===')
pseudo_df=pd.read_csv(pseudo_csv_path)
all_polys=[]
for gj_file in sorted(os.listdir(POLY_FOLDER)):
    if not gj_file.endswith('.geojson'): continue
    prefix=os.path.splitext(gj_file)[0]
    g=gpd.read_file(os.path.join(POLY_FOLDER,gj_file))
    g['_cid']=crown_id_from_gdf(g)
    g['image_name']=g['_cid'].apply(lambda x:f'{prefix}_{int(x):03d}.tif')
    g['_site']=prefix
    if 'Confidence_score' in g.columns: g=g.rename(columns={'Confidence_score':'confidence_score'})
    all_polys.append(g)
gdf_all=pd.concat(all_polys,ignore_index=True)
gdf_all=gdf_all.merge(pseudo_df[['image_name','species','cluster']],on='image_name',how='left')
print(f'Polygons: {len(gdf_all)}  Labelled: {gdf_all["species"].notna().sum()}')
if gdf_all.crs is None: gdf_all=gdf_all.set_crs(epsg=SOURCE_EPSG)
gdf_wgs=gdf_all.to_crs(epsg=4326)
species_list=sorted(gdf_wgs['species'].dropna().unique())
species_color_idx={sp:i%len(COLOR_PALETTE) for i,sp in enumerate(species_list)}
ortho_tif_list=sorted([os.path.join(ORTHO_FOLDER,f) for f in os.listdir(ORTHO_FOLDER) if f.lower().endswith('.tif')])

# ══════════════════════════════════════════════════════════════════
# STEP 3B — Ortho → PNG GroundOverlay
# ══════════════════════════════════════════════════════════════════
print('\n=== STEP 3B: Exporting orthomosaics as PNG ===')
ORTHO_PNG_DIR=os.path.join(STEP3_OUTPUT_ROOT,'ortho_overlays')
os.makedirs(ORTHO_PNG_DIR,exist_ok=True)
ortho_overlays=[]; wgs84=CRS.from_epsg(4326)
for ortho_path in ortho_tif_list:
    site_name=os.path.splitext(os.path.basename(ortho_path))[0]
    png_name=site_name+'_overlay.png'; png_path=os.path.join(ORTHO_PNG_DIR,png_name)
    print(f'  {os.path.basename(ortho_path)}')
    with rasterio.open(ortho_path) as src:
        src_crs=src.crs if src.crs else CRS.from_epsg(SOURCE_EPSG)
        tf_wgs,w_wgs,h_wgs=calculate_default_transform(src_crs,wgs84,src.width,src.height,*src.bounds)
        scale=min(ORTHO_MAX_PX/max(w_wgs,h_wgs),1.0)
        out_w=max(1,int(w_wgs*scale)); out_h=max(1,int(h_wgs*scale))
        scaled_tf=tf_wgs*Affine.scale(w_wgs/out_w,h_wgs/out_h)
        n_bands=min(src.count,3); rgb_data=np.zeros((n_bands,out_h,out_w),dtype=np.uint8)
        for bi in range(1,n_bands+1):
            reproject(source=rasterio.band(src,bi),destination=rgb_data[bi-1],
                      src_transform=src.transform,src_crs=src_crs,
                      dst_transform=scaled_tf,dst_crs=wgs84,resampling=Resampling.lanczos)
        nd=src.nodata if src.nodata is not None else 0
        if n_bands>=3:
            alpha=np.where((rgb_data[0]==nd)&(rgb_data[1]==nd)&(rgb_data[2]==nd),0,255).astype(np.uint8)
        else:
            alpha=np.where(rgb_data[0]==nd,0,255).astype(np.uint8)
        if n_bands==1: rgb_data=np.repeat(rgb_data,3,axis=0)
        elif n_bands==2: rgb_data=np.vstack([rgb_data,rgb_data[[0]]])
        Image.fromarray(np.dstack([rgb_data[0],rgb_data[1],rgb_data[2],alpha]),'RGBA').save(png_path)
        west=scaled_tf.c; north=scaled_tf.f
        east=west+scaled_tf.a*out_w; south=north+scaled_tf.e*out_h
        rotation=(math.degrees(math.atan2(-scaled_tf.d,scaled_tf.a))
                  if abs(scaled_tf.b)>1e-8 or abs(scaled_tf.d)>1e-8 else 0.0)
    print(f'    {out_w}x{out_h}px  N={north:.5f} S={south:.5f} E={east:.5f} W={west:.5f}')
    ortho_overlays.append({'site_name':site_name,'png_filename':png_name,'png_path':png_path,
                           'north':north,'south':south,'east':east,'west':west,'rotation':rotation})

# ══════════════════════════════════════════════════════════════════
# STEP 3C — Build KMZ
# ══════════════════════════════════════════════════════════════════
print('\n=== STEP 3C: Building KMZ ===')
KML_PATH=os.path.join(STEP3_OUTPUT_ROOT,'doc.kml')
KMZ_PATH=os.path.join(STEP3_OUTPUT_ROOT,'species_map.kmz')
kml=simplekml.Kml(); kml.document.name='Tree Crown Species Map'
ortho_top=kml.newfolder(name='Orthomosaics')
for ov in ortho_overlays:
    sf=ortho_top.newfolder(name=ov['site_name'])
    go=sf.newgroundoverlay(name=f"{ov['site_name']} Ortho ({ORTHO_MAX_PX}px)")
    go.draworder=ORTHO_DRAW_ORDER; go.color='ffffffff'
    go.icon.href=f"ortho_overlays/{ov['png_filename']}"
    go.latlonbox.north=ov['north']; go.latlonbox.south=ov['south']
    go.latlonbox.east=ov['east'];   go.latlonbox.west=ov['west']
    go.latlonbox.rotation=ov['rotation']
    print(f'  GroundOverlay: {ov["site_name"]}')
kml_folders={}; kml_styles={}
for sp in species_list:
    ci=species_color_idx[sp]; style=simplekml.Style()
    style.polystyle.color=COLOR_PALETTE[ci]; style.polystyle.fill=1; style.polystyle.outline=1
    style.linestyle.color='ff000000'; style.linestyle.width=1
    kml_styles[sp]=style; kml_folders[sp]=kml.newfolder(name=sp.replace('_',' ').title())
folder_unlab=kml.newfolder(name='Unlabelled')
style_unlab=simplekml.Style(); style_unlab.polystyle.color='55888888'
added=skipped_geom=unlabelled_n=0
for _,row in tqdm(gdf_wgs.iterrows(),total=len(gdf_wgs),desc='Building polygons'):
    geom=row.geometry
    if geom is None or geom.is_empty: skipped_geom+=1; continue
    if geom.geom_type=='MultiPolygon': geom=max(geom.geoms,key=lambda g:g.area)
    if geom.geom_type!='Polygon': skipped_geom+=1; continue
    sp=row.get('species'); img_name=row.get('image_name','')
    cluster=row.get('cluster',''); conf=row.get('confidence_score','')
    conf_str=f'{float(conf):.3f}' if pd.notna(conf) and conf!='' else 'N/A'
    if pd.isna(sp): folder,style,pol_name=folder_unlab,style_unlab,img_name; unlabelled_n+=1
    else: folder,style=kml_folders[sp],kml_styles[sp]; pol_name=f'{sp} | {img_name}'
    pol=folder.newpolygon(name=pol_name,outerboundaryis=list(geom.exterior.coords))
    pol.draworder=POLY_DRAW_ORDER; pol.style=style
    pol.description=(f'<b>Image:</b> {img_name}<br><b>Species:</b> {sp if pd.notna(sp) else "Unlabelled"}<br>'
                     f'<b>Cluster:</b> {cluster}<br><b>Confidence:</b> {conf_str}')
    added+=1
kml.save(KML_PATH)
with zipfile.ZipFile(KMZ_PATH,'w',zipfile.ZIP_DEFLATED) as kmz:
    kmz.write(KML_PATH,'doc.kml')
    for ov in ortho_overlays: kmz.write(ov['png_path'],os.path.join('ortho_overlays',ov['png_filename']))
os.remove(KML_PATH)
size_mb=os.path.getsize(KMZ_PATH)/1e6
print(f'Polygons added:{added} Unlabelled:{unlabelled_n} Skipped:{skipped_geom}')
print(f'KMZ size: {size_mb:.1f} MB')

# ══════════════════════════════════════════════════════════════════
# STEP 3D — Species distribution summary
# ══════════════════════════════════════════════════════════════════
print('\n=== STEP 3D: Species distribution summary ===')
sp_counts=gdf_wgs['species'].fillna('unlabelled').value_counts()
total=len(gdf_wgs)
for sp,n in sp_counts.items(): print(f'  {sp:<25}: {n:5d}  ({100*n/total:.1f}%)')
palette=['#D4A017','#2D6A4F','#2563EB','#7C3AED','#D97706','#0D9488','#DC2626','#888888']
fig,ax=plt.subplots(figsize=(max(6,len(sp_counts)*1.4),4.5))
bars=ax.bar(range(len(sp_counts)),sp_counts.values,
            color=[palette[i%len(palette)] for i in range(len(sp_counts))],alpha=0.88,edgecolor='white')
for bar,val in zip(bars,sp_counts.values):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+5,str(val),ha='center',fontsize=10,fontweight='bold')
ax.set_xticks(range(len(sp_counts)))
ax.set_xticklabels([s.replace('_',' ').title() for s in sp_counts.index],rotation=20,ha='right',fontsize=10)
ax.set_ylabel('Crown Count',fontsize=11)
ax.set_title('Final Species Distribution — All Crowns',fontsize=13,fontweight='bold')
ax.grid(axis='y',alpha=0.3); plt.tight_layout()
dist_plot=os.path.join(STEP3_OUTPUT_ROOT,'species_distribution.png')
plt.savefig(dist_plot,dpi=150,bbox_inches='tight'); plt.close()
print(f'Saved: {dist_plot}')

# ══════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('  ✅ SCENARIO 1 PIPELINE COMPLETE')
print('='*60)
print(f'\n  Crown TIFFs         : {DIR_CROWNS}')
print(f'  Features            : {DIR_FEATURES}')
print(f'  Cluster folders     : {DIR_CLUSTER}')
print(f'  k-selection plot    : {os.path.join(DIR_CLUSTER,"k_selection.png")}')
print(f'  k recommendation    : {os.path.join(DIR_CLUSTER,"k_recommendation_table.csv")}')
print(f'  Confusion matrix    : {os.path.join(DIR_VALID,f"confusion_matrix_k{CHOSEN_K}.png")}')
print(f'  Cluster purity CSV  : {os.path.join(DIR_VALID,"cluster_purity.csv")}')
print(f'  Pseudo-label CSV    : {pseudo_csv_path}')
print(f'  Species folders     : {DIR_PSEUDO}')
print(f'  KMZ (Google Earth)  : {KMZ_PATH}')
print(f'  Species chart       : {dist_plot}')
