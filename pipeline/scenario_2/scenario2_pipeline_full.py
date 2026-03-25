#!/usr/bin/env python3
"""
Forest Tree Crown Species Classification — Scenario 2
Unsupervised pipeline (no ground truth required): crop → DINOv2 embed →
multi-k cluster → human visual labelling → polygon species CSV → species
folders → KMZ export for Google Earth.

WORKFLOW:
  1. Edit USER CONFIG below and run this script  →  Step 1 runs fully
  2. Browse cluster folders visually, fill k{n}_cluster_species_map.csv
  3. Set CHOSEN_K + CLUSTER_SPECIES_CSV in USER CONFIG, set STEP='2'
  4. Run again  →  Steps 2 & 3 run

Usage:
    # Run Step 1 (set STEP = '1')
    python pipelines/scenario2_pipeline_full.py

    # After filling the CSV, run Steps 2 & 3 (set STEP = '2')
    python pipelines/scenario2_pipeline_full.py
"""

# ══════════════════════════════════════════════════════════════════
#   USER CONFIG  ←  EDIT THIS BLOCK ONLY
# ══════════════════════════════════════════════════════════════════

# Which step to run: '1' = crop+embed+cluster, '2' = label+export
STEP = '1'

# --- STEP 1 inputs ---
ORTHO_FOLDER      = '/path/to/orthomosaics'
POLY_FOLDER       = '/path/to/geojson_polygons'
STEP1_OUTPUT_ROOT = '/path/to/step1_output'

K_LIST            = [2, 4, 6, 8, 10, 12]
MODEL_NAME        = 'vit_base_patch14_dinov2.lvd142m'
IMG_SIZE          = 224
BATCH_SIZE        = 32
PCA_COMPONENTS    = 50
COPY_TO_CLUSTER_FOLDERS = True

# --- STEP 2 inputs (fill after visual inspection) ---
CHOSEN_K              = 2
CLUSTER_SPECIES_CSV   = '/path/to/step1_output/clustering/k2_cluster_species_map.csv'
STEP2_OUTPUT_ROOT     = '/path/to/step2_output'

# --- STEP 3 inputs ---
STEP3_OUTPUT_ROOT = '/path/to/step3_output'
SOURCE_EPSG       = 32643
COLOR_PALETTE = [
    '990000ff','9900ff00','99ff0000','9900ffff',
    '99ff00ff','99ff8800','9900ffff','99ffffff',
]

# ══════════════════════════════════════════════════════════════════
import os, re, shutil, zipfile, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
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
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
warnings.filterwarnings('ignore')

# ── Helpers ──────────────────────────────────────────────────────
def make_dirs(*paths):
    for p in paths: os.makedirs(p, exist_ok=True)

def crown_id_from_gdf(gdf):
    for col in ['crown_id','id']:
        if col in gdf.columns: return gdf[col].astype(int)
    return pd.Series(gdf.index.astype(int))

def normalize_label(val):
    return str(val).strip().lower().replace(' ','_').replace('-','_')

plt.rcParams.update({'font.family':'DejaVu Sans','figure.dpi':150,
                     'axes.spines.top':False,'axes.spines.right':False})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice: {device}  |  Running STEP {STEP}')

DIR_CROWNS   = os.path.join(STEP1_OUTPUT_ROOT,'crowns')
DIR_FEATURES = os.path.join(STEP1_OUTPUT_ROOT,'features')
DIR_CLUSTER  = os.path.join(STEP1_OUTPUT_ROOT,'clustering')
DIR_SPECIES  = os.path.join(STEP2_OUTPUT_ROOT,'species_folders')
make_dirs(DIR_CROWNS,DIR_FEATURES,DIR_CLUSTER)

# ══════════════════════════════════════════════════════════════════
# ─── STEP 1 ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════
if STEP in ('1','both'):

    # ── 1A Crown cropping ─────────────────────────────────────────
    print('\n=== STEP 1A: Crown cropping ===')
    ortho_srcs=[rasterio.open(os.path.join(ORTHO_FOLDER,f))
                for f in os.listdir(ORTHO_FOLDER) if f.lower().endswith('.tif')]
    print(f'Orthomosaics: {len(ortho_srcs)}')
    total_saved=total_failed=0
    for gj_file in sorted(os.listdir(POLY_FOLDER)):
        if not gj_file.endswith('.geojson'): continue
        prefix=os.path.splitext(gj_file)[0]
        gdf=gpd.read_file(os.path.join(POLY_FOLDER,gj_file))
        gdf['_cid']=crown_id_from_gdf(gdf)
        gdf=gdf.sort_values('_cid').reset_index(drop=True)
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

    # ── 1B DINOv2 features ────────────────────────────────────────
    print('\n=== STEP 1B: DINOv2 feature extraction ===')
    feat_npy=os.path.join(DIR_FEATURES,'dinov2_features.npy')
    feat_csv=os.path.join(DIR_FEATURES,'dinov2_features.csv')
    if os.path.exists(feat_npy):
        print('Cached features found — loading.')
        features=np.load(feat_npy); names=pd.read_csv(feat_csv)['image_name'].tolist()
    else:
        tf=transforms.Compose([
            transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))])
        model=timm.create_model(MODEL_NAME,pretrained=True,num_classes=0,img_size=IMG_SIZE)
        model.eval().to(device)
        img_paths=sorted([os.path.join(DIR_CROWNS,f) for f in os.listdir(DIR_CROWNS) if f.lower().endswith('.tif')])
        print(f'Images: {len(img_paths)}')
        features,names=[],[]
        for i in tqdm(range(0,len(img_paths),BATCH_SIZE),desc='Extracting'):
            bp=img_paths[i:i+BATCH_SIZE]; imgs,nms=[],[]
            for p in bp:
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

    # ── 1C Multi-k clustering ─────────────────────────────────────
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
        # Write blank species map CSV
        blank=pd.DataFrame({'cluster':list(range(k)),
                            'cluster_folder':[f'cluster_{i}' for i in range(k)],
                            'species':['']*k,'notes':['']*k})
        blank.to_csv(os.path.join(DIR_CLUSTER,f'k{k}_cluster_species_map.csv'),index=False)
        print(f'    → k{k}_cluster_species_map.csv written (fill species column)')
    print('\n✅ Clustering done.')

    # ── 1D k-selection plots ──────────────────────────────────────
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
    axes[0].set_title('Elbow Curve',fontweight='bold'); axes[0].grid(alpha=0.3)
    axes[1].plot(K_LIST,silhouette_vals,'o-',color='#40916C',lw=2)
    axes[1].axvline(best_k_auto,color='red',linestyle='--',lw=1.2,label=f'Rec. k={best_k_auto}')
    axes[1].set_title('Silhouette (higher=better)',fontweight='bold'); axes[1].legend(fontsize=9)
    axes[2].plot(K_LIST,db_vals,'o-',color='#D97706',lw=2)
    axes[2].axvline(best_k_auto,color='red',linestyle='--',lw=1.2)
    axes[2].set_title('Davies-Bouldin (lower=better)',fontweight='bold')
    for ax in axes: ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_CLUSTER,'k_selection.png'),dpi=150,bbox_inches='tight'); plt.close()
    print('Saved: clustering/k_selection.png')

    # ── 1E t-SNE ──────────────────────────────────────────────────
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
    print(f'   Cluster folders  : {DIR_CLUSTER}')
    print(f'   k recommendation : {os.path.join(DIR_CLUSTER,"k_recommendation_table.csv")}')
    print(f'\n  ─── NEXT STEPS ───')
    print(f'  1. Open clustering/k{{n}}/ folders and inspect crown images visually')
    print(f'  2. Open clustering/k{{n}}_cluster_species_map.csv')
    print(f'  3. Fill in the "species" column for each cluster row')
    print(f'  4. Set STEP = "2", CHOSEN_K, and CLUSTER_SPECIES_CSV in USER CONFIG')
    print(f'  5. Run this script again')

# ══════════════════════════════════════════════════════════════════
# ─── STEP 2 + 3 ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════
if STEP in ('2','both'):

    make_dirs(DIR_SPECIES,STEP2_OUTPUT_ROOT,STEP3_OUTPUT_ROOT)

    # ── 2A Load filled species map ────────────────────────────────
    print('\n=== STEP 2A: Loading cluster→species map ===')
    if not os.path.exists(CLUSTER_SPECIES_CSV):
        raise FileNotFoundError(
            f'CSV not found: {CLUSTER_SPECIES_CSV}\n'
            f'Run STEP=1 first, then fill the species column and set this path.')
    cluster_map_df=pd.read_csv(CLUSTER_SPECIES_CSV)
    print(f'Loaded: {CLUSTER_SPECIES_CSV}')
    print(cluster_map_df.to_string(index=False))
    blank_rows=cluster_map_df['species'].isna()|(cluster_map_df['species'].astype(str).str.strip()=='')
    if blank_rows.any():
        print(f'\n⚠️  {blank_rows.sum()} cluster(s) still blank — will be labelled "unlabelled"')
    cluster_map_df['species']=(cluster_map_df['species'].fillna('unlabelled')
                               .astype(str).str.strip().str.lower()
                               .str.replace(r'[\s\-]+','_',regex=True))
    cluster_to_species=dict(zip(cluster_map_df['cluster'].astype(int),cluster_map_df['species']))
    unique_species=sorted(set(v for v in cluster_to_species.values() if v!='unlabelled'))
    print(f'\nSpecies defined: {unique_species}')
    for c,sp in sorted(cluster_to_species.items()): print(f'  cluster {c:2d} → {sp}')

    # ── 2B Apply labels to all crown assignments ──────────────────
    print('\n=== STEP 2B: Applying species labels to all crowns ===')
    cl_csv=os.path.join(DIR_CLUSTER,f'k{CHOSEN_K}_assignments.csv')
    if not os.path.exists(cl_csv):
        raise FileNotFoundError(f'k{CHOSEN_K}_assignments.csv not found. Run STEP=1 first.')
    assign_df=pd.read_csv(cl_csv)
    assign_df['cluster']=assign_df['cluster'].astype(int)
    assign_df['species']=assign_df['cluster'].map(cluster_to_species).fillna('unlabelled')
    print(f'Total crowns: {len(assign_df)}')
    sp_counts=assign_df['species'].value_counts()
    for sp,n in sp_counts.items(): print(f'  {sp:<25}: {n:5d}  ({100*n/len(assign_df):.1f}%)')
    full_assign_path=os.path.join(STEP2_OUTPUT_ROOT,'pseudo_label_assignments.csv')
    assign_df.to_csv(full_assign_path,index=False)
    print(f'Full assignment CSV: {full_assign_path}')

    # ── 2C Build polygon_species.csv (polygon_id + species only) ──
    print('\n=== STEP 2C: Building polygon_species.csv ===')
    poly_rows=[]
    for gj_file in sorted(os.listdir(POLY_FOLDER)):
        if not gj_file.endswith('.geojson'): continue
        prefix=os.path.splitext(gj_file)[0]
        gdf_p=gpd.read_file(os.path.join(POLY_FOLDER,gj_file))
        cids=crown_id_from_gdf(gdf_p)
        for cid in cids:
            image_name=f'{prefix}_{int(cid):03d}.tif'
            poly_rows.append({'polygon_id':f'{prefix}_{int(cid):03d}','_img':image_name})
    poly_df=pd.DataFrame(poly_rows)
    poly_df=poly_df.merge(assign_df[['image_name','species']],
                          left_on='_img',right_on='image_name',how='left').drop(columns=['_img','image_name'])
    poly_df['species']=poly_df['species'].fillna('unlabelled')
    poly_csv_path=os.path.join(STEP2_OUTPUT_ROOT,'polygon_species.csv')
    poly_df[['polygon_id','species']].to_csv(poly_csv_path,index=False)
    print(f'Polygons: {len(poly_df)}  Matched: {poly_df["species"].ne("unlabelled").sum()}')
    print(f'polygon_species.csv saved: {poly_csv_path}')
    print(poly_df[['polygon_id','species']].head(6).to_string(index=False))

    # ── 2D Copy TIFFs into per-species folders ────────────────────
    print('\n=== STEP 2D: Creating per-species crown folders ===')
    for sp in unique_species+['unlabelled']: os.makedirs(os.path.join(DIR_SPECIES,sp),exist_ok=True)
    copied=skipped=0
    for _,row in tqdm(assign_df.iterrows(),total=len(assign_df),desc='Copying'):
        sp=row['species']
        src=os.path.join(DIR_CROWNS,row['image_name'])
        dst=os.path.join(DIR_SPECIES,sp,row['image_name'])
        if os.path.exists(src) and not os.path.exists(dst): shutil.copy2(src,dst); copied+=1
        else: skipped+=1
    print(f'Copied:{copied}  Skipped:{skipped}')
    for sp in sorted(os.listdir(DIR_SPECIES)):
        sp_path=os.path.join(DIR_SPECIES,sp)
        if os.path.isdir(sp_path):
            n=len([f for f in os.listdir(sp_path) if f.lower().endswith('.tif')])
            print(f'  species_folders/{sp:<25}: {n} images')
    print('\n✅ Step 2 complete.')

    # ── 3A Load polygon species + geometries ──────────────────────
    print('\n=== STEP 3A: Loading polygon species labels ===')
    poly_species_df=pd.read_csv(poly_csv_path)
    all_polys=[]
    for gj_file in sorted(os.listdir(POLY_FOLDER)):
        if not gj_file.endswith('.geojson'): continue
        prefix=os.path.splitext(gj_file)[0]
        g=gpd.read_file(os.path.join(POLY_FOLDER,gj_file))
        g['_cid']=crown_id_from_gdf(g)
        g['polygon_id']=g['_cid'].apply(lambda x:f'{prefix}_{int(x):03d}')
        if 'Confidence_score' in g.columns: g=g.rename(columns={'Confidence_score':'confidence_score'})
        all_polys.append(g)
    gdf_all=pd.concat(all_polys,ignore_index=True)
    gdf_all=gdf_all.merge(poly_species_df[['polygon_id','species']],on='polygon_id',how='left')
    gdf_all['species']=gdf_all['species'].fillna('unlabelled')
    print(f'Polygons: {len(gdf_all)}  Labelled: {gdf_all["species"].ne("unlabelled").sum()}')
    if gdf_all.crs is None: gdf_all=gdf_all.set_crs(epsg=SOURCE_EPSG)
    gdf_wgs=gdf_all.to_crs(epsg=4326)
    species_list=sorted(gdf_wgs['species'].unique())
    if 'unlabelled' in species_list: species_list.remove('unlabelled'); species_list.append('unlabelled')
    species_color_idx={sp:i%len(COLOR_PALETTE) for i,sp in enumerate(species_list)}
    for sp,ci in species_color_idx.items(): print(f'  {sp:<25}: {COLOR_PALETTE[ci]}')

    # ── 3B Build KMZ (polygons only) ─────────────────────────────
    print('\n=== STEP 3B: Building KMZ ===')
    KML_PATH=os.path.join(STEP3_OUTPUT_ROOT,'doc.kml')
    KMZ_PATH=os.path.join(STEP3_OUTPUT_ROOT,'species_map.kmz')
    kml=simplekml.Kml(); kml.document.name='Tree Crown Species Map'
    kml_folders={}; kml_styles={}
    for sp in species_list:
        ci=species_color_idx[sp]; style=simplekml.Style()
        style.polystyle.color=COLOR_PALETTE[ci]; style.polystyle.fill=1; style.polystyle.outline=1
        style.linestyle.color='ff000000'; style.linestyle.width=1
        kml_styles[sp]=style; kml_folders[sp]=kml.newfolder(name=sp.replace('_',' ').title())
    added=skipped_geom=0
    for _,row in tqdm(gdf_wgs.iterrows(),total=len(gdf_wgs),desc='Building polygons'):
        geom=row.geometry
        if geom is None or geom.is_empty: skipped_geom+=1; continue
        if geom.geom_type=='MultiPolygon': geom=max(geom.geoms,key=lambda g:g.area)
        if geom.geom_type!='Polygon': skipped_geom+=1; continue
        sp=row.get('species','unlabelled'); polygon_id=row.get('polygon_id','')
        conf=row.get('confidence_score','')
        conf_str=f'{float(conf):.3f}' if pd.notna(conf) and conf!='' else 'N/A'
        folder=kml_folders.get(sp,kml_folders.get('unlabelled'))
        style=kml_styles.get(sp,kml_styles.get('unlabelled'))
        pol=folder.newpolygon(name=f'{sp} | {polygon_id}',outerboundaryis=list(geom.exterior.coords))
        pol.style=style
        pol.description=(f'<b>Polygon ID:</b> {polygon_id}<br>'
                         f'<b>Species:</b> {sp}<br><b>Confidence:</b> {conf_str}')
        added+=1
    kml.save(KML_PATH)
    with zipfile.ZipFile(KMZ_PATH,'w',zipfile.ZIP_DEFLATED) as kmz: kmz.write(KML_PATH,'doc.kml')
    os.remove(KML_PATH)
    size_mb=os.path.getsize(KMZ_PATH)/1e6
    print(f'Polygons added:{added}  Skipped:{skipped_geom}')
    print(f'KMZ size: {size_mb:.1f} MB  →  {KMZ_PATH}')

    # ── 3D Species distribution summary ──────────────────────────
    print('\n=== STEP 3D: Species distribution summary ===')
    sp_counts=gdf_wgs['species'].value_counts()
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

    print('\n' + '='*60)
    print('  ✅ SCENARIO 2 PIPELINE COMPLETE')
    print('='*60)
    print(f'\n  Crown TIFFs         : {DIR_CROWNS}')
    print(f'  Cluster folders     : {DIR_CLUSTER}')
    print(f'  polygon_species.csv : {poly_csv_path}')
    print(f'  Species folders     : {DIR_SPECIES}')
    print(f'  KMZ (Google Earth)  : {KMZ_PATH}')
    print(f'  Species chart       : {dist_plot}')
