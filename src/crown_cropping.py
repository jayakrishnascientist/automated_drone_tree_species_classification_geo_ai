
import os
import rasterio
import geopandas as gpd
from rasterio.mask import mask
from tqdm import tqdm

def crop_crowns(ortho_folder, polygon_folder, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    orthos = []
    for f in os.listdir(ortho_folder):
        if f.endswith(".tif"):
            orthos.append((f, rasterio.open(os.path.join(ortho_folder, f))))

    for gj in os.listdir(polygon_folder):
        if not gj.endswith(".geojson"):
            continue

        prefix = os.path.splitext(gj)[0]
        gdf = gpd.read_file(os.path.join(polygon_folder, gj))

        if "crown_id" in gdf.columns:
            gdf["crown_id"] = gdf["crown_id"].astype(int)
        elif "id" in gdf.columns:
            gdf["crown_id"] = gdf["id"].astype(int)
        else:
            gdf["crown_id"] = gdf.index.astype(int)

        for _, row in tqdm(gdf.iterrows(), total=len(gdf)):
            geom = [row.geometry]
            cid = int(row["crown_id"])
            out_name = f"{prefix}_{cid:03d}.tif"
            out_path = os.path.join(output_dir, out_name)

            for _, src in orthos:
                try:
                    out_img, out_transform = mask(src, geom, crop=True)

                    meta = src.meta.copy()
                    meta.update({
                        "height": out_img.shape[1],
                        "width": out_img.shape[2],
                        "transform": out_transform
                    })

                    with rasterio.open(out_path, "w", **meta) as dst:
                        dst.write(out_img)

                    break
                except:
                    continue
