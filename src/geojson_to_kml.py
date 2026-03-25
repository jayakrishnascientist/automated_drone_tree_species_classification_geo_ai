
import geopandas as gpd
import simplekml

def geojson_to_kml(geojson_path, output_kml):

    gdf = gpd.read_file(geojson_path)
    gdf = gdf.to_crs(epsg=4326)

    kml = simplekml.Kml()

    for _, row in gdf.iterrows():
        geom = row.geometry

        pol = kml.newpolygon(
            name=row.get("species","tree"),
            outerboundaryis=list(geom.exterior.coords)
        )

    kml.save(output_kml)
    print("KML saved:", output_kml)
