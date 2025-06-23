import pandas as pd
import geopandas as gpd
from pathlib import Path
import os
from shapely.geometry import Point, Polygon
from shapely import wkt
import matplotlib.pyplot as plt

current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parents[2]


def load_data(city):

    df = pd.read_csv(os.path.join(PROJECT_ROOT,"segmentierung_csv",f"segmentierung_{city}.csv"))

    def to_geometry(x):
        if isinstance(x, Point):
            return x  # already geometry
        else:
            return wkt.loads(x)  # convert from WKT string

    df['geometry'] = df['geometry'].apply(to_geometry)

    # Create GeoDataFrame with WGS84 as CRS
    gdf = gpd.GeoDataFrame(df, geometry="geometry",crs='EPSG:4326')    
    return gdf


def request_stations(geometry, gdf, features=["trains_metro","bus"], radius=50, verbose=False):
    """
    Counts how many points in gdf fall within a given geometry and its buffer area.

    Parameters:
    - geometry: shapely Point or Polygon
    - gdf: GeoDataFrame with point geometries
    - feature: list of column names to count occurrences by (optional)
    - radius: buffer in meters if geometry is Point
    - verbose: if True or str, show plot of points and areas

    Returns:
    - results: dict with counts inside the geometry and in the buffer
    """
    if gdf.crs is None:
        raise ValueError("gdf has no valid CRS")
    
    #if not gdf.crs.is_projected:
    gdf = gdf.to_crs(epsg=25830)

    # Convert input geometry to GeoSeries with correct CRS
    if isinstance(geometry, (Point, Polygon)):
        input_crs = "EPSG:4326"
        geom_series = gpd.GeoSeries([geometry], crs=input_crs)
        geometry = geom_series.to_crs(gdf.crs).iloc[0]
    else:
        raise ValueError("geometry must be a shapely Point or Polygon")

    # If point, create a circular buffer
    if isinstance(geometry, Point):
        geom = geometry.buffer(radius)
    else:
        geom = geometry

    geom_buffer = geom.buffer(radius)

    # Points within original geometry
    in_geom = gdf[gdf.geometry.within(geom)]
    # Points within buffer only (exclude those already in original geom)
    in_buffer = gdf[gdf.geometry.within(geom_buffer) & ~gdf.geometry.within(geom)]

    results = {}

    for feature in features:
        if feature == "trains_metro":
            in_geom_trains_metro = in_geom[in_geom["railway"] == "station"]
            in_buffer_trains_metro = in_buffer[in_buffer["railway"] == "station"]

            results["trains_metro"] = len(in_geom_trains_metro)
            results["trains_metro_in_buffer"] = len(in_buffer_trains_metro)

        if feature == "bus":
            in_geom_bus = in_geom[in_geom["highway"] == "bus_stop"]
            in_buffer_bus = in_buffer[in_buffer["highway"] == "bus_stop"]
            results["bus"] = len(in_geom_bus)
            results["bus_in_buffer"] = len(in_buffer_bus)

    # If feature(s) are provided
    # for col in features:
    #     if col not in gdf.columns:
    #         raise ValueError(f"{col} is not in GeoDataFrame")

    #     results[f"{col}_in_geom"] = in_geom[col].value_counts().to_dict()
    #     results[f"{col}_in_buffer"] = in_buffer[col].value_counts().to_dict()

    if verbose:
        # print("\nBerechnete Werte:")
        # for key, val in results.items():
        #     print(f"{key}: {val}")

        fig, ax = plt.subplots(figsize=(10, 10))
        gdf.plot(ax=ax, color='lightgray', markersize=5, label="All Points")
        in_geom_trains_metro.plot(ax=ax, color='blue', markersize=10, label="Inside Geometry Trains/Metro")
        in_buffer_trains_metro.plot(ax=ax, color='red', markersize=10, label="Inside Buffer Trains/Metro")

        in_geom_bus.plot(ax=ax, color='green', markersize=10, label="Inside Geometry Bus")
        in_buffer_bus.plot(ax=ax, color='purple', markersize=10, label="Inside Buffer Bus")

        gpd.GeoSeries([geom], crs=gdf.crs).boundary.plot(ax=ax, edgecolor='blue', linewidth=2, label="Geometry")
        gpd.GeoSeries([geom_buffer], crs=gdf.crs).boundary.plot(ax=ax, edgecolor='red', linestyle='--', linewidth=2, label="Buffer")

        # Zoom in tightly around the buffer polygon with a 10% margin
        minx, miny, maxx, maxy = geom_buffer.bounds
        x_margin = (maxx - minx) * 0.1
        y_margin = (maxy - miny) * 0.1

        ax.set_xlim(minx - x_margin, maxx + x_margin)
        ax.set_ylim(miny - y_margin, maxy + y_margin)

        ax.legend()
        ax.set_title("Points in Geometry and Buffer")
        plt.tight_layout()
        plt.show()

    return results

if __name__ == "__main__":
    gdfSegmentierung = load_data("madrid")


    # return results

    poly3 = Polygon([
        (445000, 4470000),  # untere linke Ecke
        (448000, 4470000),  # untere rechte Ecke
        (448000, 4473000),  # obere rechte Ecke
        (445000, 4473000)   # obere linke Ecke
    ])

    polygon = Polygon([
        (440365, 4475480),  # NW
        (440401, 4473265),  # SW
        (446025, 4473274),  # SE
        (446004, 4475484),  # NE
        (440365, 4475480)   # zurück zum Startpunkt
    ])

    print(request_stations(polygon,gdf=gdfSegmentierung,verbose=True))