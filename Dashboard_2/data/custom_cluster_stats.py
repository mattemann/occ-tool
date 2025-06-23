import pandas as pd  
import geopandas as gpd
from shapely import wkt
import numpy as np
from shapely.geometry import MultiPoint
from data import cities
from shapely.geometry.base import BaseGeometry


def request_cluster_stats(city,dfPOIs):

    df = dfPOIs

    # print(df)

    df['geometry'] = df['geometry'].apply(
        lambda x: wkt.loads(x) if not isinstance(x, BaseGeometry) else x
    )
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    gdf['lon'] = gdf.geometry.x
    gdf['lat'] = gdf.geometry.y
    coords = gdf[['lon', 'lat']].values

    col_name = f"{city}_hdbscan_5_5"

    labels = np.full(len(gdf), 999999) # set the label as 999999
    gdf[col_name] = labels

    cluster_stats = []
    cluster_polygons = []

    for cluster_label in np.unique(labels):
        if cluster_label == -1:
            continue

        cluster_mask = labels == cluster_label
        cluster_points = coords[cluster_mask]
        lon_vals = cluster_points[:, 0]
        lat_vals = cluster_points[:, 1]

        avg_lon = lon_vals.mean()
        avg_lat = lat_vals.mean()
        median_lon = np.median(lon_vals)
        median_lat = np.median(lat_vals)
        min_lon, max_lon = lon_vals.min(), lon_vals.max()
        min_lat, max_lat = lat_vals.min(), lat_vals.max()
        bbox_area = (max_lon - min_lon) * (max_lat - min_lat)
        std_lon = lon_vals.std()
        std_lat = lat_vals.std()
        dists = np.sqrt((lon_vals - avg_lon) ** 2 + (lat_vals - avg_lat) ** 2)
        radius_avg = dists.mean()
        radius_max = dists.max()
        density = len(cluster_points) / bbox_area if bbox_area > 0 else np.nan

        shop_values = gdf.loc[cluster_mask, 'shop'].dropna()
        has_shop = "Yes" if not shop_values.empty else "No"
        amenity_values = gdf.loc[cluster_mask, 'amenity'].dropna()

        shop_counts = shop_values.value_counts()
        amenity_counts = amenity_values.value_counts()
        cluster_size = len(cluster_points)
        shop_rel = (shop_counts / cluster_size).round(3)
        amenity_rel = (amenity_counts / cluster_size).round(3)

        shop_cols = {f"shop_rel_{k}": v for k, v in shop_rel.items()}
        amenity_cols = {f"amenity_rel_{k}": v for k, v in amenity_rel.items()}

        stats = {
            'param_config': col_name,
            'cluster_label': cluster_label,
            'num_points': len(cluster_points),
            'avg_lon': avg_lon,
            'avg_lat': avg_lat,
            'median_lon': median_lon,
            'median_lat': median_lat,
            'bounding_box_area': bbox_area,
            'density': density,
            'radius_avg': radius_avg,
            'radius_max': radius_max,
            'std_lon': std_lon,
            'std_lat': std_lat,
            'has_shop': has_shop,
        }
        stats.update(shop_cols)
        stats.update(amenity_cols)
        
        cluster_stats.append(stats)

        points = gdf.loc[cluster_mask, 'geometry'].tolist()
        multipoint = MultiPoint(points)
        polygon = multipoint.convex_hull  # or use multipoint.envelope for bounding box

        stats["geometry"] = polygon

        # cluster_polygons.append({
        #     'cluster_label': cluster_label,
        #     'geometry': polygon
        # })

    cluster_df = pd.DataFrame(cluster_stats)
    return cluster_df

    print("✅ All clustering tasks completed.")
