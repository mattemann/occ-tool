import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances #, cosine_distances
import numpy as np
from shapely import wkt
from data import cities
import streamlit as st
from data.custom_cluster_stats import request_cluster_stats
from data.custom_cluster_geodata import add_geodata_custom_cluster
from data import retailers

def search(dfCluster: pd.DataFrame, dfPOIs: pd.DataFrame, retailer = "", display_N = 20, weights = {
    "num_points"            : 7.0,
    "density"               : 8.0,
    "pop_buffer"            : 3.0,
    "pp_income"             : 4.0,
    "av_age"                : 3.0,
    "public_transport"      : 3.0},
    city = "",
    from_evaluate = False):
    """
    pass the (already filterd) dfClusters as search matrix and the target retailer
    dfPOIs is used to calculate the nearest retailer out of a cluster
    """

    from shapely.geometry.base import BaseGeometry
    from shapely import wkt

    dfCluster["geometry"] = dfCluster["geometry"].apply(
        lambda x: wkt.loads(x) if isinstance(x, str) else x if isinstance(x, BaseGeometry) else None
)

    gdfCluster = gpd.GeoDataFrame(dfCluster, geometry="geometry", crs="EPSG:4326")

    # Reproject to estimated UTM CRS
    estimate_cluster_crs = gdfCluster.estimate_utm_crs()
    gdfCluster = gdfCluster.to_crs(estimate_cluster_crs)

    dfPOIs["geometry"] = dfPOIs["geometry"].apply(wkt.loads)
    gdfPOIs = gpd.GeoDataFrame(dfPOIs, geometry="geometry", crs="EPSG:4326")

    # Reproject to estimated UTM CRS from above
    gdfPOIs = gdfPOIs.to_crs(estimate_cluster_crs)

    # fillna with 0, remove unwanted columns

    dfCluster = dfCluster.fillna(0)
    to_drop = ["Unnamed: 0","param_config", "has_shop", "geometry","cluster_label",
               "male_rate","median_lon","median_lat","std_lon","std_lat","Predicted_Segment"]
    dfCluster.drop(columns=[col for col in to_drop if col in dfCluster.columns], inplace=True)

    # split data frame in has retailer: true/false
    df_true = dfCluster[dfCluster[f"has_{retailer}"] >= 1].copy()

    # df_false = dfCluster.copy() #ATTENTION <- HAS TO BE REMOVED
    # print("IMPORTANT: FROM similar_cluster.py df_false is wrongly defined for testing")

    df_false = dfCluster[dfCluster[f"has_{retailer}"] == 0].copy()

    features = [col for col in dfCluster.columns if col != f"has_{retailer}"]

    # Anteil an Nicht-Null-Werten pro Feature
    non_zero_ratio = (dfCluster[features] != 0).sum() / len(dfCluster)

 
    # Behalte nur Features, in denen >1% der Cluster einen Wert ungleich 0 haben
    filtered_features = non_zero_ratio[non_zero_ratio > 0.01].index.tolist()

    X_true = df_true[filtered_features].fillna(0) # stelle nochmal sicher dass keine NAs mehr vorhanden
    X_false = df_false[filtered_features].fillna(0)

    # Skalieren
    scaler = StandardScaler()
    X_true_scaled = scaler.fit_transform(X_true)
    X_false_scaled = scaler.transform(X_false)

    # Definiere feature weights
    feature_weights = pd.Series(1, index=filtered_features)
    feature_weights["num_points"]   = weights["num_points"]
    feature_weights["density"]      = weights["density"]
    feature_weights["pop_buffer"]   = weights["pop_buffer"]
    feature_weights["pp_income"]    = weights["pp_income"]
    feature_weights["av_age"]       = weights["av_age"]
    feature_weights["bus_in_buffer"]= weights["public_transport"]
    feature_weights["trains_metro_in_buffer"] = weights["public_transport"]

    feature_weights = feature_weights.reindex(filtered_features).fillna(1)

    # wende gewichtung an
    X_true_weighted = X_true_scaled * feature_weights.values
    X_false_weighted = X_false_scaled * feature_weights.values

    # create retailer profile
    retailer_prfofile = X_true_weighted.mean(axis=0).reshape(1, -1)

    # calculate distances
    distances = euclidean_distances(retailer_prfofile, X_false_weighted)[0]
    
    # Top N simliar clusters
    N = display_N
    top_indices = distances.argsort()[:N]

    top_distances = {   "min" : distances.min(),
                        "max"  : distances.max()}

    # results
    similar_clusters = df_false.iloc[top_indices].copy()
    similar_clusters["distance_to_retailers_profile"] = distances[top_indices]
    # similar_clusters["distance_norm"] = dist_norm[top_indices]
    # similar_clusters["similarity_score"] = 1 - similar_clusters["distance_norm"]

    print(similar_clusters)

    gdfPOIs = gdfPOIs[gdfPOIs["brand"] == retailer]

    gdfCluster_top = gdfCluster.iloc[top_indices].copy()
    gdfCluster_top["cluster_label"] = gdfCluster_top.index

    # search nearest retailer to cluster
    joined_gdf = gpd.sjoin_nearest(
            left_df=gdfCluster,
            right_df=gdfPOIs,
            how='left',
            distance_col='distance_m'
        )

    joined_gdf = joined_gdf.loc[~joined_gdf.index.duplicated(keep='first')]

    similar_clusters["cluster_label"] = similar_clusters.index

    if from_evaluate:
        similar_clusters.loc[similar_clusters["cluster_label"].idxmax(), "cluster_label"] = 999999


    similar_clusters = similar_clusters.merge(
        joined_gdf[["cluster_label", f"{city}_hdbscan_5_5", "distance_m","geometry","Predicted_Segment"]],
        on="cluster_label",
        how="left",
    )

    # print(similar_clusters)
    # print(top_distances)

    return similar_clusters[["distance_to_retailers_profile"] + features + ["cluster_label", f"{city}_hdbscan_5_5", "distance_m","geometry","Predicted_Segment"]], top_distances


def evaluate(gdfPOIs_InCluster: gpd.GeoDataFrame,dfAllCluster: pd.DataFrame, dfAllPOIs: pd.DataFrame, retailer = "", city = ""):
    # input ist ein cluster; output muss eine distance sein zu dem average cluster + 
    # vergleich mit der range

    gdfPOIs_InCluster.drop(["lon_rnd","lat_rnd","name_norm",f"{city}_hdbscan_5_5"],axis=1,inplace=True,errors='ignore')

    dfCustomClusterStats = request_cluster_stats(city=city,dfPOIs=gdfPOIs_InCluster)
    print("#### Custom cluster stats")
    print(dfCustomClusterStats[["num_points"]])
    
    dfCustomClusterStats = add_geodata_custom_cluster(dfCustomClusterStats,city = city, retailers = getattr(retailers, f"names_{city.lower()}"))

    dfCombinedCluster = pd.concat([dfAllCluster,dfCustomClusterStats], ignore_index=True, sort=False)
    dfCombinedCluster = dfCombinedCluster[dfAllCluster.columns]    

    print(dfCombinedCluster)

    results,distances = search( dfCluster    = dfCombinedCluster,
                                dfPOIs       = dfAllPOIs,
                                retailer     = retailer,
                                city         = city,
                                display_N    = 999999,
                                from_evaluate= True)
    
    print(results)
    print(distances)

    return results, distances

    pass


if __name__ == "__main__":

    dfPOIs = cities.load_pois("Madrid")
    df = cities.load_pois("Madrid").sample(20)
    dfCluster    = cities.load_hubs("Madrid")

    evaluate(df,dfCluster,dfPOIs,"Starbucks","Madrid")

    raise SystemExit

    search(dfCluster    = cities.load_hubs("Madrid"),
        dfPOIs          = cities.load_pois("Madrid"),
        retailer="Starbucks",
        display_N=20,
        city="Madrid")
