import pandas as pd
import geopandas as gpd
from pathlib import Path
import os
from data.request_geodata import request
from collections import Counter
from data.request_public_transport import load_data, request_stations
from shapely import wkt
from shapely.geometry.base import BaseGeometry


"""
* merges geojson with cluster polygons with the stats csv into one geodataframe
* adds the geodata to each polygon
"""

global dfCluster, dfPOIs, city, names_retailers

current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parents[2]  # geht eine Ebenen nach oben
# print("ROOT", PROJECT_ROOT)

# dfCluster = dfCluster.sample(50)

def add_geodata(dfCluster):
    print("start adding geodata")
    # add geodata to dfCluster (e.g. population)
    for index,row in dfCluster.iterrows():
        if index % 10 == 0: print(index)
        # if index == 5: return None
        print(row["geometry"])
        # print(type(row["geometry"]))

        geo_features = request(geometry = wkt.loads(row["geometry"]) if not isinstance(row["geometry"], BaseGeometry) else row["geometry"], 
                               gdf = gpd.read_file(os.path.join(PROJECT_ROOT,"Dashboard_2","data","result_shapefile")),
                               verbose=False,
                               radius=200)
        for feature in geo_features:
            dfCluster.at[index, feature] = geo_features[feature]

    return dfCluster


# rename retail chains  – – – – – – – – – – – – – – – – – – – – – – – – – – – –
def rename_retail_chains(map=dict, dfPOIs = ""):
    dfPOIs["brand"] = dfPOIs["brand"].map(map).fillna(dfPOIs["brand"])
    return dfPOIs
# – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – –

def add_retail_chains_exist(retail_chains = "", banks = [""], dfPOIs = "", dfCluster = "", city = ""):

    print("Start adding if retail chains exists")

    for index,row in dfCluster.iterrows():
        if index % 10 == 0: print(index)
        current_cluster_id = row["cluster_label"]
        # POIs in current Cluster
        dfPOIs_cc = dfPOIs[dfPOIs[f"{city}_hdbscan_5_5"] == current_cluster_id]

        for retailer in retail_chains:
            if retailer in dfPOIs_cc["brand"].to_list():
                dfCluster.at[index, f"has_{retailer}"] = Counter(dfPOIs_cc["brand"].to_list()).get(retailer)
            else: 
                dfCluster.at[index, f"has_{retailer}"] = 0

    return dfCluster
        # print(dfPOIs_cc)

def add_sgementation_data(city, dfCluster):
    # add the data fom the segmentation dataframe with infos like number_public_transport

    gdf = load_data(city=city)

    print("Start adding segmentation data")

    for index,row in dfCluster.iterrows():
        if index % 10 == 0: print(index)
        pub_transport_features = request_stations(geometry = wkt.loads(row["geometry"]) if not isinstance(row["geometry"], BaseGeometry) else row["geometry"],
                                                  features=["trains_metro","bus"],
                                                  gdf=gdf,verbose=False, radius=200)
        print(pub_transport_features)
        for feature in pub_transport_features:
            dfCluster.at[index, feature] = pub_transport_features[feature]

    return dfCluster

def add_geodata_custom_cluster(df: pd.DataFrame, city = "", retailers = ""):
    dfCluster = df
    city = city
    names_retailers = retailers
    dfPOIs = pd.read_csv(os.path.join(PROJECT_ROOT, "clustering_results", f"{city}_hdbscan_points.csv"))

    dfCluster = add_geodata(dfCluster)

    dfPOIs = rename_retail_chains({"La Plaza de DIA" : "Dia", "Dia Market" : "Dia", "Maxi Dia" : "Dia", "Dia & Go" : "Dia",
                      "Carrefour Express" : "Carrefour", "Carrefour Market" : "Carrefour", "Carrefour Expres" : "Carrefour"},
                      dfPOIs=dfPOIs)

    dfCluster = add_retail_chains_exist(names_retailers, dfPOIs=dfPOIs,dfCluster=dfCluster,city=city)

    dfCluster = add_sgementation_data(city, dfCluster=dfCluster)


    # dfCluster.to_csv(os.path.join(PROJECT_ROOT,"clustering_results",f"cluster_allFeatures_{city}.csv"))

    print(dfCluster)

    return dfCluster


if __name__ =="__main__":

    import cities,retailers


    add_geodata_custom_cluster(cities.load_hubs("Madrid").sample(1),city = "Madrid", retailers = getattr(retailers, f"names_madrid"))


