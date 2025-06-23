import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from pathlib import Path
import os


# current_file = Path(__file__).resolve()
# PROJECT_ROOT = current_file.parents[1]  # geht eine Ebenen nach oben

# gdf = gpd.read_file(os.path.join(PROJECT_ROOT,"Geodata/data/result_shapefile"))


def request(geometry,gdf : gpd.GeoDataFrame, features = ["population","av_age","percent_65","pp_income","male_rate"],radius = 50, verbose = False):
    """
    handles requests to get geodata information per polygon or within an radius arround an point
    """

    if gdf.crs is None:
        raise ValueError("gdf has no valid crs")
    
    if not gdf.crs.is_projected:
        gdf = gdf.to_crs(epsg=25830)

    if isinstance(geometry, (Point,Polygon)):
        # geom_series = gpd.GeoSeries([geometry], crs="EPSG:4326" if gdf.crs.is_geographic else gdf.crs)
        # geometry = geom_series.to_crs(gdf.crs).iloc[0]

        input_crs = "EPSG:4326" # This is the key change!
        
        # Create a GeoSeries with the correct *original* CRS of the input geometry
        geom_series = gpd.GeoSeries([geometry], crs=input_crs)
        
        # Now, transform this GeoSeries to the CRS of your 'gdf'
        geometry = geom_series.to_crs(gdf.crs).iloc[0]

    else:
        raise ValueError("geometry has to be a shapely point or polygon")
    

    # convert point to buffer with radius meters
    if isinstance(geometry, Point):
        geom = geometry.buffer(radius)
    else:
        geom = geometry

    gdf_copy = gdf.copy()
    gdf = gdf.copy()

    gdf["intersection"] = gdf.geometry.intersection(geom)
    gdf["inter_area"] = gdf["intersection"].area
    gdf["poly_area"] = gdf.geometry.area
    gdf["area_ratio"] = gdf["inter_area"] / gdf["poly_area"]

    relevant = gdf[gdf["inter_area"] > 0].copy()

    results = {}

    for feature in features:
        if feature not in gdf.columns:
            raise ValueError(f"{feature} is not in geodataframe")
        
        # gewichteter Durchnschitt
        if feature in ["av_age","percent_65","pp_income","male_rate"]:
            relevant[f"{feature}_weighted"] = relevant["area_ratio"] * relevant[feature]
            result = relevant[f"{feature}_weighted"].sum() / relevant["area_ratio"].sum()

        # absoluter Wert (population)
        else:
            relevant[f"{feature}_part"] = relevant["area_ratio"] * relevant[feature]
            result = relevant[f"{feature}_part"].sum()
        
        results[feature] = round(result,2)

    # zusätzlich: buffer um polygon um "einzugsgebit des clusters zu definieren"

    gdf_copy = gdf.copy()

    # print(geom)
    # print(type(geom))
    # print(radius)
    # print(type(radius))

    geomBuffer = geom.buffer(radius)
    
    gdf_copy["intersection"] = gdf_copy.geometry.intersection(geomBuffer)
    gdf_copy["inter_area"] = gdf_copy["intersection"].area
    gdf_copy["poly_area"] = gdf_copy.geometry.area
    gdf_copy["area_ratio"] = gdf_copy["inter_area"] / gdf_copy["poly_area"]

    relevantBuffer = gdf_copy[gdf_copy["inter_area"] > 0].copy()

    relevantBuffer[f"pop_buffer_part"] = relevantBuffer["area_ratio"] * relevantBuffer["population"]
    result = relevantBuffer[f"pop_buffer_part"].sum()
        
    results["pop_buffer"] = round(result - results["population"],0)

    # – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – –

    results["population"] = round(results["population"],0)

    
    # Falls verbose True oder Feature-Name ist → Plot erzeugen
    if verbose:
        print("\nBerechnete Werte:")
        for key, val in results.items():
            print(f"{key}: {val:,.2f}")

        # Bestimme das Feature für die Heatmap
        if isinstance(verbose, str):
            plot_feature = verbose
            if plot_feature not in features:
                raise ValueError(f"Feature '{plot_feature}' nicht in angefragten Features enthalten.")
        else:
            plot_feature = features[0]

        # Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        gdf.plot(ax=ax, facecolor='none', edgecolor='gray', linewidth=0.5)
        #relevant.plot(ax=ax, column=plot_feature, cmap='OrRd', legend=True, edgecolor='black')
        relevantBuffer.plot(ax=ax, column=plot_feature, cmap='OrRd', legend=True, edgecolor='black')
        gpd.GeoSeries([geom], crs=gdf.crs).plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=2)
        gpd.GeoSeries([geomBuffer], crs=gdf.crs).plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2)

        # Zoom auf Geometrie
        minx, miny, maxx, maxy = geom.bounds
        buffer = max(maxx - minx, maxy - miny) * 4
        ax.set_xlim(minx - buffer, maxx + buffer)
        ax.set_ylim(miny - buffer, maxy + buffer)
        ax.set_title(f"Überlagerung: {plot_feature}", fontsize=14)
        plt.tight_layout()
        plt.show()
    return results



if __name__ == "__main__":

    # poly2 = Polygon([(441000, 4470000), (442000, 4470000), (442000, 4471000), (441000, 4471000)])  # Madrid Ost

    poly3 = Polygon([
        (445000, 4470000),  # untere linke Ecke
        (448000, 4470000),  # untere rechte Ecke
        (448000, 4473000),  # obere rechte Ecke
        (445000, 4473000)   # obere linke Ecke
    ])

    poly3 = Polygon([(-3.554284231745193, 40.39683038034195), (-3.5533451, 40.3976789), (-3.5529744, 40.3977102), (-3.5533087, 40.3973056), (-3.554284231745193, 40.39683038034195)])


    # Falls gdf noch kein CRS hat, unbedingt zuweisen!
    # gdf.set_crs("EPSG:4326", inplace=True)

    # # Dann reprojizieren (z. B. nach EPSG:25830 für Spanien)
    # gdf = gdf.to_crs(epsg=25830)


    # print(gdf.head())

    # raise SystemExit

    print(request(poly3,gdf, verbose = "pp_income"))