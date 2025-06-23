import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
from data import request_geodata
from data import cities
import geopandas as gpd
import os
from pathlib import Path
from shapely.geometry import Polygon
import time
import geopandas as gpd
from shapely import wkt
import pandas as pd
import io # Import the io module
from assets.navigation import display_sidebar_navigation
import similar_cluster
from data import retailers
import json

BASE_URL = "http://localhost:8501" # <- change to run on server or local

st.logo("logo_locifai_tha.png", icon_image="logo_locifai_2_min.png",size="large")

current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parents[1]

# ensures a valid city is selected; otherwise force to return to home - - - - -
def store_selected_city():
    # Initialize session_state.selected_city if it doesn't exist
    if "selected_city" not in st.session_state:
        st.session_state.selected_city = None

    # 2. Check for 'city' in query parameters first.
    # If a 'city' query parameter exists, it overrides any existing session state
    # (this handles direct URL access or shared links).
    query_param_city = st.query_params.get("city", None)

    if query_param_city is not None and query_param_city != st.session_state.selected_city:
        st.session_state.selected_city = query_param_city
        # Also update the query params to ensure consistency if it was modified
        # (e.g., if a user manually typed an invalid city into the URL,
        # you might want to normalize it, though your current code doesn't do that)
        # For now, let's just make sure the URL reflects what we just pulled.
        st.query_params["city"] = query_param_city

    # 3. If no city is in query parameters, but one is in session state, set the query parameter.
    # This makes the URL shareable even if the city was selected via the selectbox and then
    # the page was refreshed.
    elif st.session_state.selected_city is not None and st.query_params.get("city") != st.session_state.selected_city:
        st.query_params["city"] = st.session_state.selected_city


    # Now, use the selected_city from session state
    selected_city = st.session_state.selected_city

    # Check if a city is actually selected by now
    if selected_city is None:
        st.error("You need to select a city first.")
        if st.button("Return home", key="return_home_button_top"): # Added unique key
            st.switch_page("home.py")
        st.stop() # Stop execution if no city is selected
    elif selected_city not in cities.names:
        st.error(f"**KeyError:** *'{selected_city}'* is not a valid city.")
        st.write(f"Available Cities are {cities.names}. **Warning:** the names are case-sensitive")

        if st.button("Return home", key="return_home_button_bottom"): # Added unique key
            st.switch_page("home.py")
        st.stop() # Stop execution if no city is selected
    return selected_city

selected_city = store_selected_city()
city = selected_city
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

dfPOIs = cities.load_pois(city)
dfCluster = cities.load_hubs(city)

display_sidebar_navigation()

output = None

# --- Tausendertrennzeichen Funktion ---
def tausendertrennzeichen(num, n_decimals=2):
    s = f"{num:,.{n_decimals}f}"
    if s.endswith(".00"):
        s = s[:-3]
    return s.replace(",", " ")

st.markdown("""<p style="font-size:16px; margin:0; padding:0;"> 🧑‍🎨 Custom view</p>""", unsafe_allow_html=True)
st.markdown(f"<h1 style='margin-top:-15px;'> Custom Hub </h1>", unsafe_allow_html=True)

if st.button("← Back to Homepage"):
        st.switch_page("home.py")

# --- Tausendertrennzeichen Funktion ---
def tausendertrennzeichen(num, n_decimals=2):
    s = f"{num:,.{n_decimals}f}"
    if s.endswith(".00"):
        s = s[:-3]
    return s.replace(",", " ")

# Create map

retailer = st.selectbox("Select a retailer to display the store portoflio",retailers.names[selected_city],index=None)


m = folium.Map(location=(cities.center[city][0],cities.center[city][1]), zoom_start=12)

# Allow drawing polygons only
Draw(
    export=False,
    draw_options={
        'polyline': False,
        'rectangle': False,
        'circle': False,
        'marker': False,
        'circlemarker': False,
        'polygon': True
    },
    edit_options={'edit': False, 'remove': True}
).add_to(m)

with open(cities.load_boundaries(city)) as f:
    geojson_data = json.load(f)

folium.GeoJson(
    geojson_data,
    name='polygon',
    style_function=lambda feature: {
        'fillColor': '#3388ff',    # You can change this to any hex color
        'color': '#3388ff',        # Border color
        'weight': 2,               # Border thickness
        'fillOpacity': 0.07         # Transparency (0 = fully transparent, 1 = fully opaque)
    }
).add_to(m)

if retailer is not None:
    df_retailer = dfPOIs[dfPOIs['brand'].str.contains(retailer, case=False, na=False)]

    for _, row in df_retailer.iterrows():
        popup_lines = []
        popup_text = row['name'] if pd.notna(row['name']) else retailer
        popup_lines.append(f"<b>Name:</b> {popup_text}")

        # Add Google Maps link
        maps_url = f"https://www.google.com/maps/search/?api=1&query={row['lat']},{row['lon']}"
        popup_lines.append(f'<br><a href="{maps_url}" target="_blank" style="color: #FF0250; text-decoration: none;">View on Google Maps</a>')
        
        full_popup_html = "<br>".join(popup_lines)

        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=folium.Popup(full_popup_html, max_width=300)
        ).add_to(m)

# Show map
output = st_folium(m, use_container_width=True,key="map")
allow_analyze = False
polygon = None # Initialize polygon

if output is not None and output["all_drawings"] is not None:
    #st.write(output["all_drawings"])
    # Process result
    if output and "all_drawings" in output:
        features = output["all_drawings"]  # It's a list
        if len(features) > 1:
            allow_analyze = False
            st.error("Only one polygon is allowed. Please remove the extra one(s).")
            st.toast("Only one polygon is allowed",icon="❌")
            # Do not st.stop() here, allow user to remove polygons
        elif len(features) == 1:
            feature = features[0]
            if feature["geometry"]["type"] not in ["Polygon"]:
                st.write(feature["geometry"]["type"] )
                st.warning("Please draw a polygon.")
                allow_analyze = False
            else:
                allow_analyze=True
                coordinates = feature["geometry"]["coordinates"][0]  # Outer ring of the polygon
                polygon = Polygon(coordinates)
                #st.json(feature)
        elif len(features) == 0: # If all polygons are removed
            allow_analyze = False
            polygon = None
    else:
        allow_analyze = False
else:
    allow_analyze = False


# Store polygon in session state so it persists across reruns
if "current_polygon" not in st.session_state:
    st.session_state.current_polygon = None
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "cluster_features" not in st.session_state:
    st.session_state.cluster_features = None
if "gdfPOIs_in_cluster" not in st.session_state:
    st.session_state.gdfPOIs_in_cluster = None
if "polygon_for_map" not in st.session_state:
    st.session_state.polygon_for_map = None


if allow_analyze and polygon:
    st.session_state.current_polygon = polygon
    st.success("Polygon captured. Start analyzing now.")
else:
    st.session_state.current_polygon = None
    # Reset analysis_done if polygon is removed or not allowed
    st.session_state.analysis_done = False
    st.info("Start your analysis by creating a hub on the map. Click the ⬟️ symbol.")


# Use a boolean to track if analysis should run this rerun
should_run_analysis = False

# "Analyze" button
if st.button(label="Analyze", disabled=not allow_analyze, key="analyze_button"):
    should_run_analysis = True

# Condition for running analysis or displaying previous results
if should_run_analysis or st.session_state.get('analysis_done', False):
    # Only run the analysis if the button was clicked OR if a polygon is present
    # AND analysis has not been done yet for this polygon (or needs re-running)
    if should_run_analysis and st.session_state.current_polygon is not None:
        with st.status("Loading Geodata...",expanded=False) as status:
            polygon_to_analyze = st.session_state.current_polygon
            if polygon_to_analyze is None:
                st.error("No polygon to analyze. Please draw one.")
                status.update(label="Error",state="error")
                st.session_state.analysis_done = False

            else:
                # geodata – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – –
                gdf_filepath = os.path.join(PROJECT_ROOT,"Dashboard_2/data/result_shapefile")
                gdf = gpd.read_file(gdf_filepath)
                # time.sleep(0.7) # UX

                status.update(label="Analyzing Geodata...")
                cluster_features = request_geodata.request( geometry=polygon_to_analyze,
                                                            gdf=gdf,
                                                            features = ["population", "av_age", "percent_65", "pp_income", "male_rate"])
                # – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – –

                # retailers – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – –
                dfPOIs['geometry'] = dfPOIs['geometry'].apply(wkt.loads)
                gdfPOIs = gpd.GeoDataFrame(dfPOIs,geometry="geometry")
                gdfPOIs_in_cluster = gdfPOIs[gdfPOIs.geometry.within(polygon_to_analyze)]

                status.update(label="Success", state="complete")
                st.session_state.cluster_features = cluster_features
                st.session_state.gdfPOIs_in_cluster = gdfPOIs_in_cluster
                st.session_state.polygon_for_map = polygon_to_analyze
                st.session_state.analysis_done = True

    # Display the results if analysis has been performed
    if st.session_state.get('analysis_done', False) and st.session_state.cluster_features is not None:
        cluster_features = st.session_state.cluster_features
        gdfPOIs_in_cluster = st.session_state.gdfPOIs_in_cluster
        polygon_for_map = st.session_state.polygon_for_map

        # KPI-Boxen
        st.header("Detail view")
        col1, col2, col3 = st.columns(3)
        col1.metric("Average age", int(cluster_features['av_age']))
        col2.metric("Average income", f"{tausendertrennzeichen(int(cluster_features['pp_income']))} €")
        col3.metric("Population 200m radius", tausendertrennzeichen(int((cluster_features['population']+cluster_features['pop_buffer']))))


        # POI-Map erstellen

        map_pois = folium.Map(location=[polygon_for_map.centroid.y, polygon_for_map.centroid.x], zoom_start=15)


        coords = [(lat, lon) for lon, lat in polygon_for_map.exterior.coords]
        folium.Polygon(
            locations=coords,
            color="blue",
            fill=True,
            fill_opacity=0.05
        ).add_to(map_pois)

        tooltip_cols = ['shop', 'amenity', 'tourism', 'healthcare', 'office', 'leisure']

        for _, row in gdfPOIs_in_cluster.iterrows():
            popup_lines = []

            # Create a copy of the row to drop columns for the popup
            poi_row_for_popup = row.copy()
            cols_to_drop = ["geometry", "lon", "lat", "lat_rnd", "lon_rnd", "name_norm", "addr:street", "addr:housenumber", "addr:postcode"]
            # Add city-specific column if it exists and is relevant
            if f"{city}_hdbscan_5_5" in poi_row_for_popup.index:
                cols_to_drop.append(f"{city}_hdbscan_5_5")

            # Drop columns if they exist in the Series
            poi_row_for_popup = poi_row_for_popup.drop(labels=[col for col in cols_to_drop if col in poi_row_for_popup.index], errors='ignore')

            for col in poi_row_for_popup.index:
                val = poi_row_for_popup[col]
                if pd.notnull(val) and str(val).strip() != "":
                    popup_lines.append(f"<b>{col}:</b> {val}")

            # Google Maps-Link
            lat = row.geometry.y
            lon = row.geometry.x
            Maps_url = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
            popup_lines.append(f'<br><a href="{Maps_url}" target="_blank" style="color: #FF0250; text-decoration: none;">View on Google Maps</a>')

            popup_html_poi = "<br>".join(popup_lines)

            tooltip_lines = []
            for col in tooltip_cols:
                if col in row and pd.notnull(row[col]):
                    tooltip_lines.append(f"{row[col]}")
            tooltip_html = " – ".join(tooltip_lines)

            folium.CircleMarker(
                location=[lat, lon],
                radius=8,  # Changed radius to 8 as in example
                color='#FF0250',
                weight=2,
                fill=True,
                fill_color='#f9ddf4',
                fill_opacity=0.8,
                popup=folium.Popup(popup_html_poi, max_width=300, min_width=200), # Added min_width
                tooltip=tooltip_html
            ).add_to(map_pois)

        map_pois.fit_bounds(map_pois.get_bounds())
        st_folium(map_pois, use_container_width=True, key="pois_map")

        # Xslx-Download der dargestellten POIs
        # Kopie der POIs, optional ohne 'geometry'
        download_df = gdfPOIs_in_cluster.copy() # Use gdfPOIs_in_cluster as cd_pois
        download_df.drop(columns=["geometry"], errors="ignore", inplace=True)

        # Excel in den Speicher schreiben
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            download_df.to_excel(writer, index=False, sheet_name="POIs")
            output.seek(0)

        c1, c2 = st.columns(2)
        with c1:
            # Google-Maps-Link Button
            # You need an average latitude and longitude for the cluster.
            # Assuming you want the centroid of the drawn polygon for the cluster link.
            avg_lat_cluster = polygon_for_map.centroid.y
            avg_lon_cluster = polygon_for_map.centroid.x
            maps_url = f"https://www.google.com/maps/search/?api=1&query={avg_lat_cluster},{avg_lon_cluster}"
            st.link_button("🔍 View on Google Maps", url=maps_url)

        with c2:
            # Download-Button anzeigen
            # You'll need a way to generate cluster_id if it's used in the file name.
            # For simplicity, I'll use a placeholder or remove it if not critical.
            # Let's use a timestamp or a generic name for now.
            file_name_suffix = int(time.time()) # Or a more meaningful ID if available
            st.download_button(
                label="📥 Export Stores (xlsx)",
                data=output,
                file_name=f"pois_{city}_cluster_custom.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


import streamlit as st
import pandas as pd # Assuming pandas is used for DataFrames

# --- Initialization of Session State Variables ---
# This block should ideally be at the very top of your Streamlit script
# to ensure all necessary session state variables are initialized before any
# part of the script tries to read them.
if "last_selected_retailer" not in st.session_state:
    st.session_state.last_selected_retailer = None

if "analysis_results_df" not in st.session_state: # Renamed for clarity (DataFrame)
    st.session_state.analysis_results_df = None

if "analysis_distances_df" not in st.session_state: # Renamed for clarity (DataFrame)
    st.session_state.analysis_distances_df = None

if "analysis_distance_metric" not in st.session_state: # Renamed for clarity (single metric)
    st.session_state.analysis_distance_metric = None

# --- Your existing code context ---
# Assuming gdfPOIs_in_cluster, dfCluster, dfPOIs, retailers, city, similar_cluster
# are defined and available from earlier parts of your script.

try:
    if gdfPOIs_in_cluster is not None:
        allow_comparsion = True
except NameError: # Catch NameError if gdfPOIs_in_cluster is not defined
    allow_comparsion = False
except Exception: # Catch any other potential error during check
    allow_comparsion = False


if allow_comparsion:
    st.subheader("Compare to a retailer's portfolio")

    selected_retailer_for_comparison = st.selectbox(
        "Select a retailer",
        retailers.names[selected_city],
        key="selected_retailer_for_comparison",
        index=retailers.names[selected_city].index(retailer) if retailer is not None else None,
        placeholder="Select a retailer"
    )

    # Logic to determine if analysis should run
    # 1. Retailer selection changed
    # 2. No results are currently stored (first run or session reset/error)
    # 3. Explicit button click
    trigger_analysis = False
    if st.session_state.last_selected_retailer != selected_retailer_for_comparison:
        trigger_analysis = True
    elif st.session_state.analysis_results_df is None: # Check if results are not yet available
        trigger_analysis = True

    if trigger_analysis and selected_retailer_for_comparison is not None:
        st.session_state.last_selected_retailer = selected_retailer_for_comparison # Update last selected retailer

        with st.status("Analyzing retailer portfolio...", expanded=True) as status:
            try:
                # Call your core function
                results, distances = similar_cluster.evaluate(
                    gdfPOIs_in_cluster,
                    dfCluster,
                    dfPOIs,
                    retailer=selected_retailer_for_comparison,
                    city=city
                )

                # Ensure results is a DataFrame and has expected columns before dropping
                if isinstance(results, pd.DataFrame) and "geometry" in results.columns:
                    results = results.drop(columns=["geometry"])
                # If results is not a DataFrame, or malformed, it will raise an error here,
                # which will be caught by the except block.

                # Store the results in session state
                st.session_state.analysis_results_df = results
                st.session_state.analysis_distances_df = distances

                # Calculate the metric, ensuring `results` is valid
                st.session_state.cc_rank  = results[results["cluster_label"] == 999999].index
                st.session_state.cc_distance_nearest = float(results.loc[results["cluster_label"] == 999999, "distance_m"].iloc[0])

                cc_stats = results[results["cluster_label"] == 999999].reset_index()

                if not cc_stats.empty and "distance_to_retailers_profile" in cc_stats.columns:
                    st.session_state.analysis_distance_metric = cc_stats.at[0, "distance_to_retailers_profile"]
                else:
                    st.session_state.analysis_distance_metric = 0.0 # Default if label not found or column missing
                    st.warning("Cluster label 999999 or 'distance_to_retailers_profile' not found. Distance defaulted to 0.0.")

                # if st.button("Rerun"):
                #     trigger_analysis = True

                status.update(label="Success", state="complete", expanded=False)


            except Exception as e:
                # Set all session state results to None on error to clear previous valid results
                st.session_state.analysis_results_df = None
                st.session_state.analysis_distances_df = None
                st.session_state.analysis_distance_metric = None
                st.session_state.last_selected_retailer = None # Reset last selected to force re-run on next select

                # Use st.error and st.exception to provide more detail
                # If 'e' truly is '0', st.exception will still show a traceback
                # that helps understand why a non-Exception is being caught.
                st.error(f"An error occurred during analysis: {e}")
                st.exception(e) # Re-adding this line in comments, as it's the best debugger if error persists.
                                # If you get "0" again, uncomment this to see the traceback.

                status.update(label="Error", state="error")

    # --- Display results if they are available in session state ---
    # This block will run on every rerun, displaying the *persisted* results
    # or nothing if results are None.
    if st.session_state.analysis_results_df is not None and st.session_state.analysis_distance_metric is not None:

        min_dis = st.session_state.analysis_distances_df["min"]
        max_dis = st.session_state.analysis_distances_df["max"]

        c1,c2,c3 =  st.columns(3)
        c1.metric(f"Distance on a scale from {round(min_dis,2)} – {round(max_dis,2)}", f"{round(st.session_state.analysis_distance_metric, 2)}")
        c2.metric(f"Rank from 0 – {len(st.session_state.analysis_results_df)} ",st.session_state.cc_rank)
        c3.metric(f"Nearest {selected_retailer_for_comparison} (Meters)",round(st.session_state.cc_distance_nearest,2))

        st.link_button(f"View Whitespace Analysis for {selected_retailer_for_comparison}",f"{BASE_URL}/retailer-detail?city={selected_city}&retailer={selected_retailer_for_comparison}")

        # st.write(st.session_state.analysis_distances_df)