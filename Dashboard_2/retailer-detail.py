import streamlit as st
from data import cities
from data import retailers
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
import folium
import altair as alt
from shapely import wkt
from shapely.ops import transform
import time
from pyproj import Transformer
from pathlib import Path
import os
from assets.navigation import display_sidebar_navigation


# BASE_URL = "http://localhost:8501" # <- change to run on server or local
BASR_URL = "https://locifai-tool-717627512976.europe-west1.run.app"

current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parents[1] 

st.logo("Dashboard_2/logo_locifai_tha.png", icon_image="Dashboard_2/logo_locifai_2_min.png",size="large")

# ensures a valid city+retailer is selected; otherwise force to return to home -
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
        if st.button("Return home"):
            st.switch_page("home.py")
        st.stop() # Stop execution if no city is selected
    elif selected_city not in cities.names:
        st.error(f"**KeyError:** *'{selected_city}'* is not a valid city.")
        st.write(f"Available Cities are {cities.names}. **Warning:** the names are case-sensitive")
    
        if st.button("Return home"):
            st.switch_page("home.py")
        st.stop() # Stop execution if no city is selected
    return selected_city

selected_city = store_selected_city()

def store_selected_retailer():
    # Initialize session_state.selected_retailer if it doesn't exist
    if "selected_retailer" not in st.session_state:
        st.session_state.selected_retailer = None

    # 2. Check for 'retailer' in query parameters first.
    # If a 'retailer' query parameter exists, it overrides any existing session state
    # (this handles direct URL access or shared links).
    query_param_retailer = st.query_params.get("retailer", None)

    if query_param_retailer is not None and query_param_retailer != st.session_state.selected_retailer:
        st.session_state.selected_retailer = query_param_retailer
        # Also update the query params to ensure consistency if it was modified
        # (e.g., if a user manually typed an invalid retailer into the URL, 
        # you might want to normalize it, though your current code doesn't do that)
        # For now, let's just make sure the URL reflects what we just pulled.
        st.query_params["retailer"] = query_param_retailer
        
    # 3. If no retailer is in query parameters, but one is in session state, set the query parameter.
    # This makes the URL shareable even if the retailer was selected via the selectbox and then
    # the page was refreshed.
    elif st.session_state.selected_retailer is not None and st.query_params.get("retailer") != st.session_state.selected_retailer:
        st.query_params["retailer"] = st.session_state.selected_retailer


    # Now, use the selected_city from session state
    selected_retailer = st.session_state.selected_retailer

    # Check if a city is actually selected by now
    if selected_retailer is None:
        st.error("You need to select a valid retailer.")
        if st.button("← Return home"):
            st.switch_page("home.py")
        st.stop() # Stop execution if no city is selected

    elif selected_retailer not in retailers.names[selected_city]:
        st.error(f"**KeyError:** *'{selected_retailer}'* is not a valid retailer.")
        st.write(f"Available retailers for {selected_city} are {retailers.names[selected_city]}. **Warning:** the names are case-sensitive")
        if st.button("← Return home"):
            st.switch_page("home.py")
        st.stop() # Stop execution if no city is selected
    return selected_retailer

selected_retailer = store_selected_retailer()
# – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – –

try:
    retailer_selected = st.query_params["retailer"]
except:
    retailer_selected = None

try:
    cluster_selected = st.session_state.selected_cluster
except:
    cluster_selected = None

# print(st.query_params["retailer"])

display_sidebar_navigation(cluster = cluster_selected, retailer = retailer_selected)

city = selected_city
retailer = selected_retailer

@st.cache_data
def load_pois(city):
    return cities.load_pois(city)

@st.cache_data
def load_hubs(city):
    return cities.load_hubs(city)

# @st.cache_data
# def load_clusters_features(city):
#     return pd.read_csv(os.path.join(PROJECT_ROOT,"clustering_results",f"cluster_allFeatures_{city}_labeled.csv"))



# Google Cloud run is case senstivie in paths

with st.spinner():
    df1 = load_pois(city)
    df2 = load_hubs(city)
    # df3 = load_clusters_features(city)



st.markdown("""<p style="font-size:16px; margin:0; padding:0;"> ℹ️ Detail View</p>""", unsafe_allow_html=True)
st.markdown(f"<h1 style='margin-top:-15px;'>{retailer}</h1>", unsafe_allow_html=True)

if st.button("← Return home"):
        st.switch_page("home.py")

description = retailers.description[selected_retailer]

st.markdown(description)

df_retailer_metrics = df2[df2[f"has_{retailer}"] >= 1]

cols = st.columns(5)
cols[0].metric("Number of Stores", df1[df1['brand'].str.contains(retailer, case=False, na=False)].shape[0])
cols[1].metric("Average Size (POIs)", round(df_retailer_metrics["num_points"].mean(),1))
cols[2].metric("Average Age",round(df_retailer_metrics["av_age"].mean()))
cols[3].metric("Average Population",round((df_retailer_metrics["population"] + df_retailer_metrics["pop_buffer"]).mean()))
cols[4].metric(f"Top Segment ({round((df_retailer_metrics["Predicted_Segment"].value_counts().max())/len(df_retailer_metrics) * 100)} %)",df_retailer_metrics["Predicted_Segment"].value_counts().idxmax())

# Initialize selected_cluster in session state if it doesn't exist
if 'selected_cluster' not in st.session_state:
    st.session_state.selected_cluster = None


with st.spinner("Karte wird geladen..."):
    # time.sleep(3)
    map_retailer = folium.Map(location=[df1.lat.mean(), df1.lon.mean()], zoom_start=12)

    # Filter auf die Spalte 'brand', fallunabhängig, NaN ausschließen
    df_retailer = df1[df1['brand'].str.contains(retailer, case=False, na=False)]


    # Marker für jede gefundene Filiale hinzufügen
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
        ).add_to(map_retailer)

    # Cluster-Bounding-Boxen für Cluster mit Retailer
    col_retailer = f"has_{retailer}"
    if col_retailer in df2.columns:
        df_cluster_filtered = df2[df2[col_retailer] >= 1.0]

        for _, row in df_cluster_filtered.iterrows():
            polygon = wkt.loads(row['geometry'])  # POLYGON -> shapely Polygon
            coords = [(lat, lon) for lon, lat in polygon.exterior.coords]  # lon/lat → lat/lon

            folium.Polygon(
                locations=coords,
                color='blue',
                fill=True,
                fill_opacity=0.1,
                weight=2,
                popup=f'Cluster {row["cluster_label"]}'
            ).add_to(map_retailer)
    

    # Map mit POIs des Hubs
    response_map_retailer = st_folium(map_retailer, use_container_width=True, returned_objects=["last_object_clicked_popup"]) #

    if response_map_retailer.get("last_object_clicked_popup"):
        selected_popup_text = response_map_retailer.get("last_object_clicked_popup")
        if selected_popup_text.startswith("Cluster "):
            try:
                cluster_id = int(selected_popup_text.split(" ")[1])
                st.session_state.selected_cluster = str(cluster_id) # Store as string to be consistent with query params
            except (ValueError, IndexError):
                pass # Handle cases where the popup text is not in the expected format for a cluster
    
    if st.session_state.selected_cluster is not None:
        # Check if the selected cluster still exists in the df_cluster_filtered data
        if int(st.session_state.selected_cluster) in df_cluster_filtered['cluster_label'].values:
            if st.button(f"ℹ️ View details for cluster {st.session_state.selected_cluster}"):
                st.query_params["cluster"] = st.session_state.selected_cluster
                st.session_state.redirected = retailer # Set redirected to current retailer
                st.switch_page("cluster-detail.py")
        else:
            st.info(f"Cluster {st.session_state.selected_cluster} is no longer visible with current filters.")
            st.session_state.selected_cluster = None # Reset selected cluster if it's filtered out
    else:
        st.info("To see details, select a cluster on the map.")

    
st.subheader("Whitespace Analysis")

with st.expander("Settings"):
    if st.toggle(label = "Show weights",value=False):
        experimental = True
    else:
        experimental = False
        weight_param_points     = 7.0
        weight_param_density    = 8.0
        weight_param_population = 3.0
        weight_param_income     = 4.0
        weight_param_age        = 3.0
        weight_param_publicTransport = 3.0

    with st.form("whitepsace-setting", border=False):
        # st.write("**Common**")
        # number of results
        NResults_slider_val = st.number_input("Select number of results",1,1000,value=20,placeholder=20)
        # checkbox_val = st.checkbox("Form checkbox")

        if experimental:
            st.divider()
            st.write("**Feature weights**")
            
            weight_param_density    = st.slider("Density",0.1,25.0,step=0.1,value=8.0)
            weight_param_points     = st.slider("Number of POIs",0.1,25.0,step=0.1,value=7.0)
            weight_param_income     = st.slider("Income",0.1,25.0,step=0.1,value=4.0)
            weight_param_population = st.slider("Population",0.1,25.0,step=0.1,value=3.0)
            weight_param_age        = st.slider("Age",0.1,25.0,step=0.1,value=3.0)
            weight_param_publicTransport = st.slider("Public Transport",0.1,25.0,step=0.1,value=3.0)


        # Every form must have a submit button.
        submitted = st.form_submit_button("Apply Settings")
        if submitted:
            whitespace = True
        else:
            whitespace = False


if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False
    st.session_state.whitespace = False

# Show the button only if it hasn't been clicked yet
if not st.session_state.button_clicked:
    if st.button(f"Start Whitespace Analysis for {retailer}"):
        st.session_state.whitespace = True
        st.session_state.button_clicked = True
        st.rerun()


if st.session_state.whitespace:
    with st.status("Finding similar clusters...",expanded=False) as status:
        from similar_cluster import search
        similar,distances = search(dfCluster  = df2,
                            dfPOIs     = df1,
                            retailer=retailer,
                            display_N=NResults_slider_val,
                            weights = {"num_points"            : weight_param_points,
                                        "density"               : weight_param_density,
                                        "pop_buffer"            : weight_param_population,
                                        "pp_income"             : weight_param_income,
                                        "av_age"                : weight_param_age,
                                        "public_transport"      : weight_param_publicTransport},
                                        city = city)
        if len(similar) == 0:
            status.update(label = "No Clusters to display: select other settings",state="error")
            st.stop()
        similar_detailed = similar.copy()

        print(similar_detailed.columns)

        similar = similar[["cluster_label", "Predicted_Segment", "distance_to_retailers_profile", "num_points", "pop_buffer", "pp_income", "av_age", "distance_m"]]
        similar["Link"] = similar["cluster_label"].apply(lambda id: f"{BASE_URL}/cluster-detail?city={city}&cluster={id}&redirected={retailer}")
        similar = similar[similar["distance_m"] != 0]
        similar["distance_m"] = similar["distance_m"].round(2)
        similar["distance_to_retailers_profile"] = similar["distance_to_retailers_profile"].round(2)
        similar = similar.rename(columns={"cluster_label" : "Cluster ID",
                        "Predicted_Segment" : "Segment type",
                        "distance_to_retailers_profile" : f"Similarity",
                        "num_points" : "Number of POIs",
                        "pop_buffer" : "Population",
                        "pp_income" : "Avg. per Person Income",
                        "av_age" : "Avg. Age",
                        "distance_m" : f"Distance to nearest {retailer} (m)"})
        status.update(label = "Success",expanded=True, state="complete")
    
        st.info(
    f"""The similarity score is a Euclidean Distance between ({round(distances["min"],2)} – {round(distances["max"],2)}).
        Be careful when comparing distances with other retailers or cities.
        [Learn More](https://en.wikipedia.org/wiki/Euclidean_distance) """
)
        
        st.button("Rerun Analysis")

    st.dataframe(data=similar,key="similar-data",hide_index=True,on_select="ignore",
                            selection_mode="single-row",column_config={
                                "Link": st.column_config.LinkColumn(display_text=f"View Cluster")}
                )
    # new map

    map_white_space = folium.Map(location=[df1.lat.mean(), df1.lon.mean()], zoom_start=12)


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
        ).add_to(map_white_space)

    transformer = Transformer.from_crs("EPSG:25830", "EPSG:4326", always_xy=True)
    
    
    # st.write(similar_detailed) <- Barcelona similar hubs dont get displayed
    for _, row in similar_detailed.iterrows():
            
        polygon = row['geometry']  # POLYGON -> shapely Polygon
        polygon_latlon = transform(transformer.transform, polygon)
        #print(polygon_latlon.is_valid)
        # print(type(polygon_latlon))
        coords2 = [(lat, lon) for lon, lat in polygon_latlon.exterior.coords]  # lon/lat → lat/lon
       
        # print(_, coords2)
        folium.Polygon(
            locations=coords2,
            color="#FF0250",
            fill=True,
            fill_opacity=0.5,
            weight=2,
            popup=f'Cluster {row["cluster_label"]}'
        ).add_to(map_white_space)

    response_ws = st_folium(map_white_space,use_container_width=True, returned_objects=["last_object_clicked_popup"])

    if response_ws.get("last_object_clicked_popup") != retailer and response_ws.get("last_object_clicked_popup") is not None:
        selected_cluster_ws = response_ws.get("last_object_clicked_popup")
        if selected_cluster_ws.startswith("Cluster "):
            selected_cluster_id_ws = selected_cluster_ws.split(" ")[1]
            st.link_button(f"Open Cluster {selected_cluster_id_ws} in new window",
                           url=f"{BASE_URL}/cluster-detail?city={city}&cluster={selected_cluster_id_ws}&redirected={retailer}")
    else:
        st.info("To see details, select a cluster on the map.")
