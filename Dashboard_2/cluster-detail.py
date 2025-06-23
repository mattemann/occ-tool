import streamlit as st
from data import cities
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from streamlit_folium import st_folium
from scipy.spatial import Delaunay
from shapely.ops import polygonize, unary_union
from shapely.geometry import MultiPoint, LineString
import folium
import altair as alt
import io
from data import retailers
from shapely import wkt
from assets.navigation import display_sidebar_navigation

st.logo("Dashboard_2/logo_locifai_tha.png", icon_image="Dashboard_2/logo_locifai_2_min.png",size="large")

# query parameters: city, cluster
# ensures a valid city+cluster is selected; otherwise force to return to home -
# checks if a redirected parameter is set
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
        st.write("Available Cities are ['Madrid', 'Barcelona', 'Valencia', 'Malaga']. **Warning:** the names are case-sensitive")
    
        if st.button("Return home"):
            st.switch_page("home.py")
        st.stop() # Stop execution if no city is selected
    return selected_city

selected_city = store_selected_city()

def store_selected_cluster():
    # Initialize session_state.selected_cluster if it doesn't exist
    if "selected_cluster" not in st.session_state:
        st.session_state.selected_cluster = None

    # 2. Check for 'cluster' in query parameters first.
    # If a 'cluster' query parameter exists, it overrides any existing session state
    # (this handles direct URL access or shared links).
    query_param_cluster = st.query_params.get("cluster", None)

    if query_param_cluster is not None and query_param_cluster != st.session_state.selected_cluster:
        st.session_state.selected_cluster = query_param_cluster
        # Also update the query params to ensure consistency if it was modified
        # (e.g., if a user manually typed an invalid cluster into the URL, 
        # you might want to normalize it, though your current code doesn't do that)
        # For now, let's just make sure the URL reflects what we just pulled.
        st.query_params["cluster"] = query_param_cluster
        
    # 3. If no cluster is in query parameters, but one is in session state, set the query parameter.
    # This makes the URL shareable even if the cluster was selected via the selectbox and then
    # the page was refreshed.
    elif st.session_state.selected_cluster is not None and st.query_params.get("cluster") != st.session_state.selected_cluster:
        st.query_params["cluster"] = st.session_state.selected_cluster


    # Now, use the selected_city from session state
    selected_cluster = st.session_state.selected_cluster

    # Check if a city is actually selected by now
    if selected_cluster is None:
        st.error("You need to select a valid city and cluster.")
        if st.button("Return home"):
            st.switch_page("home.py")
        st.stop() # Stop execution if no city is selected

        if st.button("Return home"):
            st.switch_page("home.py")
        st.stop() # Stop execution if no city is selected
    return selected_cluster

selected_cluster = store_selected_cluster()

def store_redirected():
    # Initialize session_state.selected_cluster if it doesn't exist
    if "redirected" not in st.session_state:
        st.session_state.redirected = None

    # 2. Check for 'cluster' in query parameters first.
    # If a 'cluster' query parameter exists, it overrides any existing session state
    # (this handles direct URL access or shared links).
    query_param_redirected = st.query_params.get("redirected", None)

    if query_param_redirected is not None and query_param_redirected != st.session_state.redirected:
        st.session_state.redirected = query_param_redirected
        # Also update the query params to ensure consistency if it was modified
        # (e.g., if a user manually typed an invalid cluster into the URL, 
        # you might want to normalize it, though your current code doesn't do that)
        # For now, let's just make sure the URL reflects what we just pulled.
        st.query_params["redirected"] = query_param_redirected
        
    # 3. If no cluster is in query parameters, but one is in session state, set the query parameter.
    # This makes the URL shareable even if the cluster was selected via the selectbox and then
    # the page was refreshed.
    elif st.session_state.redirected is not None and st.query_params.get("redirected") != st.session_state.redirected:
        st.query_params["redirected"] = st.session_state.redirected


    # Now, use the selected_city from session state
    redirected = st.session_state.redirected

    # Check if a city is actually selected by now
    # if redirected is None:
        
    #     st.error("You need to select a valid city and cluster.")
    #     if st.button("Return home"):
    #         st.switch_page("home.py")
    #     st.stop() # Stop execution if no city is selected

    #     if st.button("Return home"):
    #         st.switch_page("home.py")
    #     st.stop() # Stop execution if no city is selected
    return redirected

redirect = store_redirected()

city = selected_city
cluster_id = int(selected_cluster)


display_sidebar_navigation(cluster = cluster_id )


@st.cache_data
def load_pois(city):
    return cities.load_pois(city)

@st.cache_data
def load_hubs(city):
    return cities.load_hubs(city)


with st.spinner():
    df1 = load_pois(city)
    df2 = load_hubs(city)

# --- Tausendertrennzeichen Funktion ---
def tausendertrennzeichen(num, n_decimals=2):
    s = f"{num:,.{n_decimals}f}"
    if s.endswith(".00"):
        s = s[:-3]
    return s.replace(",", " ")


# df1: POIs 
cd_pois = df1[df1[f"{city}_hdbscan_5_5"] == int(cluster_id)]  # Filter auf alle POIs mit ausgewählter cluster id
# df2: Cluster features
cd_cluster = df2.loc[df2.cluster_label == cluster_id].iloc[0]  # Filter auf genau eine Zeile mit der cluster id


st.markdown("""<p style="font-size:16px; margin:0; padding:0;"> ℹ️ Detail View</p>""", unsafe_allow_html=True)
st.markdown(f"<h1 style='margin-top:-15px;'> Hub: {cluster_id}</h1>", unsafe_allow_html=True)

if redirect == "city" and city is not None:
    if st.button("← Back to City-Overview"):
        st.switch_page("city-overview.py")

elif redirect in retailers.names[city]:
    if st.button("← Back to Retailer-view"):
        st.session_state.selected_retailer = redirect
        st.query_params["retailer"] = redirect
        st.switch_page("retailer-detail.py")

else: 
    if st.button("Home"):
        st.switch_page("home.py")


# KPIs
cols = st.columns(5)
# Weise jeder Spalte eine Kennzahl zu
cols[0].metric("Number of POIs", int(cd_cluster.num_points))
cols[1].metric("Average age", int(cd_cluster.av_age))
cols[2].metric("Average income", f"{tausendertrennzeichen(int(cd_cluster.pp_income))} €")
cols[3].metric("Population 200m", tausendertrennzeichen(int(cd_cluster.population+cd_cluster.pop_buffer)))
cols[4].metric("Segment", cd_cluster.get("Predicted_Segment", "N/A"))


min_lat, max_lat = cd_pois['lat'].min(), cd_pois['lat'].max()
min_lon, max_lon = cd_pois['lon'].min(), cd_pois['lon'].max()
bounds = [[min_lat, min_lon], [max_lat, max_lon]]


with st.spinner("Map is loading..."):
    # time.sleep(3)
    map_pois = folium.Map()

    popup_html = f"""
    <div>
      <strong>Population within 200m radius:</strong> {int(cd_cluster['pop_buffer'] + cd_cluster["population"])}<br>
      <strong>Number of trains/metro:</strong> {int(cd_cluster['trains_metro_in_buffer'])}
    </div>
"""
    # 3. Erstelle das Popup
    popup = folium.Popup(
        popup_html,
        max_width=300,   # maximale Breite in Pixel
        min_width=200    # minimale Breite in Pixel
    )
    # Convert the alpha_shape string back to a list of tuples
    # coords = ast.literal_eval(row['alpha_shape'])
    polygon = wkt.loads(cd_cluster["geometry"]).buffer(0.0028)   # Anpassen auf 200m Radius
    coords  = [(lat, lon) for lon,lat in polygon.exterior.coords]
    
    # Add polygon
    folium.Polygon(
        locations=coords,
        color="blue",
        fill=True,
        fill_opacity=0.002,
        popup=popup
    ).add_to(map_pois)

    tooltip_cols = ['shop', 'amenity', 'tourism', 'healthcare', 'office', 'leisure']
    # Marker hinzufügen
    for _, row in cd_pois.iterrows():
        popup_lines = []
        cd_pois_tooltip = cd_pois.copy()
        cols_to_drop = [f"{city}_hdbscan_5_5", "geometry", "lon_rnd", "lat_rnd", "lon", "lat", "name_norm", "addr:street", "addr:housenumber", "addr:postcode"]
        cd_pois_tooltip.drop(columns=cols_to_drop, inplace=True, errors="ignore")
        for col in cd_pois_tooltip.columns:
            val = row[col]
            if pd.notnull(val) and str(val).strip() != "":
                popup_lines.append(f"<b>{col}:</b> {val}")

        # Add Google Maps link to the popup
        Maps_url = f"https://www.google.com/maps/search/?api=1&query={row['lat']},{row['lon']}"
        popup_lines.append(f'<br><a href="{Maps_url}" target="_blank" style="color: #FF0250; text-decoration: none;">View on Google Maps</a>')

        popup_html = "<br>".join(popup_lines)

        tooltip_lines = []
        for col in tooltip_cols:
            if col in row and pd.notnull(row[col]):
                tooltip_lines.append(f"{row[col]}")
        tooltip_html = " – ".join(tooltip_lines)

        folium.CircleMarker(
            location = [row['lat'], row['lon']],
            rradius=8,
            color='#FF0250',
            weight=2,
            fill=True,
            fill_color='#f9ddf4',
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=tooltip_html
        ).add_to(map_pois)


    map_pois.fit_bounds(bounds)

    # Map mit POIs des Hubs
    st_folium(map_pois, use_container_width=True)


    # Xslx-Download der dargestellten POIs
    # Kopie der POIs, optional ohne 'geometry'
    download_df = cd_pois.copy()
    download_df.drop(columns=["geometry"], errors="ignore", inplace=True)

    # Excel in den Speicher schreiben
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        download_df.to_excel(writer, index=False, sheet_name="POIs")
        output.seek(0)

    c1, c2 = st.columns(2)
    with c1:
        # Google-Maps-Link Button
        maps_url = f"https://www.google.com/maps/search/?api=1&query={cd_cluster.avg_lat},{cd_cluster.avg_lon}"
        st.link_button("🔍 View on Google Maps", url=maps_url)

    with c2:
        # Download-Button anzeigen
        st.download_button(
            label="📥 Export Stores (xlsx)",
            data=output,
            file_name=f"pois_{city}_{cluster_id}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# Relative Anteile Diagramm
rel = {
    k.split('_')[-1]: v
    for k, v in cd_cluster.items()
    if k.startswith(("shop_rel_","amenity_rel_"))
       and not pd.isna(v)
}
top5 = dict(sorted(rel.items(), key=lambda x: x[1], reverse=True)[:5])
df_top5 = pd.DataFrame(top5.items(), columns=["Kategorie", "Anteil"])
df_top5['Kategorie'] = df_top5['Kategorie'].str.capitalize()

st.write("### Top 5 Relative Shares of the Hub")

domain_order = df_top5["Kategorie"].tolist()
chart = (
    alt.Chart(df_top5)
    .mark_bar(cornerRadius=3)
    .encode(
        y=alt.Y("Kategorie:N", sort='-x', title=""),
        x=alt.X("Anteil:Q", title="relative share"),
        color=alt.Color(
            "Kategorie:N",
            title="Kategorie",
            legend=None,
            scale=alt.Scale(domain=domain_order,
                            range=["#0a50db", "#0092e8", "#00aeef", "#14c9f4", "#6bf5ff"])
        ),
        tooltip=[alt.Tooltip("Anteil", format=".2%"), "Kategorie"]
    )
    .properties(height=300)
)

st.altair_chart(chart, use_container_width=True)

st.write("### Public Transport Information")
# st.write(cd_cluster)

c_pup = st.columns(2)
c_pup[0].metric("Bus Stations", round(cd_cluster["bus"]))
c_pup[1].metric("Train / Metro Stations within 200m", round(cd_cluster["trains_metro_in_buffer"]))

# Top Retailers Table
st.write("### Top Retailers Present in this Hub")

# Filter `cd_cluster` for columns that start with 'has_' and have a value of 1
retailer_columns = [col for col in cd_cluster.index if col.startswith('has_') and cd_cluster[col] == 1]

if retailer_columns:
    # Extract retailer names by removing 'has_' prefix and replacing underscores with spaces
    top_retailers_list = [col.replace('has_', '').replace('_', ' ').title() for col in retailer_columns]
    
    st.write("Click a retailer to see its details:")
    cols_per_row = 3  # Adjust as needed
    num_retailers = len(top_retailers_list)
    
    for i in range(0, num_retailers, cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            if i + j < num_retailers:
                retailer_name_display = top_retailers_list[i + j]
                retailer_name_for_param = retailer_name_display.lower().replace(' ', '_') # Convert back to 'has_retailer_name' format if needed for retailer-detail.py
                with cols[j]:
                    if st.button(retailer_name_display, key=f"retailer_btn_{retailer_name_for_param}"):
                        st.session_state.selected_retailer = retailer_name_display
                        st.query_params["retailer"] = retailer_name_display
                        st.switch_page("retailer-detail.py")
else:
    st.info("No specific top retailers identified in this hub.")
