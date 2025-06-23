import streamlit as st
from streamlit_folium import st_folium
import folium
from data import cities
import time
import ast
from shapely import wkt
import math
from assets.navigation import display_sidebar_navigation
import altair as alt
from data import retailers
import pandas as pd


st.logo("Dashboard_2/logo_locifai_tha.png", icon_image="Dashboard_2/logo_locifai_2_min.png",size="large")

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
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# --- Tausendertrennzeichen Funktion ---
def tausendertrennzeichen(num, n_decimals=2):
    s = f"{num:,.{n_decimals}f}"
    if s.endswith(".00"):
        s = s[:-3]
    return s.replace(",", " ")

def display_introduction():
    # extract information - - - - - - - - - - - - - - - - - - - - - - - - - – – – –
    global population, tourists, description,center
    population  = cities.information[selected_city]["population"]
    tourists    = cities.information[selected_city]["tourists"]
    description = cities.description[selected_city]
    center      = cities.center[selected_city]
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # header title
    st.markdown("""<p style="font-size:16px; margin:0; padding:0;">📍 City overview</p>""", unsafe_allow_html=True)
    st.markdown(f"<h1 style='margin-top:-15px;'>{selected_city}</h1>", unsafe_allow_html=True)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    if st.button("← Return home"):
        st.switch_page("home.py")


    c1,c2, c3, c4 = st.columns(4)

    c1.metric("Population", f"{tausendertrennzeichen(int(population))}")
    c2.metric("Tourists (2023)", tausendertrennzeichen(int(tourists)))
    c3.empty()
    c4.empty()

    with st.container():
        st.markdown(
            f"""
            <div style="max-width: 800px;">
                <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

display_introduction()


# load bounding box data  - - - - - - - - - - - - - - - - - - - - - - - - - – –
@st.cache_data
def load_bounding_boxes(city):
    return cities.load_hubs(city)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# button should be removed – only for develpoment
if st.button("Reload"):
    st.cache_data.clear()
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if "selected_cluster" not in st.session_state:
    st.session_state.selected_cluster = None

df_pois = cities.load_pois(selected_city)
dfBoundingBoxes = load_bounding_boxes(selected_city)
dfBoundingBoxes["Predicted_Segment"] = (  
    dfBoundingBoxes["Predicted_Segment"]  
    .astype(str)  
    .str.strip()  
)  

dfBoundingBoxes["pop_buffer"] = dfBoundingBoxes["pop_buffer"] + dfBoundingBoxes["population"]

# Initialize filtered_df in session state if not already present
if 'dfBoundingBoxes_filtered' not in st.session_state:
    st.session_state.dfBoundingBoxes_filtered = dfBoundingBoxes.copy()

try:
    cluster_selected = st.session_state.selected_cluster
except:
    cluster_selected = None

display_sidebar_navigation(cluster = cluster_selected)


def display_sidebar():
    st.sidebar.header("Filters")
    # Initialize session state for filter values if not present
    if "num_pois_range_temp" not in st.session_state:
        st.session_state.num_pois_range_temp = (int(dfBoundingBoxes.num_points.min()), int(dfBoundingBoxes.num_points.max()))
    if "pp_income_range_temp" not in st.session_state:
        st.session_state.pp_income_range_temp = (dfBoundingBoxes.pp_income.min(), dfBoundingBoxes.pp_income.max())
    if "av_age_range_temp" not in st.session_state:
        st.session_state.av_age_range_temp = (math.floor(dfBoundingBoxes.av_age.min()), math.ceil(dfBoundingBoxes.av_age.max()))
    if "population_range_temp" not in st.session_state:
        st.session_state.population_range_temp = (int(dfBoundingBoxes.pop_buffer.min()), int(dfBoundingBoxes.pop_buffer.max()))
    if "selected_retailers_temp" not in st.session_state:
        st.session_state.selected_retailers_temp = []

    # Sliders and multiselect for temporary filter values
    st.session_state.num_pois_range_temp = st.sidebar.slider("Num of POIs", int(dfBoundingBoxes.num_points.min()), int(dfBoundingBoxes.num_points.max()), st.session_state.num_pois_range_temp, key="num_pois_slider_temp")

    # --- Avg income Slider with check for min_value == max_value ---
    min_pp_income = float(dfBoundingBoxes.pp_income.min())
    max_pp_income = float(dfBoundingBoxes.pp_income.max())
    if min_pp_income == max_pp_income:
        # st.sidebar.info(f"Average income: {min_pp_income} (only one value available)")
        st.session_state.pp_income_range_temp = (min_pp_income, max_pp_income)
    else:
        st.session_state.pp_income_range_temp = st.sidebar.slider("Average income", min_pp_income, max_pp_income, st.session_state.pp_income_range_temp, key="pp_income_slider_temp")
    # --- End Avg income Slider check ---

    min_av_age = math.floor(dfBoundingBoxes.av_age.min())
    max_av_age = math.ceil(dfBoundingBoxes.av_age.max())
    if min_av_age == max_av_age:
        # st.sidebar.info(f"Average age: {min_av_age} (only one value available)")
        st.session_state.av_age_range_temp = (min_av_age, max_av_age)
    else:
        st.session_state.av_age_range_temp = st.sidebar.slider("Average age", min_av_age, max_av_age, st.session_state.av_age_range_temp, key="av_age_slider_temp")

    min_population = int(dfBoundingBoxes.pop_buffer.min())
    max_population = int(dfBoundingBoxes.pop_buffer.max())
    if min_population == max_population:
        # st.sidebar.info(f"Population: {min_population} (only one value available)")
        st.session_state.population_range_temp = (min_population, max_population)
    else:
        st.session_state.population_range_temp = st.sidebar.slider("Population", min_population, max_population, st.session_state.population_range_temp, key="population_slider_temp")


    # --- Retailer Filter ---
    retailer_columns = [col for col in dfBoundingBoxes.columns if col.startswith('has_')]
    retailer_names_raw = [col.replace('has_', '').replace('_', ' ') for col in retailer_columns]
    retailer_names = [name for name in retailer_names_raw if name.lower() != 'shop']

    st.session_state.selected_retailers_temp = st.sidebar.multiselect(
        "Select Retailers",
        options=retailer_names,
        default=st.session_state.selected_retailers_temp,
        key="retailer_multiselect_temp"
    )
    

     # --- Segment Filter ---
    # Liste aller verfügbaren Segmente im DataFrame
    segment_options = (  
    dfBoundingBoxes  
    .loc[ dfBoundingBoxes["Predicted_Segment"].str.lower() != "ignore", "Predicted_Segment"]  
    .unique()  
)  
    segment_options = sorted(segment_options)  

    # Session-State initialisieren (temporär)
    if "selected_segments_temp" not in st.session_state:
        st.session_state.selected_segments_temp = []

    # Multiselect im Sidebar
    st.session_state.selected_segments_temp = st.sidebar.multiselect(
        "Select Segments",
        options=segment_options,
        default=st.session_state.selected_segments_temp,
        key="segments_multiselect_temp"
    )


    # Apply Filters Button
    if st.sidebar.button("✅ Apply Filters"):
        # Update the actual filter values in session state
        st.session_state.num_pois_range = st.session_state.num_pois_range_temp
        st.session_state.pp_income_range = st.session_state.pp_income_range_temp
        st.session_state.av_age_range = st.session_state.av_age_range_temp
        st.session_state.population_range = st.session_state.population_range_temp
        st.session_state.selected_retailers = st.session_state.selected_retailers_temp
        st.session_state.selected_segments = st.session_state.selected_segments_temp
        st.rerun() # Rerun to apply filters

    # Reset button
    if st.sidebar.button("❌ Reset Filters"):
        st.session_state.num_pois_range_temp = (int(dfBoundingBoxes.num_points.min()), int(dfBoundingBoxes.num_points.max()))
        st.session_state.pp_income_range_temp = (float(dfBoundingBoxes.pp_income.min()), float(dfBoundingBoxes.pp_income.max()))
        st.session_state.av_age_range_temp = (math.floor(dfBoundingBoxes.av_age.min()), math.ceil(dfBoundingBoxes.av_age.max()))
        st.session_state.population_range_temp = (int(dfBoundingBoxes.pop_buffer.min()), int(dfBoundingBoxes.pop_buffer.max()))
        st.session_state.selected_retailers_temp = []
        st.session_state.selected_segments_temp = []
        st.session_state.selected_segments = []

        # Also reset the applied filters
        st.session_state.num_pois_range = st.session_state.num_pois_range_temp
        st.session_state.pp_income_range = st.session_state.pp_income_range_temp
        st.session_state.av_age_range = st.session_state.av_age_range_temp
        st.session_state.population_range = st.session_state.population_range_temp
        st.session_state.selected_retailers = st.session_state.selected_retailers_temp

        st.rerun()

    # Return the *applied* filter values from session state
    if "num_pois_range" not in st.session_state: # Initial run before any filters are applied
        return (int(dfBoundingBoxes.num_points.min()), int(dfBoundingBoxes.num_points.max())), \
               (float(dfBoundingBoxes.pp_income.min()), float(dfBoundingBoxes.pp_income.max())), \
               (math.floor(dfBoundingBoxes.av_age.min()), math.ceil(dfBoundingBoxes.av_age.max())), \
               (int(dfBoundingBoxes.pop_buffer.min()), int(dfBoundingBoxes.pop_buffer.max())), \
               []
    else:
        return st.session_state.num_pois_range, st.session_state.pp_income_range, st.session_state.av_age_range, st.session_state.population_range, st.session_state.selected_retailers


# sidebar with advanced filter logic ensures no unwanted reruns
def display_sidebar():
    st.sidebar.header("Filters")

    # Initialize session state for filter values if not present
    if "num_pois_range_temp" not in st.session_state:
        st.session_state.num_pois_range_temp = (int(dfBoundingBoxes.num_points.min()), int(dfBoundingBoxes.num_points.max()))
    if "pp_income_range_temp" not in st.session_state:
        st.session_state.pp_income_range_temp = (dfBoundingBoxes.pp_income.min(), dfBoundingBoxes.pp_income.max())
    if "av_age_range_temp" not in st.session_state:
        st.session_state.av_age_range_temp = (math.floor(dfBoundingBoxes.av_age.min()), math.ceil(dfBoundingBoxes.av_age.max()))
    if "population_range_temp" not in st.session_state:
        st.session_state.population_range_temp = (int(dfBoundingBoxes.pop_buffer.min()), int(dfBoundingBoxes.pop_buffer.max()))
    if "selected_retailers_temp" not in st.session_state:
        st.session_state.selected_retailers_temp = []
    if "selected_segments_temp" not in st.session_state:
        st.session_state.selected_segments_temp = []

    # Start form
    with st.sidebar.form("filters_form", border=False):
        # Sliders
        st.session_state.num_pois_range_temp = st.slider(
            "Num of POIs",
            int(dfBoundingBoxes.num_points.min()),
            int(dfBoundingBoxes.num_points.max()),
            st.session_state.num_pois_range_temp,
            key="num_pois_slider_temp"
        )

        min_pp_income = float(dfBoundingBoxes.pp_income.min())
        max_pp_income = float(dfBoundingBoxes.pp_income.max())
        if min_pp_income == max_pp_income:
            st.session_state.pp_income_range_temp = (min_pp_income, max_pp_income)
        else:
            st.session_state.pp_income_range_temp = st.slider(
                "Average income",
                min_pp_income,
                max_pp_income,
                st.session_state.pp_income_range_temp,
                key="pp_income_slider_temp"
            )

        min_av_age = math.floor(dfBoundingBoxes.av_age.min())
        max_av_age = math.ceil(dfBoundingBoxes.av_age.max())
        if min_av_age == max_av_age:
            st.session_state.av_age_range_temp = (min_av_age, max_av_age)
        else:
            st.session_state.av_age_range_temp = st.slider(
                "Average age",
                min_av_age,
                max_av_age,
                st.session_state.av_age_range_temp,
                key="av_age_slider_temp"
            )

        min_population = int(dfBoundingBoxes.pop_buffer.min())
        max_population = int(dfBoundingBoxes.pop_buffer.max())
        if min_population == max_population:
            st.session_state.population_range_temp = (min_population, max_population)
        else:
            st.session_state.population_range_temp = st.slider(
                "Population",
                min_population,
                max_population,
                st.session_state.population_range_temp,
                key="population_slider_temp"
            )

        # Retailer multiselect
        retailer_columns = [col for col in dfBoundingBoxes.columns if col.startswith('has_')]
        retailer_names_raw = [col.replace('has_', '').replace('_', ' ') for col in retailer_columns]
        retailer_names = [name for name in retailer_names_raw if name.lower() != 'shop']

        st.session_state.selected_retailers_temp = st.multiselect(
            "Select Retailers",
            options=retailer_names,
            default=st.session_state.selected_retailers_temp,
            key="retailer_multiselect_temp"
        )

        # Segment multiselect
        segment_options = sorted(
            dfBoundingBoxes.loc[
                dfBoundingBoxes["Predicted_Segment"].str.lower() != "ignore",
                "Predicted_Segment"
            ].unique()
        )

        st.session_state.selected_segments_temp = st.multiselect(
            "Select Segments",
            options=segment_options,
            default=st.session_state.selected_segments_temp,
            key="segments_multiselect_temp"
        )

        # Submit buttons
        apply = st.form_submit_button("✅ Apply Filters")
        reset = st.form_submit_button("❌ Reset Filters")

    # Handle form actions outside the form context
    if apply:
        st.session_state.num_pois_range = st.session_state.num_pois_range_temp
        st.session_state.pp_income_range = st.session_state.pp_income_range_temp
        st.session_state.av_age_range = st.session_state.av_age_range_temp
        st.session_state.population_range = st.session_state.population_range_temp
        st.session_state.selected_retailers = st.session_state.selected_retailers_temp
        st.session_state.selected_segments = st.session_state.selected_segments_temp
        st.rerun()

    if reset:
        st.session_state.num_pois_range_temp = (int(dfBoundingBoxes.num_points.min()), int(dfBoundingBoxes.num_points.max()))
        st.session_state.pp_income_range_temp = (float(dfBoundingBoxes.pp_income.min()), float(dfBoundingBoxes.pp_income.max()))
        st.session_state.av_age_range_temp = (math.floor(dfBoundingBoxes.av_age.min()), math.ceil(dfBoundingBoxes.av_age.max()))
        st.session_state.population_range_temp = (int(dfBoundingBoxes.pop_buffer.min()), int(dfBoundingBoxes.pop_buffer.max()))
        st.session_state.selected_retailers_temp = []
        st.session_state.selected_segments_temp = []
        st.session_state.selected_segments = []

        st.session_state.num_pois_range = st.session_state.num_pois_range_temp
        st.session_state.pp_income_range = st.session_state.pp_income_range_temp
        st.session_state.av_age_range = st.session_state.av_age_range_temp
        st.session_state.population_range = st.session_state.population_range_temp
        st.session_state.selected_retailers = st.session_state.selected_retailers_temp

        st.rerun()

    # Return applied filters or defaults
    if "num_pois_range" not in st.session_state:
        return (int(dfBoundingBoxes.num_points.min()), int(dfBoundingBoxes.num_points.max())), \
               (float(dfBoundingBoxes.pp_income.min()), float(dfBoundingBoxes.pp_income.max())), \
               (math.floor(dfBoundingBoxes.av_age.min()), math.ceil(dfBoundingBoxes.av_age.max())), \
               (int(dfBoundingBoxes.pop_buffer.min()), int(dfBoundingBoxes.pop_buffer.max())), \
               []
    else:
        return (
            st.session_state.num_pois_range,
            st.session_state.pp_income_range,
            st.session_state.av_age_range,
            st.session_state.population_range,
            st.session_state.selected_retailers
        )


# Call display_sidebar to get the filter values
num_pois_range, pp_income_range, av_age_range, population_range, selected_retailers = display_sidebar()


def display_retail_hotspots(df_filtered):
    st.subheader("Retail Hubs")

    with st.spinner("Load Data..."):
        # Check if df_filtered is empty to avoid errors with min/max/mean
        if not df_filtered.empty:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Hubs total", tausendertrennzeichen(int(df_filtered.cluster_label.nunique())))
            c2.metric("Min POIs", df_filtered.num_points.min())
            c3.metric("Max POIs", df_filtered.num_points.max())
            c4.metric("Average POIs", int(df_filtered.num_points.mean()))
        else:
            st.warning("⚠️ No hubs match the current filter criteria.")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Hubs total", 0)
            c2.metric("Min POIs", 0)
            c3.metric("Max POIs", 0)
            c4.metric("Average POIs", 0.0)


    m = folium.Map(location = center, zoom_start=12)

    if not df_filtered.empty: # Only draw if there's data to display
        for _, row in df_filtered.iterrows():
            polygon = wkt.loads(row["geometry"])
            coords  = [(lat, lon) for lon,lat in polygon.exterior.coords]

            # Add polygon
            folium.Polygon(
                locations=coords,
                color="blue",
                fill=True,
                fill_opacity=0.5,
                popup=f"{row['cluster_label']} - {row['Predicted_Segment']}"
            ).add_to(m)


    resp = st_folium(m, use_container_width=True, key="city_map", returned_objects=["last_object_clicked_popup"])

    if resp and resp.get("last_object_clicked_popup"):
        st.session_state.selected_cluster = str(resp.get("last_object_clicked_popup")).split(" ")[0]
        st.query_params["cluster"] = str(resp["last_object_clicked_popup"]).split(" ")[0]

    if st.session_state.selected_cluster is not None:
        c1,c2 = st.columns(2)
        with c1:
            # Check if the selected cluster still exists in the filtered data
            if int(st.session_state.selected_cluster) in df_filtered['cluster_label'].values:
                if st.button(f"ℹ️ View details for hub {st.session_state.selected_cluster}"):
                    st.session_state.redirected = "city"
                    st.query_params["redirected"] = "city"
                    st.switch_page("cluster-detail.py")
            else:
                st.info(f"Cluster {st.session_state.selected_cluster} is no longer visible with current filters.")
                st.session_state.selected_cluster = None # Reset selected cluster if it's filtered out
                st.query_params["cluster"] = None
        with c2:
            # Only show Google Maps link if the cluster is still in the filtered data
            if st.session_state.selected_cluster and int(st.session_state.selected_cluster) in df_filtered['cluster_label'].values:
                cd_cluster = df_filtered[
                    df_filtered["cluster_label"] == int(st.session_state.selected_cluster)
                ].iloc[0]

                maps_url = (
                    f"https://www.google.com/maps/search/?api=1"
                    f"&query={cd_cluster.avg_lat},{cd_cluster.avg_lon}"
                )

                st.link_button(f"🔍 View hub {st.session_state.selected_cluster} on Google Maps", url=maps_url)
    else:
        st.info("To see details, select a hub on the map.")

# Filter the DataFrame based on the *applied* slider values
dfBoundingBoxes_filtered = dfBoundingBoxes[
    (dfBoundingBoxes.num_points >= num_pois_range[0]) & (dfBoundingBoxes.num_points <= num_pois_range[1]) &
    (dfBoundingBoxes.pp_income >= pp_income_range[0]) & (dfBoundingBoxes.pp_income <= pp_income_range[1]) &
    (dfBoundingBoxes.av_age >= av_age_range[0]) & (dfBoundingBoxes.av_age <= av_age_range[1]) &
    (dfBoundingBoxes.pop_buffer >= population_range[0]) & (dfBoundingBoxes.pop_buffer <= population_range[1])
]

# Apply retailer filter (robust OR logic)
if selected_retailers:
    individual_retailer_conditions = []
    for retailer in selected_retailers:
        # Convert "Starbucks" back to "has_Starbucks" for column lookup
        col_name = f"has_{retailer.replace('_', ' ')}"
        if col_name != "has_shop":
            if col_name in dfBoundingBoxes_filtered.columns:
                individual_retailer_conditions.append(dfBoundingBoxes_filtered[col_name] == 1)
        else:
            continue
    if individual_retailer_conditions:
        # Combine all individual conditions with OR
        retailer_filter_condition = individual_retailer_conditions[0]
        for condition in individual_retailer_conditions[1:]:
            retailer_filter_condition = retailer_filter_condition | condition
        dfBoundingBoxes_filtered = dfBoundingBoxes_filtered[retailer_filter_condition]
    else:
        # If selected_retailers was not empty but no matching columns were found for any of them,
        # then the result should be an empty DataFrame
        dfBoundingBoxes_filtered = dfBoundingBoxes_filtered.iloc[0:0] # Returns an empty DataFrame with same columns
 # (nach dem Retailer-Filter)
if "selected_segments" in st.session_state and st.session_state.selected_segments:
    dfBoundingBoxes_filtered = dfBoundingBoxes_filtered[
        dfBoundingBoxes_filtered["Predicted_Segment"].isin(st.session_state.selected_segments)
    ]


# Pass the filtered DataFrame to display_retail_hotspots
display_retail_hotspots(dfBoundingBoxes_filtered)

def link_to_draw():
    st.subheader("Draw a Retail Hub")


def retailer_chart():
    st.subheader(f"Leading Retail Chains")

    colors = [
        "#0a50db", "#0068df", "#007ee3", "#0092e8", "#00a5ec",
        "#00b7f1", "#14c9f4", "#37d8f8", "#52e7fb", "#6bf5ff"
    ]

    if not dfBoundingBoxes.equals(dfBoundingBoxes_filtered):
        df_pois_filterd = df_pois[df_pois[f"{selected_city}_hdbscan_5_5"].isin(dfBoundingBoxes_filtered["cluster_label"])]
    else:
        df_pois_filterd  = df_pois

    #df_pois_filterd = df_pois[]

    # Für jede Retailer-Basismarke mit .contains() die Anzahl ermitteln
    counts = []
    for retailer in retailers.names_madrid:
        n = df_pois_filterd[
            df_pois_filterd['brand']
            .str.contains(retailer, case=False, na=False)
        ].shape[0]
        counts.append({"brand": retailer, "stores": n})

    # In DataFrame, sortieren und Top 10 ziehen
    df_counts = pd.DataFrame(counts)
    df_top10_retailer = df_counts.sort_values(by='stores', ascending=False).head(10)

    # Damit der Chart die exakten Strings aus df_pois anzeigt (z. B. "Dia Market"), 
    # könntest du jetzt noch die häufigsten exakten Brand-Strings pro Basismarke holen:
    brand_labels = []
    for base in df_top10_retailer['brand']:
        matches = df_pois_filterd[df_pois_filterd['brand'].str.contains(base, case=False, na=False)]
        
        if matches.empty:
            brand_labels.append(base)
        else:
            most_common_brand = matches['brand'].value_counts().idxmax()
            brand_labels.append(most_common_brand)

    df_top10_retailer['brand_label'] = brand_labels

    # Chart
    chart = (alt.Chart(df_top10_retailer).mark_bar().encode(
        x=alt.X('stores:Q', title='Stores'),
        y=alt.Y('brand_label:N', sort='-x', title=''),
        color=alt.Color(
            'brand_label:N',
            scale=alt.Scale(domain=df_top10_retailer['brand_label'].tolist(), range=colors),
            legend=None
        ),
        tooltip=[
            alt.Tooltip('brand_label:N', title='Retailer'),
            alt.Tooltip('stores:Q', title='Stores')
        ]
    )
    .transform_filter('datum.stores > 0')
    .properties(width='container')
    )
    st.altair_chart(chart, use_container_width=True)

retailer_chart()

def share_chart():

    st.subheader(f"Category Distribution")
  
    colors = [
            "#0a50db", "#0068df", "#007ee3", "#0092e8", "#00a5ec",
            "#00b7f1", "#14c9f4", "#37d8f8", "#52e7fb", "#6bf5ff"
        ]

    # 1) Alle Spalten mit relativen Anteilen sammeln
    rel_cols = [
        col
        for col in dfBoundingBoxes_filtered.columns
        if (col.startswith("shop_rel_") or col.startswith("amenity_rel_")) and col != "shop_rel_yes" 
]
    # Gewichtung nach POI-Anzahl
    weights_pois = dfBoundingBoxes_filtered['num_points'] / dfBoundingBoxes_filtered['num_points'].sum()
    city_rel = (dfBoundingBoxes_filtered[rel_cols].multiply(weights_pois, axis=0)).sum()

    # 1. Relatives Series → DataFrame
    df_city_rel = city_rel.reset_index()
    df_city_rel.columns = ['category', 'relative']
    # Kategorie-Namen säubern
    df_city_rel['category'] = (
        df_city_rel['category']
        .str.replace('shop_rel_', '')
        .str.replace('amenity_rel_', '')
        .str.replace('_', ' ')
        .str.title()  
    )

    # 2. Top 5 auswählen
    df_top10_shares = df_city_rel.sort_values('relative', ascending=False).head(10)

    # 3. Chart bauen
    chart = (
        alt.Chart(df_top10_shares)
        .mark_bar()
        .encode(
            x=alt.X('relative:Q', title='Relative Share', axis=alt.Axis(format='%')),
            y=alt.Y('category:N', sort='-x', title=''),
            color=alt.Color(
            "category:N",
            scale=alt.Scale(domain=df_top10_shares['category'].tolist(), range=colors),
            legend=None  # Legende ausblenden, falls nicht gewünscht
        ),
            tooltip=[
                alt.Tooltip('category:N', title='Category'),
                alt.Tooltip('relative:Q', title='Share', format='.1%')
            ]
        )
        .properties(
            width='container'
        )
    )
    return chart

st.altair_chart(share_chart(), use_container_width=True)


def predicted_segment_chart():
    st.subheader("Distribution of Segment Categories")

    # Berechne Häufigkeit der Kategorien
    df_segment_counts = dfBoundingBoxes_filtered['Predicted_Segment'].value_counts(normalize=True).reset_index()
    df_segment_counts.columns = ['category', 'relative']
    df_segment_counts['category'] = df_segment_counts['category'].str.replace('_', ' ').str.title()

    colors = [
        "#0a50db", "#0068df", "#007ee3", "#0092e8", "#00a5ec",
        "#00b7f1", "#14c9f4", "#37d8f8", "#52e7fb", "#6bf5ff"
    ]

    # Farben auf Anzahl der Kategorien anpassen
    segment_colors = colors[:df_segment_counts.shape[0]]

    # Bar Chart erstellen
    chart = (
        alt.Chart(df_segment_counts)
        .mark_bar()
        .encode(
            x=alt.X('relative:Q', title='Relative Share', axis=alt.Axis(format='%')),
            y=alt.Y('category:N', sort='-x', title=''),
            color=alt.Color('category:N',
                            scale=alt.Scale(domain=df_segment_counts['category'].tolist(), range=segment_colors),
                            legend=None),
            tooltip=[
                alt.Tooltip('category:N', title='Category'),
                alt.Tooltip('relative:Q', title='Share', format='.1%')
            ]
        )
        .properties(width='container')
    )

    return chart

# Chart anzeigen
st.altair_chart(predicted_segment_chart(), use_container_width=True)
