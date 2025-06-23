import streamlit as st
from data import cities
from data import retailers

st.logo("Dashboard_2/logo_locifai_tha.png",size="large")

st.title("Welcome to LocifAI")
st.write("Helping businesses grow where it matters most.")

st.subheader("City Overview")
st.write("See details for a city you choose.")

selected_city = st.selectbox(
    "Choose your City",
    cities.names,
    index = None,
    placeholder = "Select a City"
    )

if selected_city is not None:
    st.session_state.selected_city = selected_city
    st.query_params["city"] = selected_city 
    st.switch_page("city-overview.py")

st.divider()


st.subheader("Retailer View")
st.write("View a selected retailers portofolio. Analyze the stores and gain valuable insights.")

c1,c2 = st.columns(2)
with c1:
   selected_city_r = st.selectbox(
    "Choose a City",
    cities.names,
    index = None,
    placeholder = "Select a City"
    )


with c2:
    if not selected_city_r: 
        placeholder_text = "Select a City first"
        disabeld_status = True
        options = ""
    else:
        placeholder_text = "Choose a retailer"
        disabeld_status = False
        options = sorted(retailers.names[selected_city_r])
    
    selected_retailer = st.selectbox(
    "Choose a Retailer",
    options,
    index = None,
    placeholder = placeholder_text,
    disabled=disabeld_status,
    ) 

if selected_city_r is not None and selected_retailer is not None:
    st.session_state.selected_city = selected_city_r
    st.session_state.selected_retailer = selected_retailer
    st.query_params["city"] = selected_city_r
    st.query_params["retailer"] = selected_retailer
    st.switch_page("retailer-detail.py")

st.divider()

st.subheader("Custom Hub")
st.write("Draw custom hubs. Learn how a new store could benefit your growth.")

selected_city_c = st.selectbox(
    "Choose your City",
    cities.names,
    key = "selected_city_c",
    index = None,
    placeholder = "Select a City"
    )

if selected_city_c is not None:
    st.session_state.selected_city = selected_city_c
    st.query_params["city"] = selected_city_c 
    st.switch_page("custom-cluster.py")
