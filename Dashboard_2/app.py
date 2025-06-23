import streamlit as st

def wide_space_default():
    st.set_page_config(layout="wide")

wide_space_default()

# st.button("Home")
# st.logo("logo_locifai_tha.png",size="large")
#st.logo("logo_locifai_tha.png", icon_image="logo_locifai_2_min.png",size="large")

# Define your pages
pg = st.navigation([
    st.Page("home.py", title="Home"),
    st.Page("city-overview.py",title="City overview"),
    st.Page("cluster-detail.py",title="Cluster details"),
    st.Page("retailer-detail.py",title="Retailer details"),
    st.Page("custom-cluster.py",title="Custom Cluster")
])

pg.run()
# st.set_page_config(page_title="Home")

