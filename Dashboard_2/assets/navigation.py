import streamlit as st

def display_sidebar_navigation(cluster = "",retailer = ""):
    st.sidebar.header("City overview")
    st.sidebar.write("**Navigation**")
    st.sidebar.page_link("home.py",label="Home")
    st.sidebar.page_link("city-overview.py",label="City Overview")
    if cluster not in ["","None"] and cluster is not None:
        st.sidebar.page_link("cluster-detail.py",label="Hub Detail")
    else:
        st.sidebar.page_link("cluster-detail.py",label="Hub Detail",disabled=True,help="You need to select a hub")

    if retailer not in ["","None"] and retailer is not None:
        st.sidebar.page_link("retailer-detail.py",label="Retailer Detail")
    else:
        st.sidebar.page_link("retailer-detail.py",label="Retailer Detail",disabled=True,help="You need to select a retailer")

    st.sidebar.page_link("custom-cluster.py",label="Custom Hub")
    st.sidebar.divider()
