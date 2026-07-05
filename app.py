"""
app.py
------
Entry point of the Shopper Spectrum Streamlit app.

Navigation is built explicitly with st.navigation() / st.Page(), grouped
into sections ("Overview", "Tools", "Analytics"). This is the modern,
explicit alternative to Streamlit's folder-based pages/ auto-discovery —
it gives full control over labels, icons, grouping, and default page, and
renders identically whether you run it locally or after deploying to
Streamlit Community Cloud (or any other host), as long as the app's main
file is set to `app.py`.

Run with:
    streamlit run app.py
"""

import streamlit as st
from utils import load_css

st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛍️",
    layout="centered",
    initial_sidebar_state="expanded",
)

load_css()

# ---------------------------------------------------------------------------
# Define every page once, then group them into sidebar sections.
# ---------------------------------------------------------------------------
home_page = st.Page("views/home.py", title="Home", icon="🏠", default=True)
about_page = st.Page("views/about.py", title="How It Works", icon="ℹ️")

recommend_page = st.Page(
    "views/product_recommendation.py", title="Product Recommendation", icon="🔎"
)
segment_page = st.Page(
    "views/customer_segmentation.py", title="Customer Segmentation", icon="👥"
)

eda_page = st.Page("views/eda_dashboard.py", title="EDA Dashboard", icon="📊")

pg = st.navigation(
    {
        "Overview": [home_page, about_page],
        "Tools": [recommend_page, segment_page],
        "Analytics": [eda_page],
    }
)

st.sidebar.markdown(
    "## 🛍️ Shopper Spectrum\n"
    "<span style='color:#AEB8D6;'>Customer analytics & recommendations</span>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

pg.run()
