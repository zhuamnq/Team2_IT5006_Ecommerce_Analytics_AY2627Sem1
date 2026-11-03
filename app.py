import streamlit as st


st.set_page_config(page_title="Olist Analytics", page_icon="🛒", layout="wide")

# Metric values shrink on narrow screens instead of being cut off with "...".
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: clamp(1.25rem, 2.1vw, 2.1rem); }
    [data-testid="stMetricLabel"] * { white-space: normal !important; overflow: visible !important;
                                      text-overflow: clip !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

page = st.navigation([
    st.Page("pages/EDA_Explorer.py", title="EDA Explorer", icon=":material/query_stats:", default=True),
    st.Page("pages/Delivery_Time_Predictor.py", title="Delivery Predictor",
            icon=":material/local_shipping:", url_path="Delivery_Time_Predictor"),
])
page.run()
