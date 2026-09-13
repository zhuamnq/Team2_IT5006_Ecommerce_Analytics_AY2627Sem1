from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Candidate Problems", page_icon="🎯", layout="wide")
DATA = Path(__file__).parents[1] / "data" / "olist_dashboard.csv.gz"


@st.cache_data
def load_data(version):
    return pd.read_csv(DATA, parse_dates=["order_purchase_timestamp"])


data = load_data(2)
st.title("🎯 Candidate Problem Exploration")
st.caption(
    "Preliminary evidence for two candidate Phase 2 analytics problems. "
    "These results support problem scoping and do not represent trained predictive models."
)

st.sidebar.header("Candidate problem filters")
minimum, maximum = data.order_purchase_timestamp.dt.date.agg(["min", "max"])
dates = st.sidebar.date_input("Purchase date", (minimum, maximum), min_value=minimum, max_value=maximum)
filtered = data.copy()
if len(dates) == 2:
    filtered = filtered[filtered.order_purchase_timestamp.dt.date.between(*dates)]
for label, column in {
    "Product category": "product_category",
    "Customer state": "customer_state",
    "Seller state": "seller_state",
}.items():
    selected = st.sidebar.multiselect(label, sorted(data[column].dropna().unique()), placeholder="All")
    if selected:
        filtered = filtered[filtered[column].isin(selected)]

if filtered.empty:
    st.warning("No records match these filters.")
    st.stop()

st.header("1. Freight Cost Prediction")
st.write(
    "**Question:** Can freight cost be predicted from product characteristics, order value, "
    "and other information available before shipment? This is a regression problem with "
    "freight value as the continuous target."
)
weight_corr = filtered.product_weight_g.corr(filtered.freight_value)
price_corr = filtered.price.corr(filtered.freight_value)
c1, c2, c3 = st.columns(3)
c1.metric("Weight–freight correlation", f"{weight_corr:.2f}")
c2.metric("Price–freight correlation", f"{price_corr:.2f}")
c3.metric("Order items", f"{len(filtered):,}")

x_label = st.radio("Explore freight relationship", ["Product weight", "Product price"], horizontal=True)
x_column = "product_weight_g" if x_label == "Product weight" else "price"
sample_pool = filtered.dropna(subset=[x_column, "freight_value"])
sample = sample_pool.sample(min(5000, len(sample_pool)), random_state=42)
st.plotly_chart(
    px.scatter(
        sample, x=x_column, y="freight_value", color="product_category", opacity=0.35,
        labels={x_column: x_label, "freight_value": "Freight value (R$)"},
        title=f"{x_label} vs freight value (sample up to 5,000 items)",
    ),
    width="stretch",
)
st.info(
    "Candidate predictors: product weight and dimensions, price, category, order value, "
    "item count, and seller/customer location. Outliers and skewed values require treatment during modelling."
)

st.divider()
st.header("2. Late Delivery Prediction")
st.write(
    "**Question:** Can an order's late-delivery status be predicted using information available "
    "at or shortly after purchase? This is a binary classification problem."
)
orders = filtered.drop_duplicates("order_id")
known = orders[orders.delivery_status != "Unknown"].copy()
late_rate = (known.delivery_status == "Late").mean()
reviews = known.groupby("delivery_status", as_index=False).review_score.mean()
late_review = reviews.set_index("delivery_status").review_score
c1, c2, c3 = st.columns(3)
c1.metric("Late delivery rate", f"{late_rate:.1%}")
c2.metric("Average review: late", f"{late_review.get('Late', float('nan')):.2f}")
c3.metric("Average review: on time / early", f"{late_review.get('On time / early', float('nan')):.2f}")

left, right = st.columns(2)
balance = known.delivery_status.value_counts().rename_axis("delivery_status").reset_index(name="orders")
left.plotly_chart(
    px.bar(balance, x="delivery_status", y="orders", color="delivery_status", title="Target class distribution"),
    width="stretch",
)
right.plotly_chart(
    px.bar(reviews, x="delivery_status", y="review_score", color="delivery_status", range_y=[0, 5],
           title="Average review score by delivery status"),
    width="stretch",
)

dimension_label = st.selectbox("Compare late-delivery rate by", ["Customer state", "Seller state", "Product category"])
dimension = {
    "Customer state": "customer_state",
    "Seller state": "seller_state",
    "Product category": "product_category",
}[dimension_label]
late_by_group = (
    filtered.drop_duplicates(["order_id", dimension])
    .query("delivery_status != 'Unknown'")
    .groupby(dimension)
    .agg(orders=("order_id", "nunique"), late_rate=("delivery_status", lambda x: (x == "Late").mean()))
    .query("orders >= 30")
    .nlargest(15, "late_rate")
    .sort_values("late_rate")
    .reset_index()
)
st.plotly_chart(
    px.bar(late_by_group, x="late_rate", y=dimension, orientation="h",
           labels={"late_rate": "Late-delivery rate"}, title=f"Highest late-delivery rates by {dimension_label.lower()}"),
    width="stretch",
)
st.warning(
    "Modelling risks: the target is imbalanced, and actual delivery dates, delivery duration, "
    "delay days, and review score must not be used as predictors because they are unavailable at prediction time."
)
