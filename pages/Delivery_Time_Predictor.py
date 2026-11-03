"""Milestone 3 deployed model: delivery-time regression (Problem B).

The app replays held-out historical orders (purchased 2018-08-01 to 2018-08-29,
never used for fitting) in purchase-time order, as if they were new orders
arriving on the platform. Each arriving order is scored by the saved model at
that moment; its real delivery outcome is only revealed once the simulated
clock passes the actual delivery time.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Delivery Time Predictor", page_icon="🚚", layout="wide")

ROOT = Path(__file__).parents[1]
MODEL_PATH = ROOT / "models" / "regression_final_model.joblib"
HOLDOUT_PATH = ROOT / "data" / "app" / "regression_holdout.csv.gz"
TABLES = ROOT / "report" / "tables"

FEATURES = json.loads((TABLES / "regression_selected_features.json").read_text())["tree"]
NUMERIC, CATEGORICAL = FEATURES["numeric"], FEATURES["categorical"]
MODEL_COLUMNS = NUMERIC + CATEGORICAL
TIGHT_BUFFER_DAYS = 3  # about one model MAE (2.9 days on the held-out month)


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def build_holdout():
    """Rebuild the held-out Test features exactly as in notebook 05 (Sections 12-13):
    seller history from Train + Validation, frozen at the 2018-08-01 fit cutoff."""
    import sys

    sys.path.insert(0, str(ROOT))
    from src.data import TEST_CUTOFF
    from src.features import load_feature_table

    frame = load_feature_table("regression", history_splits=("train", "validation"),
                               history_cutoff=TEST_CUTOFF)
    return frame[frame["split"] == "test"].copy()


@st.cache_data(show_spinner="Preparing held-out orders (first load only)...")
def load_holdout():
    dates = ["order_purchase_timestamp", "order_delivered_customer_date"]
    if HOLDOUT_PATH.exists():  # optional cache written by scripts/export_holdout.py
        frame = pd.read_csv(HOLDOUT_PATH, parse_dates=dates)
    else:
        frame = build_holdout()
        for column in dates:
            frame[column] = pd.to_datetime(frame[column])
    frame = frame.sort_values(["order_purchase_timestamp", "order_id"]).reset_index(drop=True)
    purchase_day = frame.order_purchase_timestamp.dt.normalize()
    frame["promised_date"] = purchase_day + pd.to_timedelta(frame.promised_delivery_days, unit="D")
    return frame


def predict_days(model, orders):
    """The model predicts log(delivery_days / promised_days); convert back to days."""
    X = orders[MODEL_COLUMNS].copy()
    for column in CATEGORICAL:
        X[column] = X[column].astype(object)
    log_ratio = model.predict(X)
    return orders.promised_delivery_days.to_numpy() * np.exp(log_ratio)


def score(model, orders):
    out = orders.copy()
    out["predicted_days"] = predict_days(model, out)
    out["predicted_date"] = out.order_purchase_timestamp + pd.to_timedelta(out.predicted_days, unit="D")
    out["buffer_days"] = out.promised_delivery_days - out.predicted_days
    out["risk"] = np.where(out.buffer_days < TIGHT_BUFFER_DAYS, "Tight", "Comfortable")
    return out


@st.cache_data
def score_all(_model, version=1):
    return score(_model, load_holdout())


try:
    model = load_model()
except Exception as error:  # usually a scikit-learn / xgboost version mismatch
    st.error(
        "Could not load the saved model. Install the versions pinned in requirements.txt "
        f"(scikit-learn 1.6.1, xgboost 3.2.0).\n\n`{type(error).__name__}: {error}`"
    )
    st.stop()

holdout = load_holdout()

# ---------------------------------------------------------------- header
st.title("🚚 Delivery Time Predictor")
st.caption(
    "Deployed regression model (tuned XGBoost) that predicts how many days an order will take "
    "to reach the customer, scored on held-out historical orders replayed as new orders."
)

with st.expander("How to use this page", expanded=False):
    st.markdown(
        f"""
**What the model does.** For each order it predicts the delivery time in days, using only
information available at purchase: Olist's promised delivery window, route and distance,
parcel size and weight, payment, and the seller's track record *before* the order.
It is trained on orders delivered before **1 Aug 2018** and has never seen the
{len(holdout):,} orders replayed here (purchased 1–29 Aug 2018).

**Live order replay** simulates a live order feed:
1. Pick a start date in the sidebar, then press **Next order** (or **Next batch**), or turn on **Auto-stream**.
2. Each arriving order is scored immediately; you see the predicted delivery date next to Olist's promised date.
   Orders whose predicted buffer is under {TIGHT_BUFFER_DAYS} days are flagged **Tight**.
3. The real outcome stays hidden ("In transit") until the simulated clock passes the actual
   delivery time, as it would in production. Running accuracy uses delivered orders only.
4. **Reset** restarts the feed.

**What-if** lets you change a few inputs of one order and re-score it.
**Batch scoring** accepts a CSV with the model columns and returns predictions.
**Model performance** shows accuracy on the full held-out month.
"""
    )

st.sidebar.header("Replay settings")
first_day, last_day = holdout.order_purchase_timestamp.dt.date.agg(["min", "max"])
start_day = st.sidebar.date_input("Start simulation on", first_day, min_value=first_day, max_value=last_day)
batch_size = st.sidebar.select_slider("Orders per batch", [5, 10, 25, 50, 100], value=10)
autoplay = st.sidebar.toggle("Auto-stream (one batch every 2 s)", value=False)

start_index = int(holdout.order_purchase_timestamp.dt.date.searchsorted(start_day))
state = st.session_state


def reset():
    state.replay_start = start_index
    state.replay_scored = holdout.iloc[0:0].assign(
        predicted_days=[], predicted_date=[], buffer_days=[], risk=[])


if state.get("replay_start") != start_index:
    reset()


def advance(count):
    done = len(state.replay_scored)
    begin = state.replay_start + done
    arriving = holdout.iloc[begin:begin + count]
    if arriving.empty:
        return
    new = score(model, arriving)
    state.replay_scored = new if done == 0 else pd.concat([state.replay_scored, new], ignore_index=True)


replay, what_if, batch, performance = st.tabs(
    ["Live order replay", "What-if", "Batch scoring", "Model performance"]
)

# ---------------------------------------------------------------- replay
with replay:

    @st.fragment(run_every=2 if autoplay else None)
    def live_feed():
        if autoplay:
            advance(batch_size)

        b1, b2, b3, _ = st.columns([1, 1, 1, 3])
        b1.button("Next order", on_click=advance, args=(1,), width="stretch")
        b2.button(f"Next batch ({batch_size})", on_click=advance, args=(batch_size,), width="stretch")
        b3.button("Reset", on_click=reset, width="stretch")

        scored = state.replay_scored
        remaining = len(holdout) - state.replay_start - len(scored)
        if scored.empty:
            st.info("No orders yet. Press **Next order** to receive the first one.")
            return

        clock = scored.order_purchase_timestamp.max()
        feed = scored.assign(
            delivered=scored.order_delivered_customer_date <= clock,
        )
        feed["status"] = np.where(feed.delivered, "Delivered", "In transit")
        feed["actual_days"] = feed.delivery_days.where(feed.delivered)
        feed["error_days"] = feed.predicted_days - feed.actual_days
        known = feed[feed.delivered]

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Simulated clock", clock.strftime("%d %b %Y %H:%M"))
        k2.metric("Orders received", f"{len(feed):,}", help=f"{remaining:,} left in the held-out month")
        k3.metric("Flagged tight", f"{(feed.risk == 'Tight').mean():.0%}")
        k4.metric("Outcomes known", f"{len(known):,}")
        k5.metric("Running MAE", f"{known.error_days.abs().mean():.2f} days" if len(known) else "—",
                  help="Mean absolute error on orders already delivered by the simulated clock")

        latest = feed.iloc[-1]
        st.subheader("Latest order")
        left, right = st.columns([3, 2])
        with left:
            st.markdown(
                f"""
| | |
|---|---|
| Order | `{latest.order_id}` |
| Purchased | {latest.order_purchase_timestamp:%a %d %b %Y, %H:%M} |
| Route | seller {latest.primary_seller_state} → customer {latest.customer_state} (~{latest.estimated_min_distance_km:,.0f} km) |
| Category | {str(latest.product_category).replace('_', ' ')} |
| Items / weight | {int(latest.item_count)} item(s), {latest.total_weight_g:,.0f} g |
| Freight / payment | R$ {latest.freight_value:,.2f}, {str(latest.payment_type).replace('_', ' ')} |
| Seller history | {int(latest.seller_past_orders)} past orders, {latest.seller_past_late_rate:.0%} late, {int(latest.seller_backlog_7d)} orders in last 7 days |
"""
            )
        with right:
            st.metric("Predicted delivery", f"{latest.predicted_days:.1f} days",
                      f"{latest.buffer_days:+.1f} days vs promise", delta_color="normal")
            st.write(f"**Predicted arrival:** {latest.predicted_date:%d %b %Y}")
            st.write(f"**Olist promised:** {latest.promised_date:%d %b %Y} ({int(latest.promised_delivery_days)} days)")
            if latest.risk == "Tight":
                st.warning(f"Tight: predicted buffer below {TIGHT_BUFFER_DAYS} days. Consider proactive follow-up.")
            else:
                st.success("Comfortable buffer before the promised date.")
            st.caption("Actual outcome: " + (f"delivered in {latest.actual_days:.1f} days"
                                              if latest.delivered else "in transit (not yet revealed)"))

        st.subheader("Order feed")
        table = feed.iloc[::-1].head(200)[[
            "order_purchase_timestamp", "order_id", "customer_state", "primary_seller_state", "product_category",
            "promised_delivery_days", "predicted_days", "predicted_date", "risk", "status", "actual_days", "error_days",
        ]]
        st.dataframe(
            table, hide_index=True, width="stretch",
            column_config={
                "order_purchase_timestamp": st.column_config.DatetimeColumn("Purchased", format="DD MMM HH:mm"),
                "order_id": "Order",
                "customer_state": "Customer",
                "primary_seller_state": "Seller",
                "product_category": "Category",
                "promised_delivery_days": "Promised (d)",
                "predicted_days": st.column_config.NumberColumn("Predicted (d)", format="%.1f"),
                "predicted_date": st.column_config.DateColumn("Predicted arrival", format="DD MMM"),
                "risk": "Buffer",
                "status": "Status",
                "actual_days": st.column_config.NumberColumn("Actual (d)", format="%.1f"),
                "error_days": st.column_config.NumberColumn("Error (d)", format="%+.1f"),
            },
        )

        if len(known) >= 5:
            chart = px.scatter(
                known, x="actual_days", y="predicted_days", color="risk", opacity=0.6,
                hover_data=["order_id", "customer_state", "promised_delivery_days"],
                labels={"actual_days": "Actual delivery (days)", "predicted_days": "Predicted (days)"},
                title="Predicted vs actual, orders delivered so far",
            )
            top = float(max(known.actual_days.max(), known.predicted_days.max()))
            chart.add_shape(type="line", x0=0, y0=0, x1=top, y1=top, line=dict(dash="dash", color="grey"))
            st.plotly_chart(chart, width="stretch")

    live_feed()

# ---------------------------------------------------------------- what-if
with what_if:
    st.caption("Change a few inputs of one held-out order and re-score it. "
               "Effects show what the model learned, not causal effects.")
    pool = state.replay_scored if len(state.replay_scored) else holdout
    order_id = st.selectbox("Order", pool.order_id.iloc[::-1], help="Defaults to the latest order in the replay")
    base = holdout[holdout.order_id == order_id].iloc[[0]]

    c1, c2, c3 = st.columns(3)
    promised = c1.number_input("Promised delivery (days)", 1, 120, int(base.promised_delivery_days.iloc[0]))
    freight = c1.number_input("Freight value (R$)", 0.0, 1000.0, float(base.freight_value.iloc[0]), step=1.0)
    backlog = c2.number_input("Seller orders in last 7 days", 0, 500, int(base.seller_backlog_7d.iloc[0]))
    late_rate = c2.slider("Seller past late rate", 0.0, 1.0, float(base.seller_past_late_rate.iloc[0]), 0.01)
    past_days = c3.number_input("Seller past avg delivery (days)", 0.0, 100.0,
                                float(base.seller_past_delivery_days.iloc[0]), step=0.5)
    payments = sorted(holdout.payment_type.dropna().unique())
    payment = c3.selectbox("Payment type", payments, index=payments.index(base.payment_type.iloc[0]))

    edited = base.assign(
        promised_delivery_days=promised,
        promised_days_per_km=promised / (base.estimated_mean_distance_km + 1),
        freight_value=freight, seller_backlog_7d=backlog, seller_past_late_rate=late_rate,
        seller_past_delivery_days=past_days, payment_type=payment,
    )
    before, after = predict_days(model, base)[0], predict_days(model, edited)[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Original prediction", f"{before:.1f} days")
    m2.metric("What-if prediction", f"{after:.1f} days", f"{after - before:+.1f} days", delta_color="inverse")
    m3.metric("Actual (held-out answer)", f"{base.delivery_days.iloc[0]:.1f} days")

# ---------------------------------------------------------------- batch
with batch:
    st.caption("Score many orders at once. The CSV needs the model columns below; "
               "download the template (20 held-out orders) to see the format.")
    template = holdout.head(20)[["order_id", *MODEL_COLUMNS]]
    st.download_button("Download template CSV", template.to_csv(index=False), "delivery_template.csv", "text/csv")
    with st.expander("Required columns"):
        st.write(", ".join(f"`{c}`" for c in MODEL_COLUMNS))
    upload = st.file_uploader("Upload orders CSV", type="csv")
    if upload is not None:
        uploaded = pd.read_csv(upload)
        missing = [c for c in MODEL_COLUMNS if c not in uploaded.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            uploaded["predicted_days"] = predict_days(model, uploaded).round(2)
            st.dataframe(uploaded, hide_index=True, width="stretch")
            st.download_button("Download predictions", uploaded.to_csv(index=False),
                               "delivery_predictions.csv", "text/csv")

# ---------------------------------------------------------------- performance
with performance:
    scored_all = score_all(model)
    errors = scored_all.predicted_days - scored_all.delivery_days
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Held-out orders", f"{len(scored_all):,}")
    p2.metric("MAE", f"{errors.abs().mean():.2f} days")
    p3.metric("RMSE", f"{np.sqrt((errors ** 2).mean()):.2f} days")
    p4.metric("Within ±3 days", f"{(errors.abs() <= 3).mean():.0%}")

    metrics_path = TABLES / "regression_test_metrics.csv"
    if metrics_path.exists():
        st.markdown("**Comparison with the other candidates (from notebook 05)**")
        st.dataframe(pd.read_csv(metrics_path).round(3), hide_index=True, width="stretch")

    left, right = st.columns(2)
    left.plotly_chart(
        px.histogram(errors.rename("error"), x="error", nbins=80, range_x=[-25, 15],
                     labels={"error": "Predicted − actual (days)"}, title="Error distribution"),
        width="stretch",
    )
    by_region = (scored_all.assign(abs_error=errors.abs())
                 .groupby("customer_region", as_index=False).abs_error.mean())
    right.plotly_chart(
        px.bar(by_region, x="customer_region", y="abs_error",
               labels={"abs_error": "MAE (days)", "customer_region": "Customer region"},
               title="MAE by customer region"),
        width="stretch",
    )
    st.caption(
        "Held-out orders were purchased after the training cutoff; seller history features are frozen at "
        "1 Aug 2018. Some late-August orders may still be undelivered in the raw data and are excluded "
        "(right-censoring), so long deliveries are under-represented."
    )
