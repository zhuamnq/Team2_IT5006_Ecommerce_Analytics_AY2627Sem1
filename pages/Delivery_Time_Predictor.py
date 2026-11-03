"""Milestone 3 deployed model: delivery-time regression (Problem B).

Held-out Test orders (purchased 1-29 Aug 2018, never used for fitting) are replayed in
purchase-time order as if they were new orders. Each order is scored when it arrives; its
real outcome is revealed only once the simulated clock passes the actual delivery time.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).parents[1]
MODEL_PATH = ROOT / "models" / "regression_final_model.joblib"
HOLDOUT_PATH = ROOT / "data" / "app" / "regression_holdout.csv.gz"
TABLES = ROOT / "report" / "tables"

FEATURES = json.loads((TABLES / "regression_selected_features.json").read_text())["tree"]
NUMERIC, CATEGORICAL = FEATURES["numeric"], FEATURES["categorical"]
MODEL_COLUMNS = NUMERIC + CATEGORICAL
TIGHT_BUFFER_DAYS = 3  # about one model MAE

STATES = {
    "AC": "Acre", "AL": "Alagoas", "AM": "Amazonas", "AP": "Amapá", "BA": "Bahia", "CE": "Ceará",
    "DF": "Distrito Federal", "ES": "Espírito Santo", "GO": "Goiás", "MA": "Maranhão",
    "MG": "Minas Gerais", "MS": "Mato Grosso do Sul", "MT": "Mato Grosso", "PA": "Pará",
    "PB": "Paraíba", "PE": "Pernambuco", "PI": "Piauí", "PR": "Paraná", "RJ": "Rio de Janeiro",
    "RN": "Rio Grande do Norte", "RO": "Rondônia", "RR": "Roraima", "RS": "Rio Grande do Sul",
    "SC": "Santa Catarina", "SE": "Sergipe", "SP": "São Paulo", "TO": "Tocantins",
}


# ---------------------------------------------------------------- data and model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def build_holdout():
    """Rebuild the held-out Test features as in notebook 05 (Sections 12-13)."""
    import sys

    sys.path.insert(0, str(ROOT))
    from src.data import TEST_CUTOFF
    from src.features import load_feature_table

    frame = load_feature_table("regression", history_splits=("train", "validation"),
                               history_cutoff=TEST_CUTOFF)
    return frame[frame["split"] == "test"].copy()


@st.cache_data(show_spinner="Preparing held-out orders...")
def load_holdout():
    dates = ["order_purchase_timestamp", "order_delivered_customer_date"]
    if HOLDOUT_PATH.exists():
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
    return orders.promised_delivery_days.to_numpy() * np.exp(model.predict(X))


def score(model, orders):
    out = orders.copy()
    out["predicted_days"] = predict_days(model, out)
    out["predicted_date"] = out.order_purchase_timestamp + pd.to_timedelta(out.predicted_days, unit="D")
    out["buffer_days"] = out.promised_delivery_days - out.predicted_days
    out["risk"] = np.where(out.buffer_days < TIGHT_BUFFER_DAYS, "Tight", "Comfortable")
    return out


@st.cache_data
def score_all(_model):
    return score(_model, load_holdout())


def state_name(code):
    return STATES.get(str(code), str(code))


def nice(text):
    return str(text).replace("_", " ").capitalize()


def big_number(value, unit):
    st.markdown(
        f"<div style='font-size:3rem;font-weight:700;line-height:1.1'>{value}"
        f"<span style='font-size:1.1rem;font-weight:400;opacity:.6'> {unit}</span></div>",
        unsafe_allow_html=True,
    )


def buffer_badge(buffer):
    if buffer < 0:
        return f":red-badge[Later than promise · {-buffer:.1f} days]"
    if buffer < TIGHT_BUFFER_DAYS:
        return f":orange-badge[Tight · {buffer:.1f}-day buffer]"
    return f":green-badge[Comfortable · {buffer:.1f}-day buffer]"


def field(column, label, value):
    column.caption(label)
    column.markdown(f"**{value}**")


try:
    model = load_model()
except Exception as error:  # usually a scikit-learn / xgboost version mismatch
    st.error(f"Could not load the model (needs scikit-learn 1.6.1, xgboost 3.2.0): {error}")
    st.stop()

holdout = load_holdout()
performance = score_all(model)
test_mae = (performance.predicted_days - performance.delivery_days).abs().mean()

# ---------------------------------------------------------------- sidebar and state
st.sidebar.header("Replay settings")
first_day, last_day = holdout.order_purchase_timestamp.dt.date.agg(["min", "max"])
start_day = st.sidebar.date_input("Start", first_day, min_value=first_day, max_value=last_day)
batch_size = st.sidebar.select_slider("Batch size", [10, 50, 100, 250], value=50)
autoplay = st.sidebar.toggle("Auto-stream", value=False, help="Adds one batch every 2 seconds")

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


if autoplay:
    state.full_run = True

    @st.fragment(run_every=2)
    def ticker():
        if not state.get("full_run"):  # timer tick, not the page's own run
            advance(batch_size)
            st.rerun()

    ticker()
    state.full_run = False

# ---------------------------------------------------------------- header
st.title("Delivery Time Predictor")
st.caption("Replay unseen orders and see the delivery estimate available at purchase time.")
st.markdown(
    f":gray-badge[Held-out test · {len(holdout):,} orders] :gray-badge[Tuned XGBoost] "
    f":gray-badge[Test MAE · {test_mae:.2f} days]"
)

scored = state.replay_scored
clock = scored.order_purchase_timestamp.max() if len(scored) else None
feed = scored.assign(delivered=scored.order_delivered_customer_date <= clock) if len(scored) else scored
if len(scored):
    feed["status"] = np.where(feed.delivered, "Delivered", "In transit")
    feed["actual_days"] = feed.delivery_days.where(feed.delivered)
    feed["error_days"] = feed.predicted_days - feed.actual_days
known = feed[feed.delivered] if len(scored) else feed

# ---------------------------------------------------------------- live order card
with st.container(border=True):
    st.markdown("**LIVE ORDER REPLAY**")
    st.caption(f"Simulated clock: {clock:%d %b %Y, %H:%M}" if clock is not None
               else f"Starts {start_day:%d %b %Y}")
    b1, b2, b3, _ = st.columns([1.3, 1.1, 0.9, 2.5])
    b1.button("Next order", type="primary", on_click=advance, args=(1,), width="stretch")
    b2.button(f"Next {batch_size}", on_click=advance, args=(batch_size,), width="stretch")
    b3.button("Reset", type="tertiary", on_click=reset, width="stretch")

    if not len(scored):
        st.info("Press **Next order** to receive the first order.")
    else:
        latest = feed.iloc[-1]
        left, right = st.columns([3, 2])
        with left.container(border=True):
            st.caption(f"Order {latest.order_id[:4]}…{latest.order_id[-4:]} · "
                       f"{latest.order_purchase_timestamp:%d %b %Y, %H:%M}")
            st.markdown(f"#### {state_name(latest.primary_seller_state)} → "
                        f"{state_name(latest.customer_state)}")
            c1, c2 = st.columns(2)
            field(c1, "Category", nice(latest.product_category))
            field(c2, "Distance", f"{latest.estimated_min_distance_km:,.0f} km")
            field(c1, "Items / weight", f"{int(latest.item_count)} item{'s' if latest.item_count > 1 else ''} · {latest.total_weight_g / 1000:.1f} kg")
            field(c2, "Freight", f"R$ {latest.freight_value:,.2f}")
        with right.container(border=True):
            st.caption("Predicted delivery")
            big_number(f"{latest.predicted_days:.1f}", "days")
            st.markdown(f"Expected {latest.predicted_date:%d %b} · Promised {latest.promised_date:%d %b}")
            st.markdown(buffer_badge(latest.buffer_days))
            st.caption(f"Actual: {latest.actual_days:.1f} days" if latest.delivered else "Actual: in transit")

# ---------------------------------------------------------------- tabs
replay_tab, scenario_tab, performance_tab, batch_tab = st.tabs(
    ["Replay", "Try a scenario", "Model performance", "Batch tools"]
)

with replay_tab:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Orders received", f"{len(feed):,}")
    k2.metric("Outcomes known", f"{len(known):,}")
    k3.metric("Running MAE", f"{known.error_days.abs().mean():.2f} d" if len(known) else "—")
    k4.metric("Flagged tight", f"{(feed.risk == 'Tight').mean():.0%}" if len(feed) else "—")

    if len(known) >= 5:
        chart = px.scatter(
            known, x="actual_days", y="predicted_days", color="risk", opacity=0.6,
            hover_data=["order_id", "customer_state", "promised_delivery_days"],
            labels={"actual_days": "Actual (days)", "predicted_days": "Predicted (days)", "risk": ""},
            title="Predicted vs actual, delivered orders",
        )
        top = float(max(known.actual_days.max(), known.predicted_days.max()))
        chart.add_shape(type="line", x0=0, y0=0, x1=top, y1=top, line=dict(dash="dash", color="grey"))
        st.plotly_chart(chart, width="stretch")

    if len(feed):
        table = feed.iloc[::-1].head(200).copy()
        for column in ("actual_days", "error_days"):  # blank until delivered
            table[column] = table[column].map(lambda v: "" if pd.isna(v) else f"{v:+.1f}"
                                              if column == "error_days" else f"{v:.1f}")
        st.dataframe(
            table[[
                "order_purchase_timestamp", "customer_state", "primary_seller_state", "product_category",
                "promised_delivery_days", "predicted_days", "risk", "status", "actual_days", "error_days",
            ]],
            hide_index=True, width="stretch",
            column_config={
                "order_purchase_timestamp": st.column_config.DatetimeColumn("Purchased", format="DD MMM HH:mm"),
                "customer_state": "Customer", "primary_seller_state": "Seller", "product_category": "Category",
                "promised_delivery_days": "Promised (d)",
                "predicted_days": st.column_config.NumberColumn("Predicted (d)", format="%.1f"),
                "risk": "Buffer", "status": "Status",
                "actual_days": "Actual (d)", "error_days": "Error (d)",
            },
        )

with scenario_tab:
    st.caption("Adjust inputs and compare the prediction.")
    pool = scored if len(scored) else holdout
    order_id = st.selectbox("Order", pool.order_id.iloc[::-1])
    base = holdout[holdout.order_id == order_id].iloc[[0]]

    c1, c2, c3 = st.columns(3)
    promised = c1.number_input("Promised delivery (days)", 1, 120, int(base.promised_delivery_days.iloc[0]))
    freight = c1.number_input("Freight (R$)", 0.0, 1000.0, float(base.freight_value.iloc[0]), step=1.0)
    backlog = c2.number_input("Seller orders, last 7 days", 0, 500, int(base.seller_backlog_7d.iloc[0]))
    late_rate = c2.slider("Seller past late rate", 0.0, 1.0, float(base.seller_past_late_rate.iloc[0]), 0.01)
    past_days = c3.number_input("Seller past avg delivery (days)", 0.0, 100.0,
                                float(base.seller_past_delivery_days.iloc[0]), step=0.5)
    payments = sorted(holdout.payment_type.dropna().unique())
    payment = c3.selectbox("Payment type", payments, index=payments.index(base.payment_type.iloc[0]),
                           format_func=nice)

    edited = base.assign(
        promised_delivery_days=promised,
        promised_days_per_km=promised / (base.estimated_mean_distance_km + 1),
        freight_value=freight, seller_backlog_7d=backlog, seller_past_late_rate=late_rate,
        seller_past_delivery_days=past_days, payment_type=payment,
    )
    before, after = predict_days(model, base)[0], predict_days(model, edited)[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Original", f"{before:.1f} days")
    m2.metric("Scenario", f"{after:.1f} days", f"{after - before:+.1f} days", delta_color="inverse")
    m3.metric("Buffer vs promise", f"{promised - after:+.1f} days")

with performance_tab:
    errors = performance.predicted_days - performance.delivery_days
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Orders", f"{len(performance):,}")
    p2.metric("MAE", f"{errors.abs().mean():.2f} d")
    p3.metric("RMSE", f"{np.sqrt((errors ** 2).mean()):.2f} d")
    p4.metric("Within ±3 days", f"{(errors.abs() <= 3).mean():.0%}")

    left, right = st.columns(2)
    left.plotly_chart(
        px.histogram(errors.rename("error"), x="error", nbins=80, range_x=[-25, 15],
                     labels={"error": "Predicted − actual (days)"}, title="Error distribution"),
        width="stretch",
    )
    by_region = (performance.assign(abs_error=errors.abs())
                 .groupby("customer_region", as_index=False).abs_error.mean())
    right.plotly_chart(
        px.bar(by_region, x="customer_region", y="abs_error",
               labels={"abs_error": "MAE (days)", "customer_region": "Customer region"},
               title="MAE by customer region"),
        width="stretch",
    )
    metrics_path = TABLES / "regression_test_metrics.csv"
    if metrics_path.exists():
        st.markdown("**All candidates on the test month**")
        st.dataframe(pd.read_csv(metrics_path).round(3), hide_index=True, width="stretch")

with batch_tab:
    st.caption("Upload orders with the model columns to score them in bulk.")
    template = holdout.head(20)[["order_id", *MODEL_COLUMNS]]
    st.download_button("Download template CSV", template.to_csv(index=False), "delivery_template.csv", "text/csv")
    upload = st.file_uploader("Orders CSV", type="csv")
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
