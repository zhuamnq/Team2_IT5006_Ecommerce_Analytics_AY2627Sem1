import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import BR_HOLIDAYS, PEAK_SEASONS, RAW
from src.data import read_processed

# Columns known only after delivery; never use as features.
LEAKAGE_COLUMNS = [
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "delivery_days",
    "is_late",
    "delay_days",
    "delivery_status",
    "review_score",
]


def load_purchase_times():
    """Purchase timestamps of every raw order, any status, sorted."""
    orders = pd.read_csv(
        RAW / "olist_orders_dataset.csv",
        usecols=["order_purchase_timestamp"],
        parse_dates=["order_purchase_timestamp"],
    )
    return np.sort(orders["order_purchase_timestamp"].to_numpy())


def count_before(times, at, days):
    """Number of `times` in [at - days, at), for each value of `at`."""
    end = np.searchsorted(times, at, side="left")
    start = np.searchsorted(times, at - np.timedelta64(days, "D"), side="left")
    return end - start


def add_features(frame, purchase_times=None):
    """Add engineered features available at purchase time.

    `purchase_times` defaults to every raw order, so platform load includes
    orders that were later cancelled or never delivered.
    """
    out = frame.copy()
    purchased = pd.to_datetime(out["order_purchase_timestamp"])
    purchase_date = purchased.dt.normalize()

    # +1 km avoids dividing by zero for same-zip orders.
    out["promised_days_per_km"] = out["promised_delivery_days"] / (
        out["estimated_mean_distance_km"] + 1
    )

    # From Part B. Installments of 0 (2 orders) have no per-installment value.
    out["payment_per_installment"] = out["payment_value"] / out["payment_installments"].replace(0, np.nan)

    month = purchased.dt.month
    out["purchase_month_of_year"] = month
    out["purchase_month_sin"] = np.sin(2 * np.pi * month / 12)
    out["purchase_month_cos"] = np.cos(2 * np.pi * month / 12)
    out["purchase_weekday"] = purchased.dt.dayofweek
    out["purchase_hour"] = purchased.dt.hour

    holidays = pd.to_datetime(BR_HOLIDAYS).to_numpy()
    position = np.searchsorted(holidays, purchase_date.to_numpy())
    if (position == len(holidays)).any():
        raise ValueError("Purchase after the last BR_HOLIDAYS date; extend it in src/config.py")
    next_holiday = holidays[position]
    out["days_to_holiday"] = (next_holiday - purchase_date.to_numpy()) // np.timedelta64(1, "D")

    out["peak_season_flag"] = 0
    for start, end in PEAK_SEASONS:
        out.loc[purchase_date.between(start, end), "peak_season_flag"] = 1

    # Olist never promises a weekend date, so count the weekend days the
    # promise has to span instead.
    first_day = (purchase_date + pd.Timedelta(days=1)).to_numpy().astype("datetime64[D]")
    after_promise = first_day + out["promised_delivery_days"].to_numpy()
    out["weekend_days_to_estimate"] = out["promised_delivery_days"] - np.busday_count(
        first_day, after_promise
    )

    if purchase_times is None:
        purchase_times = load_purchase_times()
    at = purchased.to_numpy()
    out["platform_orders_7d"] = count_before(purchase_times, at, 7)
    weekly_average_28d = count_before(purchase_times, at, 28) / 4
    out["platform_load_ratio"] = np.where(
        weekly_average_28d > 0, out["platform_orders_7d"] / weekly_average_28d, np.nan
    )
    return out


# Seller history (Part B), built from order_seller_bridge: one row per order
# and seller, reduced to one row per order before joining (plan Section 5).
#   Point-in-time, from purchase times only, so every split may use every
#   earlier order: seller_past_orders, seller_backlog_7d, seller_is_new,
#   seller_past_avg_freight.
#   Outcome-based, from HISTORY_SPLITS orders only (training by default) and
#   only those already delivered at this order's purchase time
#   (delivered = purchase + delivery_days): seller_past_late_rate,
#   seller_past_delivery_days, route_past_late_rate, smoothed towards the
#   history average by SMOOTHING pseudo-orders, so new or rarely seen sellers
#   and routes get the history average.
#   For the final model, refitted on train + validation and scored on test,
#   pass history_splits=("train", "validation") so July outcomes count as
#   history, as they would in production.
SMOOTHING = 10
NEW_SELLER_DAYS = 30
HISTORY_SPLITS = ("train",)


def load_seller_bridge():
    return read_processed("order_seller_bridge.csv.gz", parse_dates=["order_purchase_timestamp"])


def load_outcomes():
    """is_late and delivery_days for every order, from the two cleaned files."""
    late = read_processed("late_delivery_classification.csv.gz", usecols=["order_id", "is_late"])
    days = read_processed("delivery_time_regression.csv.gz", usecols=["order_id", "delivery_days"])
    return late.merge(days, on="order_id", how="inner", validate="one_to_one")


def seller_point_in_time(bridge):
    """Features from the seller's earlier purchases; ties at the same timestamp are not earlier."""
    out = []
    for _, group in bridge.groupby("seller_id", sort=False):
        group = group.sort_values("order_purchase_timestamp")
        times = group["order_purchase_timestamp"].to_numpy()
        past = np.searchsorted(times, times, side="left")
        freight_sum = np.concatenate([[0], np.cumsum(group["seller_freight_value"].to_numpy())])
        out.append(pd.DataFrame({
            "seller_past_orders": past,
            "seller_backlog_7d": past - np.searchsorted(times, times - np.timedelta64(7, "D"), side="left"),
            "seller_is_new": (times - times[0] < np.timedelta64(NEW_SELLER_DAYS, "D")).astype(int),
            "seller_past_avg_freight": np.where(past > 0, freight_sum[past] / np.maximum(past, 1), np.nan),
        }, index=group.index))
    return pd.concat(out)


def past_outcome(keys, at, history, value, prior, smoothing=SMOOTHING):
    """Smoothed mean of `value` over `history` rows with the same key delivered before `at`.

    Returns (smoothed mean, number of history rows used) for each query.
    """
    mean = np.full(len(keys), prior, dtype=float)
    count = np.zeros(len(keys), dtype=int)
    groups = {key: g.sort_values("delivered_at") for key, g in history.groupby("key")}
    for key, positions in pd.Series(np.arange(len(keys))).groupby(keys.to_numpy()).groups.items():
        if key not in groups:
            continue
        g = groups[key]
        n = np.searchsorted(g["delivered_at"].to_numpy(), at[positions], side="left")
        total = np.concatenate([[0], np.cumsum(g[value].to_numpy())])[n]
        # smoothing=0 (no smoothing) leaves keys with no history at the prior
        with np.errstate(invalid="ignore", divide="ignore"):
            smoothed = (total + smoothing * prior) / (n + smoothing)
        mean[positions] = np.where(n + smoothing > 0, smoothed, prior)
        count[positions] = n
    return mean, count


def add_seller_features(
    frame, bridge=None, outcomes=None, history_splits=HISTORY_SPLITS, smoothing=SMOOTHING
):
    """Join seller history features to an order-level frame (one row per order).

    Outcome-based features use only orders whose split is in `history_splits`,
    smoothed towards their average by `smoothing` pseudo-orders.
    """
    bridge = load_seller_bridge() if bridge is None else bridge
    outcomes = load_outcomes() if outcomes is None else outcomes
    rows = bridge.merge(outcomes, on="order_id", how="left", validate="many_to_one").merge(
        frame[["order_id", "customer_state"]], on="order_id", how="inner", validate="many_to_one"
    )
    rows = rows.join(seller_point_in_time(rows))

    unknown = set(history_splits) - set(rows["split"])
    if unknown:
        raise ValueError(f"Unknown history splits: {sorted(unknown)}")
    rows["route"] = rows["seller_state"] + ">" + rows["customer_state"]
    past = rows[rows["split"].isin(history_splits)].assign(
        delivered_at=lambda d: d["order_purchase_timestamp"] + pd.to_timedelta(d["delivery_days"], unit="D")
    )
    at = rows["order_purchase_timestamp"].to_numpy()
    for name, key, value in [
        ("seller_past_late_rate", "seller_id", "is_late"),
        ("seller_past_delivery_days", "seller_id", "delivery_days"),
        ("route_past_late_rate", "route", "is_late"),
    ]:
        history = past[[key, "delivered_at", value]].rename(columns={key: "key"})
        rows[name], count = past_outcome(rows[key], at, history, value, past[value].mean(), smoothing)
        if key == "seller_id":
            rows["seller_past_deliveries"] = count

    # An order arrives with its last parcel, so take the worst seller.
    per_order = rows.groupby("order_id").agg(
        seller_past_late_rate=("seller_past_late_rate", "max"),
        seller_past_delivery_days=("seller_past_delivery_days", "max"),
        route_past_late_rate=("route_past_late_rate", "max"),
        seller_backlog_7d=("seller_backlog_7d", "max"),
        seller_past_orders=("seller_past_orders", "min"),
        seller_past_deliveries=("seller_past_deliveries", "min"),
        seller_is_new=("seller_is_new", "max"),
        seller_past_avg_freight=("seller_past_avg_freight", "mean"),
    )
    out = frame.merge(per_order, on="order_id", how="left", validate="one_to_one")
    assert len(out) == len(frame)
    return out


TABLES = {
    "classification": "late_delivery_classification.csv.gz",
    "regression": "delivery_time_regression.csv.gz",
}


def load_feature_table(problem, history_splits=HISTORY_SPLITS, smoothing=SMOOTHING):
    """Cleaned order file for "classification" or "regression" with Part A and
    Part B features added. See add_seller_features for `history_splits` and
    `smoothing`."""
    frame = read_processed(TABLES[problem])
    return add_seller_features(add_features(frame), history_splits=history_splits, smoothing=smoothing)


# Fixed band edges, not fitted; each band is [edge, next edge).
BANDS = {
    "days_to_holiday": [8, 15, 31, 61],  # 0-7, 8-14, 15-30, 31-60, 61+
    "estimated_mean_distance_km": [100, 500, 1500],  # <100, 100-500, 500-1,500, 1,500+
    "purchase_hour": [6, 12, 18],  # night, morning, afternoon, evening
}


def to_band(values, edges):
    return np.digitize(values, edges)


def build_preprocessor(numeric, categorical, skewed=(), banded=(), scale=True, min_frequency=20):
    """Return a ColumnTransformer for imputing, transforming and encoding.

    numeric: imputed with the training median, then scaled.
    skewed: as numeric, but log1p first (values must be >= 0).
    banded: median-imputed, cut at the fixed BANDS edges, then one-hot encoded.
    categorical: one-hot encoded; categories seen fewer than `min_frequency`
        times in training, and unseen ones, share one "infrequent" column.
    scale=False skips the scaler, for tree models.

    No missing-value indicators are added: geolocation_missing and
    payment_missing already flag every row with a gap.
    """
    def numeric_steps(*steps):
        steps = [*steps, SimpleImputer(strategy="median")]
        if scale:
            steps.append(StandardScaler())
        return make_pipeline(*steps)

    one_hot = dict(handle_unknown="infrequent_if_exist", min_frequency=min_frequency)
    transformers = [
        ("numeric", numeric_steps(), list(numeric)),
        ("skewed", numeric_steps(FunctionTransformer(np.log1p, feature_names_out="one-to-one")), list(skewed)),
        ("categorical", make_pipeline(
            SimpleImputer(strategy="most_frequent"), OneHotEncoder(**one_hot)
        ), list(categorical)),
    ]
    for column in banded:
        band = FunctionTransformer(to_band, kw_args={"edges": BANDS[column]}, feature_names_out="one-to-one")
        transformers.append(
            (f"{column}_band", make_pipeline(
                SimpleImputer(strategy="median"), band, OneHotEncoder(handle_unknown="ignore")
            ), [column])
        )
    # Always dense: HistGradientBoosting rejects sparse input, which a mostly
    # one-hot column set would otherwise produce.
    return ColumnTransformer(
        [t for t in transformers if t[2]], remainder="drop", sparse_threshold=0,
        verbose_feature_names_out=False,
    )


# Feature sets, grouped as build_preprocessor arguments:
#     build_preprocessor(**FEATURES_LINEAR) / build_preprocessor(**FEATURES_TREE, scale=False)
# Both exclude IDs, the raw timestamp, "YYYY-MM" purchase_month (test months
# never occur in training), split and both targets. Seller features need
# load_feature_table (or add_seller_features).

# Deduplicated for linear models, which need scaled inputs and are hurt by
# near-duplicate columns. Left out, with the column that stays in brackets:
#   any_seller_same_state, any_seller_same_city (same_state, same_city);
#   seller_count, seller_state_count (multi_seller_flag);
#   estimated_min/max_distance_km (mean); volumetric_weight_kg (volume / 6000);
#   billable_weight_kg (max of weight and volume / 6000; 0.95 correlated with
#   total_volume_cm3 on train);
#   density_g_cm3 (weight / volume, and has impossible values up to 66 g/cm3);
#   payment_value (order_value + freight_value); mean/max_item_price,
#   unique_product_count, product_category_count, max_item_weight_g,
#   max_sum_of_dims_cm (order_value, item_count, weights, longest_side_cm);
#   customer_region, primary_seller_region, primary_seller_state (region_pair,
#   customer_state); raw coordinates and zip prefix (distance, region);
#   promised_days_per_km (after logs, log promised days minus log distance);
#   purchase_month_of_year (sin/cos); weekend_days_to_estimate (0.97
#   correlated with promised_delivery_days); platform_orders_7d (grows with
#   the platform, so test values sit beyond training; platform_load_ratio);
#   payment_per_installment (payment_value / installments);
#   seller_past_deliveries (seller_past_orders).
FEATURES_LINEAR = {
    "numeric": [
        "purchase_month_sin", "purchase_month_cos",
        "peak_season_flag", "platform_load_ratio", "customer_is_capital",
        "remote_flag", "same_state", "same_city", "multi_seller_flag",
        "used_voucher", "payment_installments", "longest_side_cm",
        "geolocation_missing", "payment_missing",
        "seller_is_new", "seller_past_late_rate", "seller_past_delivery_days",
        "route_past_late_rate",
    ],
    "skewed": [
        "promised_delivery_days", "estimated_mean_distance_km", "freight_value",
        "order_value", "item_count", "total_weight_g", "total_volume_cm3",
        "customer_zip_density",
        "seller_past_orders", "seller_backlog_7d", "seller_past_avg_freight",
    ],
    "categorical": [
        "product_category", "region_pair", "customer_state", "payment_type",
        "purchase_weekday",
    ],
    "banded": ["days_to_holiday", "estimated_mean_distance_km", "purchase_hour"],
}

# Everything, untransformed: trees split on raw values and are unaffected by
# scale, skew or redundant columns. Except platform_orders_7d: it grows with
# the platform, so validation and test values sit beyond the training range,
# where a tree predicts as for the largest training values
# (platform_load_ratio stays).
FEATURES_TREE = {
    "numeric": [
        "promised_delivery_days", "promised_days_per_km", "purchase_month_of_year",
        "purchase_weekday", "purchase_hour", "days_to_holiday", "peak_season_flag",
        "weekend_days_to_estimate", "platform_load_ratio",
        "customer_zip_lat", "customer_zip_lng", "mean_seller_zip_lat",
        "mean_seller_zip_lng", "customer_zip_density", "customer_is_capital",
        "remote_flag", "same_state", "any_seller_same_state", "same_city",
        "any_seller_same_city", "estimated_min_distance_km",
        "estimated_mean_distance_km", "estimated_max_distance_km",
        "geolocation_missing", "order_value", "mean_item_price", "max_item_price",
        "freight_value", "item_count", "unique_product_count", "seller_count",
        "seller_state_count", "product_category_count", "multi_seller_flag",
        "total_weight_g", "total_volume_cm3", "max_item_weight_g",
        "longest_side_cm", "max_sum_of_dims_cm", "volumetric_weight_kg",
        "billable_weight_kg", "density_g_cm3", "payment_value",
        "payment_installments", "payment_type_count", "used_voucher",
        "payment_missing", "payment_per_installment",
        "seller_past_orders", "seller_past_deliveries", "seller_backlog_7d",
        "seller_is_new", "seller_past_avg_freight", "seller_past_late_rate",
        "seller_past_delivery_days", "route_past_late_rate",
    ],
    "categorical": [
        "product_category", "region_pair", "customer_state", "customer_region",
        "primary_seller_state", "primary_seller_region", "payment_type",
    ],
}


def feature_columns(features):
    """Input columns a feature set reads, each once."""
    return list(dict.fromkeys(column for group in features.values() for column in group))


def output_sources(preprocessor):
    """Input column behind each output column of a fitted build_preprocessor,
    e.g. "customer_state" for "customer_state_SP"."""
    names = preprocessor.get_feature_names_out()
    sources = []
    for name, transformer, columns in preprocessor.transformers_:
        if transformer == "drop":
            continue
        outputs = names[preprocessor.output_indices_[name]]
        if len(columns) == 1 or len(outputs) == len(columns):
            sources += list(columns) * (len(outputs) // len(columns))
        else:
            # One-hot columns are named <column>_<category>
            sources += [max((c for c in columns if n.startswith(f"{c}_")), key=len) for n in outputs]
    return sources
