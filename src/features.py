# Columns known only after delivery; never use as features.
LEAKAGE_COLUMNS = [
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "delivery_days",
    "delay_days",
    "delivery_status",
    "review_score",
]


def add_features(frame):
    """Add engineered features available at purchase time."""
    raise NotImplementedError("TODO: volume, same-state flag, purchase month/weekday, ...")


def build_preprocessor(numeric, categorical):
    """Return a ColumnTransformer for scaling and encoding."""
    raise NotImplementedError("TODO: scaling + one-hot encoding inside the pipeline")
