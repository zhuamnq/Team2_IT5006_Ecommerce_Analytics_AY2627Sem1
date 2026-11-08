"""Export the held-out Test orders (purchased from 2018-08-01) for the Streamlit app.

The app replays these orders as if they were arriving live. Features are built
exactly as in Section 12-13 of notebooks/05_regression_delivery_time.ipynb:
seller history comes from Train + Validation orders and is frozen at the
2018-08-01 fit cutoff, so no outcome from the replayed orders leaks into
their own features.

Run from the repository root after 02_preprocessing.ipynb:

    python src/scripts/export_holdout.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import TABLES  # noqa: E402
from src.data import DELIVERY_TIME, TEST_CUTOFF  # noqa: E402
from src.features import load_feature_table  # noqa: E402

OUT = ROOT / "data" / "app" / "regression_holdout.csv.gz"

# Shown in the app but not used by the model.
DISPLAY = [
    "order_id", "order_purchase_timestamp", DELIVERY_TIME, "delivery_days",
    "customer_region", "primary_seller_region", "estimated_mean_distance_km",
    "order_value", "item_count", "total_weight_g", "payment_value", "payment_installments",
]

features = json.loads((TABLES / "regression_selected_features.json").read_text())["tree"]
model_columns = features["numeric"] + features["categorical"]

frame = load_feature_table("regression", history_splits=("train", "validation"),
                           history_cutoff=TEST_CUTOFF)
test = frame[frame["split"] == "test"].sort_values(["order_purchase_timestamp", "order_id"])
columns = list(dict.fromkeys(DISPLAY + model_columns))
OUT.parent.mkdir(parents=True, exist_ok=True)
test[columns].to_csv(OUT, index=False, compression="gzip")
print(f"{OUT.relative_to(ROOT)}: {len(test):,} orders, {len(columns)} columns")
