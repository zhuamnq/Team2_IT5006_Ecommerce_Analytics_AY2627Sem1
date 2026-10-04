import random
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FIGURES = ROOT / "report" / "figures"
TABLES = ROOT / "report" / "tables"
MODELS = ROOT / "models"

RAW_FILES = [
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)


def validate_data_path():
    missing = [name for name in RAW_FILES if not (RAW / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing files in {RAW}: {', '.join(missing)}")
