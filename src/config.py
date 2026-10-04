import random
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
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

# Brazilian national holidays covering the data (2016-09 to 2018-08) plus the
# next few, so every purchase has an upcoming holiday. Includes Carnival
# Monday/Tuesday and Corpus Christi, which most states observe.
BR_HOLIDAYS = [
    "2016-09-07", "2016-10-12", "2016-11-02", "2016-11-15", "2016-12-25",
    "2017-01-01", "2017-02-27", "2017-02-28", "2017-04-14", "2017-04-21",
    "2017-05-01", "2017-06-15", "2017-09-07", "2017-10-12", "2017-11-02",
    "2017-11-15", "2017-12-25",
    "2018-01-01", "2018-02-12", "2018-02-13", "2018-03-30", "2018-04-21",
    "2018-05-01", "2018-05-31", "2018-09-07", "2018-10-12", "2018-11-02",
    "2018-11-15", "2018-12-25",
]

# Peak shopping periods, inclusive: Black Friday week (Monday to Cyber Monday)
# and Carnival (Saturday to Ash Wednesday). Christmas is left out because late
# rates in the training data are not raised from Dec 10 to Dec 24.
PEAK_SEASONS = [
    ("2016-11-21", "2016-11-28"),
    ("2017-02-25", "2017-03-01"),
    ("2017-11-20", "2017-11-27"),
    ("2018-02-10", "2018-02-14"),
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
