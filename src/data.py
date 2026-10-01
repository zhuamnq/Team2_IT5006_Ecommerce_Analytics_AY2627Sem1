import pandas as pd

from src.config import RAW


def load_orders():
    """Join the raw Olist tables needed for modelling."""
    raise NotImplementedError("TODO: load and merge raw tables from RAW")


def clean(frame):
    """Drop invalid records and handle missing values."""
    raise NotImplementedError("TODO: document each cleaning rule in the report")


def split(frame, target):
    """Return X_train, X_test, y_train, y_test without leakage."""
    raise NotImplementedError("TODO: time-based or stratified split")
