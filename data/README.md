# Data

## Source

The project uses the Olist Brazilian E-Commerce public dataset supplied in the
IT5006 Canvas module. The nine source CSV files in `raw/` are treated as
read-only inputs so every team member works from the same copy.

## Directory policy

| Location | Committed? | Produced by |
| --- | --- | --- |
| `raw/` | Yes, never edited | Original Olist download |
| `processed/` | No (git-ignored) | Preprocessing and feature-engineering notebooks, see below |
| `app/regression_holdout.csv.gz` | Yes | `src/scripts/export_holdout.py`, after notebook 05 |
| `olist_dashboard.csv.gz` | Yes | `src/scripts/prepare_data.py` |

The two committed generated files are small, frozen copies that the deployed
Streamlit app reads, so the app runs without re-executing the notebooks.
Generated files must not be treated as independent source data.

## Generated modelling tables (`processed/`)

These files are not in Git because they are large and fully reproducible from
`raw/`. Run the notebooks in the order given in the root README to rebuild them.
Rebuilding the three `.csv.gz` files from `raw/` on Python 3.12 gives the same
tables as the team's working copies (same rows and columns, values equal to
floating-point precision).

| File | Written by | Read by | Contents |
| --- | --- | --- | --- |
| `delivery_time_regression.csv.gz` | `02_preprocessing` | `03_feature_engineering`, `05_regression_delivery_time`, `src/scripts/export_holdout.py` | One row per delivered order with the regression target `delivery_days` and purchase-time inputs |
| `late_delivery_classification.csv.gz` | `02_preprocessing` | `src/features.py` (`load_outcomes`) | Same orders with the binary `is_late` outcome; used for leakage-safe seller and route late-rate history |
| `order_seller_bridge.csv.gz` | `02_preprocessing` | `src/features.py` | One row per order + seller with seller location, for seller-level history features |
| `promise_band_base.csv` | `02_preprocessing_problemA` | `03_feature_engineering_problemA` | Problem A cohort with the three-class promise-band target and split labels |
| `promise_band_split_summary.csv` | `02_preprocessing_problemA` | Reference only | Row counts per split, including label-maturity exclusions |
| `order_seller_bridge.csv` | `02_preprocessing_problemA` | `03_feature_engineering_problemA` | Problem A's own order + seller bridge (uncompressed; separate from the `.csv.gz` above) |
| `promise_band_feature_table.csv` | `03_feature_engineering_problemA` | `04_classification_promise_band_modeling` | Problem A feature table |
| `promise_band_feature_dictionary.csv` | `03_feature_engineering_problemA` | Reference only | Feature definitions and rationale |
| `selected_features.csv` | `03_feature_engineering_problemA` | `04_classification_promise_band_modeling` | Problem A selected feature list |
| `feature_*.csv` | `03_feature_engineering_problemA` | Reference only | Feature-selection diagnostics (mutual information, permutation importance, correlated pairs, validation metrics, training profile) |

Problem B (regression) writes its results to `docs/report/tables/regression_*`
and `deployment/models/regression_final_model.joblib` instead of `processed/`.

## Rebuild commands

```bash
python src/scripts/prepare_data.py       # data/olist_dashboard.csv.gz
python src/scripts/export_holdout.py     # data/app/regression_holdout.csv.gz (after notebook 05)
```

## Raw tables

| File | Purpose |
| --- | --- |
| `olist_orders_dataset.csv` | Order status and purchase/delivery timestamps |
| `olist_order_items_dataset.csv` | Products, sellers, prices, and freight by order item |
| `olist_customers_dataset.csv` | Customer identifiers and locations |
| `olist_sellers_dataset.csv` | Seller identifiers and locations |
| `olist_products_dataset.csv` | Product attributes and categories |
| `olist_order_payments_dataset.csv` | Payment types, instalments, and values |
| `olist_order_reviews_dataset.csv` | Review scores and review metadata |
| `olist_geolocation_dataset.csv` | Brazilian ZIP-prefix coordinates |
| `product_category_name_translation.csv` | Portuguese-to-English category names |
