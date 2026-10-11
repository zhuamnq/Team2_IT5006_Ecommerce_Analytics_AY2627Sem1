# Olist E-commerce Explorer

Interactive exploratory dashboard for IT5006 Milestone 1, built with Streamlit.

**Live app:** https://team2-it5006-dashboard.streamlit.app/

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The committed compressed dataset is derived from the public Olist dataset. The nine source CSV files are stored in `data/raw/`; do not modify them directly. To rebuild the dashboard dataset:

```bash
python prepare_data.py
```

The duplicate SQLite copy is intentionally excluded because the modelling and dashboard code use the CSV files.

## Deploy

In Streamlit Community Cloud, select this repository and set the entry point to `app.py`.

## Milestone 2: Modelling

```bash
pip install -r requirements-modelling.txt
jupyter notebook notebooks/
```

`data/processed/` is not committed. Each problem has its own notebook sequence; run each in order:

- **Problem A (promise-band classification):** `02_preprocessing_problemA` → `03_feature_engineering_problemA` → `04_classification_promise_band_modeling`
- **Problem B (delivery-time regression):** `02_preprocessing` → `03_feature_engineering` → `05_regression_delivery_time`

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/02_preprocessing.ipynb
```

| Path | Contents |
| --- | --- |
| `notebooks/01_problem_scoping.ipynb` | Early problem scoping and leakage rules. Problem A is framed here as binary late delivery; it was later reframed as the three-band task below |
| `notebooks/02_preprocessing_problemA.ipynb` | Problem A: order-level cohort, three-band target (`under_half`, `half_to_full`, `late`), label-maturity chronological split, order–seller bridge |
| `notebooks/03_feature_engineering_problemA.ipynb` | Problem A: point-in-time seller, customer, category, route and marketplace histories; feature screening and validation late-AP ranking; exports the selected 16 features |
| `notebooks/04_classification_promise_band_modeling.ipynb` | Problem A: Dummy, Logistic Regression and tree models under 4-fold temporal CV; Logistic Regression tuning; refit and final test evaluation; error analysis |
| `notebooks/02_preprocessing.ipynb` | Problem B: cleaning, time-based train/validation/test split, export to `data/processed/` |
| `notebooks/03_feature_engineering.ipynb` | Problem B: Part 1 builds the feature tables with `src/features.py` and checks them; Part 2 selects features (Lasso stability selection, permutation importance) |
| `notebooks/05_regression_delivery_time.ipynb` | Problem B: baseline and Linear Regression, Lasso, Decision Tree, Random Forest and XGBoost under 5-fold time-series CV; tuning; final test evaluation; error analysis; feature importance |
| `src/` | Reusable config, data loading and splitting, features, models and evaluation code |
| `report/` | Report outline, feature engineering plan, figures and tables |
| `data/raw/` | Original Olist CSV files shared by all notebooks and scripts |
