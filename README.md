# Olist E-commerce Explorer

Interactive Streamlit dashboard for IT5006: exploratory analysis (Milestone 1) and the deployed delivery-time regression model (Milestone 3). The repository also contains the Milestone 2 modelling notebooks for both prediction problems; see [Milestone 2: Modelling](#milestone-2-modelling).

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

## Milestone 3: Deployed model

The **Delivery Time Predictor** page (`pages/Delivery_Time_Predictor.py`) serves the final tuned XGBoost regression model from notebook 05 (`models/regression_final_model.joblib`).

Olist data ends in 2018, so the app does not connect to a live order feed. Instead it **replays the held-out Test orders** (6,351 orders purchased 1–29 Aug 2018, never used for fitting) in purchase-time order, as if they were new orders arriving:

- **Live order replay:** press *Next order* / *Next batch* or turn on *Auto-stream*. Each arriving order is scored at that moment and shown with its predicted and promised delivery dates. Its real outcome stays hidden ("In transit") until the simulated clock passes the actual delivery time; running MAE uses delivered orders only.
- **What-if:** edit promised days, freight, seller history or payment type for one order and re-score it.
- **Batch scoring:** upload a CSV with the model columns (template provided) and download predictions.
- **Model performance:** MAE / RMSE and error breakdown on the whole held-out month.

The model predicts `log(delivery_days / promised_delivery_days)`; the app converts back with `promised_delivery_days * exp(prediction)`. Held-out features use seller history frozen at the 2018-08-01 cutoff, exactly as in the notebook's final test evaluation, so the app reproduces the reported Test MAE (2.88 days).

On first load the page rebuilds the held-out features from `data/processed/` and `data/raw/` with `src/features.py` (about 30 s, then cached). Optionally, write them to a file so the page loads faster:

```bash
python scripts/export_holdout.py   # writes data/app/regression_holdout.csv.gz
```

`scikit-learn` and `xgboost` are pinned in `requirements.txt` to the versions that saved the model. On Streamlit Community Cloud choose **Python 3.12** under *Advanced settings* (the model was trained with 3.12).

## Milestone 2: Modelling

```bash
pip install -r requirements-modelling.txt
jupyter notebook notebooks/
```

The notebooks need scikit-learn 1.6.1 or later. The saved outputs were produced with Python 3.9 (Problem A notebooks) and Python 3.11 (Problem B notebooks).

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
| `src/` | Reusable config, data loading and splitting, features and evaluation code |
| `report/` | Report outline, feature engineering plan, and the regression result tables exported by notebook 05 (`report/tables/`) |
| `data/raw/` | Original Olist CSV files shared by all notebooks and scripts |
