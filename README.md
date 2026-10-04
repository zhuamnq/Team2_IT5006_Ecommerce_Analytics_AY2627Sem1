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

| Path | Contents |
| --- | --- |
| `notebooks/01_problem_scoping.ipynb` | Problem statements, stakeholders, targets, leakage rules |
| `notebooks/02_preprocessing.ipynb` | Cleaning and train/test split |
| `notebooks/03_feature_engineering.ipynb` | Feature construction |
| `notebooks/04_classification_late_delivery.ipynb` | Late-delivery classification |
| `notebooks/05_regression_freight.ipynb` | Freight-cost regression |
| `notebooks/06_model_comparison.ipynb` | CV comparison, ensembles, feature importance, final selection |
| `src/` | Reusable config, data, feature, model and evaluation code |
| `report/` | Report outline, figures and tables |
| `data/raw/` | Original Olist CSV files shared by all notebooks and scripts |
