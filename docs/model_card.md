# Delivery-time model card

## Intended use

The deployed model estimates delivery duration for an Olist order at checkout.
It is a proof of concept for customer-success and operations teams, not a live
production service.

## Model and target

- Model: tuned XGBoost regression pipeline.
- Target: `log(delivery_days / promised_delivery_days)`.
- Output: converted back to predicted delivery days and a predicted date.
- Test period: 1–29 August 2018, 6,351 held-out orders.

## Held-out performance

- MAE: 2.883 days.
- RMSE: 4.086 days.
- R²: 0.112.
- Bias: -0.070 days.
- MAE improvement over the promise baseline: 4.8%.

## Limitations

- The dataset covers historical Brazilian orders from 2016–2018.
- Results may not transfer to a different marketplace or a later period.
- Right-censoring and temporal drift remain possible sources of error.
- Historical Test results were inspected during earlier project iterations;
  rerunning the notebook cannot restore a completely independent holdout.
- The application replays historical orders and does not connect to a live
  order feed.
