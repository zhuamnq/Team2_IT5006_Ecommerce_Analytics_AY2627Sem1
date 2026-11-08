# Project overview

This project uses the Olist Brazilian e-commerce dataset to study delivery
performance from the perspective of a marketplace customer-success and
operations team.

## Analytics problems

1. **Promise-band classification:** classify an order as delivered in less than
   half of its promised window, between half and the full window, or late. The
   revised evaluation selects models using late-class average precision and
   reports an operational precision/recall threshold analysis.
2. **Delivery-time regression:** predict delivery duration relative to the
   promised delivery window. The final model is a tuned XGBoost regressor.

Only the regression model is deployed. It is the stronger proof of concept and
can provide a revised expected delivery date at checkout. The classification
experiment remains part of the analysis but did not meet its provisional
business success criteria.

## End-to-end flow

Raw Olist CSV files are cleaned into chronological modelling cohorts. Feature
engineering uses only information available at purchase time, and historical
outcomes are restricted to deliveries completed before each cutoff. Models are
selected with temporal cross-validation, evaluated on later calendar periods,
and the chosen regression model is served through Streamlit using a frozen
August 2018 holdout that simulates incoming orders.
