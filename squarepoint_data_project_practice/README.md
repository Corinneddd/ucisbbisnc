# Squarepoint-style Dataset Project practice (UNOFFICIAL)

**This is an independently constructed exercise, not an actual Squarepoint assessment.**

## Your assignment
It is the end of hour **t**. Predict the market electricity price during **hour t+1** using data that would be observable at or before hour t. You are given a synthetic hourly dataset with temperature, demand and price. Some fields contain data-quality problems and a future-information leak. You have **six hours** to produce a rigorous, defensible model. Strong methodology matters more than a high score.

## Deliverables
1. Data audit: missing values, duplicates, gaps, inconsistent calculated columns, leakage candidate, suspicious price spikes.
2. Training-only EDA: distribution, skew/outliers, autocorrelation, daily/weekly seasonality. Optionally use STL for **descriptive EDA**, not as leaked full-sample input.
3. A defensible imputation and cleaning strategy, documenting decisions.
4. Causal features: lag 1/24/168, trailing rolling statistics, known calendar features, available exogenous variables.
5. Strict chronological approximately 60% train / 20% validation / 20% final test, with a one-observation gap at split boundaries so next-hour target labels do not cross segments.
6. Naive persistence benchmark + regularized linear model + one tree model.
7. Validation MAE and RMSE, selected model evaluated only once on untouched test.
8. Residual diagnosis, written assumptions, caveats and possible improvements.

## Run
```bash
python -m pip install numpy pandas scipy matplotlib scikit-learn statsmodels
python generate_data.py
python starter.py  # This is intentionally unfinished; code your own solution.
python reference_solution.py  # Only AFTER you have attempted it yourself.
```

The dataset is generated with a fixed seed and requires no network access. The reference solution produces figures and a validation comparison table under `output/`.

## Six-hour mock interview clock
- 00:00–00:40: dataset definitions, audit, missingness, integrity, leakage.
- 00:40–01:35: EDA, distributions, outliers, seasonality, plots.
- 01:35–02:25: target, chronological splits, causal features, baseline.
- 02:25–03:45: Ridge/tree pipelines, imputation, validation.
- 03:45–04:30: residuals, failure cases, robustness check.
- 04:30–05:30: clean code, figures, results table, limitations.
- 05:30–06:00: rehearse presenting assumptions and possible follow-ups.

## Deliberate traps (SPOILERS)
If you want to practice unassisted, stop reading here.

- Exactly duplicated timestamps are mixed into an unsorted file.
- Some missing observed prices must **not** be converted into fake target labels.
- `temperature_c_squared` contains deliberate mismatches.
- `next_hour_settlement_hint` is unavailable at prediction time, despite looking useful; do not include it.
- STL often uses centered information and is unsuitable as a naive full-sample backtest feature.
- Price spikes should not automatically be deleted; electricity prices genuinely spike.
- Chronological model comparison is not a trading P&L backtest.

## Questions a senior researcher might ask
1. When is price_t actually published? Would it truly be available at the forecast cutoff?
2. Why does chronological splitting help? What about label overlap at the split boundary?
3. Why is an imputer fitted only on train? What about fitting it on train+validation later?
4. Why might log-transforming electricity prices be invalid? (Prices can be zero/negative.)
5. How would you replace the naive persistence model for a 24-hour forecast?
6. Why might the validation winner lose on test? What is distribution shift?
7. Why does a high Pearson correlation not necessarily imply predictive value?
8. How do you handle real spikes versus sensor/record errors?
9. How would you perform rolling-origin CV without the current train/validation split?
10. What if the dataset were independent boat listings rather than hourly prices?

## A note on prediction cutoffs
The exercise **assumes** observed values for hour t are available at the end of hour t before making the t+1 forecast. If electricity data is published later in reality, shift the features to their true release times. This is an essential modeling assumption, not a guarantee about actual exchanges.
