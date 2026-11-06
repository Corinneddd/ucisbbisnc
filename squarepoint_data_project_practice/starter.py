"""Practice skeleton: implement the full project WITHOUT opening reference_solution.py.
Prediction: at end of hour t, forecast the observed price in hour t+1.
Run generate_data.py once before starting.
"""
import numpy as np
import pandas as pd

df = pd.read_csv("electricity_hourly_practice.csv", parse_dates=['timestamp'])

# TODO 1: print df.shape, df.info(), null counts, duplicates, and descriptive statistics.

# TODO 2: sort by timestamp, remove exact duplicates, confirm hourly spacing.

# TODO 3: check whether temperature_c_squared equals temperature_c ** 2.
#          Identify the column that leaks future target information.

# TODO 4: target_next_hour = next observed price. Do not fill any missing labels!

# TODO 5: create causal lag features (1,24,168), rolling mean/std(24),
#          time-of-day cyclic features for hour t+1, plus temperature and demand.

# TODO 6: chronological 60/20/20 train/validation/test split.

# TODO 7: train-only EDA: histogram, series plot, STL(period=24).
#          Describe skew, spikes and daily pattern.

# TODO 8: benchmark 'next price = current known price'; then Ridge and
#          HistGradientBoostingRegressor using Pipeline + SimpleImputer.
#          Fit every preprocessing step on TRAIN only.

# TODO 9: compare MAE and RMSE on VALIDATION; choose model by MAE.

# TODO 10: test chosen model ONCE; plot actual vs predicted and residuals.

# TODO 11: write one paragraph explaining assumptions, leakage prevention,
#          what you would do next, and a limitation.
