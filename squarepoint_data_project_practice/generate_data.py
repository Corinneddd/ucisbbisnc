"""Synthetic interview practice data, not real Squarepoint interview material."""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / 'electricity_hourly_practice.csv'
rng = np.random.default_rng(20261022)
n = 24 * 120
idx = pd.date_range('2025-01-01 00:00:00', periods=n, freq='h')
hour = idx.hour.to_numpy()
dow = idx.dayofweek.to_numpy()
daily = np.sin(2 * np.pi * (hour - 8) / 24)
weekly = (dow < 5).astype(float)
trend = np.linspace(0, 2, n)
temperature = 13 + 8 * np.sin(2 * np.pi * np.arange(n) / (24 * 30)) + 3 * daily + rng.normal(0, 2.2, n)
demand = 1300 + 175 * weekly + 90 * np.sin(2 * np.pi * (hour - 6) / 24) + 5 * np.maximum(18 - temperature, 0) + rng.normal(0, 40, n)
price = np.empty(n)
price[0] = 52
for t in range(1, n):
    equilibrium = 20 + 0.021 * demand[t] + 0.6 * max(5 - temperature[t], 0) + 5 * np.sin(2 * np.pi * hour[t] / 24) + trend[t]
    price[t] = 0.62 * price[t - 1] + 0.38 * equilibrium + rng.normal(0, 4.3)
spike_indices = rng.choice(np.arange(24, n), size=23, replace=False)
price[spike_indices] += rng.choice([-1, 1], size=23) * rng.uniform(25, 75, size=23)
raw = pd.DataFrame({
    'timestamp': idx.astype(str),
    'temperature_c': temperature,
    'demand_mw': demand,
    'price_usd_mwh': price,
    'temperature_c_squared': temperature**2,
    'irrelevant_noise': rng.standard_normal(n),
    'next_hour_settlement_hint': np.r_[price[1:] + rng.normal(0, 0.1, n-1), np.nan],
})
# Insert deliberately bad calculated values, missing records, and exact duplicates.
corrupt = rng.choice(n, size=30, replace=False)
raw.loc[corrupt, 'temperature_c_squared'] += rng.uniform(5, 25, size=30)
for col, count in [('temperature_c', 75), ('demand_mw', 45), ('price_usd_mwh', 35)]:
    raw.loc[rng.choice(n, size=count, replace=False), col] = np.nan
raw = pd.concat([raw, raw.sample(9, random_state=7)], ignore_index=True)
raw = raw.sample(frac=1, random_state=11).reset_index(drop=True)
raw.to_csv(OUT, index=False)
print(f'Saved {OUT} with {len(raw)} rows and {len(raw.columns)} columns')
