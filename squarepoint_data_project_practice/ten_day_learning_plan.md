# 10-day hand-coding study plan (2.5–4 hours/day)

All stages are **without AI** for the final timed drill. At the learning stage, read docs, but type code yourself. This is a training roadmap, not Squarepoint's official content.

| Day | Goal | Commands to write WITHOUT COPY/PASTE | End-of-day evidence |
|---|---|---|---|
| 1 | Python and Pandas basic data loading | `import`, `read_csv`, `head`, `shape`, `info`, `describe`, `dtypes`, column selection, Boolean filtering | Load CSV and explain every column |
| 2 | Data audit & cleaning | `isna().sum()`, `duplicated`, `sort_values`, `drop_duplicates`, `to_datetime`, `np.isclose` | Detect duplicates, bad squared field and leakage hint |
| 3 | Exploratory data analysis | `hist`, `plot`, `corr`, `groupby`, `quantile`, `skew`, `kurtosis` | Describe skew, spikes and hourly seasonality |
| 4 | Time-series transformations | `.shift`, `.rolling`, `.diff`, `np.log1p` (only when valid), `STL` | Produce lag/rolling features; plot training-only STL |
| 5 | Target and leakage | `shift(-1)` for LABEL only, train/val/test `iloc`, assert timestamp order | Explain every feature's availability at hour t |
| 6 | Baselines and metrics | `MAE`, `RMSE`, `mean_absolute_error`, `mean_squared_error` | Evaluate persistence baseline on validation |
| 7 | Linear model | `SimpleImputer`, `StandardScaler`, `Pipeline`, `Ridge.fit/predict` | Compare Ridge with baseline; never fit preprocessing on test |
| 8 | Nonlinear model | `HistGradientBoostingRegressor`, validation model comparison | Compare one tree model with Ridge; inspect errors |
| 9 | Reporting and defense | plots, residual autocorrelation, MAE by hour, markdown writeup | 5-minute explanation + answers to 10 critique questions |
| 10 | Six-hour mock interview | Complete `starter.py` from scratch, **do not** open reference solution | A reproducible script, plots, metrics, method and caveats |

## Recommended daily schedule
- 30 min: read the API documentation and type 10–15 syntax drills.
- 75 min: practice on the exercise data without AI or code completion.
- 45 min: close all guides, retype the day’s operations from memory.
- 20 min: say out loud in English what you did and why.
- Remaining time: fix mistakes and record a one-page error log.

## Pre-mock code checklist
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Audit
print(df.shape)
print(df.info())
print(df.isna().sum())
print(df.duplicated().sum())
print(df.describe(include='all'))

# Time series
df = df.sort_values('timestamp')
df['lag1'] = df['price'].shift(1)
df['ma24'] = df['price'].rolling(24).mean()
df['target'] = df['price'].shift(-1)

# Evaluation
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
```

Note: The `price` example in the checklist is pseudocode for syntax memorization. In the full task the real column is `price_usd_mwh`, and it must be handled according to the forecast cutoff. Never use a shifted future observation as an input feature.
