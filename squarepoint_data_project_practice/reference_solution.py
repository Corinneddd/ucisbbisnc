"""End-to-end 1-hour-ahead electricity price forecast.

Run: python reference_solution.py
Requires: numpy, pandas, scipy, matplotlib, scikit-learn, statsmodels
No internet; does not download any data. All outputs go to output/.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from statsmodels.tsa.seasonal import STL
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
DATA = ROOT / 'electricity_hourly_practice.csv'
if not DATA.exists():
    raise FileNotFoundError('Run python generate_data.py first.')

# 1. Initial audit: use only information observable at time t for predicting t+1.
df = pd.read_csv(DATA, parse_dates=['timestamp'])
print('Raw shape:', df.shape)
print('Column types:\n', df.dtypes.to_string())
print('Missing ratios:\n', df.isna().mean().round(3).to_string())
print('Duplicate timestamps:', df['timestamp'].duplicated().sum())
df = df.sort_values('timestamp').drop_duplicates('timestamp', keep='first').reset_index(drop=True)
assert df['timestamp'].is_monotonic_increasing
assert not df['timestamp'].duplicated().any()
assert df['timestamp'].diff().dropna().eq(pd.Timedelta(hours=1)).all(), 'Missing hourly timestamps: reindex and handle separately'

# 2. Validate deterministic transformations, with a mask excluding missing values.
valid = df['temperature_c'].notna() & df['temperature_c_squared'].notna()
wrong = valid & ~np.isclose(df['temperature_c_squared'], df['temperature_c']**2, atol=1e-6, rtol=1e-6)
print('Inconsistent temperature-squared entries:', wrong.sum())
# We DO NOT use that column, because it is unreliable. We also remove the
# settlement hint: it includes future information and causes severe leakage.
assert 'next_hour_settlement_hint' in df.columns

# 3. TARGET: the observed next-hour market price. Never fill missing targets.
df['target_next_hour'] = df['price_usd_mwh'].shift(-1)
# Only historical/current prices can be used as predictors.
# Forward-fill feature prices CAUSALLY; this must happen after ordering timestamps.
df['known_price_at_t'] = df['price_usd_mwh'].ffill()
df['price_lag_1'] = df['known_price_at_t'].shift(1)
df['price_lag_24'] = df['known_price_at_t'].shift(24)
df['price_lag_168'] = df['known_price_at_t'].shift(168)
# Trailing statistics ONLY (no centered rolling windows).
df['price_ma_24'] = df['known_price_at_t'].rolling(24, min_periods=24).mean()
df['price_std_24'] = df['known_price_at_t'].rolling(24, min_periods=24).std()
# Time-of-day at target timestamp (t+1) is known at prediction time.
future_time = df['timestamp'] + pd.Timedelta(hours=1)
df['next_hour_sin'] = np.sin(2 * np.pi * future_time.dt.hour / 24)
df['next_hour_cos'] = np.cos(2 * np.pi * future_time.dt.hour / 24)
df['next_dow'] = future_time.dt.dayofweek

FEATURES = ['known_price_at_t', 'price_lag_1', 'price_lag_24', 'price_lag_168',
            'price_ma_24', 'price_std_24', 'temperature_c', 'demand_mw',
            'next_hour_sin', 'next_hour_cos', 'next_dow']
# Drop history warm-up and rows without REAL target measurements.
model_df = df.iloc[168:].dropna(subset=['target_next_hour', 'known_price_at_t']).copy()
assert len(model_df) > 1000
assert 'next_hour_settlement_hint' not in FEATURES

# 4. Chronological 60/20/20 split (test remains untouched until final).
n = len(model_df)
i_train, i_val = int(n*0.6), int(n*0.8)
# One-row purge: final train label would fall inside the validation period;
# final validation label would fall inside the test period.
train = model_df.iloc[:i_train-1]
val = model_df.iloc[i_train:i_val-1]
test = model_df.iloc[i_val:]
print('Train/val/test:', len(train), len(val), len(test))
print('Ranges:', train['timestamp'].min(), train['timestamp'].max(), ' | ',
      val['timestamp'].min(), val['timestamp'].max(), ' | ',
      test['timestamp'].min(), test['timestamp'].max())

# 5. EDA on TRAIN ONLY (distribution and seasonal decomposition).
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
train['price_usd_mwh'].plot(ax=ax[0], title='Train: hourly electricity price')
train['price_usd_mwh'].dropna().hist(bins=50, ax=ax[1])
ax[1].set_title('Train: price distribution')
fig.tight_layout()
fig.savefig(OUT / 'eda_train.png', dpi=140)
plt.close(fig)
observed_train = df.loc[df['timestamp'].between(train['timestamp'].min(), train['timestamp'].max()),
                        'price_usd_mwh'].ffill().bfill().to_numpy()
# STL is only for descriptive training EDA here. Do not feed full-sample STL
# components into the forecasting model: centered filters leak future data.
stl = STL(observed_train, period=24, robust=True).fit()
fig = stl.plot()
fig.set_size_inches(10, 7)
fig.tight_layout()
fig.savefig(OUT / 'stl_train.png', dpi=140)
plt.close(fig)
print('Train price skew:', round(float(skew(train['price_usd_mwh'].dropna())), 3))
print('Train price excess kurtosis:', round(float(kurtosis(train['price_usd_mwh'].dropna())), 3))

X_train, y_train = train[FEATURES], train['target_next_hour']
X_val, y_val = val[FEATURES], val['target_next_hour']
X_test, y_test = test[FEATURES], test['target_next_hour']

# 6. Benchmark + two models. Imputer/scaler fit exclusively on training data.
models = {
    'ridge': Pipeline([
        ('impute', SimpleImputer(strategy='median', add_indicator=True)),
        ('scale', StandardScaler()),
        ('model', Ridge(alpha=10.0)),
    ]),
    'boosted_trees': Pipeline([
        ('impute', SimpleImputer(strategy='median', add_indicator=True)),
        ('model', HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15,
                                                learning_rate=0.06, random_state=42)),
    ]),
}

def metrics(y_true, y_pred):
    return {'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred))}

# Naive persistence = next hour is the same as currently observed price.
naive_val = val['known_price_at_t'].to_numpy()
records = [('persistence', *metrics(y_val, naive_val).values())]
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    result = metrics(y_val, pred)
    records.append((name, result['MAE'], result['RMSE']))
summary = pd.DataFrame(records, columns=['model', 'val_MAE', 'val_RMSE']).sort_values('val_MAE')
print('\nValidation comparison:\n', summary.to_string(index=False))
summary.to_csv(OUT / 'validation_comparison.csv', index=False)
best = summary.iloc[0]['model']

# 7. Final TEST: choose based on validation only. No further model tuning.
if best == 'persistence':
    test_pred = test['known_price_at_t'].to_numpy()
else:
    # Keep the fitted TRAIN model here; optionally refit on train+val in a
    # production workflow, but predeclare the procedure and never use test to tune.
    test_pred = models[best].predict(X_test)
print('\nChosen model:', best)
print('Final held-out test:', metrics(y_test, test_pred))
print('Test persistence benchmark:', metrics(y_test, test['known_price_at_t']))

residuals = y_test.to_numpy() - test_pred
print('Test residual mean:', round(float(residuals.mean()), 3))
print('Test residual lag-1 autocorrelation:',
      round(float(pd.Series(residuals).autocorr(lag=1)), 3))
fig, ax = plt.subplots(2, 1, figsize=(11, 6))
ax[0].plot(test['timestamp'], y_test, label='Actual', linewidth=0.8)
ax[0].plot(test['timestamp'], test_pred, label='Predicted', linewidth=0.8)
ax[0].legend()
ax[0].set_title('Held-out test: prediction vs actual')
ax[1].plot(test['timestamp'], residuals, linewidth=0.7)
ax[1].axhline(0, linestyle='--', color='black')
ax[1].set_title('Held-out test residuals')
fig.tight_layout()
fig.savefig(OUT / 'test_results.png', dpi=140)
plt.close(fig)
print('Wrote output/eda_train.png, stl_train.png, test_results.png and validation_comparison.csv')
