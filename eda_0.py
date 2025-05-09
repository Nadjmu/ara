# ======================
# 0. Setup & Data Loading
# ======================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from scipy.stats import zscore, kurtosis, skew

# Load your data (replace with your file)
df = pd.read_csv('ARA_Region_Lenzburg.xls', 
                 parse_dates=['date_column'], 
                 index_col='date_column')

# Ensure proper datetime index
df = df.asfreq('D')  # Set daily frequency
print(f"Data from {df.index.min()} to {df.index.max()}")

# ======================
# 1. Data Quality Check
# ======================
print("\n=== Data Quality ===")
print(f"Total records: {len(df)}")
print(f"Missing values:\n{df.isna().sum()}")
print(f"Duplicate dates: {df.index.duplicated().sum()}")

# Visualize missing data
plt.figure(figsize=(10, 3))
sns.heatmap(df.isna().T, cbar=False, cmap='viridis')
plt.title("Missing Data Pattern")
plt.show()

# ======================
# 2. Descriptive Statistics
# ======================
print("\n=== Basic Stats ===")
stats = df.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
stats.loc['skewness'] = df.skew()
stats.loc['kurtosis'] = df.kurtosis()
print(stats)

# ======================
# 3. Time-Series Metrics
# ======================
def ts_metrics(series, name):
    """Calculate key time-series metrics"""
    metrics = {
        'ADF p-value': adfuller(series)[1],
        '1-day autocorr': series.autocorr(1),
        '7-day autocorr': series.autocorr(7),
        '30-day autocorr': series.autocorr(30),
        'Annual seasonality': series.autocorr(365)
    }
    return pd.Series(metrics, name=name)

ts_results = df.apply(ts_metrics)
print("\n=== Time-Series Metrics ===")
print(ts_results)

# ======================
# 4. Correlation Analysis
# ======================
print("\n=== Correlation ===")
corr_matrix = df.corr(method='pearson')
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Pearson Correlation")
plt.show()

# For non-linear relationships
from sklearn.feature_selection import mutual_info_regression
mi_matrix = pd.DataFrame(
    {col: mutual_info_regression(df.fillna(df.mean()), df[col]) 
     for col in df.columns},
    index=df.columns
)
sns.heatmap(mi_matrix, annot=True, cmap='Blues')
plt.title("Mutual Information (Non-linear Dependence)")
plt.show()

# ======================
# 5. Trend & Seasonality
# ======================
# Decomposition for each parameter
for col in df.columns:
    try:
        result = seasonal_decompose(df[col].interpolate(), model='additive', period=365)
        result.plot()
        plt.suptitle(f"Decomposition of {col}")
        plt.tight_layout()
        plt.show()
    except:
        print(f"Could not decompose {col}")

# ======================
# 6. Outlier Detection
# ======================
def detect_outliers(series, window=30, threshold=3):
    """Identify outliers using rolling Z-score"""
    roll_mean = series.rolling(window).mean()
    roll_std = series.rolling(window).std()
    z_scores = (series - roll_mean) / roll_std
    return series[abs(z_scores) > threshold]

outliers = {col: detect_outliers(df[col]) for col in df.columns}
print("\n=== Outliers ===")
for col, outs in outliers.items():
    print(f"{col}: {len(outs)} outliers")

# Visualize outliers
fig, axes = plt.subplots(len(df.columns), 1, figsize=(10, 2*len(df.columns)))
for ax, col in zip(axes, df.columns):
    df[col].plot(ax=ax, alpha=0.5, label='Normal')
    outliers[col].plot(ax=ax, style='ro', label='Outliers')
    ax.legend()
    ax.set_title(col)
plt.tight_layout()
plt.show()

# ======================
# 7. Advanced Analysis
# ======================
# A. Cross-correlation (lagged relationships)
max_lag = 30  # days
for target in df.columns:
    for feature in df.columns:
        if target != feature:
            ccf = [df[target].corr(df[feature].shift(lag)) for lag in range(max_lag)]
            plt.plot(ccf, label=f"{feature} → {target}")
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f"Cross-correlation with {target}")
    plt.xlabel("Lag (days)")
    plt.ylabel("Correlation")
    plt.legend()
    plt.show()

# B. Rolling statistics
window = 365  # 1-year window
for col in df.columns:
    plt.figure(figsize=(12, 4))
    df[col].plot(alpha=0.3, label='Daily')
    df[col].rolling(window).mean().plot(label=f'{window}-day mean')
    df[col].rolling(window).std().plot(label=f'{window}-day std')
    plt.title(f"Rolling Statistics for {col}")
    plt.legend()
    plt.show()

# ======================
# 8. Final Report
# ======================
print("\n=== Key Findings ===")
print(f"1. Data spans {len(df)} days with {df.isna().sum().sum()} missing values")
print(f"2. Strongest correlation: {corr_matrix.abs().stack().nlargest(3)}")
print(f"3. Most seasonal parameter: {ts_results.loc['Annual seasonality'].idxmax()}")
print(f"4. Parameter with most outliers: {max(outliers, key=lambda k: len(outliers[k]))}")