import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 1. LOAD DATA
df = pd.read_csv("demand.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["sku", "store", "date"]
)

# Optional columns
for col in [
    "price",
    "promotion",
    "temperature",
    "rainfall"
]:
    if col not in df.columns:
        df[col] = 0

    df[col] = df[col].fillna(
        df[col].median()
    )

# ==========================================
# 2. ENTER FORECASTING MONTH
# ==========================================

month = input(
    "Enter forecasting month (YYYY-MM): "
)

try:
    forecast_start = pd.to_datetime(
        month + "-01"
    )
except:
    print("Invalid format. Use YYYY-MM")
    exit()

forecast_end = (
    forecast_start
    + pd.offsets.MonthEnd(1)
)

forecast_dates = pd.date_range(
    forecast_start,
    forecast_end,
    freq="D"
)

print(
    "\nForecasting:",
    forecast_start.strftime("%B %Y")
)

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================

group = df.groupby(
    ["sku", "store"]
)["demand"]

df["lag1"] = group.shift(1)

df["lag7"] = group.shift(7)

df["rolling7"] = group.transform(
    lambda x:
    x.shift(1)
    .rolling(7)
    .mean()
)

# Calendar features

df["dow"] = (
    df["date"].dt.dayofweek
)

df["month_num"] = (
    df["date"].dt.month
)

df["dow_sin"] = np.sin(
    2 * np.pi * df["dow"] / 7
)

df["dow_cos"] = np.cos(
    2 * np.pi * df["dow"] / 7
)

# Spatial features

df["store_avg"] = df.groupby(
    ["date", "store"]
)["demand"].transform("mean")

df["region_avg"] = df.groupby(
    ["date", "region"]
)["demand"].transform("mean")

# Remove missing lag values

df = df.dropna()

# ==========================================
# 4. FEATURES
# ==========================================

features = [
    "lag1",
    "lag7",
    "rolling7",
    "price",
    "promotion",
    "temperature",
    "rainfall",
    "dow_sin",
    "dow_cos",
    "store_avg",
    "region_avg"
]

# ==========================================
# 5. NUMPY RIDGE REGRESSION
# ==========================================

X = df[features].values

y = df["demand"].values

# Add intercept

X = np.c_[
    np.ones(len(X)),
    X
]

# Ridge parameter

lam = 1.0

I = np.eye(
    X.shape[1]
)

# Don't regularize intercept

I[0, 0] = 0

# Calculate coefficients

beta = np.linalg.pinv(
    X.T @ X + lam * I
) @ X.T @ y

print(
    "\nModel trained successfully."
)

# ==========================================
# 6. GENERATE MONTHLY FORECAST
# ==========================================

forecasts = []

for (sku, store), data in df.groupby(
    ["sku", "store"]
):

    data = data.sort_values(
        "date"
    )

    last = data.iloc[-1]

    # Historical demand

    history = data[
        "demand"
    ].tolist()

    for date in forecast_dates:

        # Lags

        lag1 = history[-1]

        lag7 = (
            history[-7]
            if len(history) >= 7
            else history[-1]
        )

        rolling7 = np.mean(
            history[-7:]
        )

        # Calendar

        dow = date.dayofweek

        dow_sin = np.sin(
            2 * np.pi * dow / 7
        )

        dow_cos = np.cos(
            2 * np.pi * dow / 7
        )

        # Use latest known external values

        price = last["price"]

        promotion = last["promotion"]

        temperature = last[
            "temperature"
        ]

        rainfall = last[
            "rainfall"
        ]

        store_avg = last[
            "store_avg"
        ]

        region_avg = last[
            "region_avg"
        ]

        # Feature vector

        x = np.array([
            lag1,
            lag7,
            rolling7,
            price,
            promotion,
            temperature,
            rainfall,
            dow_sin,
            dow_cos,
            store_avg,
            region_avg
        ])

        # Prediction

        prediction = (
            np.r_[
                1,
                x
            ] @ beta
        )

        # No negative demand

        prediction = max(
            0,
            prediction
        )

        forecasts.append([
            sku,
            store,
            last["rdc"],
            last["region"],
            date,
            prediction
        ])

        # Recursive forecasting

        history.append(
            prediction
        )

# ==========================================
# 7. FORECAST DATAFRAME
# ==========================================

forecast = pd.DataFrame(
    forecasts,
    columns=[
        "sku",
        "store",
        "rdc",
        "region",
        "date",
        "forecast"
    ]
)

# ==========================================
# 8. HIERARCHICAL AGGREGATION
# ==========================================

sku_forecast = forecast.groupby(
    ["sku", "date"]
)["forecast"].sum()

store_forecast = forecast.groupby(
    ["store", "date"]
)["forecast"].sum()

rdc_forecast = forecast.groupby(
    ["rdc", "date"]
)["forecast"].sum()

region_forecast = forecast.groupby(
    ["region", "date"]
)["forecast"].sum()

national_forecast = forecast.groupby(
    "date"
)["forecast"].sum()

# ==========================================
# 9. RECONCILIATION
# ==========================================

# Make bottom-level forecasts
# sum exactly to national forecast

reconciled = forecast.copy()

for date in forecast_dates:

    mask = (
        reconciled["date"]
        == date
    )

    total = reconciled.loc[
        mask,
        "forecast"
    ].sum()

    if total > 0:

        target = national_forecast[
            date
        ]

        factor = (
            target / total
        )

        reconciled.loc[
            mask,
            "forecast"
        ] *= factor

# ==========================================
# 10. RMSSE
# ==========================================

actual = df["demand"].values

# Historical average prediction

prediction = np.repeat(
    df["demand"].mean(),
    len(actual)
)

scale = np.mean(
    np.diff(actual) ** 2
)

if scale > 0:

    rmsse = np.sqrt(
        np.mean(
            (actual - prediction) ** 2
        ) / scale
    )

else:

    rmsse = 0

# ==========================================
# 11. DISPLAY RESULTS
# ==========================================

print("\n" + "=" * 50)

print(
    "FORECAST RESULT -",
    forecast_start.strftime(
        "%B %Y"
    )
)

print("=" * 50)

print(
    "\nNational Forecast:"
)

print(
    national_forecast.round(2)
)

print(
    "\nRMSSE:",
    round(rmsse, 3)
)

print(
    "\nSample Detailed Forecast:"
)

print(
    reconciled.head(20).round(2)
)

# ==========================================
# 12. SAVE FORECAST
# ==========================================

reconciled.to_csv(
    "reconciled_forecast.csv",
    index=False
)

print(
    "\nSaved:"
)

print(
    "reconciled_forecast.csv"
)

# ==========================================
# 13. PLOT NATIONAL FORECAST
# ==========================================

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    national_forecast.index,
    national_forecast.values,
    marker="o",
    color="blue",
    linewidth=2
)

plt.title(
    "National Demand Forecast - "
    + forecast_start.strftime(
        "%B %Y"
    )
)

plt.xlabel("Date")

plt.ylabel("Forecast Demand")

plt.xticks(
    rotation=45
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "forecast.png",
    dpi=300
)

plt.show()

# ==========================================
# 14. PROJECT SUMMARY
# ==========================================

print("\n" + "=" * 50)

print("PROJECT COMPLETED")

print("=" * 50)

print("""
Hierarchy:

SKU
 ↓
Store
 ↓
RDC
 ↓
Region
 ↓
National

Features:
- Historical demand
- Lag demand
- Rolling demand
- Price
- Promotion
- Temperature
- Rainfall
- Calendar features
- Spatial aggregation

Output:
- Monthly daily forecasts
- Hierarchical forecasts
- Reconciled forecasts
- RMSSE
- Forecast graph
""")
