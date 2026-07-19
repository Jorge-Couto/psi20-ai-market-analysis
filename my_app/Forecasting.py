# =========================================================
# 0) LIBRARY IMPORTS
# =========================================================
# Core data handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning utilities
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Deep learning (LSTM model)
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

# Time utilities for business-day indexing
from pandas.tseries.offsets import BDay

# fix random seed for reproducibility
tf.random.set_seed(66)

# =========================================================
# 1) DATA CLEANING
# =========================================================
def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw Yahoo Finance stock data and ensures:
    - Proper header formatting
    - Correct datetime indexing
    - Numeric consistency
    - Basic outlier handling on returns

    Returns:
        Cleaned pandas DataFrame indexed by Date
    """
    df = df.copy()

    # -------------------------------------------------
    # FIX YAHOO MULTI-ROW HEADER
    # -------------------------------------------------
    # Row 0 = Ticker row
    # Row 1 = Date row
    # Row 2 onwards = actual values
    if str(df.iloc[0, 0]).lower() == "ticker":
        real_cols = ["Date"] + list(df.columns[1:])
        df = df.iloc[2:].copy()
        df.columns = real_cols

    # Standardize column naming (e.g., "close" → "Close")
    df.columns = [str(c).strip().title() for c in df.columns]

    # Parse dates
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        # if Date is already index, reset it safely
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce") # Convert Date column to datetime format
    df = df.dropna(subset=["Date"]) # Remove invalid or missing dates
    df = df.sort_values("Date") # Sort chronologically
    df = df.set_index("Date") # Set Date as index

    # ---------------------------------------------------------
    # ENSURE NUMERIC CONSISTENCY
    # ---------------------------------------------------------
    numeric_cols = ["Close", "High", "Low", "Open", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove invalid rows
    df = df.dropna(subset=["Close", "High", "Low"])

    # ---------------------------------------------------------
    # OUTLIER HANDLING VIA RETURN WINSORIZATION
    # ---------------------------------------------------------
    # Instead of clipping prices directly, we:
    # 1. Compute log returns
    # 2. Clip extreme returns
    # 3. Reconstruct price series from clipped returns
    returns = np.log(df["Close"]).diff()
    lower = returns.quantile(0.005)
    upper = returns.quantile(0.995)
    
    # Store cleaned returns as optional feature
    df["ret_1_clean"] = returns.clip(
        lower,
        upper
    )

    return df

# =========================================================
# 2) FEATURE ENGINEERING
# =========================================================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates predictive features for the LSTM model.

    The goal is to expose the model to multiple dimensions of market behavior:
    - Returns (momentum & short-term direction)
    - Volatility (risk/regime)
    - Trend (moving averages)
    - Relative positioning (overbought/oversold)
    - Volume dynamics (market participation)

    Also defines the forward-looking target variable (5-day return).
    """

    df = df.copy()

    # =========================================================
    # 1) RETURN-BASED FEATURES (Momentum & short-term movement)
    # =========================================================

    # 1-day log return (basic building block of most financial models)
    df["ret_1"] = df["ret_1_clean"]

    # 1-day log return (raw, not cleaned through smooth winsorization
    df["ret_raw"] = np.log(df["Close"]).diff()

    # 5-day cumulative return (captures short-term trend)
    df["ret_5"] = df["ret_1"].rolling(5).sum()

    # 10-day momentum (medium-term directional signal)
    df["momentum_10"] = df["Close"] / df["Close"].shift(10) - 1

    # =========================================================
    # 2) VOLATILITY FEATURES (Risk & regime detection)
    # =========================================================

    # Short-term volatility (5-day rolling std of returns)
    df["volatility_5"] = df["ret_1"].rolling(5).std()

    # Medium-term volatility (20-day rolling std)
    df["volatility_20"] = df["ret_1"].rolling(20).std()

    # Daily price range relative to close (intraday volatility proxy)
    df["range_pct"] = (df["High"] - df["Low"]) / df["Close"]

    # =========================================================
    # 3) TREND FEATURES (Market direction & structure)
    # =========================================================

    # Moving averages
    df["ma_10"] = df["Close"].rolling(10).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()

    # Trend ratio:
    # > 1 → short-term bullish trend
    # < 1 → bearish trend
    df["trend_ratio"] = (df["ma_10"] / (df["ma_20"] + 1e-8))

    # Distance from moving average (mean reversion signal)
    df["price_vs_ma20"] = df["Close"] / df["ma_20"] - 1

    # =========================================================
    # 4) RELATIVE POSITIONING (Overbought / oversold)
    # =========================================================

    # Z-score relative to 20-day rolling window
    # Measures how far price is from its recent "normal"
    rolling_mean_20 = df["Close"].rolling(20).mean()
    rolling_std_20 = df["Close"].rolling(20).std()

    df["zscore_20"] = (df["Close"] - rolling_mean_20) / rolling_std_20

    # =========================================================
    # 5) VOLUME FEATURES (Market participation & anomalies)
    # =========================================================

    if "Volume" in df.columns:
        # Volume shock:
        # > 1 means unusually high volume
        # < 1 means low activity
        df["volume_shock"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-8)

        # Volume momentum (change in trading activity)
        df["volume_momentum"] = df["Volume"] / df["Volume"].shift(5) - 1
    else:
        # Fallback if volume data is unavailable
        df["volume_shock"] = 0.0
        df["volume_momentum"] = 0.0

    # =========================================================
    # 6) ATR (Average True Range)
    # =========================================================
    
    tr1 = df["High"] - df["Low"]
    
    tr2 = abs(
        df["High"] - df["Close"].shift(1)
    )
    
    tr3 = abs(
        df["Low"] - df["Close"].shift(1)
    )
    
    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)
    
    df["atr_14"] = true_range.rolling(14).mean()

    # =========================================================
    # 7) TARGET VARIABLE (Volatility-Standardized Returns)
    # =========================================================
    
    for h in range(1, 6):
        # Forward log return
        raw_return = np.log(df["Close"].shift(-h) / df["Close"])
        # Volatility-standardized target
        df[f"target_{h}d"] = (raw_return / (df["volatility_20"] + 1e-8))
    
    return df

def create_sequences(features, targets, lookback=90):
    """
    Converts time-series data into supervised learning sequences.

    Each sample:
    - X: past `lookback` feature window
    - y: correctly aligned target (vector of size 5)
    """
    X, y = [], []

    for i in range(lookback, len(features)):
        X.append(features[i - lookback:i])
        y.append(targets[i])   # now a vector of size 5

    return np.array(X), np.array(y)

# =========================================================
# 4) TRAIN LSTM FORECASTER
# =========================================================
def train_lstm_forecaster(df: pd.DataFrame, lookback=90, horizon=5):
    """
    Full end-to-end training pipeline:

    Steps:
    1. Clean raw data
    2. Generate engineered features
    3. Normalize feature space
    4. Convert to LSTM sequences
    5. Train neural network model
    """

    # ---------------------------------------------------------
    # DATA PREPARATION
    # ---------------------------------------------------------
    df = clean_stock_data(df)
    df = add_features(df)

    feature_cols = [
    "ret_1",          # short-term movement
    "ret_5",          # short-term trend
    "momentum_10",    # medium-term momentum
    "volatility_5",   # recent realized volatility
    "volatility_20",  # broader volatility regime
    "range_pct",      # intraday volatility
    "atr_14",         # market stress / expansion
    "trend_ratio",    # MA10 / MA20
    "price_vs_ma20",  # mean reversion distance
    "zscore_20",      # overbought / oversold
    "volume_shock",       # abnormal participation
    "volume_momentum"     # acceleration/deceleration in volume
    ]
    
    # Remove rows with missing feature or target values
    train_df = df.dropna(subset=feature_cols + ["target_5d"]).copy()

    # ---------------------------------------------------------
    # REMOVE INF / EXTREME VALUES (TO ENSURE NO ERRORS OCCUR)
    # ---------------------------------------------------------
    
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    
    # Remove rows containing NaNs after feature engineering
    train_df = train_df.dropna(
        subset=feature_cols + [f"target_{h}d" for h in range(1, 6)]
    )
    
    # Optional extra protection:
    # clip absurdly large feature magnitudes
    train_df[feature_cols] = train_df[feature_cols].clip(
        lower=-1e6,
        upper=1e6
    )

    X_raw = train_df[feature_cols].values
    y_raw = train_df[[f"target_{h}d" for h in range(1, 6)]].values

    # ---------------------------------------------------------
    # FEATURE SCALING
    # ---------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # ---------------------------------------------------------
    # SEQUENCE CONSTRUCTION
    # ---------------------------------------------------------
    X, y = create_sequences(X_scaled, y_raw, lookback=lookback)

    # Train/test split (time-ordered, no shuffling)
    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # ---------------------------------------------------------
    # MODEL INITIALIZATION
    # ---------------------------------------------------------
    model = build_mc_lstm(
        input_shape=(X_train.shape[1], X_train.shape[2])
    )

    # Early stopping prevents overfitting
    es = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    # ---------------------------------------------------------
    # MODEL TRAINING
    # ---------------------------------------------------------
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=40,
        batch_size=32,
        callbacks=[es],
        verbose=1
    )

    return model, scaler, df, feature_cols

# =========================================================
# 5) MODEL ARCHITECTURE
# =========================================================
def build_mc_lstm(input_shape):
    """
    Defines an LSTM model with dropout layers for
    Monte Carlo Dropout uncertainty estimation.
    """
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dropout(0.2),
        Dense(5) # Predict log return for the next 5 days simultaneously
    ])

    model.compile(
        optimizer="adam",
        loss="mae", # or "tf.keras.losses.Huber()" or "mse", testing Huber for better forecasts
        metrics=["mae"]
    )
    return model

def compute_var_cvar(pred_samples, alpha=0.05):
    """
    Computes financial risk metrics:
    - VaR (Value at Risk)
    - CVaR (Conditional Value at Risk)

    These are derived from simulated return distributions.
    """
    var = np.percentile(pred_samples, alpha * 100)
    cvar = pred_samples[pred_samples <= var].mean()
    return var, cvar

# =========================================================
# 6) MONTE CARLO FORECAST (FIXED MULTI-HORIZON VERSION)
# =========================================================
def mc_dropout_forecast_path(
    model,
    scaler,
    df,
    feature_cols,
    lookback=90,
    horizon=5,
    n_samples=200
):
    """
    Generates probabilistic multi-horizon forecasts.

    Unlike recursive forecasting:
    - each horizon is predicted independently
    - uncertainty is estimated separately per horizon

    This produces much more realistic confidence bands.
    """

    # ---------------------------------------------------------
    # PREPARE LATEST FEATURE WINDOW
    # ---------------------------------------------------------
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.clip(-1e6, 1e6)
    df = df.dropna(subset=feature_cols)
    latest_features = df[feature_cols].values[-lookback:]

    latest_scaled = scaler.transform(latest_features)

    X_input = latest_scaled.reshape(
        1,
        lookback,
        len(feature_cols)
    )

    # ---------------------------------------------------------
    # MONTE CARLO DROPOUT SAMPLING
    # ---------------------------------------------------------
    # Shape:
    # (n_samples, horizon)
    #
    # Example:
    # sample 0 -> [d1, d2, d3, d4, d5]
    # sample 1 -> [d1, d2, d3, d4, d5]
    #
    pred_samples = []

    for _ in range(n_samples):

        pred = model(
            X_input,
            training=True
        ).numpy()[0]

        pred_samples.append(pred)

    pred_samples = np.array(pred_samples)

    # ---------------------------------------------------------
    # LAST OBSERVED MARKET PRICE
    # ---------------------------------------------------------
    last_price = df["Close"].iloc[-1]

    # ---------------------------------------------------------
    # FUTURE BUSINESS DATES
    # ---------------------------------------------------------
    future_dates = pd.bdate_range(
        start=df.index[-1] + BDay(1),
        periods=horizon
    )

    rows = []

    # ---------------------------------------------------------
    # COMPUTE FORECAST STATISTICS PER HORIZON
    # ---------------------------------------------------------
    for h in range(horizon):

        # Distribution ONLY for this horizon
        latest_vol = df["volatility_20"].iloc[-1]
        horizon_returns = pred_samples[:, h] * latest_vol

        # Mean prediction
        mean_ret = horizon_returns.mean()

        # Confidence interval
        #lower_ret = np.percentile(
        #    horizon_returns,
        #    2.5
        #)

        #upper_ret = np.percentile(
        #    horizon_returns,
        #    97.5
        #)

        # ---------------------------------------------------------
        # OPTIONAL HORIZON-BASED UNCERTAINTY SCALING
        # ---------------------------------------------------------
        spread_scale = 1 + (h * 0.15)
        mean_ret = horizon_returns.mean()
        
        lower_ret = np.percentile(horizon_returns, 2.5)
        upper_ret = np.percentile(horizon_returns, 97.5)
        
        # Distance from mean
        lower_dev = mean_ret - lower_ret
        upper_dev = upper_ret - mean_ret
        
        # Expand uncertainty with horizon
        lower_ret = mean_ret - lower_dev * spread_scale
        upper_ret = mean_ret + upper_dev * spread_scale

        # Convert log returns -> prices
        mean_price = last_price * np.exp(mean_ret)

        lower_price = last_price * np.exp(lower_ret)

        upper_price = last_price * np.exp(upper_ret)

        rows.append({
            "date": future_dates[h],
            "forecast_day": h + 1,
            "mean_return": mean_ret,
            "lower_return": lower_ret,
            "upper_return": upper_ret,
            "forecast_price": mean_price,
            "lower_ci": lower_price,
            "upper_ci": upper_price
        })

    # ---------------------------------------------------------
    # RISK METRICS USING FINAL HORIZON ONLY
    # ---------------------------------------------------------
    final_horizon_returns = pred_samples[:, -1]

    var_95 = np.percentile(
        final_horizon_returns,
        5
    )

    cvar_95 = final_horizon_returns[
        final_horizon_returns <= var_95
    ].mean()

    print(f"5D VaR(95%): {var_95:.4f}")
    print(f"5D CVaR(95%): {cvar_95:.4f}")

    rows.append({
        "5D VaR(95%)": var_95,
        "5D CVaR(95%)": cvar_95
    })

    forecast_df = pd.DataFrame(rows)

    return forecast_df

# =========================================================
# 7) HISTORICAL PLOT (MULTI-HORIZON VERSION)
# =========================================================
def plot_historical_model_fit(
    model,
    scaler,
    df,
    feature_cols,
    lookback=90,
    history_window=250
):
    """
    Compares predicted vs realized future prices
    using the multi-horizon forecast model.

    The model simultaneously predicts:
    - t+1
    - t+2
    - t+3
    - t+4
    - t+5 returns

    For visualization simplicity,
    we plot ONLY the t+5 prediction historically,
    since it represents the final forecast horizon.
    """

    # ---------------------------------------------------------
    # REMOVE ROWS WITH MISSING VALUES
    # ---------------------------------------------------------
    target_cols = [f"target_{h}d" for h in range(1, 6)]
    
    eval_df = df.copy()
    
    # Remove infinities
    eval_df = eval_df.replace([np.inf, -np.inf], np.nan)
    
    # Remove absurdly large values
    eval_df = eval_df.clip(-1e6, 1e6)
    
    # Drop invalid rows
    eval_df = eval_df.dropna(
        subset=feature_cols + target_cols
    ).copy()

    # ---------------------------------------------------------
    # SCALE FEATURES
    # ---------------------------------------------------------
    features = scaler.transform(
        eval_df[feature_cols].values
    )

    # ---------------------------------------------------------
    # CREATE LSTM SEQUENCES
    # ---------------------------------------------------------
    X, y = create_sequences(
        features,
        eval_df[target_cols].values,
        lookback
    )

    # ---------------------------------------------------------
    # MODEL PREDICTIONS
    # ---------------------------------------------------------
    # Shape:
    # (samples, 5)
    preds = model.predict(X, verbose=0)

    # ---------------------------------------------------------
    # USE ONLY THE 5-DAY HORIZON
    # ---------------------------------------------------------
    # Index 4 corresponds to t+5
    vol = eval_df["volatility_20"].iloc[lookback:].values
    preds_5d = preds[:, 4] * vol

    # ---------------------------------------------------------
    # ALIGN DATES
    # ---------------------------------------------------------
    base_dates = eval_df.index[lookback:]

    # Realization dates occur 5 business days later
    future_dates = base_dates + BDay(5)

    # ---------------------------------------------------------
    # CONVERT RETURNS -> PRICES
    # ---------------------------------------------------------
    base_prices = eval_df["Close"].values[lookback:]

    pred_prices = base_prices * np.exp(preds_5d)

    # ---------------------------------------------------------
    # REAL FUTURE PRICES
    # ---------------------------------------------------------
    real_prices = (
        eval_df["Close"]
        .reindex(future_dates, method="nearest")
        .values
    )

    # ---------------------------------------------------------
    # PLOT
    # ---------------------------------------------------------
    plt.figure(figsize=(14, 6))

    plt.plot(
        future_dates[-history_window:],
        real_prices[-history_window:],
        label="Realized Future Price"
    )

    plt.plot(
        future_dates[-history_window:],
        pred_prices[-history_window:],
        label="Predicted Future Price"
    )

    plt.title("Historical Fit (5-Day Horizon)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# =========================================================
# 8) FORECAST PLOT (MULTI-HORIZON VERSION)
# =========================================================
def plot_mc_forecast(df, forecast_df, history_window=60):
    """
    Visualizes:

    - Historical stock prices
    - Multi-horizon forecast trajectory
    - Confidence intervals for each horizon

    Unlike recursive forecasting,
    each future point is predicted simultaneously
    by the neural network.
    """
    forecast_df = forecast_df.copy()

    # Remove invalid rows
    forecast_df = forecast_df.replace([np.inf, -np.inf], np.nan)

    forecast_df = forecast_df.dropna(
        subset=[
            "date",
            "forecast_price",
            "lower_ci",
            "upper_ci"
        ]
    )

    # ---------------------------------------------------------
    # INITIALIZE FIGURE
    # ---------------------------------------------------------
    plt.figure(figsize=(14, 6))

    # ---------------------------------------------------------
    # HISTORICAL PRICE WINDOW
    # ---------------------------------------------------------
    hist = (
        df["Close"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .iloc[-history_window:]
    )

    plt.plot(
        hist.index,
        hist.values,
        label="Historical Close"
    )

    # ---------------------------------------------------------
    # FORECASTED PRICE PATH
    # ---------------------------------------------------------
    plt.plot(
        forecast_df["date"],
        forecast_df["forecast_price"],
        marker="o",
        label="Forecasted Price Path"
    )

    # ---------------------------------------------------------
    # CONFIDENCE INTERVALS
    # ---------------------------------------------------------
    plt.fill_between(
        forecast_df["date"],
        forecast_df["lower_ci"],
        forecast_df["upper_ci"],
        alpha=0.25,
        label="95% Confidence Band"
    )

    # ---------------------------------------------------------
    # SEPARATION BETWEEN HISTORY & FORECAST
    # ---------------------------------------------------------
    plt.axvline(
        df.index[-1],
        linestyle="--",
        alpha=0.7
    )

    # ---------------------------------------------------------
    # COSMETICS
    # ---------------------------------------------------------
    plt.title("LSTM Multi-Horizon Forecast")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()

# =========================================================
# 9) RUN MULTI-STOCK PIPELINE
# =========================================================
def run_forecast_from_csv(csv_path):
    """
    Streamlit-friendly wrapper.

    Loads CSV data, trains the model,
    generates forecasts, and returns:
    - forecast dataframe
    - cleaned dataframe used internally
    """

    # Load raw data
    df = pd.read_csv(csv_path)

    # Train pipeline
    model, scaler, df_ready, feature_cols = train_lstm_forecaster(
        df,
        lookback=120
    )

    # Generate forecast
    forecast_df = mc_dropout_forecast_path(
        model,
        scaler,
        df_ready,
        feature_cols,
        lookback=120,
        horizon=5,
        n_samples=200
    )

    return forecast_df, df_ready