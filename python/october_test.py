import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timezone
from db import DbProvider


# === Aggregate higher timeframes ===
def aggregate_timeframe(df, hours):
    df = df.copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="us", errors="coerce")
    df = df.set_index("open_time").sort_index()
    agg = df.resample(f"{hours}h").agg({
        "open": "first",
        "close": "last",
    }).dropna().reset_index()
    agg["dir"] = (agg["close"] > agg["open"]).astype(int)
    return agg


# === Align higher TF directions back to 1h ===
def align_dirs(base_len, higher_df):
    return np.interp(
        np.arange(base_len),
        np.linspace(0, base_len - 1, len(higher_df)),
        higher_df["dir"]
    ).round().astype(int)


# === Build features ===
def build_features(df, lookback_hours=24):
    df = df.copy()
    df["dir"] = (df["close"] > df["open"]).astype(int)
    df_4h = aggregate_timeframe(df, 4)
    df_12h = aggregate_timeframe(df, 12)
    df_24h = aggregate_timeframe(df, 24)
    df["dir_4h"] = align_dirs(len(df), df_4h)
    df["dir_12h"] = align_dirs(len(df), df_12h)
    df["dir_24h"] = align_dirs(len(df), df_24h)

    X, y, times = [], [], []
    for i in range(lookback_hours, len(df) - 1):
        features = np.concatenate([df["dir"].iloc[i - lookback_hours:i].values,
                                   df["dir_4h"].iloc[i - lookback_hours // 4:i].values,
                                   df["dir_12h"].iloc[i - lookback_hours // 12:i].values,
                                   df["dir_24h"].iloc[i - lookback_hours // 24:i].values])
        X.append(features)
        y.append(df["dir"].iloc[i + 1])
        times.append(df["open_time"].iloc[i + 1])
    return np.array(X), np.array(y), np.array(times)


# === Train RandomForest ===
def train_rf(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# === Predict the next hour's candle ===
def predict_next_candle(model, df, lookback_hours=48, last_known_ts=None):

    # Select the data up to the last known timestamp (October 27 in this case)
    df_last = df[df["open_time"] < last_known_ts]

    # Build features for the most recent data
    X_live, _, _ = build_features(df_last, lookback_hours=lookback_hours)
    if len(X_live) == 0:
        print("Not enough data to predict.")
        return None

    # Reshape for prediction (the most recent features)
    X_pred = X_live[-1].reshape(1, -1)

    # Predict the next candle
    proba = model.predict_proba(X_pred)[0]
    pred = int(proba[1] > 0.5)
    conf = proba[1]

    # Calculate the start time of the predicted candle (next hour)
    predicted_time = last_known_ts # + 3_600_000_000

    # Map the prediction to U (Up) or D (Down)
    prediction = "U" if pred == 1 else "D"

    # Print the prediction and the predicted candle start time
    print(f"Prediction for the candle starting at {predicted_time}: {prediction} (Confidence: {conf:.2f})")
    return prediction, conf, predicted_time


# === Main async run to predict the next candle ===
async def run():
    db = DbProvider()
    res = await db.select("candles", ["open_time", "open_price", "close_price", "dir"],
                          None, 0, "ASC")
    df = pd.DataFrame(res, columns=["open_time", "open", "close", "dir"]).reset_index(
        drop=True)

    # Use data up to the end of October 27 (2025-10-27 23:00:00)
    cutoff_ts = 1759276800_000_000  # timestamp 2025-10-27 23:00:00 UTC in microseconds


    # Filter the data for the training and test sets
    df_train = df[df["open_time"] < cutoff_ts]
    df_test = df[df["open_time"] >= cutoff_ts]

    # Train the model on data up to the cutoff point
    X_train, y_train, _ = build_features(df_train, lookback_hours=48)
    model = train_rf(X_train, y_train)

    # Predict the first hour candle of October 28, 2025 (starting from 00:00:00)
    prediction, confidence, predicted_time = predict_next_candle(model, df, lookback_hours=48,
                                                                 last_known_ts=cutoff_ts)

    timestamp_s = predicted_time / 1_000_000
    utc_datetime = datetime.fromtimestamp(timestamp_s, timezone.utc)

    # Output the prediction, confidence, and predicted candle start time
    print(f"Predicted candle for the next hour ({utc_datetime}): {prediction} (Confidence: {confidence:.2f})")


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
