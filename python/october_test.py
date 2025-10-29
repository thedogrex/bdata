import time

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timezone, timedelta
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
        n_estimators=100,
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# === Predict next hour candle ===
def predict_next_candle(model, df, lookback_hours=48, last_known_ts=None):

    tbf = time.time()
    df_last = df[df["open_time"] < last_known_ts]
    print(f'{time.time() - tbf} dataframe by time sort')

    tb = time.time()
    X_live, _, _ = build_features(df_last, lookback_hours=lookback_hours)
    if len(X_live) == 0:
        print("Not enough data to predict.")
        return None
    print(f'{time.time()-tb} features build')

    X_pred = X_live[-1].reshape(1, -1)
    proba = model.predict_proba(X_pred)[0]
    pred = int(proba[1] > 0.5)
    conf = proba[1]

    prediction = "U" if pred == 1 else "D"
    return prediction, conf


# === Main async simulation for October 1st to 25th ===
async def run():
    db = DbProvider()
    res = await db.select("candles", ["open_time", "open_price", "close_price", "dir"],
                          None, 0, "ASC")
    df = pd.DataFrame(res, columns=["open_time", "open", "close", "dir"]).reset_index(drop=True)

    # === Train the model up to the start of October 1 ===
    df_train = df[df["open_time"] < int(datetime(2025, 10, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1_000_000)]
    X_train, y_train, _ = build_features(df_train, lookback_hours=48)
    model = train_rf(X_train, y_train)

    # === Loop through each day from October 1st to October 25th ===
    overall_wins = 0
    overall_losses = 0
    overall_balance = 0.0

    for day in range(1, 26):
        # === Filter data for the current day ===
        start_time = datetime(2025, 10, day, 0, 0, 0, tzinfo=timezone.utc)
        end_time = start_time + timedelta(days=1)

        start_ts = int(start_time.timestamp() * 1_000_000)
        end_ts = int(end_time.timestamp() * 1_000_000)

        df_day = df[(df["open_time"] >= start_ts) & (df["open_time"] < end_ts)]

        # === Betting simulation parameters ===
        balance = 0.0
        bet_size = 100
        win_coef = 1.96
        wins = 0
        losses = 0

        # === Loop over each hour in the current day ===
        for hour in range(24):
            current_ts = df_day["open_time"].iloc[hour]

            # Predict next candle
            tt = time.time()
            prediction, confidence = predict_next_candle(model, df, lookback_hours=48, last_known_ts=current_ts)
            print(f'prediction time: {time.time()-tt}')

            pred_dir = 1 if prediction == "U" else 0

            # Display prediction info
            ts_seconds = current_ts / 1_000_000
            candle_time = datetime.fromtimestamp(ts_seconds, timezone.utc)
            print(f"\n[{hour+1}/24] Candle at {candle_time} — Predicted: {prediction} (Conf: {confidence:.2f})")

            # Get the real direction from the dataframe (actual values for the current day)
            real_value = 1 if df_day["dir"].iloc[hour]=='U' else 0

            # Simulate bet
            if confidence < 0.53:
                print('Skip due to low confidence')
                continue

            if real_value == pred_dir:
                profit = bet_size * (win_coef - 1)
                balance += profit
                wins += 1
                result = "WIN"
            else:
                balance -= bet_size
                losses += 1
                result = "LOSS"

            print(f"→ Result: {result} | Current Balance: ${balance:.2f} pred: {pred_dir} real: {real_value}")

        # === Daily Summary ===
        daily_total_bets = wins + losses
        daily_winrate = (wins / daily_total_bets) * 100 if daily_total_bets > 0 else 0

        print("\n=== Daily Summary ===")
        print(f"Day: {start_time.date()}")
        print(f"Total Bets: {daily_total_bets}")
        print(f"Wins: {wins} | Losses: {losses}")
        print(f"Winrate: {daily_winrate:.2f}%")
        print(f"Final Balance for the Day: ${balance:.2f}")

        # Update overall stats
        overall_wins += wins
        overall_losses += losses
        overall_balance += balance

    # === Overall Summary ===
    total_bets = overall_wins + overall_losses
    overall_winrate = (overall_wins / total_bets) * 100 if total_bets > 0 else 0

    print("\n=== Overall Simulation Summary ===")
    print(f"Total Bets: {total_bets}")
    print(f"Wins: {overall_wins} | Losses: {overall_losses}")
    print(f"Winrate: {overall_winrate:.2f}%")
    print(f"Final Overall Balance: ${overall_balance:.2f}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
