import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def aggregate_timeframe(df, hours):
    df = df.copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="s", errors="coerce")
    df = df.set_index("open_time").sort_index()
    agg = df.resample(f"{hours}h").agg({
        "open": "first",
        "close": "last",
        "high": "max",
        "low": "min",
        "volume": "sum"
    }).dropna().reset_index()
    agg["dir"] = (agg["close"] > agg["open"]).astype(int)
    return agg

def martingale_multi_tf(res):
    # Convert DB tuples → DataFrame
    df = pd.DataFrame(res, columns=["open_time", "open", "high", "low", "close", "volume"])
    df = df.sort_values("open_time").reset_index(drop=True)

    # Aggregate to higher TFs
    df_4h = aggregate_timeframe(df, 4)
    df_12h = aggregate_timeframe(df, 12)
    df_24h = aggregate_timeframe(df, 24)

    # Base direction (1h)
    df["dir"] = (df["close"] > df["open"]).astype(int)

    # Align aggregated frames back to 1h via interpolation
    def align_dirs(base_len, higher_df):
        return np.interp(
            np.arange(base_len),
            np.linspace(0, base_len - 1, len(higher_df)),
            higher_df["dir"]
        ).round().astype(int)

    df["dir_4h"] = align_dirs(len(df), df_4h)
    df["dir_12h"] = align_dirs(len(df), df_12h)
    df["dir_24h"] = align_dirs(len(df), df_24h)

    # === Build features for training ===
    X, y = [], []
    for i in range(10, len(df) - 1):
        features = np.concatenate([
            df["dir"].iloc[i-6:i].values,        # last 6 1h
            df["dir_4h"].iloc[i-3:i].values,     # last 3 4h
            df["dir_12h"].iloc[i-2:i].values,    # last 2 12h
            [df["dir_24h"].iloc[i]]              # last 1 daily
        ])
        X.append(features)
        y.append(df["dir"].iloc[i + 1])

    X, y = np.array(X), np.array(y)

    # === Split for training/testing ===
    split_idx = int(len(X) * 0.98)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # === Normalize and fit ===
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    # === Predict and simulate ===
    probas = model.predict_proba(X_test)
    preds = (probas[:, 1] > 0.5).astype(int)
    conf = np.max(probas, axis=1)

    payout = 1.92
    balance = 1000
    bet = 20
    base_bet = 20
    wins = 0
    trades = 0

    for p, real, c in zip(preds, y_test, conf):
        if c < 0.6:
            continue
        trades += 1
        if p == real:
            balance += bet * (payout - 1)
            wins += 1
            bet = base_bet  # reset after win
        else:
            balance -= bet
            bet *= 2
            if bet > base_bet * 2 * 2 * 2 * 2 * 2:
                bet = base_bet  # stop-loss reset

    winrate = wins / trades if trades else 0
    roi = (balance - 1000) / trades if trades else 0

    print(f"\n=== Multi-TF Martingale Results ===")
    print(f"Trades: {trades}")
    print(f"Winrate: {winrate:.3f}")
    print(f"Total Profit: {balance - 1000:.2f}")
    print(f"Average Profit/Trade (ROI): {roi:.3f}")
    print(f"Balance: {balance:.2f}")

    return model, scaler

# Example usage:
async def run():
    # Example `res` structure from your DB
    # res = [(timestamp, open, high, low, close, volume), ...]
    # Make sure timestamps are in seconds (UNIX)
    import random, time
    now = int(time.time())
    res = [
        (now - i * 3600, 100 + random.random(), 101 + random.random(), 99 - random.random(), 100 + random.uniform(-1, 1), random.random()*100)
        for i in range(40000, 0, -1)
    ]
    martingale_multi_tf(res)

if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
