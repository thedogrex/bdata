import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from db import DbProvider

lookback_hours = 48

def aggregate_timeframe(df, hours):
    df = df.copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="us", errors="coerce")
    df = df.set_index("open_time").sort_index()

    # Resample strictly by time
    agg = df.resample(f"{hours}H").agg({
        "open": "first",
        "close": "last",
    }).dropna().reset_index()

    agg["dir"] = (agg["close"] > agg["open"]).astype(int)
    return agg

def martingale_multi_tf(res):
    # Convert DB tuples → DataFrame
    df = pd.DataFrame(res, columns=["open_time", "open", "close", "dir"])
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
    for i in range(lookback_hours, len(df) - 1):
        features = np.concatenate([
            df["dir"].iloc[i - lookback_hours:i].values,  # last 1h candles
            df["dir_4h"].iloc[i - lookback_hours//4:i].values,  # last 4h candles
            df["dir_12h"].iloc[i - lookback_hours//12:i].values, # last 12h candles
            df["dir_24h"].iloc[i - lookback_hours//24:i].values  # last 24h candles
        ])
        X.append(features)
        y.append(df["dir"].iloc[i + 1])

    X, y = np.array(X), np.array(y)

    # === Split for training/testing ===
    split_idx = int(len(X) * 1) - 24*7*4
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # === Normalize and fit ===
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #est:100 max_depth:16 min_split:2 min_leaf:1
    model = RandomForestClassifier(n_estimators=200,
                                   max_depth=8,
                                   min_samples_split=4,
                                   min_samples_leaf=1,
                                   random_state=42)
    model.fit(X_train, y_train)

    # === Predict and simulate ===
    probas = model.predict_proba(X_test)
    preds = (probas[:, 1] > 0.5).astype(int)
    conf = np.max(probas, axis=1)

    payout = 1.9607
    balance = 0
    bet = 100
    wins = 0
    trades = 0

    konf = 0.54

    n_hours = 0
    correct = 0
    not_correct = 0
    not_correct_row = 0
    max_not_correct_row = 0
    for p, real, c in zip(preds, y_test, conf):
        n_hours += 1
        if c < konf:
            continue  # skip low-confidence trades

        trades += 1
        if p == real:
            correct += 1
            wins += 1
            balance += bet * (payout - 1)
            if max_not_correct_row < not_correct_row:
                max_not_correct_row = not_correct_row
            not_correct_row = 0
            print(f'[{trades}] prediction: {p} | real: {real} | balance: {balance}')
        else:
            not_correct += 1
            not_correct_row +=1
            balance -= bet  # lose only the fixed bet amount
            print(f'[{trades}] prediction: {p} | real: {real} | balance: {balance}')

    winrate = wins / trades if trades else 0
    roi = (balance) / trades if trades else 0

    print(f'max_not_correct_in_row: {max_not_correct_row}')

    print(f"\n=== Multi-TF Martingale Results ===")
    print(f"Trades: {trades}")
    print(f'hours: {n_hours}')
    print(f"Winrate: {winrate:.3f}")
    print(f'correct/not correct: {correct}/{not_correct}')
    print(f"Total Profit: {balance:.2f}")
    print(f"Average Profit/Trade (ROI): {roi:.3f}")
    print(f"Balance: {balance:.2f}")

    return model, scaler

# Example usage:
async def run():
    db = DbProvider()
    res = await db.select("candles", ["open_time", "open_price", "close_price", "dir"],
                          None, 0, "ASC")
    martingale_multi_tf(res)

if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
