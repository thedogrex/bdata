import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import itertools
from db import DbProvider

# === Aggregate higher timeframes ===
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

# === Align higher TF directions back to 1h ===
def align_dirs(base_len, higher_df):
    return np.interp(
        np.arange(base_len),
        np.linspace(0, base_len - 1, len(higher_df)),
        higher_df["dir"]
    ).round().astype(int)

# === Build features for prediction ===
def build_features(df, lookback_hours=24):
    # Base 1h direction
    df["dir"] = (df["close"] > df["open"]).astype(int)

    # Aggregate higher TFs
    df_4h = aggregate_timeframe(df, 4)
    df_12h = aggregate_timeframe(df, 12)
    df_24h = aggregate_timeframe(df, 24)

    df["dir_4h"] = align_dirs(len(df), df_4h)
    df["dir_12h"] = align_dirs(len(df), df_12h)
    df["dir_24h"] = align_dirs(len(df), df_24h)

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

    return np.array(X), np.array(y)

# === Grid search RandomForest for best parameters ===
def grid_search_rf(X_train, y_train, X_test, y_test):
    n_estimators_list = [100, 200, 300]
    max_depth_list = [8, 12, 16]
    min_samples_split_list = [2, 4]
    min_samples_leaf_list = [1, 2]

    best_winrate = 0
    best_balance = 0
    best_params = None

    for n_estimators, max_depth, min_split, min_leaf in itertools.product(
        n_estimators_list, max_depth_list, min_samples_split_list, min_samples_leaf_list
    ):
        print(f'CHECK HYPER: est:{n_estimators} max_depth:{max_depth} min_split:{min_split} min_leaf:{min_leaf}')

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_split,
            min_samples_leaf=min_leaf,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        probas = model.predict_proba(X_test)
        preds = (probas[:, 1] > 0.5).astype(int)
        conf = np.max(probas, axis=1)

        for konff in [0.51,0.52,0.53,0.54,0.55]:
            # Simulate simple fixed-bet strategy
            payout = 1.923
            balance = 1000
            bet = 20
            wins = 0
            trades = 0

            total_hours = len(preds)

            for p, real, c in zip(preds, y_test, conf):
                if c < konff:  # skip low-confidence
                    continue
                trades += 1
                if p == real:
                    wins += 1
                    balance += bet * (payout - 1)
                else:
                    balance -= bet

            winrate = wins / trades if trades else 0

            print(f'Session ends: trades: {trades}/{total_hours} {trades/total_hours} winrate:{winrate} balance: {balance} konfidence: {konff}')

            if winrate > best_winrate:
                best_winrate = winrate
                best_balance = balance
                best_params = (n_estimators, max_depth, min_split, min_leaf, konff)

    print(f"\nBest RF parameters:")
    print(f"n_estimators={best_params[0]}, max_depth={best_params[1]}, "
          f"min_samples_split={best_params[2]}, min_samples_leaf={best_params[3]}")
    print(f"Winrate: {best_winrate:.3f}, Final Balance: {best_balance:.2f}")
    return best_params

# === Main martingale simulation ===
def simulate_trading(X_train, y_train, X_test, y_test, rf_params):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=rf_params[0],
        max_depth=rf_params[1],
        min_samples_split=rf_params[2],
        min_samples_leaf=rf_params[3],
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    probas = model.predict_proba(X_test)
    preds = (probas[:, 1] > 0.5).astype(int)
    conf = np.max(probas, axis=1)

    payout = 1.923
    balance = 1000
    bet = 20
    wins = 0
    trades = 0

    for p, real, c in zip(preds, y_test, conf):
        if c < 0.54:  # skip low-confidence
            continue
        trades += 1
        if p == real:
            wins += 1
            balance += bet * (payout - 1)
        else:
            balance -= bet

    winrate = wins / trades if trades else 0
    print(f"\n=== Final Simulation ===")
    print(f"Trades: {trades}, Winrate: {winrate:.3f}, Balance: {balance:.2f}")
    return model, scaler

# === Run ===
async def run():
    db = DbProvider()
    res = await db.select("candles", ["open_time", "open_price", "close_price", "dir"],
                          None, 0, "ASC")
    df = pd.DataFrame(res, columns=["open_time", "open", "close", "dir"]).sort_values("open_time").reset_index(drop=True)

    # Build features
    X, y = build_features(df, lookback_hours=48)  # last 1 day; can increase

    # Split train/test (e.g., last 1000 samples for testing)
    split_idx = len(X) - 1000
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Grid search best RF parameters
    best_params = grid_search_rf(X_train, y_train, X_test, y_test)

    # Simulate trading with best parameters
    simulate_trading(X_train, y_train, X_test, y_test, best_params)

if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
