import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import itertools
from datetime import datetime
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
        features = np.concatenate([
            df["dir"].iloc[i - lookback_hours:i].values,
            df["dir_4h"].iloc[i - lookback_hours // 4:i].values,
            df["dir_12h"].iloc[i - lookback_hours // 12:i].values,
            df["dir_24h"].iloc[i - lookback_hours // 24:i].values
        ])
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

# === Predict one-step-ahead ===
def backtest_hourly(df, model, lookback_hours=48, start_timestamp=None):
    df = df.copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="us", errors="coerce")
    df["dir"] = (df["close"] > df["open"]).astype(int)

    # выбираем октябрьские данные
    df_oct = df[df["open_time"] >= pd.to_datetime(start_timestamp, unit="us")]
    df_train = df[df["open_time"] < pd.to_datetime(start_timestamp, unit="us")]

    results = []
    payout = 1.96
    bet = 100

    print('backtest starts')

    for i in range(len(df_oct)):
        current_row = df_oct.iloc[i]
        ts = current_row["open_time"]

        # Берём историю до текущего момента
        hist = df[df["open_time"] < ts].tail(lookback_hours * 2)
        X_live, _, _ = build_features(hist, lookback_hours=lookback_hours)
        if len(X_live) == 0:
            continue
        X_pred = X_live[-1].reshape(1, -1)

        proba = model.predict_proba(X_pred)[0]
        pred = int(proba[1] > 0.5)
        conf = proba[1]

        print(f'predict {proba}')

        real = int(current_row["dir"])
        results.append({
            "timestamp": ts,
            "pred": pred,
            "real": real,
            "conf": conf
        })

    return pd.DataFrame(results)

# === Daily report ===
def daily_stats(pred_df):
    pred_df["date"] = pred_df["timestamp"].dt.date
    out_lines = []
    for day, group in pred_df.groupby("date"):
        day_num = day.day
        pred_seq = []
        real_seq = []
        stat_seq = []
        wins, total = 0, 0
        balance = 0
        payout = 1.96
        bet = 100

        for i, row in enumerate(group.itertuples(), start=1):
            p = "U" if row.pred == 1 else "D"
            r = "U" if row.real == 1 else "D"
            if row.pred == row.real:
                s = "+"
                wins += 1
                balance += bet * (payout - 1)
            else:
                s = "-"
                balance -= bet
            total += 1
            pred_seq.append(f"{i:02d}.{p}")
            real_seq.append(f"{i:02d}.{r}")
            stat_seq.append(f"{i:02d}.{s}")

        winrate = wins / total * 100 if total > 0 else 0
        out_lines.append(f"\nOCT: {day_num}")
        out_lines.append("PRED: " + " ".join(pred_seq))
        out_lines.append("RESU: " + " ".join(real_seq))
        out_lines.append("STATS: " + " ".join(stat_seq))
        out_lines.append(f"Winrate: {winrate:.1f}%  Balance: {balance:+.0f}")
    return "\n".join(out_lines)

# === Main async run ===
async def run():
    db = DbProvider()
    res = await db.select("candles", ["open_time", "open_price", "close_price", "dir"],
                          None, 0, "ASC")
    df = pd.DataFrame(res, columns=["open_time", "open", "close", "dir"]).sort_values("open_time").reset_index(drop=True)

    # Разделение по времени (до 1 октября тренировка)
    cutoff_ts = 1759276800000000  # timestamp 2024-09-30 00:00:00 UTC в микросекундах
    df_train = df[df["open_time"] < cutoff_ts]
    df_test = df[df["open_time"] >= cutoff_ts]

    print(f'train len: {len(df_train)}')

    # Обучаем модель на данных до октября
    X_train, y_train, _ = build_features(df_train, lookback_hours=48)
    model = train_rf(X_train, y_train)

    print(f'model trained')

    # Прогоняем бэктест по октябрю
    pred_df = backtest_hourly(df, model, lookback_hours=48, start_timestamp=cutoff_ts)
    report = daily_stats(pred_df)

    print(report)


if __name__ == "__main__":
    import asyncio
    asyncio.run(run())