import time

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timezone, timedelta
from db import DbProvider
import pytz

N_ESTIMATORS = 200
MAX_DEPTH = 11
SPLIT = 2
LEAFS = 1
RANDOM_STATE = 42

KONF = 0.54
LOOKBACK_HOURS = 48

train_start = datetime(2022, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
#train_start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
train_end = datetime(2025, 8, 1, 0, 0, 0, tzinfo=timezone.utc)

# бэктест
global start_date
start_date = datetime(2025, 8, 1, tzinfo=pytz.utc)

end_date = datetime(2025, 10, 25, tzinfo=pytz.utc)


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

    # Convert relevant columns to NumPy arrays for faster slicing
    df_4h = aggregate_timeframe(df, 4)
    df_12h = aggregate_timeframe(df, 12)
    df_24h = aggregate_timeframe(df, 24)
    df["dir_4h"] = align_dirs(len(df), df_4h)
    df["dir_12h"] = align_dirs(len(df), df_12h)
    df["dir_24h"] = align_dirs(len(df), df_24h)

    dir_1h = df["dir"].to_numpy()
    dir_4h = df["dir_4h"].to_numpy()
    dir_12h = df["dir_12h"].to_numpy()
    dir_24h = df["dir_24h"].to_numpy()

    times_arr = df["open_time"].to_numpy()

    # Precompute lengths
    n = len(df)
    L1, L4, L12, L24 = lookback_hours, lookback_hours // 4, lookback_hours // 12, lookback_hours // 24

    # Preallocate lists for features, target and times
    X = []
    y = []
    times = []

    # Vectorized slicing for feature extraction
    for i in range(L1, n - 1):
        # Create feature vector using slicing
        features = np.concatenate([
            dir_1h[i - L1:i],  # dir_1h lookback
            dir_4h[i - L4:i],  # dir_4h lookback
            dir_12h[i - L12:i],  # dir_12h lookback
            dir_24h[i - L24:i]  # dir_24h lookback
        ])

        # Append to lists
        X.append(features)
        y.append(dir_1h[i + 1])  # Target for the next time step
        times.append(times_arr[i + 1])

    # Convert X, y, times to NumPy arrays after loop
    X = np.array(X)
    y = np.array(y)
    times = np.array(times)


    return np.array(X), np.array(y), np.array(times)


# === Train RandomForest ===
def train_rf(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=SPLIT,
        min_samples_leaf=LEAFS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# === Predict next hour candle ===
def predict_next_candle(model, df, lookback_hours=48, last_known_ts=None):

    #tbf = time.time()
    df_last = df[df["open_time"] < last_known_ts]
    #print(f'{time.time() - tbf} dataframe by time sort')

    #tb = time.time()
    X_live, _, _ = build_features(df_last, lookback_hours=lookback_hours)
    if len(X_live) == 0:
        print("Not enough data to predict.")
        return None
    #print(f'{time.time()-tb} features build')

    X_pred = X_live[-1].reshape(1, -1)
    proba = model.predict_proba(X_pred)[0]
    pred = int(proba[1] > 0.5)
    conf = proba[1]

    prediction = "U" if pred == 1 else "D"
    return prediction, conf


# === Main async simulation for October 1st to 25th ===
async def run():

    global start_date

    db = DbProvider()
    res = await db.select("candles", ["open_time", "open_price", "close_price", "dir"],
                          None, 0, "ASC")
    df = pd.DataFrame(res, columns=["open_time", "open", "close", "dir"]).reset_index(drop=True)

    # === Train the model up to the start of October 1 ===
    df_train = df[df["open_time"] < int(train_end.timestamp() * 1_000_000)]
    df_train = df_train[df_train["open_time"] > int(train_start.timestamp() * 1_000_000)]
    X_train, y_train, _ = build_features(df_train, lookback_hours=LOOKBACK_HOURS)
    model = train_rf(X_train, y_train)


    # Инициализация общего баланса и статистики
    overall_wins = 0
    overall_losses = 0
    overall_balance = 0.0

    # Перебор месяцев с июня 2025 года до текущего месяца
    while start_date <= end_date:
        # Начало и конец месяца
        start_time = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = (start_date + timedelta(days=32)).replace(day=1).replace(hour=0, minute=0, second=0, microsecond=0)

        # Убедимся, что мы не превысили текущий месяц
        if end_time > end_date:
            end_time = end_date

        # Преобразуем время в таймстампы (в микро-секундах)
        start_ts = int(start_time.timestamp() * 1_000_000)
        end_ts = int(end_time.timestamp() * 1_000_000)

        # Фильтруем данные для текущего месяца
        df_month = df[(df["open_time"] >= start_ts) & (df["open_time"] < end_ts)]

        # === Параметры для симуляции ставок ===

        bet_size = 100
        win_coef = 1.96

        # Перебор дней в текущем месяце
        for day in range((end_time - start_time).days):

            wins = 0
            losses = 0
            balance = 0.0

            current_day = start_time + timedelta(days=day)

            # Фильтруем данные для текущего дня
            df_day = df_month[(df_month["open_time"] >= start_ts + day * 24 * 60 * 60 * 1_000_000) &
                              (df_month["open_time"] < start_ts + (day + 1) * 24 * 60 * 60 * 1_000_000)]

            # === Симуляция ставок для каждого дня ===
            for hour in range(24):
                current_ts = df_day["open_time"].iloc[hour]

                # Прогнозирование следующей свечи
                prediction, confidence = predict_next_candle(model, df, lookback_hours=LOOKBACK_HOURS, last_known_ts=current_ts)

                pred_dir = 1 if prediction == "U" else 0

                # Отображаем информацию о прогнозе
                ts_seconds = current_ts / 1_000_000
                candle_time = datetime.fromtimestamp(ts_seconds, timezone.utc)
                print(f"\n[{hour + 1}/24] Candle at {candle_time} — Predicted: {prediction} (Conf: {confidence:.2f})")

                # Получаем реальное значение направления
                real_value = 1 if df_day["dir"].iloc[hour] == 'U' else 0

                # Симулируем ставку
                if confidence < KONF:
                    total_bets2 = overall_wins + overall_losses
                    overall_winrate2 = (overall_wins / total_bets2) * 100 if total_bets2 > 0 else 0
                    print(
                        f'Skip due to low confidence: day balance {balance}$ {overall_winrate2:.2f}% overall balance: {overall_balance}')
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

                total_bets2 = overall_wins + overall_losses
                overall_winrate2 = (overall_wins / total_bets2) * 100 if total_bets2 > 0 else 0
                print(
                    f"→ Result: {result} | Current Balance: ${balance:.2f} pred: {pred_dir} real: {real_value} {overall_winrate2:.2f}%")

            # === Ежедневный отчет ===
            daily_total_bets = wins + losses
            daily_winrate = (wins / daily_total_bets) * 100 if daily_total_bets > 0 else 0

            print("\n=== Daily Summary ===")
            print(f"Day: {current_day.date()}")
            print(f"Total Bets: {daily_total_bets}")
            print(f"Wins: {wins} | Losses: {losses}")
            print(f"Winrate: {daily_winrate:.2f}%")
            print(f"Final Balance for the Day: ${balance:.2f}")

            # Обновляем общую статистику
            overall_wins += wins
            overall_losses += losses
            overall_balance += balance

        # Переходим к следующему месяцу
        start_date = end_time

    # === Общий итог ===
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
