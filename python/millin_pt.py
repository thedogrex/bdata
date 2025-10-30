import asyncio
import pandas as pd
import numpy as np
from collections import Counter
from tensorflow import keras
from tensorflow.keras import layers
from db import DbProvider
from datetime import datetime


# === 1. Загрузка данных из базы ===
async def load_candles(start_date=None, end_date=None):
    db = DbProvider()

    res = await db.select('c_15m', ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                    'quota_volume', 'trades', 'taker_base_volume', 'taker_quota_volume'], None, 0, "ASC")

    df = pd.DataFrame(res, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                    'quota_volume', 'trades', 'taker_base_volume', 'taker_quota_volume']).reset_index(
        drop=True)

    start_dt = pd.to_datetime(start_date) if start_date else None
    end_dt = pd.to_datetime(end_date)  if end_date else None

    # Convert timestamps from microseconds to datetime
    df['open_time'] = pd.to_datetime(df['open_time'], unit='us')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='us')

    if start_dt:
        df = df[(df['open_time'] >= start_dt)].reset_index(drop=True)

    if end_dt:
        df = df[(df['open_time'] <= end_dt)].reset_index(drop=True)

    return df


# === 2. Преобразуем свечи в направление (U/D) ===
def add_direction(df):
    df['dir'] = np.where(df['close'] > df['open'], 'U', 'D')
    return df


# === 3. Находим все паттерны и вероятности UP ===
def build_pattern_stats(df, min_len=4, max_len=14):
    dir_seq = df['dir'].tolist()
    stats = {L: Counter() for L in range(min_len, max_len + 1)}

    for L in range(min_len, max_len + 1):
        for i in range(len(dir_seq) - L - 1):
            pattern = ''.join(dir_seq[i:i + L])
            next_dir = dir_seq[i + L]
            stats[L][(pattern, next_dir)] += 1

    # Считаем вероятность UP после каждого паттерна
    prob_up = {}
    for L, counter in stats.items():
        pattern_stats = {}
        for (pattern, next_dir), count in counter.items():
            up_count = counter.get((pattern, 'U'), 0)
            down_count = counter.get((pattern, 'D'), 0)
            total = up_count + down_count
            if total > 0:
                pattern_stats[pattern] = {
                    'prob_up': up_count / total,
                    'freq': total
                }
        prob_up[L] = pattern_stats
    return prob_up


# === 4. Генерация фичей для обучения ===
def build_features(df, prob_up):
    dir_seq = df['dir'].tolist()
    feature_rows = []
    target_rows = []

    for i in range(14, len(dir_seq) - 1):
        features = {}
        for L in range(4, 15):
            pattern = ''.join(dir_seq[i - L:i])
            info = prob_up[L].get(pattern, {'prob_up': 0.5, 'freq': 0})
            features[f'prob_up_{L}'] = info['prob_up']
            features[f'freq_{L}'] = info['freq']
        feature_rows.append(features)
        target_rows.append(1 if dir_seq[i + 1] == 'U' else 0)

    df_features = pd.DataFrame(feature_rows)
    df_target = pd.Series(target_rows)
    return df_features, df_target


# === 5. Обучение нейросети ===
def train_model(X, y):
    model = keras.Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=2, batch_size=128, validation_split=0.2, verbose=1)
    return model


# === 6. Предсказание направления следующей свечи с использованием порога уверенности ===
def predict_next(df, prob_up, model, confidence_threshold=0.6):
    dir_seq = df['dir'].tolist()
    features = {}
    for L in range(4, 15):
        pattern = ''.join(dir_seq[-L:])
        info = prob_up[L].get(pattern, {'prob_up': 0.5, 'freq': 0})
        features[f'prob_up_{L}'] = info['prob_up']
        features[f'freq_{L}'] = info['freq']

    X = pd.DataFrame([features])
    pred = model.predict(X)
    confidence = pred[0][0]
    predicted_direction = "UP" if confidence > 0.5 else "DOWN"

    print(confidence)
    # Only predict if confidence is above the threshold
    if confidence < confidence_threshold:
        predicted_direction = "NO_TRADE"
        confidence = 0.5  # Neutral confidence if no trade

    return predicted_direction, confidence


# === 7. Backtesting Function ===
async def backtest(start_train, end_train, start_backtest, end_backtest, confidence_threshold=0.6):
    # 1. Load the training and backtest data
    df_train = await load_candles(start_train, end_train)
    df_train = add_direction(df_train)

    df_backtest = await load_candles(start_backtest, end_backtest)
    df_backtest = add_direction(df_backtest)

    # 2. Build pattern statistics using the training data
    print("Building pattern statistics...")
    prob_up = build_pattern_stats(df_train)

    # 3. Build features for training
    print("Building features for training...")
    df_features, df_target = build_features(df_train, prob_up)

    # 4. Train the model
    print("Training model...")
    model = train_model(df_features.values, df_target.values)

    # 5. Start the backtest simulation
    print("Backtesting...")
    total_days = len(df_backtest) - 1
    correct_predictions = 0
    win_count = 0
    lose_count = 0

    print(f'total days: {total_days}')

    for i in range(14, len(df_backtest) - 1):
        # Generate features for backtest day
        predicted_direction, confidence = predict_next(df_backtest, prob_up, model, confidence_threshold)
        actual_direction = df_backtest['dir'][i + 1]

        # Track performance for the current day
        win = (predicted_direction == 'UP' and actual_direction == 'U') or (
                    predicted_direction == 'DOWN' and actual_direction == 'D')
        if win and predicted_direction != "NO_TRADE":
            correct_predictions += 1
            win_count += 1
            print(f"Day {i} - Predicted: {predicted_direction} (Confidence: {confidence:.2f}), Actual: {actual_direction} - Win")
        elif predicted_direction != "NO_TRADE":
            lose_count += 1
            print(f"Day {i} - Predicted: {predicted_direction} (Confidence: {confidence:.2f}), Actual: {actual_direction} - Lose")

        # Print dynamic win rate and counts
        win_rate = correct_predictions / (i - 13) * 100  # i-13 because of training window
        print(f"Current Win Rate: {win_rate:.2f}% | Wins: {win_count} | Losses: {lose_count}")

    # 6. Final Stats
    win_rate = correct_predictions / total_days * 100
    print("\n=== Final Backtest Stats ===")
    print(f"Total days: {total_days}")
    print(f"Total wins: {win_count}")
    print(f"Total losses: {lose_count}")
    print(f"Win rate: {win_rate:.2f}%")


# === 8. Main function for backtesting ===
async def main():
    # Define the training and backtest periods (these should be strings in the format 'YYYY-MM-DD HH:MM:SS')
    start_train = '2020-07-01 00:00:00'
    end_train = '2025-07-30 23:59:59'
    start_backtest = '2025-08-01 00:00:00'
    end_backtest = '2025-08-15 23:59:59'

    # Run the backtest
    await backtest(start_train, end_train, start_backtest, end_backtest, confidence_threshold=0.6)


# === Запуск ===
if __name__ == "__main__":
    asyncio.run(main())
