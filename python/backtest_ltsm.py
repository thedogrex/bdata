#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
from db import DbProvider
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler

# ============================
# НАСТРОЙКИ
# ============================

TRAIN_START_DATE = "2023-01-01"
TRAIN_END_DATE   = "2025-06-30"

TEST_START_DATE  = "2025-07-01"
TEST_END_DATE    = "2025-12-31"

SEQUENCE_LENGTH  = 24
THRESHOLD_SIGNAL = 0.55

TABLE_NAME = "c_15m"

# ============================

db = DbProvider()
app = FastAPI()

# ============================
# UTILS
# ============================

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def date_to_ts(date_str, end_of_day=False):
    t = "23:59:59" if end_of_day else "00:00:00"
    return int(pd.Timestamp(f"{date_str} {t}").timestamp() * 1_000_000)

def fmt_ts(ts):
    return pd.to_datetime(ts//1_000_000, unit='s').strftime("%Y-%m-%d %H:%M")

def filter_df_by_dates(df, start_date, end_date):
    return df[
        (df.open_time >= date_to_ts(start_date, False)) &
        (df.open_time <= date_to_ts(end_date, True))
    ].copy()

# ============================
# LOAD DATA
# ============================

async def load_candles(table, start_date, end_date):
    log(f"SQL load {start_date} → {end_date}")
    query = f"""
        SELECT open_time, open, high, low, close, volume
        FROM {table}
        WHERE open_time BETWEEN
            {date_to_ts(start_date, False)} AND {date_to_ts(end_date, True)}
        ORDER BY open_time
    """
    t0 = time.time()
    rows = await db.fetchall(query)
    log(f"SQL rows={len(rows)} loaded in {time.time()-t0:.2f}s")

    df = pd.DataFrame(rows, columns=['open_time','open','high','low','close','volume'])

    # Приведение Decimal → float
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)

    df['future_direction'] = (df['close'].shift(-1) > df['open'].shift(-1)).astype(np.int8)
    df.dropna(inplace=True)
    log(f"DataFrame prepared: {len(df)} rows after dropna")
    return df

# ============================
# FEATURES
# ============================

def add_features(df):
    log("Feature engineering started")
    df = df.copy()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['direction_last'] = (df['close'] > df['open']).astype(np.int8)
    df['run_up6']   = df['direction_last'].rolling(6).sum().eq(6).astype(np.int8)
    df['run_down6'] = df['direction_last'].rolling(6).sum().eq(0).astype(np.int8)

    # новые признаки
    df['dist_sma10'] = (df['close'] - df['sma_10']) / df['sma_10']
    df['dist_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
    df['volatility_1h'] = df['close'].rolling(4).std()
    df['momentum_1h'] = df['close'] - df['close'].shift(4)
    df.fillna(0, inplace=True)
    log("Feature engineering completed")
    log(f"Features: {list(df.columns)}")
    log(f"Data snapshot:\n{df.head(5)}")
    return df

# ============================
# MAKE SEQUENCES
# ============================

def make_sequences(df, feature_cols, seq_len):
    """
    Быстрая генерация последовательностей для LSTM.
    df           : DataFrame с исходными данными
    feature_cols : список колонок для признаков
    seq_len      : длина последовательности (например, 24)
    """
    data = df[feature_cols].to_numpy(dtype=np.float32)   # (n_rows, n_features)
    targets = df['future_direction'].to_numpy(dtype=np.int8)
    times = df['open_time'].to_numpy()

    n_samples = len(data) - seq_len
    if n_samples <= 0:
        return np.empty((0, seq_len, len(feature_cols))), np.empty(0), np.empty(0)

    # Создаем последовательности через сдвиги (без циклов на каждую строку)
    X = np.zeros((n_samples, seq_len, len(feature_cols)), dtype=np.float32)
    for i in range(seq_len):
        X[:, i, :] = data[i:i+n_samples, :]

    y = targets[seq_len:]
    times_seq = times[seq_len:]

    return X, y, times_seq

# ============================
# MODEL
# ============================

def build_lstm(input_shape):
    log("Building LSTM model")
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    log("LSTM model built")
    return model

# ============================
# WALK-FORWARD ONLINE
# ============================

def walk_forward_online(model, X_test, y_test, times_test, threshold):
    preds = []
    correct = []
    probs = []

    correct_count = 0
    incorrect_count = 0

    log(f"Walk-forward start: {len(X_test)} samples")
    for i in range(len(X_test)):
        x_input = X_test[i:i+1]
        prob = model.predict(x_input, verbose=0)[0,0]
        if prob > threshold:
            pred = 1
        elif prob < (1-threshold):
            pred = 0
        else:
            pred = -1

        preds.append(pred)
        probs.append(prob)

        if pred != -1:
            is_correct = int(pred == y_test[i])
            correct.append(is_correct)
            if is_correct:
                correct_count += 1
            else:
                incorrect_count += 1
        else:
            correct.append(np.nan)

        # Лог каждой свечи
        log(f"{fmt_ts(times_test[i])} | prob={prob:.3f} | pred={pred} | real={y_test[i]} | "
            f"correct_count={correct_count} | incorrect_count={incorrect_count}")

    log("Walk-forward completed")
    return np.array(preds), np.array(correct), np.array(probs)

# ============================
# PIPELINE
# ============================

async def train_and_predict():
    log("PIPELINE START")
    global_start = min(TRAIN_START_DATE, TEST_START_DATE)
    global_end   = max(TRAIN_END_DATE, TEST_END_DATE)

    df = await load_candles(TABLE_NAME, global_start, global_end)
    df = add_features(df)

    feature_cols = ['open','high','low','close','volume','sma_10','sma_50',
                    'run_up6','run_down6','dist_sma10','dist_sma50','volatility_1h','momentum_1h']

    df_train = filter_df_by_dates(df, TRAIN_START_DATE, TRAIN_END_DATE)
    df_test  = filter_df_by_dates(df, TEST_START_DATE, TEST_END_DATE)

    log("Creating training sequences...")
    X_train, y_train, _ = make_sequences(df_train, feature_cols, SEQUENCE_LENGTH)
    log("Creating test sequences...")
    X_test, y_test, times_test = make_sequences(df_test, feature_cols, SEQUENCE_LENGTH)

    log(f"Train samples={len(X_train)}, Test samples={len(X_test)}")

    log("Scaling features...")
    scaler = StandardScaler()
    f = X_train.shape[-1]
    X_train = scaler.fit_transform(X_train.reshape(-1,f)).reshape(X_train.shape)
    log("Training features scaled")
    X_test  = scaler.transform(X_test.reshape(-1,f)).reshape(X_test.shape)
    log("Test features scaled")

    model = build_lstm((SEQUENCE_LENGTH,len(feature_cols)))

    log("Training model...")
    for epoch in range(40):
        model.fit(X_train, y_train, epochs=1, batch_size=64, verbose=1)
        log(f"Epoch {epoch+1}/10 completed")
    log("Training completed")

    preds, correct, probs = walk_forward_online(model, X_test, y_test, times_test, THRESHOLD_SIGNAL)
    log("PIPELINE END")
    return df_test, preds, correct, y_test, probs, times_test

# ============================
# PLOT
# ============================

def render_chart(df, preds, probs, y):
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(
        x=pd.to_datetime(df['open_time'], unit='us'),
        open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='Candles'
    ))
    pred_x = pd.to_datetime(df['open_time'].iloc[SEQUENCE_LENGTH:], unit='us')
    fig.add_trace(go.Scatter(
        x=pred_x,
        y=df['close'].iloc[SEQUENCE_LENGTH:],
        mode='markers',
        marker=dict(color=['green' if p==1 else 'red' for p in preds], size=8),
        text=[f"prob={pr:.2f}, real={r}" for pr,r in zip(probs,y)],
        name='Predictions',
        hoverinfo='text'
    ))
    fig.update_layout(height=600, width=1000, title="Candlestick + Predictions")
    fig.show()

# ============================
# FASTAPI
# ============================

@app.get("/", response_class=HTMLResponse)
async def index():
    df_test, preds, correct_flags, y_test, probs, times_test = await train_and_predict()
    mask = preds!=-1
    acc = np.mean(preds[mask]==y_test[mask])
    render_chart(df_test, preds, probs, y_test)

    return HTMLResponse(f"""
        <h2>Backtest finished</h2>
        <p>Accuracy: {acc:.3f}</p>
        <p>Samples: {len(y_test)}</p>
    """)

# ============================
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
