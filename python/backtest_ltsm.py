#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
from db import DbProvider
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# ML
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler

# ============================
# НАСТРОЙКИ
# ============================

TRAIN_START_DATE = "2020-01-01"
TRAIN_END_DATE   = "2025-11-30"

TEST_START_DATE  = "2025-12-01"
TEST_END_DATE    = "2025-12-30"

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
    log(f"SQL rows={len(rows)} time={time.time()-t0:.2f}s")

    df = pd.DataFrame(rows, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume'
    ])

    df['direction'] = (df['close'].shift(-1) > df['open'].shift(-1)).astype(np.int8)
    df.dropna(inplace=True)

    return df

# ============================
# FEATURES
# ============================

def add_features(df):
    log("Feature engineering")

    df = df.copy()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_50'] = df['close'].rolling(50).mean()

    d = (df['close'] > df['open']).astype(np.int8)
    df['run_up6']   = d.rolling(6).sum().eq(6).astype(np.int8)
    df['run_down6'] = d.rolling(6).sum().eq(0).astype(np.int8)

    df.fillna(0, inplace=True)
    return df

# ============================
# FAST SEQUENCES
# ============================

def make_sequences_fast(df, feature_cols, seq_len):
    log(f"Sequences fast: rows={len(df)}")

    X_raw = df[feature_cols].values
    y_raw = df['direction'].values
    times = df['open_time'].values


    X = np.lib.stride_tricks.sliding_window_view(
        X_raw, window_shape=(seq_len, X_raw.shape[1]))[:-1, 0, :, :]

    y = y_raw[seq_len:]
    times = times[seq_len:]

    log(f"Sequences created: {len(X)}")
    return X, y, times

# ============================
# MODEL
# ============================

def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# ============================
# WALK FORWARD
# ============================

def walk_forward(model, X, y, threshold):
    log(f"Walk-forward samples={len(X)}")

    probs = model.predict(X, verbose=0).ravel()
    preds = np.full_like(probs, -1, dtype=np.int8)

    preds[probs > threshold] = 1
    preds[probs < (1 - threshold)] = 0

    correct = np.where(preds != -1, preds == y, np.nan)
    return preds, correct

# ============================
# PIPELINE
# ============================

async def train_and_predict():
    log("PIPELINE START")

    global_start = min(TRAIN_START_DATE, TEST_START_DATE)
    global_end   = max(TRAIN_END_DATE,   TEST_END_DATE)

    df = await load_candles(TABLE_NAME, global_start, global_end)
    df = add_features(df)

    feature_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'sma_10', 'sma_50', 'run_up6', 'run_down6'
    ]

    df_train = filter_df_by_dates(df, TRAIN_START_DATE, TRAIN_END_DATE)
    df_test  = filter_df_by_dates(df, TEST_START_DATE,  TEST_END_DATE)

    X_train, y_train, _ = make_sequences_fast(
        df_train, feature_cols, SEQUENCE_LENGTH
    )
    X_test, y_test, times_test = make_sequences_fast(
        df_test, feature_cols, SEQUENCE_LENGTH
    )

    # scale (fit only train)
    log("Scaling")
    scaler = StandardScaler()
    f = X_train.shape[-1]

    X_train = scaler.fit_transform(
        X_train.reshape(-1, f)
    ).reshape(X_train.shape)

    X_test = scaler.transform(
        X_test.reshape(-1, f)
    ).reshape(X_test.shape)

    model = build_lstm((SEQUENCE_LENGTH, len(feature_cols)))

    log("Training")
    model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)

    preds, correct = walk_forward(
        model, X_test, y_test, THRESHOLD_SIGNAL
    )

    log("PIPELINE END")
    return times_test, y_test, preds, correct

# ============================
# FASTAPI
# ============================

@app.get("/", response_class=HTMLResponse)
async def index():
    times, y, preds, correct = await train_and_predict()
    acc = np.nanmean(correct)

    return HTMLResponse(f"""
    <html>
    <body>
        <h2>Fast backtest</h2>
        <p>Accuracy: {acc:.3f}</p>
        <p>Samples: {len(y)}</p>
    </body>
    </html>
    """)

# ============================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
