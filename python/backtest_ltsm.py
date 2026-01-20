#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from db import DbProvider
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# ML
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler

# Plotly
import plotly.graph_objects as go

# ============================
# НАСТРОЙКИ
# ============================

TRAIN_START_DATE = "2021-01-01"
TRAIN_END_DATE   = "2023-12-30"

TEST_START_DATE  = "2024-01-01"
TEST_END_DATE    = "2025-12-31"

SEQUENCE_LENGTH  = 24
THRESHOLD_SIGNAL = 0.55

TABLE_NAME = "c_15m"
FUTURE_HORIZON = 4  # 4 * 15m = 1 час

# ============================

db = DbProvider()
app = FastAPI()

# ============================
# UTILS
# ============================

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def fmt_ts(ts):
    return datetime.utcfromtimestamp(ts / 1_000_000).strftime("%Y-%m-%d %H:%M")

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

    # --- приведение к float ---
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    return df

# ============================
# FEATURES
# ============================

def add_features(df):
    log("Feature engineering")

    df = df.copy()

    # SMA
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_50'] = df['close'].rolling(50).mean()

    # distance to SMA
    df['dist_sma_10'] = df['close'] - df['sma_10']
    df['dist_sma_50'] = df['close'] - df['sma_50']

    # direction placeholder (будет новый target)
    df['direction'] = 0

    df.fillna(0, inplace=True)
    return df

# ============================
# HOURLY FEATURES
# ============================

def compute_hourly_features(df):
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['open_time'], unit='us')
    df.set_index('datetime', inplace=True)

    # resample 1H
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    df_hour = df.resample('1H').apply(ohlc_dict)
    df_hour = df_hour.fillna(method='ffill')

    # map hourly features back to 15m candles
    df['hour_open']   = df_hour['open'].reindex(df.index, method='ffill').values
    df['hour_high']   = df_hour['high'].reindex(df.index, method='ffill').values
    df['hour_low']    = df_hour['low'].reindex(df.index, method='ffill').values
    df['hour_close']  = df_hour['close'].reindex(df.index, method='ffill').values
    df['hour_volume'] = df_hour['volume'].reindex(df.index, method='ffill').values

    df.reset_index(drop=True, inplace=True)
    return df

# ============================
# NEW TARGET
# ============================

def add_future_target(df, horizon=4):
    df = df.copy()
    df['future_close'] = df['close'].shift(-horizon)
    df['direction'] = (df['future_close'] > df['close']).astype(np.int8)
    df.dropna(inplace=True)
    return df

# ============================
# FAST SEQUENCES
# ============================

def make_sequences_fast(df, feature_cols, seq_len):
    X_raw = df[feature_cols].values
    y_raw = df['direction'].values
    times = df['open_time'].values

    X = np.lib.stride_tricks.sliding_window_view(
        X_raw, window_shape=(seq_len, X_raw.shape[1])
    )[:-1, 0]

    y = y_raw[seq_len:]
    times = times[seq_len:]

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
# PIPELINE
# ============================

async def train_and_predict():
    log("PIPELINE START")

    global_start = min(TRAIN_START_DATE, TEST_START_DATE)
    global_end   = max(TRAIN_END_DATE,   TEST_END_DATE)

    df = await load_candles(TABLE_NAME, global_start, global_end)
    df = add_features(df)
    df = compute_hourly_features(df)
    df = add_future_target(df, FUTURE_HORIZON)

    feature_cols = [
        'open','high','low','close','volume',
        'sma_10','sma_50','dist_sma_10','dist_sma_50',
        'hour_open','hour_high','hour_low','hour_close','hour_volume'
    ]

    df_train = filter_df_by_dates(df, TRAIN_START_DATE, TRAIN_END_DATE)
    df_test  = filter_df_by_dates(df, TEST_START_DATE,  TEST_END_DATE)

    log(f"Train rows={len(df_train)} Test rows={len(df_test)}")

    X_train, y_train, _ = make_sequences_fast(df_train, feature_cols, SEQUENCE_LENGTH)
    X_test, y_test, times_test = make_sequences_fast(df_test, feature_cols, SEQUENCE_LENGTH)

    log(f"Train seq={len(X_train)} Test seq={len(X_test)}")

    # scale
    log("Scaling")
    scaler = StandardScaler()
    f = X_train.shape[-1]
    X_train = scaler.fit_transform(X_train.reshape(-1,f)).reshape(X_train.shape)
    X_test  = scaler.transform(X_test.reshape(-1,f)).reshape(X_test.shape)

    # train
    model = build_lstm((SEQUENCE_LENGTH, len(feature_cols)))
    log("Training")
    model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)

    # backtest
    log("BACKTEST START")
    probs = model.predict(X_test, verbose=0).ravel()
    correct = 0
    wrong = 0
    preds = []

    for i in range(len(probs)):
        prob = probs[i]
        ts = times_test[i]
        real = y_test[i]

        if prob > THRESHOLD_SIGNAL:
            pred = 1
        elif prob < (1 - THRESHOLD_SIGNAL):
            pred = 0
        else:
            preds.append(-1)
            continue

        if pred == real:
            correct += 1
            mark = "✔"
        else:
            wrong += 1
            mark = "✘"

        preds.append(pred)
        log(
            f"{fmt_ts(ts)} | prob={prob:.2f} | pred={'UP' if pred else 'DOWN'} | "
            f"real={'UP' if real else 'DOWN'} | ✔={correct} ✘={wrong} {mark}"
        )

    log("BACKTEST END")
    return df_test.iloc[SEQUENCE_LENGTH:], preds, probs, y_test

# ============================
# CHART
# ============================

def render_chart(df, preds, probs, y):
    fig = go.Figure()
    fig.add_candlestick(
        x=pd.to_datetime(df.open_time, unit="us"),
        open=df.open,
        high=df.high,
        low=df.low,
        close=df.close,
        name="Candles"
    )

    mask = np.array(preds) != -1
    fig.add_scatter(
        x=pd.to_datetime(df.open_time[mask], unit="us"),
        y=df.close[mask],
        mode="markers",
        marker=dict(
            size=9,
            color=np.where(np.array(preds)[mask]==1,"green","red")
        ),
        hovertemplate=
        "Time: %{x|%Y-%m-%d %H:%M}<br>"
        "Close: %{y}<br>"
        "Prob: %{customdata[0]:.2f}<br>"
        "Pred: %{customdata[1]}<br>"
        "Real: %{customdata[2]}",
        customdata=np.column_stack([
            probs[mask],
            np.where(np.array(preds)[mask]==1,"UP","DOWN"),
            np.where(y[mask]==1,"UP","DOWN")
        ]),
        name="Predictions"
    )

    fig.update_layout(title="LSTM Backtest", xaxis_rangeslider_visible=False)
    fig.show()

# ============================
# FASTAPI
# ============================

@app.get("/", response_class=HTMLResponse)
async def index():
    df_test, preds, probs, y = await train_and_predict()

    preds_arr = np.array(preds)
    mask = preds_arr != -1
    acc = np.mean(preds_arr[mask] == y[mask])

    render_chart(df_test, preds, probs, y)

    return HTMLResponse(f"""
        <h2>Backtest finished</h2>
        <p>Accuracy: {acc:.3f}</p>
        <p>Samples: {len(y)}</p>
    """)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
