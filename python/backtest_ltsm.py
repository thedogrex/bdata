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
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# ============================
# НАСТРОЙКИ
# ============================

TRAIN_START_DATE = "2021-01-01"
TRAIN_END_DATE   = "2025-06-30"

TEST_START_DATE  = "2025-07-01"
TEST_END_DATE    = "2025-12-31"

SEQUENCE_LENGTH  = 24
THRESHOLD_SIGNAL = 0.56

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
    return pd.to_datetime(ts // 1_000_000, unit='s').strftime("%Y-%m-%d %H:%M")

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

    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)

    df['future_direction'] = (df['close'].shift(-1) > df['open'].shift(-1)).astype(np.int8)
    df.dropna(inplace=True)

    log(f"DataFrame prepared: {len(df)} rows")
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

    df['dist_sma10'] = (df['close'] - df['sma_10']) / df['sma_10']
    df['dist_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
    df['volatility_1h'] = df['close'].rolling(4).std()
    df['momentum_1h'] = df['close'] - df['close'].shift(4)

    lag_features = ['open','high','low','close','volume']
    for col in lag_features:
        for lag in range(1, SEQUENCE_LENGTH + 1):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    df.fillna(0, inplace=True)

    log("Feature engineering completed")
    return df

# ============================
# WALK-FORWARD XGBOOST
# ============================

def walk_forward_online_xgb(model, X_test, y_test, times_test, threshold):
    preds = []
    probs = []
    correct = []

    correct_count = 0
    incorrect_count = 0

    log(f"Walk-forward start: {len(X_test)} samples")

    for i in range(len(X_test)):
        x_input = X_test[i].reshape(1, -1)
        prob = model.predict_proba(x_input)[0, 1]

        if prob > threshold:
            pred = 1
        elif prob < (1 - threshold):
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

        log(
            f"{fmt_ts(times_test[i])} | "
            f"prob={prob:.3f} | pred={pred} | real={y_test[i]} | "
            f"correct={correct_count} | incorrect={incorrect_count}"
        )

    log("Walk-forward completed")
    return np.array(preds), np.array(correct), np.array(probs)

# ============================
# PIPELINE
# ============================

async def train_and_predict():
    log("PIPELINE START")

    df = await load_candles(
        TABLE_NAME,
        min(TRAIN_START_DATE, TEST_START_DATE),
        max(TRAIN_END_DATE, TEST_END_DATE)
    )

    df = add_features(df)

    feature_cols = [c for c in df.columns if c not in ['open_time','future_direction']]

    df_train = filter_df_by_dates(df, TRAIN_START_DATE, TRAIN_END_DATE)
    df_test  = filter_df_by_dates(df, TEST_START_DATE, TEST_END_DATE)

    X_train = df_train[feature_cols].to_numpy()
    y_train = df_train['future_direction'].to_numpy()

    X_test  = df_test[feature_cols].to_numpy()
    y_test  = df_test['future_direction'].to_numpy()
    times_test = df_test['open_time'].to_numpy()

    log(f"Train samples={len(X_train)}, Test samples={len(X_test)}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    log("Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    log("Training completed")

    preds, correct, probs = walk_forward_online_xgb(
        model,
        X_test,
        y_test,
        times_test,
        THRESHOLD_SIGNAL
    )

    mask = preds != -1
    acc = np.mean(preds[mask] == y_test[mask])
    log(f"Accuracy (signals only): {acc:.4f}")

    return df_test, preds, correct, y_test, probs

# ============================
# PLOT
# ============================

def render_chart(df, preds, probs, y):
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(
        x=pd.to_datetime(df['open_time'], unit='us'),
        open=df['open'], high=df['high'], low=df['low'], close=df['close']
    ))
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(df['open_time'], unit='us'),
        y=df['close'],
        mode='markers',
        marker=dict(color=['green' if p==1 else 'red' for p in preds], size=7),
        text=[f"prob={p:.2f}, real={r}" for p,r in zip(probs,y)],
        hoverinfo='text'
    ))
    fig.show()

# ============================
# FASTAPI
# ============================

@app.get("/", response_class=HTMLResponse)
async def index():
    df_test, preds, correct, y_test, probs = await train_and_predict()
    render_chart(df_test, preds, probs, y_test)
    return HTMLResponse("<h2>Backtest finished</h2>")

# ============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
