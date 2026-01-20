#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import itertools
import numpy as np
import pandas as pd
from db import DbProvider
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# ============================
# НАСТРОЙКИ
# ============================

TRAIN_START_DATE = "2021-01-01"
TRAIN_END_DATE   = "2025-06-30"

TEST_START_DATE  = "2025-07-01"
TEST_END_DATE    = "2025-12-31"

TABLE_NAME = "c_15m"

SEQUENCE_LENGTHS = [12, 24, 48]
THRESHOLDS = [0.52, 0.55, 0.58, 0.6]

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
    query = f"""
        SELECT open_time, open, high, low, close, volume
        FROM {table}
        WHERE open_time BETWEEN
            {date_to_ts(start_date, False)} AND {date_to_ts(end_date, True)}
        ORDER BY open_time
    """
    rows = await db.fetchall(query)
    df = pd.DataFrame(rows, columns=['open_time','open','high','low','close','volume'])

    for c in ['open','high','low','close','volume']:
        df[c] = df[c].astype(float)

    df['future_direction'] = (df['close'].shift(-1) > df['open'].shift(-1)).astype(np.int8)
    df.dropna(inplace=True)

    return df

# ============================
# FEATURES
# ============================

def add_features(df, sequence_length):
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

    for col in ['open','high','low','close','volume']:
        for lag in range(1, sequence_length + 1):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    df.fillna(0, inplace=True)
    return df

# ============================
# WALK FORWARD
# ============================

def walk_forward(model, X_test, y_test, threshold):
    preds = []

    for i in range(len(X_test)):
        prob = model.predict_proba(X_test[i].reshape(1, -1))[0, 1]

        if prob > threshold:
            preds.append(1)
        elif prob < 1 - threshold:
            preds.append(0)
        else:
            preds.append(-1)

    preds = np.array(preds)
    mask = preds != -1

    if mask.sum() == 0:
        return 0.0, 0

    acc = np.mean(preds[mask] == y_test[mask])
    return acc, mask.sum()

# ============================
# GRID SEARCH
# ============================

async def grid_search():
    df_raw = await load_candles(
        TABLE_NAME,
        min(TRAIN_START_DATE, TEST_START_DATE),
        max(TRAIN_END_DATE, TEST_END_DATE)
    )

    xgb_grid = {
        "n_estimators": [200, 400],
        "max_depth": [3, 5],
        "learning_rate": [0.03, 0.06],
        "subsample": [0.7, 0.85],
        "colsample_bytree": [0.7, 0.85],
    }

    xgb_keys = list(xgb_grid.keys())
    xgb_combos = list(itertools.product(*xgb_grid.values()))

    results = []

    total = len(SEQUENCE_LENGTHS) * len(THRESHOLDS) * len(xgb_combos)
    step = 0

    for seq_len in SEQUENCE_LENGTHS:
        log(f"FEATURES: sequence_length={seq_len}")
        df = add_features(df_raw, seq_len)

        feature_cols = [c for c in df.columns if c not in ['open_time','future_direction']]

        train = filter_df_by_dates(df, TRAIN_START_DATE, TRAIN_END_DATE)
        test  = filter_df_by_dates(df, TEST_START_DATE, TEST_END_DATE)

        X_train = train[feature_cols].to_numpy()
        y_train = train['future_direction'].to_numpy()
        X_test  = test[feature_cols].to_numpy()
        y_test  = test['future_direction'].to_numpy()

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        for params_vals in xgb_combos:
            params = dict(zip(xgb_keys, params_vals))

            model = xgb.XGBClassifier(
                **params,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train, y_train)

            for threshold in THRESHOLDS:
                step += 1
                acc, signals = walk_forward(model, X_test, y_test, threshold)

                result = {
                    "accuracy": acc,
                    "signals": signals,
                    "sequence_length": seq_len,
                    "threshold": threshold,
                    **params
                }

                results.append(result)

                log(f"[{step}/{total}] acc={acc:.4f} signals={signals} {result}")

    results = sorted(results, key=lambda x: (x["accuracy"], x["signals"]), reverse=True)

    log("===== 🥇 BEST RESULT =====")
    log(results[0])

    log("===== 🏆 TOP 5 =====")
    for r in results[:5]:
        log(r)

    return results

# ============================
# FASTAPI
# ============================

@app.get("/", response_class=HTMLResponse)
async def index():
    await grid_search()
    return HTMLResponse("<h2>Grid search finished. Check console.</h2>")

# ============================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
