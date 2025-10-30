import asyncio
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timezone
from db import DbProvider


EPOCH = 8

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# =========================
# Multi-Timeframe Feature Builder
# =========================
def add_multi_tf_features(df):
    """
    Add 4h, 12h, 24h directional averages.
    Assume df is 1h candles.
    """
    df = df.copy()
    df["dir"] = (df["close"] > df["open"]).astype(int)
    for tf in [4, 12, 24]:
        df[f"dir_{tf}h"] = df["dir"].rolling(tf).mean().fillna(0)
    return df

# =========================
# Sequence Builder
# =========================
def build_sequences(df, lookback=24):
    df = add_multi_tf_features(df)
    features = ["dir", "dir_4h", "dir_12h", "dir_24h"]

    # Используем numpy для векторизации процесса
    X = []
    y = []

    seqs = df[features].values
    labels = (df["close"].shift(-1) > df["open"].shift(-1)).astype(int).values  # Предсказание направленности

    for i in range(lookback, len(df) - 1):
        X.append(seqs[i - lookback:i].T)  # Строим последовательности
        y.append(labels[i + 1])  # Следующая свеча

    return np.array(X), np.array(y)

class CandleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

# =========================
# TCN with multi-channel input
# =========================
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=(kernel_size-1)*dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.net = nn.Sequential(self.conv, self.relu)

    def forward(self, x):
        out = self.net(x)
        return out[:, :, :x.size(2)]

class TCN(nn.Module):
    def __init__(self, seq_len, in_channels=4, num_classes=2, num_channels=[16,32], kernel_size=2):
        super().__init__()
        layers = []
        ch_in = in_channels
        for i, ch in enumerate(num_channels):
            layers.append(TCNBlock(ch_in, ch, kernel_size, dilation=2**i))
            ch_in = ch
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(ch_in * seq_len, num_classes)

    def forward(self, x):
        out = self.network(x)
        out = out.flatten(1)
        return self.fc(out)

# =========================
# Training
# =========================
def train_tcn(df_train, lookback=24, num_channels=[16,32], kernel_size=2, epochs=5, lr=0.001, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train = build_sequences(df_train, lookback)
    dataset = CandleDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    model = TCN(seq_len=lookback, num_channels=num_channels, kernel_size=kernel_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss/len(dataloader):.4f}")
    return model, device

# =========================
# Prediction
# =========================
def predict_next_candle(model, df, lookback=24, device=None):
    X, _ = build_sequences(df, lookback)
    if len(X) == 0:
        return None, None
    model.eval()
    x_pred = torch.tensor(X[-1], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x_pred)
        proba = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred = int(proba[1] > 0.5)
        conf = proba[1]
    return "U" if pred == 1 else "D", conf

# =========================
# Backtesting
# =========================
 # =========================
# Backtesting with live stats
# =========================
async def main(backtest_start, backtest_end, train_start, train_end, lookback=24, num_channels=[16,32], kernel_size=2, print_interval=50):
    db = DbProvider()
    print("Init Mysql provider")
    res = await db.select("candles", ["open_time", "open_price", "close_price"], None, 0, "ASC")
    df = pd.DataFrame(res, columns=["open_time", "open", "close"]).reset_index(drop=True)

    # Filter training data
    df_train = df[(df["open_time"] >= train_start) & (df["open_time"] < train_end)]
    print(f"Training on {len(df_train)} candles")
    model, device = train_tcn(df_train, lookback, num_channels, kernel_size, epochs=EPOCH)

    # Backtesting
    df_backtest = df[(df["open_time"] >= backtest_start) & (df["open_time"] < backtest_end)]
    balance = 0
    bet_size = 100
    win_coef = 1.96
    wins = 0
    losses = 0
    confidences = []

    for idx in range(lookback, len(df_backtest)-1):
        df_window = df_backtest.iloc[:idx+1]
        prediction, confidence = predict_next_candle(model, df_window, lookback, device)
        if prediction is None or confidence is None or confidence < 0.53:
            continue

        confidences.append(confidence)
        real = 1 if df_backtest["close"].iloc[idx+1] > df_backtest["open"].iloc[idx+1] else 0
        pred = 1 if prediction == "U" else 0

        if real == pred:
            balance += bet_size*(win_coef-1)
            wins += 1
        else:
            balance -= bet_size
            losses += 1

        # Print stats every 'print_interval' candles
        if (idx % print_interval == 0) and (idx != lookback):
            total_bets = wins + losses
            winrate = (wins/total_bets)*100 if total_bets > 0 else 0
            avg_conf = np.mean(confidences) if confidences else 0
            print(f"[Candle {idx}] Balance: {balance:.2f} | Wins: {wins} | Losses: {losses} | Winrate: {winrate:.2f}% | Avg Conf: {avg_conf:.2f}")

    # Final summary
    total_bets = wins + losses
    winrate = (wins/total_bets)*100 if total_bets > 0 else 0
    avg_conf = np.mean(confidences) if confidences else 0
    print("\n=== Backtest Summary ===")
    print(f"Total Bets: {total_bets} | Wins: {wins} | Losses: {losses} | Winrate: {winrate:.2f}% | Final Balance: {balance:.2f} | Avg Confidence: {avg_conf:.2f}")

# =========================
# Brute-force parameter search
# =========================
async def brute_force():
    lookbacks = [24, 48, 72]
    kernels = [2, 3]
    channels_options = [[16,32], [32,64]]
    best_balance = float('-inf')
    best_params = None

    for lb in lookbacks:
        for k in kernels:
            for ch in channels_options:
                print(f"Testing lookback={lb}, kernel={k}, channels={ch}")
                try:
                    await main(backtest_start, backtest_end, train_start, train_end, lookback=lb, num_channels=ch, kernel_size=k)
                except Exception as e:
                    print("Error:", e)

if __name__ == "__main__":
    train_start = int(datetime(2018, 7, 1, tzinfo=timezone.utc).timestamp() * 1_000_000)
    train_end = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1_000_000)

    backtest_start = int(datetime(2025, 9, 1, tzinfo=timezone.utc).timestamp() * 1_000_000)
    backtest_end = int(datetime(2025, 10, 25, tzinfo=timezone.utc).timestamp() * 1_000_000)

    asyncio.run(brute_force())
