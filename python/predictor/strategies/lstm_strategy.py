import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from predictor.strategies.base import BaseStrategy
from predictor.features import add_technical_features, get_feature_columns

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class _LSTMNet(nn.Module):
    """Simple LSTM binary classifier."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # Take last timestep
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)


class LSTMStrategy(BaseStrategy):

    def __init__(self, params: dict):
        super().__init__(params)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = []
        self.device = "cpu"

    @staticmethod
    def description() -> str:
        return (
            "LSTM (Long Short-Term Memory) neural network for sequential candle prediction. "
            "Feeds sequences of N candles with all technical features into an LSTM, "
            "predicting the direction of the candle at +horizon. Captures temporal patterns "
            "that tree-based models may miss. Requires PyTorch."
        )

    @staticmethod
    def default_params() -> dict:
        return {
            "seq_len": 10,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "epochs": 15,
            "batch_size": 256,
            "learning_rate": 0.001,
            "threshold": 0.52,
        }

    @staticmethod
    def param_docs() -> dict:
        return {
            "seq_len": "Number of past candles in each input sequence. Typical: 5-20.",
            "hidden_size": "LSTM hidden state dimension. Higher = more capacity. Typical: 32-128.",
            "num_layers": "Number of stacked LSTM layers. Typical: 1-3.",
            "dropout": "Dropout between LSTM layers. Prevents overfitting. Typical: 0.1-0.4.",
            "epochs": "Training epochs. More = better fit but slower. Typical: 5-30.",
            "batch_size": "Mini-batch size for training. Typical: 64-512.",
            "learning_rate": "Adam optimizer learning rate. Typical: 0.0005-0.005.",
            "threshold": "Minimum probability to emit a signal. Typical: 0.50-0.55.",
        }

    def fit(self, df: pd.DataFrame, horizon: int = 1) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for LSTM strategy. Install with: pip install torch")

        if "rsi_14" not in df.columns:
            df = add_technical_features(df)
        target_col = f"future_dir_{horizon}"
        if target_col not in df.columns:
            df[target_col] = (df["close"].shift(-horizon) > df["open"].shift(-horizon)).astype(np.int8)

        df = df.dropna()
        self.feature_cols = get_feature_columns(df)

        X_all = df[self.feature_cols].values.astype(np.float32)
        y_all = df[target_col].values.astype(np.float32)

        self.scaler.fit(X_all)
        X_scaled = self.scaler.transform(X_all)

        seq_len = self.params["seq_len"]
        # Build sequences
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X_scaled)):
            X_seq.append(X_scaled[i - seq_len:i])
            y_seq.append(y_all[i])

        if len(X_seq) < 100:
            return

        X_t = torch.tensor(np.array(X_seq), dtype=torch.float32)
        y_t = torch.tensor(np.array(y_seq), dtype=torch.float32)

        input_size = X_t.shape[2]
        net = _LSTMNet(
            input_size=input_size,
            hidden_size=self.params["hidden_size"],
            num_layers=self.params["num_layers"],
            dropout=self.params["dropout"],
        )
        net.train()

        optimizer = torch.optim.Adam(net.parameters(), lr=self.params["learning_rate"])
        loss_fn = nn.BCEWithLogitsLoss()
        batch_size = self.params["batch_size"]

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(self.params["epochs"]):
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = net(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

        net.eval()
        self.model = net

    def predict_proba(self, df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        if self.model is None:
            return np.full(len(df), 0.5)

        if "rsi_14" not in df.columns:
            df = add_technical_features(df)
        df = df.fillna(0)

        X_all = df[self.feature_cols].values.astype(np.float32)
        X_scaled = self.scaler.transform(X_all)

        seq_len = self.params["seq_len"]
        n = len(X_scaled)

        if n < seq_len:
            return np.full(n, 0.5)

        # Build sequences for all rows that have enough history
        probas = np.full(n, 0.5)
        seqs = []
        valid_indices = []
        for i in range(seq_len, n):
            seqs.append(X_scaled[i - seq_len:i])
            valid_indices.append(i)

        # Also handle the last row if n == seq_len (single prediction)
        if n == seq_len:
            seqs.append(X_scaled[0:seq_len])
            valid_indices.append(n - 1)

        if not seqs:
            return probas

        X_t = torch.tensor(np.array(seqs), dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.sigmoid(logits).numpy()

        for idx, vi in enumerate(valid_indices):
            probas[vi] = float(probs[idx])

        return probas

    def predict(self, df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        proba = self.predict_proba(df, horizon)
        threshold = self.params["threshold"]
        preds = np.full(len(proba), -1, dtype=np.int8)
        preds[proba > threshold] = 1
        preds[proba < (1 - threshold)] = 0
        return preds
