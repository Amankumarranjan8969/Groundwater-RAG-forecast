"""deep_models.py — PINN, reservoir computing (ESN), SARIMA, and LSTM models
for the groundwater dashboard.

These sit alongside — not instead of — the six headline scikit-learn /
boosted models already trained in pipeline.py. They use the same temporal
train/validation/test split and the same selected features, but they are
scored and shown in their own dashboard tab ("12 · Deep learning &
physics-informed") so the two families never compete for the same plots or
the same Taylor diagram.

Honest design notes
--------------------
- PINNRegressor is a small NumPy feed-forward network whose training loss
  combines the usual data term with a soft physics residual based on the
  seasonal water balance (Rainfall − ET informs the expected change in
  DTWL). That makes it a physics-*informed* regulariser rather than a full
  PDE-constrained network — there's no spatial mesh here, just tabular
  village-year rows. When the water-balance / lag-1 columns aren't part of
  the selected features for a season, the physics term is simply switched
  off and the model behaves as a plain regularised MLP.
- EchoStateRegressor is a classic echo-state network: a large, fixed, random
  recurrent reservoir plus a small trained linear read-out (Ridge). Only the
  read-out is learned, which is the defining trait of reservoir computing.
  The "sequence" each row rides through the reservoir is built from that
  row's own lag columns (…_lag4 → … → …_lag1 → present), so the reservoir
  genuinely sees a short time history per sample.
- SarimaxRegressor wraps statsmodels SARIMAX. Because the modelling table is
  a village/year panel and not one clean ordered series, the wrapper uses
  the full ARIMA/seasonal fit for in-sample (training) rows, and falls back
  to the fitted exogenous-regression relationship for any other call shape —
  validation, test, a recursive scenario-forecast step, or a
  leave-one-feature-out refit — rather than guessing at a Kalman state to
  propagate for an arbitrary, possibly single, row. That keeps `.predict`
  safe for every calling pattern already used elsewhere in this dashboard.
- LSTMRegressor uses PyTorch when it's installed, and is skipped gracefully
  (exactly like CatBoost/XGBoost/LightGBM/Cubist/M5 already are) when it
  isn't. It sees the same lag sequence as the reservoir model.

Every model here follows the scikit-learn estimator contract
(`fit(X, y)` / `predict(X)`), so they drop into the same evaluation and
metrics code as the rest of the dashboard.
"""
from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge

DEEP_MODEL_LIST = [
    "PINN (Physics-Informed NN)",
    "Reservoir Computing (ESN)",
    "SARIMA (SARIMAX)",
    "LSTM (Deep Learning)",
]


# --------------------------------------------------------------------------- #
# Optional-dependency helpers (same philosophy as pipeline._optional_model_status)
# --------------------------------------------------------------------------- #
def _torch_available() -> bool:
    try:
        importlib.import_module("torch")
        return True
    except Exception:
        return False


def _deep_model_status() -> dict[str, str]:
    statuses = {name: "Available" for name in DEEP_MODEL_LIST}
    if not _torch_available():
        statuses["LSTM (Deep Learning)"] = "Not installed (optional `torch` package missing)"
    return statuses


# --------------------------------------------------------------------------- #
# Shared helper: build a short per-row lag sequence for the sequence models
# --------------------------------------------------------------------------- #
def _lag_sequence_groups(columns: list[str]) -> dict[str, list[tuple[int, str]]]:
    """Group '<base>_lag<k>' columns by base name, oldest lag first.

    e.g. {'DTWL (mbgl)': [(4, 'DTWL (mbgl)_lag4'), (3, '...lag3'), (2, ...), (1, ...)]}
    """
    groups: dict[str, list[tuple[int, str]]] = {}
    for col in columns:
        if "_lag" not in col:
            continue
        base, _, tail = col.rpartition("_lag")
        if not tail.isdigit():
            continue
        groups.setdefault(base, []).append((int(tail), col))
    for base in groups:
        groups[base].sort(key=lambda item: item[0], reverse=True)
    return groups


def build_lag_sequence(frame: pd.DataFrame) -> np.ndarray:
    """Turn a feature table into (n_samples, n_steps, n_series) sequences.

    Each '<base>_lagK' family becomes one channel, walked from the oldest
    lag to the most recent. Static (non-lag) columns are appended as extra
    channels at the final ("present") time step, so the sequence models
    still see the current-period context alongside the short history.

    Falls back to a single time step (T=1) built from every column when no
    lag columns are present at all — the reservoir/LSTM then behaves like a
    plain random/learned projection of the feature row, which still works,
    it just isn't drawing on an explicit time history.
    """
    columns = list(frame.columns)
    groups = _lag_sequence_groups(columns)
    lag_columns = {col for family in groups.values() for _, col in family}
    static_columns = [c for c in columns if c not in lag_columns]

    n_samples = len(frame)
    if not groups:
        base = frame[static_columns] if static_columns else frame
        series = base.to_numpy(dtype=float)
        if series.ndim == 1:
            series = series.reshape(-1, 1)
        return series.reshape(n_samples, 1, max(series.shape[1], 1))

    n_steps = max(len(family) for family in groups.values()) + 1  # +1 for "present"
    n_series = len(groups) + len(static_columns)
    sequence = np.zeros((n_samples, n_steps, max(n_series, 1)), dtype=float)

    channel = 0
    for family in groups.values():
        values = [frame[col].to_numpy(dtype=float) for _, col in family]
        pad = n_steps - len(values) - 1
        for t, arr in enumerate(values):
            sequence[:, pad + t, channel] = arr
        channel += 1
    for col in static_columns:
        sequence[:, -1, channel] = frame[col].to_numpy(dtype=float)
        channel += 1
    return sequence


# --------------------------------------------------------------------------- #
# 1. Physics-informed neural network (pure NumPy — always available)
# --------------------------------------------------------------------------- #
class PINNRegressor(BaseEstimator, RegressorMixin):
    """A compact NumPy feed-forward network with a water-balance physics prior."""

    def __init__(self, hidden_units: int = 24, epochs: int = 300, lr: float = 0.02,
                 physics_weight: float = 0.2, l2: float = 1e-4, random_state: int = 42):
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.lr = lr
        self.physics_weight = physics_weight
        self.l2 = l2
        self.random_state = random_state

    @staticmethod
    def _find_physics_columns(columns: list[str]) -> tuple[str | None, str | None, str | None, str | None]:
        lag1 = next((c for c in columns if c.startswith("DTWL") and c.endswith("_lag1")), None)
        balance = next((c for c in columns if c.lower() == "water_balance"), None)
        rainfall = next((c for c in columns if c.lower() == "rainfall"), None)
        et = next((c for c in columns if c.lower() == "et"), None)
        return lag1, balance, rainfall, et

    def _physics_target(self, X_df: pd.DataFrame, y: np.ndarray | None) -> np.ndarray | None:
        lag1_col, balance_col, rainfall_col, et_col = self._find_physics_columns(list(X_df.columns))
        if lag1_col is None:
            return None
        if balance_col is not None:
            balance = X_df[balance_col].to_numpy(dtype=float)
        elif rainfall_col is not None and et_col is not None:
            balance = X_df[rainfall_col].to_numpy(dtype=float) - X_df[et_col].to_numpy(dtype=float)
        else:
            return None
        lag1 = X_df[lag1_col].to_numpy(dtype=float)
        if y is not None:
            # Closed-form recharge coefficient k in the soft physics prior:
            #   (DTWL − DTWL_lag1)  ≈  −k · (Rainfall − ET)
            # i.e. a positive water balance (more rain than ET) is expected
            # to push DTWL down (shallower water table) by roughly k per unit.
            diff = np.asarray(y, dtype=float) - lag1
            denom = float(np.dot(balance, balance)) or 1.0
            self._recharge_k = -float(np.dot(balance, diff)) / denom
        k = getattr(self, "_recharge_k", 0.0)
        return lag1 + k * balance

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        X_df = pd.DataFrame(X).reset_index(drop=True)
        y_arr = np.asarray(y, dtype=float).ravel()
        self._fit_columns = list(X_df.columns)

        self._y_mean, self._y_std = float(y_arr.mean()), float(y_arr.std() or 1.0)
        y_scaled = (y_arr - self._y_mean) / self._y_std

        physics_target = self._physics_target(X_df, y_arr)
        self.physics_active_ = physics_target is not None
        physics_scaled = (
            (physics_target - self._y_mean) / self._y_std if physics_target is not None else None
        )

        n_features = max(X_df.shape[1], 1)
        Xv = X_df.to_numpy(dtype=float) if X_df.shape[1] else np.zeros((len(X_df), 1))
        h = self.hidden_units
        self.W1 = rng.randn(n_features, h) * np.sqrt(2.0 / n_features)
        self.b1 = np.zeros(h)
        self.W2 = rng.randn(h, 1) * np.sqrt(2.0 / h)
        self.b2 = np.zeros(1)

        params = ["W1", "b1", "W2", "b2"]
        m = {p: np.zeros_like(getattr(self, p)) for p in params}
        v = {p: np.zeros_like(getattr(self, p)) for p in params}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        n = max(len(Xv), 1)

        for epoch in range(1, self.epochs + 1):
            z1 = Xv @ self.W1 + self.b1
            a1 = np.tanh(z1)
            z2 = a1 @ self.W2 + self.b2
            pred = z2.ravel()

            grad_out = 2.0 * (pred - y_scaled) / n
            if self.physics_active_:
                grad_out = grad_out + self.physics_weight * 2.0 * (pred - physics_scaled) / n

            d_z2 = grad_out.reshape(-1, 1)
            d_W2 = a1.T @ d_z2 + self.l2 * self.W2
            d_b2 = d_z2.sum(axis=0)
            d_a1 = d_z2 @ self.W2.T
            d_z1 = d_a1 * (1 - a1 ** 2)
            d_W1 = Xv.T @ d_z1 + self.l2 * self.W1
            d_b1 = d_z1.sum(axis=0)

            grads = {"W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2}
            for p in params:
                m[p] = beta1 * m[p] + (1 - beta1) * grads[p]
                v[p] = beta2 * v[p] + (1 - beta2) * (grads[p] ** 2)
                m_hat = m[p] / (1 - beta1 ** epoch)
                v_hat = v[p] / (1 - beta2 ** epoch)
                setattr(self, p, getattr(self, p) - self.lr * m_hat / (np.sqrt(v_hat) + eps))
        return self

    def predict(self, X):
        X_df = pd.DataFrame(X).reindex(columns=self._fit_columns, fill_value=0.0)
        Xv = X_df.to_numpy(dtype=float) if X_df.shape[1] else np.zeros((len(X_df), 1))
        a1 = np.tanh(Xv @ self.W1 + self.b1)
        pred_scaled = (a1 @ self.W2 + self.b2).ravel()
        return pred_scaled * self._y_std + self._y_mean


# --------------------------------------------------------------------------- #
# 2. Reservoir computing / Echo State Network (pure NumPy — always available)
# --------------------------------------------------------------------------- #
class EchoStateRegressor(BaseEstimator, RegressorMixin):
    """A fixed, random, sparse recurrent reservoir with a trained Ridge read-out."""

    def __init__(self, reservoir_size: int = 80, spectral_radius: float = 0.9,
                 sparsity: float = 0.15, leak_rate: float = 0.35,
                 ridge_alpha: float = 2.0, random_state: int = 42):
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leak_rate = leak_rate
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state

    def _init_reservoir(self, n_inputs: int) -> None:
        rng = np.random.RandomState(self.random_state)
        w_in = rng.uniform(-1.0, 1.0, size=(self.reservoir_size, max(n_inputs, 1)))
        w_res = rng.uniform(-1.0, 1.0, size=(self.reservoir_size, self.reservoir_size))
        mask = rng.random_sample(w_res.shape) < self.sparsity
        w_res = w_res * mask
        radius = float(np.max(np.abs(np.linalg.eigvals(w_res)))) or 1.0
        self.w_in_ = w_in
        self.w_res_ = w_res * (self.spectral_radius / radius)

    def _run(self, sequences: np.ndarray) -> np.ndarray:
        n_samples, n_steps, _ = sequences.shape
        states = np.zeros((n_samples, self.reservoir_size))
        for i in range(n_samples):
            state = np.zeros(self.reservoir_size)
            for t in range(n_steps):
                pre = self.w_in_ @ sequences[i, t] + self.w_res_ @ state
                state = (1 - self.leak_rate) * state + self.leak_rate * np.tanh(pre)
            states[i] = state
        return states

    def fit(self, X, y):
        X_df = pd.DataFrame(X).reset_index(drop=True)
        self._fit_columns = list(X_df.columns)
        sequences = build_lag_sequence(X_df)
        self._init_reservoir(sequences.shape[-1])
        states = self._run(sequences)
        self.readout_ = Ridge(alpha=self.ridge_alpha)
        self.readout_.fit(states, np.asarray(y, dtype=float).ravel())
        return self

    def predict(self, X):
        X_df = pd.DataFrame(X).reindex(columns=self._fit_columns, fill_value=0.0)
        sequences = build_lag_sequence(X_df)
        states = self._run(sequences)
        return self.readout_.predict(states)


# --------------------------------------------------------------------------- #
# 3. SARIMA / SARIMAX (statsmodels — already a hard dependency of this app)
# --------------------------------------------------------------------------- #
class SarimaxRegressor(BaseEstimator, RegressorMixin):
    """SARIMAX (ARIMA plus exogenous drivers), robust to small, imbalanced panels.

    See the module docstring for why `.predict` behaves differently for the
    training rows (full fitted values) versus any other call shape (the
    fitted exogenous-regression relationship).
    """

    def __init__(self, order: tuple[int, int, int] = (2, 0, 1),
                 seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
                 trend: str = "c"):
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend

    def fit(self, X, y):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        X_df = pd.DataFrame(X).reset_index(drop=True)
        y_arr = np.asarray(y, dtype=float).ravel()
        self._fit_columns = list(X_df.columns)
        self._train_index = pd.DataFrame(X).index
        self._y_mean = float(y_arr.mean())
        exog = X_df if len(X_df.columns) else None

        last_exc: Exception | None = None
        for order in (self.order, (1, 0, 0), (0, 0, 0)):
            try:
                model = SARIMAX(
                    y_arr, exog=exog, order=order, seasonal_order=self.seasonal_order,
                    trend=self.trend, enforce_stationarity=False, enforce_invertibility=False,
                )
                self.results_ = model.fit(disp=False, maxiter=200)
                self._fitted_order_ = order
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001 - deliberately broad, we retry simpler orders
                last_exc = exc
        if last_exc is not None:
            raise last_exc
        return self

    def predict(self, X):
        X_original = pd.DataFrame(X)
        if X_original.index.equals(self._train_index):
            return np.asarray(self.results_.fittedvalues, dtype=float)

        X_df = X_original.reindex(columns=self._fit_columns, fill_value=0.0)
        params = self.results_.params
        intercept = float(params.get("const", self._y_mean)) if "const" in params.index else self._y_mean
        coefs = np.array([float(params.get(col, 0.0)) for col in self._fit_columns])
        if coefs.size and coefs.size == X_df.shape[1]:
            return intercept + X_df.to_numpy(dtype=float) @ coefs
        return np.full(len(X_df), intercept)


# --------------------------------------------------------------------------- #
# 4. LSTM (PyTorch — optional, skipped gracefully when not installed)
# --------------------------------------------------------------------------- #
class LSTMRegressor(BaseEstimator, RegressorMixin):
    """A small single-layer LSTM + linear head, trained with Adam in PyTorch.

    Sees the same per-row lag sequence as `EchoStateRegressor`. Only ever
    instantiated by `build_deep_models` when PyTorch is importable.
    """

    def __init__(self, hidden_size: int = 32, epochs: int = 150, lr: float = 0.01,
                 weight_decay: float = 1e-4, random_state: int = 42):
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_state = random_state

    def fit(self, X, y):
        import torch
        from torch import nn

        torch.manual_seed(self.random_state)
        X_df = pd.DataFrame(X).reset_index(drop=True)
        self._fit_columns = list(X_df.columns)
        sequences = build_lag_sequence(X_df)
        y_arr = np.asarray(y, dtype=float).ravel()
        self._y_mean, self._y_std = float(y_arr.mean()), float(y_arr.std() or 1.0)
        y_scaled = (y_arr - self._y_mean) / self._y_std

        n_series = sequences.shape[-1]
        seq_tensor = torch.tensor(sequences, dtype=torch.float32)
        target_tensor = torch.tensor(y_scaled, dtype=torch.float32).view(-1, 1)

        class _Net(nn.Module):
            def __init__(self, input_size: int, hidden_size: int):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.head = nn.Linear(hidden_size, 1)

            def forward(self, sequence):
                out, _ = self.lstm(sequence)
                return self.head(out[:, -1, :])

        net = _Net(n_series, self.hidden_size)
        optimiser = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        net.train()
        for _ in range(self.epochs):
            optimiser.zero_grad()
            loss = loss_fn(net(seq_tensor), target_tensor)
            loss.backward()
            optimiser.step()
        net.eval()
        self._net = net
        return self

    def predict(self, X):
        import torch

        X_df = pd.DataFrame(X).reindex(columns=self._fit_columns, fill_value=0.0)
        sequences = build_lag_sequence(X_df)
        with torch.no_grad():
            seq_tensor = torch.tensor(sequences, dtype=torch.float32)
            pred_scaled = self._net(seq_tensor).numpy().ravel()
        return pred_scaled * self._y_std + self._y_mean


# --------------------------------------------------------------------------- #
# Public factory — mirrors pipeline.build_models
# --------------------------------------------------------------------------- #
def build_deep_models(reservoir_size: int = 80) -> tuple[dict[str, Any], pd.DataFrame]:
    """Instantiate the PINN, reservoir-computing, SARIMA, and (if available)
    LSTM models, plus an availability table matching pipeline.build_models'
    format so the dashboard can render it the same way."""
    models: dict[str, Any] = {
        "PINN (Physics-Informed NN)": PINNRegressor(),
        "Reservoir Computing (ESN)": EchoStateRegressor(reservoir_size=reservoir_size),
        "SARIMA (SARIMAX)": SarimaxRegressor(),
    }
    if _torch_available():
        models["LSTM (Deep Learning)"] = LSTMRegressor()
    statuses = _deep_model_status()
    availability = pd.DataFrame(
        [{"Model": name, "Status": "Included" if name in models else statuses.get(name, "Not installed")}
         for name in DEEP_MODEL_LIST]
    )
    return models, availability
