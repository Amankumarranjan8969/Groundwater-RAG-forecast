"""rag/context.py — summarise a dashboard run for grounding the assistant."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
        return "—" if not np.isfinite(numeric) else f"{numeric:.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def summarize_results(results: dict[str, dict[str, Any]]) -> str:
    """A compact text summary of every season's headline result."""
    if not results:
        return "No analysis has been run yet."

    lines: list[str] = []
    for season, result in results.items():
        top = result["metrics"].iloc[0]
        lines.append(
            f"{season}: best model = {result['best_model']}, "
            f"bias-corrected test R² = {_fmt(top['Bias-corrected test R2'])}, "
            f"RMSE = {_fmt(top['Bias-corrected test RMSE'])} mbgl, "
            f"selected features = {', '.join(result['selected_features'])}."
        )
        if result.get("low_data_rescue_used"):
            lines.append(
                f"  ({season} had a small dataset — rows with an incomplete lag/rolling "
                "value were imputed instead of dropped so there was enough data to train.)"
            )
    return "\n".join(lines)


def summarize_deep_results(results: dict[str, dict[str, Any]]) -> str:
    """A compact text summary of the separate PINN / ESN / SARIMA / LSTM tab."""
    lines: list[str] = []
    for season, result in results.items():
        deep_metrics = result.get("deep_metrics")
        if deep_metrics is None or deep_metrics.empty:
            continue
        top = deep_metrics.iloc[0]
        ranked = ", ".join(
            f"{row['Model']} (R²={_fmt(row['Bias-corrected test R2'])})"
            for _, row in deep_metrics.iterrows()
        )
        lines.append(
            f"{season} (deep-learning tab): best = {result.get('deep_best_model')}, "
            f"bias-corrected test R² = {_fmt(top['Bias-corrected test R2'])}, "
            f"RMSE = {_fmt(top['Bias-corrected test RMSE'])} mbgl. All models: {ranked}."
        )
    if not lines:
        return ""
    return "\n".join(lines)


def summarize_dataset(payload: dict[str, Any]) -> str:
    """A compact text summary of the uploaded workbook."""
    data = payload.get("clean_data")
    if data is None or data.empty:
        return "No dataset is loaded."
    return (
        f"Uploaded workbook: {len(data):,} rows, "
        f"{data['VILLAGE'].nunique()} villages, "
        f"years {int(data['YEAR'].min())}–{int(data['YEAR'].max())}, "
        f"seasons present: {', '.join(sorted(data['Season'].unique()))}."
    )


def build_job_context(payload: dict[str, Any] | None) -> str:
    """Assemble the full grounding context: dataset + per-season results."""
    if not payload:
        return "No analysis has been run in this session yet."
    parts = [summarize_dataset(payload)]
    results = payload.get("results") or {}
    if results:
        parts.append("Season-wise best results (six headline models):")
        parts.append(summarize_results(results))
        deep_summary = summarize_deep_results(results)
        if deep_summary:
            parts.append("Season-wise results — separate deep-learning / physics-informed "
                         "tab (PINN, reservoir computing, SARIMA, and LSTM if installed):")
            parts.append(deep_summary)
    return "\n\n".join(parts)