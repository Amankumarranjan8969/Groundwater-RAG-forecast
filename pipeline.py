"""Reusable groundwater modelling workflow extracted from the rajasthan notebook.

The dashboard keeps the notebook's seasonal workflow, but makes its assumptions
explicit and safeguards optional third-party model libraries.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable
import importlib
import math
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from deep_models import DEEP_MODEL_LIST, build_deep_models

TARGET = "DTWL (mbgl)"
SEASON_ORDER = ["Pre-Monsoon", "Monsoon", "Post-Monsoon", "Non-Monsoon"]

# Canonical (normalised) sheet-name aliases -> canonical season label.
# Keys must be run through _normalise_name() at lookup time.
SEASON_SHEET_ALIASES = {
    "monsoon": "Monsoon",
    "pre_monsoon": "Pre-Monsoon",
    "premonsoon": "Pre-Monsoon",
    "post_monsoon": "Post-Monsoon",
    "postmonsoon": "Post-Monsoon",
    "non_monsoon": "Non-Monsoon",
    "nonmonsoon": "Non-Monsoon",
    "non_monsoon_season": "Non-Monsoon",
}

REQUIRED_COLUMNS = {
    "VILLAGE", "YEAR", TARGET, "Tmax", "Tmin", "SM_10cm", "SM_40cm", "SM_100cm",
    "NDVI", "NDMI", "NDBI", "ET", "Rainfall",
}

COLUMN_ALIASES = {
    # target
    "dtwl (mbgl)": "DTWL (mbgl)",
    "dtwl": "DTWL (mbgl)",
    "wl (mbgl)": "DTWL (mbgl)",
    "wl": "DTWL (mbgl)",
    "water level (mbgl)": "DTWL (mbgl)",

    # met
    "tmax (°c)": "Tmax",
    "tmax(c)": "Tmax",
    "tmax": "Tmax",
    "tmin (°c)": "Tmin",
    "tmin(c)": "Tmin",
    "tmin": "Tmin",
    "rh (%)": "RH",
    "rh": "RH",

    # fluxes
    "et_mm": "ET",
    "et (mm)": "ET",
    "et": "ET",
    "rainfall (mm)": "Rainfall",
    "rainfall_mm": "Rainfall",
    "rainfall": "Rainfall",

    # soil moisture
    "sm_10cm": "SM_10cm",
    "sm_40cm": "SM_40cm",
    "sm_100cm": "SM_100cm",

    # vegetation / indices
    "ndvi": "NDVI",
    "ndmi": "NDMI",
    "ndbi": "NDBI",
    "ndwi": "NDWI",

    # geometry
    "slope_degree(12/05/2026)": "Slope_Degree",
    "slope_degree": "Slope_Degree",
    "slope_percent": "Slope_Percent",
    "plain curvature": "Plain curvature",
    "profile curvature": "Profile Curvature",
    "profile_curvature": "Profile Curvature",
    "aaspect": "Aspect",
    "aspect": "Aspect",
    "drainage density": "Drainage density",
    "distance to drainage": "Distance to drainage",
}


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map common column-name variants to the canonical names used by the pipeline."""
    rename_map = {}
    for col in frame.columns:
        key = str(col).strip()
        canonical = COLUMN_ALIASES.get(key.lower())
        if canonical and canonical != col:
            rename_map[col] = canonical
    return frame.rename(columns=rename_map)


META_COLUMNS = ["STATE_UT", "DISTRICT", "BLOCK", "LATITUDE", "LONGITUDE"]


@dataclass(frozen=True)
class RunConfig:
    train_end: int
    validation_end: int
    selected_feature_count: int = 6
    tree_count: int = 250
    include_optional_models: bool = True
    run_lofo: bool = True
    forecast_steps: int = 5
    include_deep_models: bool = True
    reservoir_size: int = 80


def _normalise_name(name: str) -> str:
    """Lowercase, strip, and collapse separators so sheet-name matching is tolerant."""
    cleaned = str(name).strip().lower()
    for separator in (" ", "-", "\t", "\n", "\r"):
        cleaned = cleaned.replace(separator, "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def _as_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce possible numeric Excel fields while leaving descriptive columns alone."""
    result = frame.copy()
    required_numeric = REQUIRED_COLUMNS - {"VILLAGE"}
    for column in result.columns:
        if column in required_numeric:
            result[column] = pd.to_numeric(result[column], errors="coerce")
        elif column not in {"VILLAGE", "Season", "STATE_UT", "DISTRICT", "BLOCK", "DATE"}:
            converted = pd.to_numeric(result[column], errors="coerce")
            # Preserve a genuine categorical predictor; convert a numeric-looking field.
            if converted.notna().sum() == result[column].notna().sum():
                result[column] = converted
    return result


def load_seasonal_workbook(source: Any) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Read all recognised season sheets and return data, a workbook report, and notices."""
    workbook = pd.ExcelFile(source)
    frames: list[pd.DataFrame] = []
    report_rows: list[dict[str, Any]] = []
    notices: list[str] = []

    aliases = {_normalise_name(key): value for key, value in SEASON_SHEET_ALIASES.items()}

    for sheet_name in workbook.sheet_names:
        normalised = _normalise_name(sheet_name)
        season = aliases.get(normalised)
        if season is None:
            notices.append(
                f"Ignored sheet '{sheet_name}' because it is not a recognised season sheet. "
                f"Recognised names include: {sorted(set(aliases.values()))}."
            )
            continue

        # Single read of the sheet (previous version read it twice).
        frame = pd.read_excel(workbook, sheet_name=sheet_name)
        frame.columns = [str(column).strip() for column in frame.columns]
        frame = frame.rename(columns={"Slope_Degree(12/05/2026)": "Slope_Degree"})
        frame = _normalise_columns(frame)

        # Drop fully empty rows and rows missing the two key identifiers.
        frame = frame.dropna(how="all").reset_index(drop=True)
        frame = frame.dropna(subset=["VILLAGE", "YEAR"], how="any").reset_index(drop=True)

        missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
        report_rows.append({
            "Sheet": sheet_name,
            "Mapped season": season,
            "Rows": len(frame),
            "Columns": len(frame.columns),
            "Missing required columns": ", ".join(missing) if missing else "None",
        })
        if missing:
            notices.append(
                f"Sheet '{sheet_name}' mapped to '{season}' is missing required columns: "
                f"{', '.join(missing)}. This sheet was skipped."
            )
            continue

        frame = frame.drop(
            columns=[column for column in META_COLUMNS if column in frame.columns],
            errors="ignore",
        )
        frame["Season"] = season
        frames.append(_as_numeric(frame))

    report = pd.DataFrame(report_rows)
    if not frames:
        detail = "No recognised season sheet contained all required columns."
        if not report.empty:
            detail += " Review the 'Missing required columns' field in the upload report."
        if notices:
            detail += " Notices: " + " | ".join(notices)
        raise ValueError(detail)

    data = pd.concat(frames, ignore_index=True)
    data["YEAR"] = pd.to_numeric(data["YEAR"], errors="coerce")
    data = data.dropna(subset=["YEAR"]).copy()
    data["YEAR"] = data["YEAR"].astype(int)
    return data, report, notices


def validate_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the notebook's validation corrections and make them auditable."""
    frame = data.copy()
    issues: list[dict[str, Any]] = []

    missing = frame.isna().sum()
    for column, count in missing[missing.gt(0)].items():
        issues.append({"Check": "Missing values", "Field": column, "Affected rows": int(count),
                       "Action": "Reported; incomplete rows are excluded only at modelling stage"})

    negative_target = frame[TARGET] < 0
    if negative_target.any():
        median = frame.loc[~negative_target, TARGET].median()
        frame.loc[negative_target, TARGET] = median
        issues.append({"Check": "Negative DTWL", "Field": TARGET, "Affected rows": int(negative_target.sum()),
                       "Action": f"Replaced with dataset median ({median:.3f})"})

    negative_rainfall = frame["Rainfall"] < 0
    if negative_rainfall.any():
        frame.loc[negative_rainfall, "Rainfall"] = 0
        issues.append({"Check": "Negative rainfall", "Field": "Rainfall", "Affected rows": int(negative_rainfall.sum()),
                       "Action": "Clipped to zero"})

    soil_violation = (frame["SM_10cm"] > frame["SM_40cm"]) | (frame["SM_40cm"] > frame["SM_100cm"])
    if soil_violation.any():
        affected = frame.loc[soil_violation, ["SM_10cm", "SM_40cm", "SM_100cm"]]
        frame.loc[soil_violation, "SM_10cm"] = affected.min(axis=1)
        frame.loc[soil_violation, "SM_40cm"] = affected.median(axis=1)
        frame.loc[soil_violation, "SM_100cm"] = affected.max(axis=1)
        issues.append({"Check": "Soil-moisture depth order", "Field": "SM_10cm / SM_40cm / SM_100cm",
                       "Affected rows": int(soil_violation.sum()), "Action": "Reordered low / middle / high values"})

    duplicate_count = int(frame.duplicated(["VILLAGE", "YEAR", "Season"]).sum())
    if duplicate_count:
        issues.append({"Check": "Duplicate village-year-season", "Field": "VILLAGE / YEAR / Season",
                       "Affected rows": duplicate_count, "Action": "Reported; retained for traceability"})

    if not issues:
        issues.append({"Check": "Validation", "Field": "All input fields", "Affected rows": 0,
                       "Action": "No automatic correction was required"})
    return frame, pd.DataFrame(issues)


def engineer_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = data.copy()
    frame["Temp_Range"] = frame["Tmax"] - frame["Tmin"]
    frame["Avg_SM"] = (frame["SM_10cm"] + frame["SM_40cm"] + frame["SM_100cm"]) / 3
    frame["SM_Gradient"] = frame["SM_100cm"] - frame["SM_10cm"]
    frame["Veg_Moisture"] = frame["NDVI"] * frame["NDMI"]
    frame["GW_Stress"] = frame["ET"] / (frame["Rainfall"] + 1)
    frame["Water_Balance"] = frame["Rainfall"] - frame["ET"]
    frame["Veg_Health"] = frame["NDVI"] * (1 - frame["NDBI"])
    season_codes = {season: number for number, season in enumerate(SEASON_ORDER)}
    encoded = frame["Season"].map(season_codes)
    frame["Season_Sin"] = np.sin(2 * np.pi * encoded / 4)
    frame["Season_Cos"] = np.cos(2 * np.pi * encoded / 4)
    created = [
        "Temp_Range", "Avg_SM", "SM_Gradient", "Veg_Moisture", "GW_Stress", "Water_Balance",
        "Veg_Health", "Season_Sin", "Season_Cos",
    ]
    return frame, created


def create_lag_features(
    data: pd.DataFrame, rescue_low_data: bool = False,
) -> tuple[pd.DataFrame, dict[str, list[str]], int, bool]:
    """Create the lag, rolling, and trend variables from the notebook workflow.

    By default this drops every row with any incomplete lag/rolling/trend
    value, exactly like the notebook. When `rescue_low_data=True` and that
    strict drop would leave too few rows to model, incomplete *engineered*
    values (lags, rolling stats, trends — never the target itself) are
    imputed instead of dropped: forward/back-filled within each village,
    then any still-missing cells filled with the column median. This only
    changes behaviour when the strict path is not enough data on its own —
    see `run_season_pipeline`, which tries the strict path first and only
    retries with `rescue_low_data=True` if that path can't support the
    configured train/validation/test split.

    Returns (complete_frame, feature_groups, rows_removed, rescue_used).
    """
    frame = data.sort_values(["VILLAGE", "YEAR"]).copy()
    grouped = frame.groupby("VILLAGE", sort=False)
    lag_config = {
        TARGET: [1, 2, 3, 4], "Rainfall": [1, 2, 3], "ET": [1, 2], "Avg_SM": [1, 2, 3],
        "NDVI": [1, 2], "NDWI": [1, 2], "NDMI": [1, 2],
    }
    lag_features: list[str] = []
    rolling_features: list[str] = []
    trend_features: list[str] = []

    for variable, lags in lag_config.items():
        if variable not in frame.columns:
            continue
        for lag in lags:
            name = f"{variable}_lag{lag}"
            frame[name] = grouped[variable].shift(lag)
            lag_features.append(name)

    for variable in ["Rainfall", "ET", "Avg_SM"]:
        if variable not in frame.columns:
            continue
        frame[f"{variable}_roll3"] = grouped[variable].transform(lambda values: values.rolling(3, min_periods=1).mean())
        frame[f"{variable}_roll5"] = grouped[variable].transform(lambda values: values.rolling(5, min_periods=1).mean())
        frame[f"{variable}_roll3_std"] = grouped[variable].transform(lambda values: values.rolling(3, min_periods=1).std())
        rolling_features.extend([f"{variable}_roll3", f"{variable}_roll5", f"{variable}_roll3_std"])

    frame[f"{TARGET}_roll3"] = grouped[TARGET].transform(lambda values: values.shift(1).rolling(3, min_periods=1).mean())
    frame[f"{TARGET}_roll5"] = grouped[TARGET].transform(lambda values: values.shift(1).rolling(5, min_periods=1).mean())
    frame[f"{TARGET}_roll3_std"] = grouped[TARGET].transform(lambda values: values.shift(1).rolling(3, min_periods=1).std())
    rolling_features.extend([f"{TARGET}_roll3", f"{TARGET}_roll5", f"{TARGET}_roll3_std"])

    for variable in ["Rainfall", "Avg_SM"]:
        if variable not in frame.columns:
            continue
        name = f"{variable}_trend"
        frame[name] = grouped[variable].transform(lambda values: values.diff(2))
        trend_features.append(name)
    target_trend = f"{TARGET}_trend"
    frame[target_trend] = grouped[TARGET].transform(lambda values: values.shift(1).diff(2))
    trend_features.append(target_trend)

    initial_rows = len(frame)
    engineered_columns = lag_features + rolling_features + trend_features
    complete = frame.dropna().reset_index(drop=True)
    rescue_used = False

    if rescue_low_data and len(frame) > len(complete):
        # Rescue rows whose only problem is a missing *engineered* value —
        # never touch the raw target or the original measured columns.
        rescued = frame.copy()
        existing_engineered = [c for c in engineered_columns if c in rescued.columns]
        if existing_engineered:
            rescued[existing_engineered] = (
                rescued.groupby("VILLAGE")[existing_engineered]
                .transform(lambda block: block.ffill().bfill())
            )
            still_missing = rescued[existing_engineered].isna()
            if still_missing.to_numpy().any():
                medians = rescued[existing_engineered].median(numeric_only=True)
                rescued[existing_engineered] = rescued[existing_engineered].fillna(medians)
        essential_columns = [c for c in rescued.columns if c not in existing_engineered]
        rescued = rescued.dropna(subset=essential_columns).reset_index(drop=True)
        if len(rescued) > len(complete):
            complete = rescued
            rescue_used = True

    feature_groups = {
        "lag": lag_features,
        "rolling": rolling_features,
        "trend": trend_features,
        "all": lag_features + rolling_features + trend_features,
    }
    return complete, feature_groups, initial_rows - len(complete), rescue_used


def prepare_modelling_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    excluded = {"VILLAGE", "YEAR", "Season", TARGET, "Slope_Degree", "DATE"}
    feature_columns = [column for column in data.columns if column not in excluded]
    features = data[feature_columns].copy()
    for column in features.select_dtypes(include=["object", "category"]).columns:
        features[column] = pd.factorize(features[column].astype(str), sort=True)[0]
    features = features.apply(pd.to_numeric, errors="coerce")
    valid_rows = features.notna().all(axis=1) & data[TARGET].notna()
    features = features.loc[valid_rows].copy()
    target = data.loc[valid_rows, TARGET].copy()
    years = data.loc[valid_rows, "YEAR"].copy()
    return features, target, years, list(features.columns)


def temporal_split(features: pd.DataFrame, target: pd.Series, years: pd.Series,
                   config: RunConfig) -> dict[str, pd.DataFrame | pd.Series]:
    if config.train_end >= config.validation_end:
        raise ValueError("Validation end year must be later than train end year.")
    masks = {
        "train": years <= config.train_end,
        "validation": (years > config.train_end) & (years <= config.validation_end),
        "test": years > config.validation_end,
    }
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    if counts["train"] < 8 or counts["validation"] < 2 or counts["test"] < 2:
        raise ValueError(
            "This seasonal split is too small after lag creation. Use earlier split years or upload more annual observations. "
            f"Current counts: train {counts['train']}, validation {counts['validation']}, test {counts['test']}."
        )
    result: dict[str, pd.DataFrame | pd.Series] = {}
    for name, mask in masks.items():
        result[f"X_{name}"] = features.loc[mask].copy()
        result[f"y_{name}"] = target.loc[mask].copy()
    result["counts"] = pd.Series(counts)
    return result


def select_features(features: pd.DataFrame, target: pd.Series, requested_count: int) -> pd.DataFrame:
    """Use mRMR when available, otherwise mutual information plus correlation."""
    count = min(max(1, requested_count), features.shape[1])
    mi_values = mutual_info_regression(features, target, random_state=42)
    correlation = features.corrwith(target).abs().fillna(0)
    ranking = pd.DataFrame({"Feature": features.columns, "Mutual information": mi_values,
                            "Absolute correlation": [correlation[column] for column in features.columns]})
    ranking["Selection method"] = "Mutual information"
    try:
        from mrmr import mrmr_regression  # type: ignore
        selected = mrmr_regression(features, target, K=count, show_progress=False)
        ranking["Selected"] = ranking["Feature"].isin(selected)
        ranking.loc[ranking["Selected"], "Selection method"] = "mRMR"
        ordering = {feature: position for position, feature in enumerate(selected)}
        ranking["Selection rank"] = ranking["Feature"].map(ordering)
        ranking = ranking.sort_values(["Selected", "Selection rank", "Mutual information"],
                                       ascending=[False, True, False])
    except Exception:
        ranking = ranking.sort_values("Mutual information", ascending=False).reset_index(drop=True)
        ranking["Selected"] = False
        ranking.loc[: count - 1, "Selected"] = True
        ranking["Selection rank"] = np.where(ranking["Selected"], ranking.index.to_numpy() + 1, np.nan)
    return ranking.reset_index(drop=True)


def _optional_model_status() -> dict[str, str]:
    statuses = {
        "CatBoost": "Not installed", "XGBoost": "Not installed", "LightGBM": "Not installed",
        "Cubist": "Not installed", "M5 Model Tree": "Not installed",
    }
    packages = {"CatBoost": "catboost", "XGBoost": "xgboost", "LightGBM": "lightgbm",
                "Cubist": "cubist", "M5 Model Tree": "m5py"}
    for display_name, package_name in packages.items():
        try:
            importlib.import_module(package_name)
            statuses[display_name] = "Available"
        except Exception:
            pass
    return statuses


def build_models(tree_count: int, include_optional: bool) -> tuple[dict[str, Any], pd.DataFrame]:
    models: dict[str, Any] = {
        "Random Forest": RandomForestRegressor(n_estimators=tree_count, max_depth=15, min_samples_split=5,
                                                 random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=tree_count, max_depth=15, min_samples_split=5,
                                             random_state=42, n_jobs=-1),
        "AdaBoost + CART": AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=3),
                                               n_estimators=min(tree_count, 200), learning_rate=1.0, random_state=42),
        "Linear Regression": LinearRegression(),
        "Gaussian Process (RBF)": GaussianProcessRegressor(
            kernel=1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0), alpha=1e-10,
            normalize_y=True, random_state=42,
        ),
        "SVR (RBF)": SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale"),
        "Bayesian Ridge": BayesianRidge(alpha_1=1e-7, alpha_2=1e-7, lambda_1=1e-5, lambda_2=1e-5),
        "Ridge Regression": Ridge(alpha=1.0),
    }
    statuses = _optional_model_status()
    if include_optional:
        try:
            from catboost import CatBoostRegressor  # type: ignore
            models["CatBoost"] = CatBoostRegressor(iterations=tree_count, learning_rate=0.05, depth=5,
                                                     l2_leaf_reg=5, random_seed=42, verbose=False)
        except Exception:
            pass
        try:
            from xgboost import XGBRegressor  # type: ignore
            models["XGBoost"] = XGBRegressor(n_estimators=tree_count, learning_rate=0.05, max_depth=5,
                                              min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                                              reg_alpha=1.0, reg_lambda=1.0, random_state=42, n_jobs=-1)
        except Exception:
            pass
        try:
            from lightgbm import LGBMRegressor  # type: ignore
            models["LightGBM"] = LGBMRegressor(n_estimators=min(tree_count, 200), learning_rate=0.05,
                                                num_leaves=15, min_child_samples=20, subsample=0.7,
                                                colsample_bytree=0.7, reg_alpha=2.0, reg_lambda=2.0,
                                                random_state=42, n_jobs=-1, verbosity=-1)
        except Exception:
            pass
        try:
            from cubist import Cubist  # type: ignore
            models["Cubist"] = Cubist(n_rules=100, n_committees=10, unbiased=True, random_state=42, verbose=0)
        except Exception:
            pass
        try:
            from m5py import M5Prime  # type: ignore
            models["M5 Model Tree"] = M5Prime(random_state=42)
        except Exception:
            pass
    availability = pd.DataFrame(
        [{"Model": name, "Status": "Included" if name in models else state} for name, state in statuses.items()]
        + [{"Model": name, "Status": "Included"} for name in models if name not in statuses]
    ).drop_duplicates(subset=["Model"], keep="first")
    return models, availability


def _safe_pearson(observed: np.ndarray, predicted: np.ndarray) -> float:
    if len(observed) < 2 or np.std(observed) == 0 or np.std(predicted) == 0:
        return float("nan")
    return float(pearsonr(observed, predicted)[0])


def calculate_metrics(observed: Any, predicted: Any) -> dict[str, float]:
    actual = np.asarray(observed, dtype=float).ravel()
    estimate = np.asarray(predicted, dtype=float).ravel()
    valid = np.isfinite(actual) & np.isfinite(estimate)
    actual, estimate = actual[valid], estimate[valid]
    if len(actual) < 2:
        return {key: float("nan") for key in ["R2", "RMSE", "MAE", "Pearson r", "KGE", "NSE",
                                              "NRMSE", "WI", "LMI", "Bias"]}
    mean_actual = float(np.mean(actual))
    rmse = float(math.sqrt(mean_squared_error(actual, estimate)))
    mae = float(mean_absolute_error(actual, estimate))
    correlation = _safe_pearson(actual, estimate)
    std_actual = float(np.std(actual, ddof=0))
    std_estimate = float(np.std(estimate, ddof=0))
    ratio_std = std_estimate / std_actual if std_actual else float("nan")
    ratio_mean = float(np.mean(estimate)) / mean_actual if mean_actual else float("nan")
    kge = (1 - math.sqrt((correlation - 1) ** 2 + (ratio_std - 1) ** 2 + (ratio_mean - 1) ** 2)
           if np.isfinite(correlation) else float("nan"))
    squared_error = float(np.sum((actual - estimate) ** 2))
    nse_denom = float(np.sum((actual - mean_actual) ** 2))
    wi_denom = float(np.sum((np.abs(estimate - mean_actual) + np.abs(actual - mean_actual)) ** 2))
    lmi_denom = float(np.sum(np.abs(actual - mean_actual)))
    return {
        "R2": float(r2_score(actual, estimate)), "RMSE": rmse, "MAE": mae, "Pearson r": correlation,
        "KGE": float(kge), "NSE": 1 - squared_error / nse_denom if nse_denom else float("nan"),
        "NRMSE": rmse / (float(np.max(actual) - np.min(actual)) or float("nan")),
        "WI": 1 - squared_error / wi_denom if wi_denom else float("nan"),
        "LMI": 1 - float(np.sum(np.abs(actual - estimate))) / lmi_denom if lmi_denom else float("nan"),
        "Bias": float(np.mean(estimate - actual)),
    }


def train_and_evaluate(split: dict[str, pd.DataFrame | pd.Series], feature_names: list[str],
                       config: RunConfig) -> dict[str, Any]:
    x_train = split["X_train"][feature_names]  # type: ignore[index]
    x_validation = split["X_validation"][feature_names]  # type: ignore[index]
    x_test = split["X_test"][feature_names]  # type: ignore[index]
    y_train = split["y_train"]  # type: ignore[assignment]
    y_validation = split["y_validation"]  # type: ignore[assignment]
    y_test = split["y_test"]  # type: ignore[assignment]
    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=feature_names, index=x_train.index)
    x_validation_scaled = pd.DataFrame(scaler.transform(x_validation), columns=feature_names, index=x_validation.index)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=feature_names, index=x_test.index)
    models, availability = build_models(config.tree_count, config.include_optional_models)
    rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    prediction_table = pd.DataFrame({"Actual DTWL (mbgl)": y_test})
    fitted: dict[str, Any] = {}
    errors: list[str] = []

    for name, model in models.items():
        try:
            model.fit(x_train_scaled, y_train)
            fitted[name] = model
            raw_train = model.predict(x_train_scaled)
            raw_validation = model.predict(x_validation_scaled)
            inference_start = perf_counter()
            raw_test = model.predict(x_test_scaled)
            inference_ms_per_row = (perf_counter() - inference_start) * 1000 / max(len(x_test_scaled), 1)
            correction = float(np.mean(y_train) - np.mean(raw_train))
            corrected_test = raw_test + correction
            metrics = {f"Train {key}": value for key, value in calculate_metrics(y_train, raw_train).items()}
            metrics.update({f"Validation {key}": value for key, value in
                            calculate_metrics(y_validation, raw_validation).items()})
            metrics.update({f"Test {key}": value for key, value in
                            calculate_metrics(y_test, raw_test).items()})
            metrics.update({f"Bias-corrected test {key}": value for key, value in
                            calculate_metrics(y_test, corrected_test).items()})
            rows.append({"Model": name, "Training mean correction": correction,
                         "Test inference (ms/row)": inference_ms_per_row, **metrics})
            prediction_table[f"{name} prediction"] = raw_test
            prediction_table[f"{name} bias-corrected prediction"] = corrected_test
            if hasattr(model, "get_params"):
                try:
                    for parameter, value in model.get_params(deep=False).items():
                        parameter_rows.append({"Model": name, "Parameter": parameter, "Value": str(value)})
                except Exception:
                    pass
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    if not rows:
        raise ValueError("Every selected model failed to train. Check that each input column is numeric "
                         "and has enough variation.")
    metrics_table = pd.DataFrame(rows).sort_values("Bias-corrected test R2", ascending=False).reset_index(drop=True)
    return {
        "metrics": metrics_table, "predictions": prediction_table, "models": fitted, "scaler": scaler,
        "X_train": x_train, "X_test": x_test, "X_train_scaled": x_train_scaled, "X_test_scaled": x_test_scaled,
        "y_train": y_train, "y_test": y_test, "availability": availability, "errors": errors,
        "parameters": pd.DataFrame(parameter_rows),
    }


def train_and_evaluate_deep(split: dict[str, pd.DataFrame | pd.Series], feature_names: list[str],
                            config: RunConfig) -> dict[str, Any]:
    """Train and score the PINN / reservoir-computing / SARIMA / LSTM family.

    Mirrors `train_and_evaluate` exactly (same scaler, same split, same
    bias-correction and metric set) so the two families are directly
    comparable, but keeps its own dict of fitted models and predictions —
    it never touches `MODEL_LIST`, so the existing "six headline models"
    Taylor diagrams and cross-season views are completely unaffected.
    """
    x_train = split["X_train"][feature_names]  # type: ignore[index]
    x_validation = split["X_validation"][feature_names]  # type: ignore[index]
    x_test = split["X_test"][feature_names]  # type: ignore[index]
    y_train = split["y_train"]  # type: ignore[assignment]
    y_validation = split["y_validation"]  # type: ignore[assignment]
    y_test = split["y_test"]  # type: ignore[assignment]
    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=feature_names, index=x_train.index)
    x_validation_scaled = pd.DataFrame(scaler.transform(x_validation), columns=feature_names, index=x_validation.index)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=feature_names, index=x_test.index)

    models, availability = build_deep_models(reservoir_size=config.reservoir_size)
    rows: list[dict[str, Any]] = []
    prediction_table = pd.DataFrame({"Actual DTWL (mbgl)": y_test})
    fitted: dict[str, Any] = {}
    errors: list[str] = []

    for name, model in models.items():
        try:
            model.fit(x_train_scaled, y_train)
            fitted[name] = model
            raw_train = np.asarray(model.predict(x_train_scaled)).ravel()
            raw_validation = np.asarray(model.predict(x_validation_scaled)).ravel()
            inference_start = perf_counter()
            raw_test = np.asarray(model.predict(x_test_scaled)).ravel()
            inference_ms_per_row = (perf_counter() - inference_start) * 1000 / max(len(x_test_scaled), 1)
            correction = float(np.mean(y_train) - np.mean(raw_train))
            corrected_test = raw_test + correction
            metrics = {f"Train {key}": value for key, value in calculate_metrics(y_train, raw_train).items()}
            metrics.update({f"Validation {key}": value for key, value in
                            calculate_metrics(y_validation, raw_validation).items()})
            metrics.update({f"Test {key}": value for key, value in
                            calculate_metrics(y_test, raw_test).items()})
            metrics.update({f"Bias-corrected test {key}": value for key, value in
                            calculate_metrics(y_test, corrected_test).items()})
            rows.append({"Model": name, "Training mean correction": correction,
                         "Test inference (ms/row)": inference_ms_per_row, **metrics})
            prediction_table[f"{name} prediction"] = raw_test
            prediction_table[f"{name} bias-corrected prediction"] = corrected_test
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    metrics_table = (
        pd.DataFrame(rows).sort_values("Bias-corrected test R2", ascending=False).reset_index(drop=True)
        if rows else pd.DataFrame()
    )
    return {
        "metrics": metrics_table, "predictions": prediction_table, "models": fitted, "scaler": scaler,
        "y_test": y_test, "availability": availability, "errors": errors,
    }


def leave_one_feature_out(model: Any, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame,
                          y_test: pd.Series, baseline_prediction: np.ndarray) -> pd.DataFrame:
    baseline = _safe_pearson(np.asarray(y_test), baseline_prediction)
    records: list[dict[str, Any]] = []
    for feature in x_train.columns:
        try:
            candidate = clone(model)
            candidate.fit(x_train.drop(columns=feature), y_train)
            predicted = candidate.predict(x_test.drop(columns=feature))
            correlation = _safe_pearson(np.asarray(y_test), predicted)
            records.append({"Feature": feature, "Baseline Pearson r": baseline, "Pearson r after removal": correlation,
                            "Correlation drop": baseline - correlation})
        except Exception as exc:
            records.append({"Feature": feature, "Baseline Pearson r": baseline, "Pearson r after removal": np.nan,
                            "Correlation drop": np.nan, "Note": f"Could not refit: {type(exc).__name__}"})
    return pd.DataFrame(records).sort_values("Correlation drop", ascending=False,
                                             na_position="last").reset_index(drop=True)


def recursive_forecast(model: Any, scaler: StandardScaler, x_reference: pd.DataFrame, target_history: pd.Series,
                       start_year: int, steps: int) -> pd.DataFrame:
    """Forecast from the last feature row, updating target-history features when they are selected."""
    if steps < 1:
        return pd.DataFrame(columns=["Year", "Scenario forecast DTWL (mbgl)"])
    current = x_reference.iloc[-1].astype(float).copy()
    history = list(pd.Series(target_history).dropna().astype(float).tail(5))
    forecasts: list[float] = []
    for _ in range(steps):
        for lag in range(1, 5):
            field = f"{TARGET}_lag{lag}"
            if field in current.index and len(history) >= lag:
                current[field] = history[-lag]
        if f"{TARGET}_roll3" in current.index and history:
            current[f"{TARGET}_roll3"] = float(np.mean(history[-3:]))
        if f"{TARGET}_roll5" in current.index and history:
            current[f"{TARGET}_roll5"] = float(np.mean(history[-5:]))
        if f"{TARGET}_roll3_std" in current.index and len(history) > 1:
            current[f"{TARGET}_roll3_std"] = float(np.std(history[-3:], ddof=1))
        if f"{TARGET}_trend" in current.index and len(history) >= 3:
            current[f"{TARGET}_trend"] = history[-1] - history[-3]
        scaled = pd.DataFrame(scaler.transform(pd.DataFrame([current], columns=x_reference.columns)),
                              columns=x_reference.columns)
        prediction = float(np.asarray(model.predict(scaled)).ravel()[0])
        forecasts.append(prediction)
        history.append(prediction)
    return pd.DataFrame({"Year": range(start_year + 1, start_year + steps + 1),
                         "Scenario forecast DTWL (mbgl)": forecasts})


def make_full_timeline(lagged: pd.DataFrame, features: pd.DataFrame, selected_features: list[str],
                       model_result: dict[str, Any]) -> pd.DataFrame:
    """Create the full season timeline used by the notebook's annual and rainfall plots."""
    x_all = features[selected_features].copy()
    scaled = pd.DataFrame(model_result["scaler"].transform(x_all), columns=selected_features, index=x_all.index)
    metadata_columns = [column for column in ["YEAR", "VILLAGE", "Season", "Rainfall", TARGET]
                        if column in lagged.columns]
    timeline = lagged.loc[x_all.index, metadata_columns].copy().rename(columns={
        "YEAR": "Year", "VILLAGE": "Village", TARGET: "Observed DTWL (mbgl)",
    })
    corrections = model_result["metrics"].set_index("Model")["Training mean correction"].to_dict()
    for name, model in model_result["models"].items():
        try:
            raw_prediction = np.asarray(model.predict(scaled)).ravel()
            timeline[f"{name} prediction"] = raw_prediction
            timeline[f"{name} bias-corrected prediction"] = raw_prediction + float(corrections.get(name, 0.0))
        except Exception:
            continue
    return timeline.sort_values(["Year", "Village"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Notebook helper functions (moved above run_season_pipeline so it can use them).
# --------------------------------------------------------------------------- #
def filter_models(predictions: dict[str, Any], wanted: list[str]) -> dict[str, Any]:
    """Case/whitespace-tolerant model-name filter, matching the notebook helper."""
    out: dict[str, Any] = {}
    for target in wanted:
        key = target.strip().lower()
        for name, values in predictions.items():
            if name.strip().lower() == key:
                out[name] = values
                break
    return out


def season_predictions_dict(result: dict[str, Any],
                            bias_corrected: bool = True) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Convert a dashboard result into (obs, {'Model': np.array}).

    The dict is compatible with the notebook's `plot_taylor_latex_style` and
    `plot_observed_vs_predicted_on_ax` helpers.
    """
    predictions = result["predictions"]
    observed = predictions["Actual DTWL (mbgl)"].to_numpy(dtype=float)
    suffix = " bias-corrected prediction" if bias_corrected else " prediction"
    preds: dict[str, np.ndarray] = {}
    for column in predictions.columns:
        if not column.endswith(suffix):
            continue
        if not bias_corrected and column.endswith(" bias-corrected prediction"):
            continue
        model = column[: -len(suffix)]
        preds[model] = predictions[column].to_numpy(dtype=float)
    return observed, preds


def run_season_pipeline(data: pd.DataFrame, season: str, config: RunConfig,
                        progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Execute every dashboard stage for one season and return all labelled outputs."""
    def update(message: str) -> None:
        if progress:
            progress(message)

    update("Filtering the selected season")
    seasonal = data.loc[data["Season"] == season].copy()
    if seasonal.empty:
        raise ValueError(f"The uploaded workbook does not contain '{season}' records.")

    update("Creating derived hydrology and vegetation features")
    engineered, derived_features = engineer_features(seasonal)

    update("Creating lag, rolling, and trend features")
    lagged, feature_groups, removed_rows, low_data_rescue_used = create_lag_features(engineered)

    update("Preparing the temporal train, validation, and test data")
    features, target, years, all_feature_names = prepare_modelling_data(lagged)
    try:
        split = temporal_split(features, target, years, config)
    except ValueError:
        # The strict (notebook-faithful) drop-NaN pass left too few rows for
        # the configured split. Retry once, this time imputing incomplete
        # engineered (lag/rolling/trend) values within each village instead
        # of dropping those rows outright — the raw measurements and the
        # target itself are never touched. Only kicks in when the dataset
        # is genuinely too small on its own; otherwise nothing changes.
        update("Dataset is small — recovering rows by imputing missing lag values")
        lagged, feature_groups, removed_rows, low_data_rescue_used = create_lag_features(
            engineered, rescue_low_data=True,
        )
        if not low_data_rescue_used:
            raise
        features, target, years, all_feature_names = prepare_modelling_data(lagged)
        split = temporal_split(features, target, years, config)

    update("Selecting the strongest features")
    feature_ranking = select_features(split["X_train"], split["y_train"],
                                      config.selected_feature_count)  # type: ignore[arg-type]
    selected_features = feature_ranking.loc[feature_ranking["Selected"], "Feature"].tolist()

    update("Training the model suite and calculating performance metrics")
    model_result = train_and_evaluate(split, selected_features, config)
    best_model_name = str(model_result["metrics"].iloc[0]["Model"])
    best_model = model_result["models"][best_model_name]

    update("Training the deep-learning and physics-informed model family")
    deep_result: dict[str, Any] | None = None
    if config.include_deep_models:
        deep_result = train_and_evaluate_deep(split, selected_features, config)

    update("Creating complete season timelines for comparative analysis")
    full_timeline = make_full_timeline(lagged, features, selected_features, model_result)

    update("Calculating feature sensitivity for the best model")
    lofo = pd.DataFrame()
    if config.run_lofo:
        lofo = leave_one_feature_out(
            best_model,
            model_result["X_train_scaled"], model_result["y_train"],
            model_result["X_test_scaled"], model_result["y_test"],
            model_result["predictions"][f"{best_model_name} prediction"].to_numpy(),
        )

    update("Generating the scenario forecast")
    forecast = recursive_forecast(
        best_model, model_result["scaler"], model_result["X_test"], target,
        int(years.max()), config.forecast_steps,
    )

    test_metadata = lagged.loc[model_result["X_test"].index,
                               ["YEAR", "VILLAGE", "Season", TARGET]].copy()
    test_metadata = test_metadata.rename(
        columns={"YEAR": "Year", "VILLAGE": "Village", TARGET: "Actual DTWL (mbgl)"}
    )
    predictions = test_metadata.join(
        model_result["predictions"].drop(columns=["Actual DTWL (mbgl)"], errors="ignore")
    )

    deep_predictions = pd.DataFrame()
    deep_best_model_name: str | None = None
    if deep_result is not None and not deep_result["metrics"].empty:
        deep_predictions = test_metadata.join(
            deep_result["predictions"].drop(columns=["Actual DTWL (mbgl)"], errors="ignore")
        )
        deep_best_model_name = str(deep_result["metrics"].iloc[0]["Model"])

    # Extra artefacts for the notebook-style 2×2 figures.
    x_full = features[selected_features].copy()
    x_full_scaled = pd.DataFrame(
        model_result["scaler"].transform(x_full),
        columns=selected_features, index=x_full.index,
    )
    df_lag_full = lagged.loc[x_full.index].copy()
    y_full = lagged.loc[x_full.index, TARGET].copy()

    _, model_predictions = season_predictions_dict(
        {"predictions": predictions}, bias_corrected=True
    )

    return {
        "season": season,
        "raw_rows": len(seasonal),
        "lagged_rows": len(lagged),
        "rows_removed": removed_rows,
        "derived_features": derived_features,
        "feature_groups": feature_groups,
        "lagged_data": lagged,
        "all_feature_names": all_feature_names,
        "feature_ranking": feature_ranking,
        "selected_features": selected_features,
        "split_counts": split["counts"],
        "metrics": model_result["metrics"],
        "predictions": predictions,
        "availability": model_result["availability"],
        "model_errors": model_result["errors"],
        "lofo": lofo,
        "forecast": forecast,
        "best_model": best_model_name,
        "parameters": model_result["parameters"],
        "full_timeline": full_timeline,
        "X_full_scaled": x_full_scaled,
        "y_full": y_full,
        "df_lag_full": df_lag_full,
        "fitted_models": model_result["models"],
        "feature_cols": selected_features,
        "model_predictions": model_predictions,
        "low_data_rescue_used": low_data_rescue_used,
        "deep_metrics": deep_result["metrics"] if deep_result is not None else pd.DataFrame(),
        "deep_predictions": deep_predictions,
        "deep_availability": deep_result["availability"] if deep_result is not None else pd.DataFrame(),
        "deep_model_errors": deep_result["errors"] if deep_result is not None else [],
        "deep_best_model": deep_best_model_name,
        "deep_fitted_models": deep_result["models"] if deep_result is not None else {},
    }


def season_model_metrics(season_results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Return the notebook-style long table of model metrics across seasons."""
    records: list[dict[str, Any]] = []
    fields = ["R2", "RMSE", "MAE", "Pearson r", "KGE", "NSE", "NRMSE", "WI", "LMI", "Bias"]
    for season, result in season_results.items():
        for _, row in result["metrics"].iterrows():
            record = {"Season": season, "Model": row["Model"],
                      "Test inference (ms/row)": row.get("Test inference (ms/row)")}
            for field in fields:
                record[f"Test {field}"] = row.get(f"Test {field}")
                record[f"Bias-corrected test {field}"] = row.get(f"Bias-corrected test {field}")
            record["Train–test R2 gap"] = row.get("Train R2", np.nan) - row.get("Test R2", np.nan)
            records.append(record)
    return pd.DataFrame(records)


def residual_long_table(season_results: dict[str, dict[str, Any]],
                        bias_corrected: bool = True) -> pd.DataFrame:
    """Build a long residual table for violin/box diagnostics across all seasons and models."""
    records: list[dict[str, Any]] = []
    suffix = " bias-corrected prediction" if bias_corrected else " prediction"
    for season, result in season_results.items():
        predictions = result["predictions"]
        actual = predictions["Actual DTWL (mbgl)"].to_numpy(dtype=float)
        for column in predictions.columns:
            if not column.endswith(suffix):
                continue
            if not bias_corrected and column.endswith(" bias-corrected prediction"):
                continue
            model = column.removesuffix(suffix)
            values = predictions[column].to_numpy(dtype=float)
            for observation, prediction in zip(actual, values):
                records.append({"Season": season, "Model": model,
                                "Residual (prediction − observation)": prediction - observation})
    return pd.DataFrame(records)


def taylor_statistics(season_results: dict[str, dict[str, Any]], bias_corrected: bool = True) -> pd.DataFrame:
    """Calculate the statistics needed for an interactive Taylor-style polar diagram."""
    records: list[dict[str, Any]] = []
    suffix = " bias-corrected prediction" if bias_corrected else " prediction"
    for season, result in season_results.items():
        predictions = result["predictions"]
        observed = predictions["Actual DTWL (mbgl)"].to_numpy(dtype=float)
        observed_std = float(np.std(observed, ddof=0))
        for column in predictions.columns:
            if not column.endswith(suffix):
                continue
            if not bias_corrected and column.endswith(" bias-corrected prediction"):
                continue
            estimate = predictions[column].to_numpy(dtype=float)
            correlation = _safe_pearson(observed, estimate)
            if not np.isfinite(correlation) or not observed_std:
                continue
            std_ratio = float(np.std(estimate, ddof=0) / observed_std)
            centered_rmsd = float(np.sqrt(np.mean(((estimate - np.mean(estimate)) -
                                                   (observed - np.mean(observed))) ** 2)))
            records.append({
                "Season": season, "Model": column.removesuffix(suffix), "Correlation": correlation,
                "Angle (degrees)": float(np.degrees(np.arccos(np.clip(correlation, -1, 1)))),
                "Standard deviation ratio": std_ratio, "Centered RMSD (mbgl)": centered_rmsd,
            })
    return pd.DataFrame(records)


def taylor_statistics_matrix(season_results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Reuse taylor_statistics but expose the columns the matplotlib Taylor helper needs.

    `Std dev (model)` is the ratio σ_model / σ_obs (dimensionless) because the
    Taylor diagram is drawn with observed_std = 1.0.
    """
    df = taylor_statistics(season_results, bias_corrected=True).copy()
    if df.empty:
        return df
    df["Std dev (model)"] = df["Standard deviation ratio"]
    df["Std dev (obs)"] = 1.0
    return df


def acf_pacf_table(values: Any, max_lag: int = 10) -> pd.DataFrame:
    """Compute ACF and an autoregression-based PACF without requiring statsmodels."""
    series = np.asarray(values, dtype=float).ravel()
    series = series[np.isfinite(series)]
    limit = min(max_lag, max(0, len(series) - 2))
    rows: list[dict[str, float]] = []
    for lag in range(1, limit + 1):
        acf_value = _safe_pearson(series[:-lag], series[lag:])
        y = series[lag:]
        design = np.column_stack([series[lag - step: len(series) - step] for step in range(1, lag + 1)])
        try:
            pacf_value = float(np.linalg.lstsq(design, y, rcond=None)[0][-1])
        except np.linalg.LinAlgError:
            pacf_value = float("nan")
        rows.append({"Lag": lag, "ACF": acf_value, "PACF": pacf_value})
    return pd.DataFrame(rows)


def rainfall_cross_correlation(rainfall: Any, dtwl: Any, max_lag: int = 8) -> pd.DataFrame:
    """Correlation between rainfall and DTWL at positive and negative annual lags."""
    rain = np.asarray(rainfall, dtype=float).ravel()
    target = np.asarray(dtwl, dtype=float).ravel()
    length = min(len(rain), len(target))
    rain, target = rain[:length], target[:length]
    rows: list[dict[str, float]] = []
    for lag in range(-min(max_lag, length - 2), min(max_lag, length - 2) + 1):
        if lag < 0:
            x, y = rain[-lag:], target[:lag]
        elif lag > 0:
            x, y = rain[:-lag], target[lag:]
        else:
            x, y = rain, target
        rows.append({"Lag (years; + means rainfall leads DTWL)": lag, "Correlation": _safe_pearson(x, y)})
    return pd.DataFrame(rows)


def stationarity_summary(values: Any) -> dict[str, Any]:
    """Run the notebook's Augmented Dickey–Fuller check when statsmodels is available."""
    series = np.asarray(values, dtype=float).ravel()
    series = series[np.isfinite(series)]
    if len(series) < 8:
        return {"Status": "Not run", "Reason": "At least eight complete observations are needed."}
    try:
        from statsmodels.tsa.stattools import adfuller  # type: ignore
        statistic, p_value, used_lag, observations, critical_values, _ = adfuller(series, autolag="AIC")
        return {
            "Status": "Completed", "ADF statistic": float(statistic), "p-value": float(p_value),
            "Used lags": int(used_lag), "Observations": int(observations),
            "5% critical value": float(critical_values.get("5%", np.nan)),
            "Interpretation": ("Evidence of stationarity (p < 0.05)" if p_value < 0.05
                               else "No stationarity evidence at 5% (p ≥ 0.05)"),
        }
    except Exception as exc:
        return {"Status": "Not run", "Reason": f"statsmodels is unavailable: {type(exc).__name__}"}


def build_excel_export(workbook_report: pd.DataFrame, validation_report: pd.DataFrame,
                       season_results: dict[str, dict[str, Any]]) -> bytes:
    """Create one downloadable Excel workbook containing every tabular dashboard result."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        workbook_report.to_excel(writer, sheet_name="Upload report", index=False)
        validation_report.to_excel(writer, sheet_name="Data quality", index=False)
        season_model_metrics(season_results).to_excel(writer, sheet_name="All-season metrics", index=False)
        residual_long_table(season_results).to_excel(writer, sheet_name="All residuals", index=False)
        taylor_statistics(season_results).to_excel(writer, sheet_name="Taylor statistics", index=False)
        summary_rows = []
        for season, result in season_results.items():
            top = result["metrics"].iloc[0]
            summary_rows.append({
                "Season": season, "Input rows": result["raw_rows"], "Modelling rows": result["lagged_rows"],
                "Best model": result["best_model"], "Bias-corrected test R2": top["Bias-corrected test R2"],
                "Bias-corrected test RMSE": top["Bias-corrected test RMSE"],
            })
            token = season.replace("-", "")[:12]
            result["feature_ranking"].to_excel(writer, sheet_name=f"{token} features", index=False)
            result["metrics"].to_excel(writer, sheet_name=f"{token} metrics", index=False)
            result["parameters"].to_excel(writer, sheet_name=f"{token} parameters", index=False)
            result["predictions"].to_excel(writer, sheet_name=f"{token} predictions", index=False)
            result["full_timeline"].to_excel(writer, sheet_name=f"{token} timeline", index=False)
            result["forecast"].to_excel(writer, sheet_name=f"{token} forecast", index=False)
            if not result["lofo"].empty:
                result["lofo"].to_excel(writer, sheet_name=f"{token} sensitivity", index=False)
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Season summary", index=False)
    return output.getvalue()


# --------------------------------------------------------------------------- #
# Wavelet (CWT) analysis — aggregated seasonal DTWL series
# --------------------------------------------------------------------------- #
def build_aggregated_series(seasonal_data: pd.DataFrame,
                            season_order: list[str] | None = None) -> pd.DataFrame:
    """Quarterly aggregated (median across villages) DTWL series.

    Positions each season inside the year so that Pre-Monsoon = .00,
    Monsoon = .25, Post-Monsoon = .50, Non-Monsoon = .75.
    """
    season_order = season_order or SEASON_ORDER
    offsets = {s: i / len(season_order) for i, s in enumerate(season_order)}
    rows = []
    for season in season_order:
        block = seasonal_data.loc[seasonal_data["Season"] == season]
        if block.empty:
            continue
        grouped = block.groupby("YEAR")[TARGET].median().reset_index()
        grouped["t"] = grouped["YEAR"] + offsets.get(season, 0.0)
        grouped["Season"] = season
        rows.append(grouped[["t", "YEAR", "Season", TARGET]])
    if not rows:
        return pd.DataFrame(columns=["t", "YEAR", "Season", TARGET])
    combined = pd.concat(rows, ignore_index=True).sort_values("t").reset_index(drop=True)
    return combined.dropna(subset=[TARGET])


def wavelet_power_spectrum(t: np.ndarray, signal: np.ndarray,
                           dt: float = 0.25,
                           mother_name: str = "Morlet",
                           mother_param: float = 6.0,
                           dj: float = 1.0 / 12.0,
                           detrend: bool = True) -> dict[str, Any]:
    """Continuous wavelet transform of a 1-D signal.

    Returns a dict with everything the dashboard needs to plot the scalogram
    and global spectrum: wave, scales, period, coi, power, significance.
    """
    import pycwt as wavelet  # local import so the rest of the pipeline keeps working without it

    series = np.asarray(signal, dtype=float)
    series = series[np.isfinite(series)]
    if len(series) < 16:
        raise ValueError("At least 16 complete observations are needed for a CWT.")

    if detrend:
        x_idx = np.arange(len(series))
        trend = np.polyval(np.polyfit(x_idx, series, 1), x_idx)
        series = series - trend
    series = (series - series.mean()) / (series.std() or 1.0)

    if mother_name.lower().startswith("morlet"):
        mother = wavelet.Morlet(mother_param)
    elif mother_name.lower().startswith("paul"):
        mother = wavelet.Paul(mother_param)
    else:
        mother = wavelet.DOG(mother_param)

    s0 = 2 * dt
    J = int(np.log2(len(series) * dt / s0) / dj)
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(series, dt, dj, s0, J, mother)

    power = np.abs(wave) ** 2
    power_norm = power / (np.var(series) or 1.0)
    period = 1.0 / freqs

    # 95 % significance against a red-noise (AR1) background
    try:
        alpha, _, _ = wavelet.ar1(series)
        signif, fft_theor = wavelet.significance(
            1.0, dt, scales, 0, alpha, significance_level=0.95,
            wavelet=mother,
        )
        sig95 = np.ones([1, len(scales)]) * signif[:, None]
        sig95 = power / sig95
    except Exception:
        sig95 = np.zeros_like(power)
        fft_theor = np.zeros_like(scales)

    # Global wavelet spectrum (time-averaged power, scaled by scale)
    global_ws = power.sum(axis=1) / len(series)
    dof = 2 * np.sqrt(1 - (np.abs(wave) ** 2 / (np.abs(wave) ** 2).max()) ** 2)
    global_signif = None
    try:
        dof_min = dof.min(axis=1)
        global_signif = wavelet.significance(
            1.0, dt, scales, 1, alpha, significance_level=0.95,
            wavelet=mother, dof=dof_min,
        )[0]
        global_signif = global_signif * (np.var(series) or 1.0)
    except Exception:
        pass

    return {
        "t": np.asarray(t, dtype=float),
        "series": series,
        "wave": wave,
        "scales": scales,
        "freqs": freqs,
        "period": period,
        "coi": coi,
        "power": power,
        "power_norm": power_norm,
        "sig95": sig95,
        "global_ws": global_ws,
        "global_signif": global_signif,
        "fft_theor": fft_theor,
        "dt": dt,
    }


# --------------------------------------------------------------------------- #
# Notebook 2×2 figure helpers
# --------------------------------------------------------------------------- #
MODEL_LIST = [
    "Cubist",
    "Random Forest",
    "XGBoost",
    "CatBoost",
    "AdaBoost + CART",
    "Extra Trees",
]

SEASON_INDEX = {"Pre-Monsoon": 1, "Monsoon": 2, "Post-Monsoon": 3, "Non-Monsoon": 4}


def fractional_year(year: int, season: str) -> float:
    """Fractional year with the notebook's convention (Pre-Monsoon = x.00)."""
    return float(year) + (SEASON_INDEX.get(season, 1) - 1) / 4.0


def recursive_forecast_constant(model: Any, x_init: pd.DataFrame,
                                feature_names: list[str],
                                n_steps: int = 6) -> np.ndarray:
    """Notebook-faithful recursive forecast: hold exogenous features fixed and
    roll the DTWL lags forward one step at a time.

    `x_init` is expected to be a single-row DataFrame in the *scaled* feature
    space used to fit the models (that's what the notebook passes).
    """
    x_forecast = x_init.copy()
    forecasts: list[float] = []
    lag_columns = [c for c in feature_names if f"{TARGET}_lag" in c]
    lag_columns_sorted = sorted(
        lag_columns,
        key=lambda x: int(x.split("lag")[-1]) if x.split("lag")[-1].isdigit() else 0,
    )
    for step in range(n_steps):
        pred = float(np.asarray(model.predict(x_forecast.values.reshape(1, -1))).ravel()[0])
        forecasts.append(pred)
        if step < n_steps - 1:
            x_new = x_forecast.copy()
            if lag_columns_sorted:
                for i in range(len(lag_columns_sorted) - 1, 0, -1):
                    x_new[lag_columns_sorted[i]] = x_new[lag_columns_sorted[i - 1]]
                x_new[lag_columns_sorted[0]] = pred
            x_forecast = x_new
    return np.asarray(forecasts)


def build_season_forecast_table(model: Any,
                                scaled_full: pd.DataFrame,
                                feature_names: list[str],
                                meta_last_row: pd.Series,
                                n_steps: int = 3) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the notebook-shaped forecast table for a single model.

    Returns (timeline_table, predictions_array). The timeline table has columns
    YEAR, Season, Step, Frac. The predictions array is length n_steps.
    """
    last_row = scaled_full.iloc[-1:].copy()
    preds = recursive_forecast_constant(model, last_row, feature_names, n_steps=n_steps)

    current_year = int(meta_last_row["YEAR"])
    current_season = str(meta_last_row["Season"])

    rows = []
    for step in range(n_steps):
        forecast_year = current_year + (step + 1)
        rows.append({
            "YEAR": forecast_year,
            "Season": current_season,
            "Step": step + 1,
            "Frac": fractional_year(forecast_year, current_season),
        })
    table = pd.DataFrame(rows)
    return table, preds