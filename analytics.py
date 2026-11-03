"""Reusable dataset, prediction, drift, and lock-time analytics."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import BytesIO, StringIO
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_utils import (
    FEATURE_COLUMNS,
    GLOBAL_SAFETY_INHIBIT_REASON,
    ILLUSTRATIVE_JITTER_USL_PS,
    ILLUSTRATIVE_SCREEN_THRESHOLD,
    LOCKTIME_FIXED_WAIT_US,
    LOCKTIME_MARGIN_US,
    TARGET_COLUMN,
    adaptive_wait_us,
    decide_jitter_test,
    _safety_inhibit_reason,
)
from train import create_model


def compare_regression_models(
    partitions: Mapping[str, pd.DataFrame],
    saved_model: Any,
    saved_metrics: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Compare training-only baselines with a read-only saved model on one test split."""
    required_partitions = ("train", "calibration", "test")
    missing_partitions = [
        name for name in required_partitions if name not in partitions
    ]
    if missing_partitions:
        raise ValueError(
            "Missing required partition(s): " + ", ".join(missing_partitions)
        )

    clean_partitions: dict[str, pd.DataFrame] = {}
    seen_die_ids: set[Any] = set()
    for name in required_partitions:
        frame = partitions[name]
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(f"The {name} partition must be a non-empty DataFrame.")
        if "split" in frame.columns:
            labels = frame["split"].dropna().astype(str).str.lower().unique()
            if len(labels) != 1 or labels[0] != name:
                raise ValueError(
                    f"The {name} partition contains inconsistent split labels."
                )
        required_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
        missing_columns = [
            column for column in required_columns if column not in frame.columns
        ]
        if missing_columns:
            raise ValueError(
                f"The {name} partition is missing required columns: "
                + ", ".join(missing_columns)
            )

        model_frame = frame.loc[:, list(FEATURE_COLUMNS)].apply(
            pd.to_numeric, errors="coerce"
        )
        targets = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
        x_values = model_frame.to_numpy(dtype=float)
        y_values = targets.to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
            raise ValueError(f"The {name} partition contains non-finite model data.")
        if model_frame["lock_time_us"].lt(0).any():
            raise ValueError(f"The {name} partition contains negative lock times.")
        if "die_id" in frame.columns:
            if frame["die_id"].duplicated().any():
                raise ValueError(f"The {name} partition contains duplicate die IDs.")
            ids = set(frame["die_id"].tolist())
            if seen_die_ids.intersection(ids):
                raise ValueError("Die IDs overlap across the supplied partitions.")
            seen_die_ids.update(ids)

        clean_frame = frame.copy(deep=True)
        clean_frame.loc[:, list(FEATURE_COLUMNS)] = model_frame
        clean_frame[TARGET_COLUMN] = targets
        clean_partitions[name] = clean_frame

    train = clean_partitions["train"]
    test = clean_partitions["test"]
    x_train = train.loc[:, list(FEATURE_COLUMNS)]
    y_train = train[TARGET_COLUMN].astype(float)
    x_test = test.loc[:, list(FEATURE_COLUMNS)]
    y_test = test[TARGET_COLUMN].astype(float).to_numpy()

    fitted_models = (
        ("DummyRegressor (mean)", DummyRegressor(strategy="mean")),
        ("LinearRegression", LinearRegression()),
        ("HistGradientBoostingRegressor (comparison fit)", create_model()),
    )
    rows: list[dict[str, Any]] = []
    for model_name, estimator in fitted_models:
        estimator.fit(x_train, y_train)
        predictions = np.asarray(estimator.predict(x_test), dtype=float).reshape(-1)
        if predictions.size != len(y_test) or not np.isfinite(predictions).all():
            raise ValueError(
                f"{model_name} returned invalid or misaligned test predictions."
            )
        rows.append(
            {
                "Model": model_name,
                "MAE (ps)": float(mean_absolute_error(y_test, predictions)),
                "RMSE (ps)": float(np.sqrt(mean_squared_error(y_test, predictions))),
                "R²": float(r2_score(y_test, predictions)),
                "Training rows": len(x_train),
                "Test rows": len(x_test),
                "Evaluation": "Fresh estimator fit on train only",
            }
        )

    saved_predictions = np.asarray(
        saved_model.predict(x_test), dtype=float
    ).reshape(-1)
    if (
        saved_predictions.size != len(y_test)
        or not np.isfinite(saved_predictions).all()
    ):
        raise ValueError(
            "The saved model returned invalid or misaligned test predictions."
        )
    saved_row: dict[str, Any] = {
        "Model": "Existing saved HistGradientBoostingRegressor",
        "MAE (ps)": float(mean_absolute_error(y_test, saved_predictions)),
        "RMSE (ps)": float(np.sqrt(mean_squared_error(y_test, saved_predictions))),
        "R²": float(r2_score(y_test, saved_predictions)),
        "Training rows": (
            int(saved_metrics["train_rows"])
            if saved_metrics and "train_rows" in saved_metrics
            else len(train)
        ),
        "Test rows": len(x_test),
        "Evaluation": "Saved model; read-only inference",
    }
    if saved_metrics:
        saved_row["Previously reported MAE (ps)"] = float(
            saved_metrics["test_mae_ps"]
        )
        saved_row["Previously reported RMSE (ps)"] = float(
            saved_metrics["test_rmse_ps"]
        )
        saved_row["Previously reported R²"] = float(saved_metrics["test_r2"])
    rows.append(saved_row)
    return pd.DataFrame(rows)


def read_csv_upload(contents: bytes) -> pd.DataFrame:
    """Parse CSV bytes in memory and surface malformed uploads as clear errors."""
    if not contents:
        raise ValueError("The uploaded file is empty.")
    try:
        text = contents.decode("utf-8-sig")
        records = csv.reader(StringIO(text))
        header = next(records, None)
        if not header or not any(name.strip() for name in header):
            raise ValueError("The uploaded CSV has no usable header row.")
        if len(set(header)) != len(header):
            raise ValueError("The uploaded CSV contains duplicate column names.")
        expected_fields = len(header)
        for line_number, row in enumerate(records, start=2):
            if not row:
                continue
            if len(row) != expected_fields:
                raise ValueError(
                    f"Malformed CSV row {line_number}: expected {expected_fields} "
                    f"fields, found {len(row)}."
                )
        frame = pd.read_csv(BytesIO(contents))
    except (csv.Error, pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"Could not parse the uploaded CSV: {error}") from error
    if frame.empty:
        raise ValueError("The uploaded CSV contains no data rows.")
    return frame


def safe_csv_download(frame: pd.DataFrame) -> bytes:
    """Serialize a CSV while neutralizing formula-leading text cells."""
    safe_frame = frame.copy(deep=True)
    formula_prefixes = ("=", "+", "-", "@")
    text_columns = safe_frame.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        safe_frame[column] = safe_frame[column].map(
            lambda value: (
                f"'{value}"
                if isinstance(value, str)
                and value.lstrip().startswith(formula_prefixes)
                else value
            )
        )
    return safe_frame.to_csv(index=False).encode("utf-8")


def validate_dataset(
    frame: pd.DataFrame,
    feature_ranges: Mapping[str, Sequence[float]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return row-level validation results without mutating the caller's frame."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Uploaded data must be provided as a pandas DataFrame.")
    if frame.empty:
        raise ValueError("The uploaded CSV contains no data rows.")

    result = frame.copy(deep=True).reset_index(drop=True)
    original_columns = list(frame.columns)
    missing_columns = [name for name in FEATURE_COLUMNS if name not in original_columns]
    invalid_type_counts: dict[str, int] = {}
    missing_value_counts: dict[str, int] = {}
    row_reasons: list[list[str]] = [[] for _ in range(len(result))]
    feature_valid = np.ones(len(result), dtype=bool)
    out_of_range = np.zeros(len(result), dtype=bool)

    for name in FEATURE_COLUMNS:
        if name not in result.columns:
            result[name] = np.nan
            missing_value_counts[name] = len(result)
            invalid_type_counts[name] = 0
            feature_valid[:] = False
            for reasons in row_reasons:
                reasons.append(f"missing column: {name}")
            continue

        original = result[name]
        missing_mask = original.isna() | original.astype("string").str.strip().eq("")
        numeric = pd.to_numeric(original, errors="coerce")
        numeric_array = numeric.to_numpy(dtype=float, na_value=np.nan)
        invalid_type = ~missing_mask.to_numpy() & ~np.isfinite(numeric_array)
        missing_value_counts[name] = int(missing_mask.sum())
        invalid_type_counts[name] = int(invalid_type.sum())
        invalid_value = missing_mask.to_numpy() | ~np.isfinite(numeric_array)
        feature_valid &= ~invalid_value
        result[name] = numeric_array
        for index in np.flatnonzero(missing_mask.to_numpy()):
            row_reasons[index].append(f"missing value: {name}")
        for index in np.flatnonzero(invalid_type):
            row_reasons[index].append(f"invalid numeric value: {name}")

        if name in feature_ranges:
            lower, upper = map(float, feature_ranges[name])
            valid_numeric = np.isfinite(numeric_array)
            out_of_range |= valid_numeric & (
                (numeric_array < lower) | (numeric_array > upper)
            )

    if "lock_time_us" in result.columns:
        negative_lock = (
            pd.to_numeric(result["lock_time_us"], errors="coerce").lt(0).fillna(False)
        ).to_numpy()
        feature_valid &= ~negative_lock
        for index in np.flatnonzero(negative_lock):
            row_reasons[index].append("negative lock_time_us")

    if "die_id" in result.columns:
        duplicate_mask = result["die_id"].duplicated(keep=False).to_numpy()
        duplicate_rule = "duplicate die_id (all occurrences flagged)"
    else:
        duplicate_mask = result.duplicated(keep=False).to_numpy()
        duplicate_rule = "duplicate rows (all occurrences flagged)"
    for index in np.flatnonzero(duplicate_mask):
        row_reasons[index].append(duplicate_rule)
    feature_valid &= ~duplicate_mask

    screen_flagged = np.zeros(len(result), dtype=bool)
    if "leak_screen_norm" in result.columns:
        leak = pd.to_numeric(result["leak_screen_norm"], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        screen_flagged = np.isfinite(leak) & (
            leak >= ILLUSTRATIVE_SCREEN_THRESHOLD
        )

    if TARGET_COLUMN in result.columns:
        raw_actual = result[TARGET_COLUMN]
        actual_missing = raw_actual.isna() | raw_actual.astype("string").str.strip().eq("")
        actual_numeric = pd.to_numeric(raw_actual, errors="coerce")
        actual_values = actual_numeric.to_numpy(dtype=float, na_value=np.nan)
        result[TARGET_COLUMN] = actual_values
        missing_value_counts[TARGET_COLUMN] = int(actual_missing.sum())
        invalid_type_counts[TARGET_COLUMN] = int(
            (~actual_missing.to_numpy() & ~np.isfinite(actual_values)).sum()
        )

    result["feature_valid"] = feature_valid
    result["out_of_training_range"] = out_of_range
    result["screen_flagged"] = screen_flagged
    result["validation_reasons"] = [
        "; ".join(reasons) if reasons else "all required model features valid"
        for reasons in row_reasons
    ]
    statuses = np.full(len(result), "VALID", dtype=object)
    statuses[screen_flagged] = "SCREEN FLAGGED"
    statuses[out_of_range] = "OUT OF TRAINING RANGE"
    statuses[~feature_valid] = "INVALID"
    result["validation_status"] = statuses

    report = {
        "rows": int(len(result)),
        "columns": int(len(original_columns)),
        "column_names": original_columns,
        "missing_columns": missing_columns,
        "missing_value_counts": missing_value_counts,
        "invalid_type_counts": invalid_type_counts,
        "duplicate_row_count": int(duplicate_mask.sum()),
        "invalid_feature_row_count": int((~feature_valid).sum()),
        "out_of_training_range_row_count": int(out_of_range.sum()),
        "screen_flagged_row_count": int(screen_flagged.sum()),
        "valid_feature_row_count": int(feature_valid.sum()),
        "duplicate_rule": duplicate_rule,
    }
    return result, report


def predict_batch(
    frame: pd.DataFrame,
    model: Any,
    artifact: Mapping[str, Any],
    global_safety_inhibit: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any] | None]:
    """Validate, score valid feature rows, and fail safely on every unsafe case."""
    inhibit_reason = _safety_inhibit_reason(global_safety_inhibit)
    result, report = validate_dataset(frame, artifact["feature_ranges"])
    if TARGET_COLUMN not in result.columns and "actual_jitter_ps" in result.columns:
        result[TARGET_COLUMN] = pd.to_numeric(
            result["actual_jitter_ps"], errors="coerce"
        ).to_numpy(dtype=float, na_value=np.nan)
    result["predicted_jitter_ps"] = np.nan
    result["upper_prediction_bound_ps"] = np.nan
    result["guard_band_ps"] = float(artifact["guard_band_ps"])
    result["illustrative_usl_ps"] = float(artifact["illustrative_usl_ps"])
    result["illustrative_candidate_skip"] = False
    result["recommendation"] = "RUN FULL JITTER TEST"
    result["safety_inhibit_active"] = True
    result["recommendation_reason"] = result["validation_reasons"]

    valid_indices = np.flatnonzero(result["feature_valid"].to_numpy())
    if valid_indices.size:
        inputs = result.iloc[valid_indices].loc[:, FEATURE_COLUMNS]
        predictions = np.asarray(model.predict(inputs), dtype=float).reshape(-1)
        if predictions.size != valid_indices.size or not np.isfinite(predictions).all():
            raise ValueError("Model returned missing, non-finite, or misaligned predictions.")
        guard_band = float(artifact["guard_band_ps"])
        upper = predictions + guard_band
        result.loc[valid_indices, "predicted_jitter_ps"] = predictions
        result.loc[valid_indices, "upper_prediction_bound_ps"] = upper

        for offset, index in enumerate(valid_indices):
            if bool(result.at[index, "out_of_training_range"]):
                result.at[index, "recommendation_reason"] = (
                    "One or more features are outside training ranges."
                )
                continue
            if bool(result.at[index, "screen_flagged"]):
                result.at[index, "recommendation_reason"] = (
                    "Illustrative leakage screen flagged this row."
                )
                continue
            if not upper[offset] < float(artifact["illustrative_usl_ps"]):
                result.at[index, "recommendation_reason"] = (
                    "Upper prediction bound is not below the illustrative USL."
                )
                continue

            result.at[index, "illustrative_candidate_skip"] = True
            result.at[index, "recommendation_reason"] = inhibit_reason

    result["predicted_above_illustrative_limit"] = (
        result["predicted_jitter_ps"] >= float(artifact["illustrative_usl_ps"])
    )
    result["upper_bound_above_illustrative_limit"] = (
        result["upper_prediction_bound_ps"] >= float(artifact["illustrative_usl_ps"])
    )
    result["die_id_display"] = (
        result["die_id"] if "die_id" in result.columns else "—"
    )

    summary = {
        "total_records": int(len(result)),
        "successfully_predicted_records": int(result["predicted_jitter_ps"].notna().sum()),
        "invalid_records": int((~result["feature_valid"]).sum()),
        "mean_predicted_jitter_ps": (
            float(result["predicted_jitter_ps"].mean())
            if result["predicted_jitter_ps"].notna().any()
            else None
        ),
        "maximum_predicted_jitter_ps": (
            float(result["predicted_jitter_ps"].max())
            if result["predicted_jitter_ps"].notna().any()
            else None
        ),
        "records_above_illustrative_limit": int(
            result["predicted_above_illustrative_limit"].sum()
        ),
        "upper_bounds_above_illustrative_limit": int(
            result["upper_bound_above_illustrative_limit"].sum()
        ),
        "illustrative_candidate_skip_records": int(
            result["illustrative_candidate_skip"].sum()
        ),
    }
    comparison = compare_actual_predictions(result)
    return result, summary, comparison


def compute_global_feature_importance(
    model: Any,
    frame: pd.DataFrame,
    target: Sequence[float] | pd.Series,
    sample_size: int = 1200,
    n_repeats: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Estimate model-dependent permutation importance using only model features."""
    missing = [name for name in FEATURE_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"Importance data is missing model features: {', '.join(missing)}.")
    if sample_size < 2:
        raise ValueError("sample_size must be at least 2.")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")

    features = frame.loc[:, list(FEATURE_COLUMNS)].reset_index(drop=True)
    target_values = np.asarray(target, dtype=float).reshape(-1)
    if len(features) != target_values.size:
        raise ValueError("Importance target length must match the feature rows.")
    if not np.isfinite(target_values).all():
        raise ValueError("Importance target values must be finite.")
    feature_values = features.to_numpy(dtype=float)
    if not np.isfinite(feature_values).all():
        raise ValueError("Importance feature values must be finite.")

    sample_indices = features.sample(
        n=min(sample_size, len(features)), random_state=random_state
    ).index.to_numpy()
    importance = permutation_importance(
        model,
        features.iloc[sample_indices],
        target_values[sample_indices],
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "feature": list(FEATURE_COLUMNS),
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)


def explain_prediction(
    features: Mapping[str, Any],
    model: Any,
    artifact: Mapping[str, Any],
    reference_values: Mapping[str, Any],
) -> dict[str, Any]:
    """Explain one prediction by single-feature replacement against a reference row."""
    decision = decide_jitter_test(
        features,
        model,
        artifact,
        global_safety_inhibit=GLOBAL_SAFETY_INHIBIT_REASON,
    )
    if decision["predicted_jitter_ps"] is None:
        return {
            "valid": False,
            "validation_status": decision["validation_status"],
            "reason": decision["reason"],
            "prediction_ps": None,
            "reference_prediction_ps": None,
            "effects": pd.DataFrame(columns=["feature", "effect_ps", "direction"]),
        }

    missing_reference = [
        name for name in FEATURE_COLUMNS
        if name not in reference_values or reference_values[name] is None
    ]
    if missing_reference:
        raise ValueError(
            "Explanation reference is missing model features: "
            + ", ".join(missing_reference)
            + "."
        )
    try:
        reference = {name: float(reference_values[name]) for name in FEATURE_COLUMNS}
    except (TypeError, ValueError) as error:
        raise ValueError("Explanation reference values must be numeric.") from error
    if not np.isfinite(list(reference.values())).all():
        raise ValueError("Explanation reference values must be finite.")

    observed = {
        name: float(features[name])
        for name in FEATURE_COLUMNS
    }
    counterfactuals = []
    for feature in FEATURE_COLUMNS:
        row = dict(observed)
        row[feature] = reference[feature]
        counterfactuals.append(row)
    prediction_rows = pd.DataFrame(
        [reference, *counterfactuals],
        columns=list(FEATURE_COLUMNS),
    )
    predictions = np.asarray(model.predict(prediction_rows), dtype=float).reshape(-1)
    if predictions.size != len(FEATURE_COLUMNS) + 1 or not np.isfinite(predictions).all():
        raise ValueError("Model returned missing, non-finite, or misaligned explanation predictions.")

    observed_prediction = float(decision["predicted_jitter_ps"])
    effects = pd.DataFrame(
        {
            "feature": list(FEATURE_COLUMNS),
            "effect_ps": observed_prediction - predictions[1:],
        }
    )
    effects["direction"] = np.select(
        [effects["effect_ps"] > 0, effects["effect_ps"] < 0],
        ["Raises prediction", "Lowers prediction"],
        default="No measured change",
    )
    effects = effects.reindex(
        effects["effect_ps"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)
    return {
        "valid": True,
        "validation_status": decision["validation_status"],
        "out_of_training_range_features": decision["out_of_training_range_features"],
        "reason": decision["reason"],
        "prediction_ps": observed_prediction,
        "reference_prediction_ps": float(predictions[0]),
        "reference_values": reference,
        "effects": effects,
        "method": (
            "For each feature, replace only that input with its training-median reference "
            "and compare the resulting prediction with the observed-input prediction. "
            "Effects are model-dependent, non-causal, and non-additive."
        ),
    }


def compare_what_if_predictions(
    baseline_features: Mapping[str, Any],
    modified_features: Mapping[str, Any],
    model: Any,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two validated scenarios while retaining the full-test inhibit."""
    baseline = decide_jitter_test(
        baseline_features,
        model,
        artifact,
        global_safety_inhibit=GLOBAL_SAFETY_INHIBIT_REASON,
    )
    modified = decide_jitter_test(
        modified_features,
        model,
        artifact,
        global_safety_inhibit=GLOBAL_SAFETY_INHIBIT_REASON,
    )
    baseline_prediction = baseline["predicted_jitter_ps"]
    modified_prediction = modified["predicted_jitter_ps"]
    both_predicted = baseline_prediction is not None and modified_prediction is not None
    candidate_changed = (
        baseline["illustrative_candidate_skip"]
        != modified["illustrative_candidate_skip"]
    )
    validation_changed = baseline["validation_status"] != modified["validation_status"]
    if candidate_changed or validation_changed:
        change_summary = (
            "Illustrative eligibility or validation status changed between scenarios. "
            "The application still keeps the full jitter test enabled."
        )
    else:
        change_summary = (
            "Illustrative eligibility and validation status remained unchanged. "
            "The application still keeps the full jitter test enabled."
        )
    return {
        "baseline": baseline,
        "modified": modified,
        "prediction_delta_ps": (
            float(modified_prediction - baseline_prediction)
            if both_predicted
            else None
        ),
        "upper_bound_delta_ps": (
            float(modified["upper_bound_ps"] - baseline["upper_bound_ps"])
            if both_predicted
            else None
        ),
        "illustrative_candidate_changed": candidate_changed,
        "validation_status_changed": validation_changed,
        "change_summary": change_summary,
        "recommendation": "RUN FULL JITTER TEST",
    }


def compare_actual_predictions(
    predictions: pd.DataFrame,
    minimum_observations: int = 2,
) -> dict[str, Any] | None:
    """Calculate actual-vs-predicted metrics only for real supplied labels."""
    actual_column = (
        TARGET_COLUMN
        if TARGET_COLUMN in predictions.columns
        else "actual_jitter_ps"
        if "actual_jitter_ps" in predictions.columns
        else None
    )
    if actual_column is None:
        return None
    working = predictions.copy(deep=True)
    if actual_column != TARGET_COLUMN:
        working[TARGET_COLUMN] = working[actual_column]
    actual = pd.to_numeric(working[TARGET_COLUMN], errors="coerce")
    predicted = pd.to_numeric(predictions["predicted_jitter_ps"], errors="coerce")
    valid = np.isfinite(actual.to_numpy(dtype=float, na_value=np.nan)) & np.isfinite(
        predicted.to_numpy(dtype=float, na_value=np.nan)
    )
    subset = working.loc[valid].copy()
    if len(subset) < minimum_observations:
        return {
            "count": int(len(subset)),
            "sufficient": False,
            "reason": (
                f"At least {minimum_observations} valid actual/predicted pairs "
                "are required for comparison metrics."
            ),
            "table": subset,
        }

    actual_values = actual.loc[valid].to_numpy(dtype=float)
    predicted_values = predicted.loc[valid].to_numpy(dtype=float)
    subset["signed_error_ps"] = predicted_values - actual_values
    subset["absolute_error_ps"] = np.abs(predicted_values - actual_values)
    denominator = np.abs(actual_values)
    percentage_error = np.full(denominator.shape, np.nan, dtype=float)
    np.divide(
        (predicted_values - actual_values) * 100,
        denominator,
        out=percentage_error,
        where=denominator > np.finfo(float).eps,
    )
    subset["percentage_error"] = percentage_error
    largest_errors = subset.nlargest(min(10, len(subset)), "absolute_error_ps")
    return {
        "count": int(len(subset)),
        "sufficient": True,
        "mae_ps": float(mean_absolute_error(actual_values, predicted_values)),
        "rmse_ps": float(np.sqrt(mean_squared_error(actual_values, predicted_values))),
        "r2": float(r2_score(actual_values, predicted_values)),
        "signed_error_mean_ps": float(subset["signed_error_ps"].mean()),
        "largest_errors": largest_errors,
        "table": subset,
    }


def dataset_profile(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize an uploaded or reference dataset for interactive inspection."""
    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "preview": frame.head(100),
        "dtypes": frame.dtypes.astype(str).rename("dtype").to_frame(),
        "missing": frame.isna().sum().rename("missing_values").to_frame(),
        "statistics": frame.describe(include="all").transpose(),
    }


def compare_feature_distributions(
    uploaded: pd.DataFrame,
    training: pd.DataFrame,
    feature_ranges: Mapping[str, Sequence[float]],
) -> pd.DataFrame:
    """Compare feature moments, empirical KS distance, and training-range rates."""
    rows: list[dict[str, Any]] = []
    for feature in FEATURE_COLUMNS:
        if feature not in uploaded.columns or feature not in training.columns:
            continue
        new_values = pd.to_numeric(uploaded[feature], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        train_values = pd.to_numeric(training[feature], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        new_values = new_values[np.isfinite(new_values)]
        train_values = train_values[np.isfinite(train_values)]
        if not len(new_values) or not len(train_values):
            rows.append(
                {
                    "feature": feature,
                    "uploaded_count": int(len(new_values)),
                    "training_count": int(len(train_values)),
                    "mean_change": np.nan,
                    "std_change": np.nan,
                    "ks_statistic": np.nan,
                    "ks_p_value": np.nan,
                    "out_of_training_range_percent": np.nan,
                    "drift_warning": "INSUFFICIENT DATA",
                }
            )
            continue

        lower, upper = map(float, feature_ranges[feature])
        out_of_range = np.mean((new_values < lower) | (new_values > upper)) * 100
        ks_result = ks_2samp(
            new_values,
            train_values,
            alternative="two-sided",
            method="auto",
        )
        train_mean = float(np.mean(train_values))
        uploaded_mean = float(np.mean(new_values))
        train_std = float(np.std(train_values, ddof=1)) if len(train_values) > 1 else 0.0
        uploaded_std = (
            float(np.std(new_values, ddof=1)) if len(new_values) > 1 else 0.0
        )
        mean_change = uploaded_mean - train_mean
        std_change = uploaded_std - train_std
        warning = (
            "SHIFT WARNING"
            if out_of_range > 0 or ks_result.pvalue < 0.01
            else "NO STRONG WARNING"
        )
        rows.append(
            {
                "feature": feature,
                "uploaded_count": int(len(new_values)),
                "training_count": int(len(train_values)),
                "training_mean": train_mean,
                "uploaded_mean": uploaded_mean,
                "mean_change": mean_change,
                "mean_change_percent": (
                    mean_change / abs(train_mean) * 100
                    if abs(train_mean) > np.finfo(float).eps
                    else np.nan
                ),
                "training_std": train_std,
                "uploaded_std": uploaded_std,
                "std_change": std_change,
                "ks_statistic": float(ks_result.statistic),
                "ks_p_value": float(ks_result.pvalue),
                "out_of_training_range_percent": float(out_of_range),
                "drift_warning": warning,
            }
        )
    return pd.DataFrame(rows)


def summarize_drift(comparison: pd.DataFrame) -> dict[str, Any]:
    """Summarize warning counts without treating drift as proof of model failure."""
    if comparison.empty:
        return {
            "status": "NOT ENOUGH COMPARABLE FEATURES",
            "warning_features": 0,
            "compared_features": 0,
            "insufficient_features": 0,
        }
    warning_count = int(comparison["drift_warning"].eq("SHIFT WARNING").sum())
    insufficient_count = int(
        comparison["drift_warning"].eq("INSUFFICIENT DATA").sum()
    )
    compared_count = int(len(comparison) - insufficient_count)
    return {
        "status": (
            "NOT ENOUGH COMPARABLE FEATURES"
            if compared_count == 0
            else "DISTRIBUTION SHIFT WARNING"
            if warning_count
            else "NO STRONG SHIFT WARNING"
        ),
        "warning_features": warning_count,
        "compared_features": compared_count,
        "insufficient_features": insufficient_count,
        "interpretation": (
            "Distribution checks are monitoring indicators, not proof that model "
            "performance has changed."
        ),
    }


def analyze_locktime_values(
    values: Sequence[Any],
    fixed_wait_us: float = LOCKTIME_FIXED_WAIT_US,
    margin_us: float = LOCKTIME_MARGIN_US,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Calculate adaptive waits for manual or uploaded lock-time values."""
    if not np.isfinite(fixed_wait_us) or fixed_wait_us <= 0:
        raise ValueError("fixed_wait_us must be finite and positive.")
    if not np.isfinite(margin_us) or margin_us < 0:
        raise ValueError("margin_us must be finite and non-negative.")
    frame = pd.DataFrame({"lock_time_us": list(values)})
    numeric = pd.to_numeric(frame["lock_time_us"], errors="coerce")
    valid = np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)) & numeric.ge(
        0
    ).to_numpy()
    frame["lock_time_us"] = numeric
    frame["valid"] = valid
    frame["fixed_wait_us"] = float(fixed_wait_us)
    frame["adaptive_wait_us"] = np.nan
    frame["time_saved_us"] = np.nan
    if valid.any():
        frame.loc[valid, "adaptive_wait_us"] = [
            adaptive_wait_us(
                value,
                margin_us=margin_us,
                fixed_wait_us=fixed_wait_us,
            )
            for value in numeric.loc[valid].to_numpy(dtype=float)
        ]
        frame.loc[valid, "time_saved_us"] = (
            float(fixed_wait_us) - frame.loc[valid, "adaptive_wait_us"]
        )

    valid_rows = frame.loc[valid]
    if valid_rows.empty:
        summary = {
            "total_records": int(len(frame)),
            "valid_records": 0,
            "invalid_records": int(len(frame)),
            "mean_fixed_wait_us": None,
            "mean_adaptive_wait_us": None,
            "mean_time_saved_us": None,
            "savings_percent": None,
            "minimum_adaptive_wait_us": None,
            "maximum_adaptive_wait_us": None,
            "minimum_fixed_wait_us": None,
            "maximum_fixed_wait_us": None,
            "minimum_time_saved_us": None,
            "maximum_time_saved_us": None,
        }
    else:
        mean_adaptive = float(valid_rows["adaptive_wait_us"].mean())
        summary = {
            "total_records": int(len(frame)),
            "valid_records": int(valid.sum()),
            "invalid_records": int((~valid).sum()),
            "mean_fixed_wait_us": float(fixed_wait_us),
            "mean_adaptive_wait_us": mean_adaptive,
            "mean_time_saved_us": float(valid_rows["time_saved_us"].mean()),
            "savings_percent": float(
                (1 - mean_adaptive / float(fixed_wait_us)) * 100
            ),
            "minimum_adaptive_wait_us": float(valid_rows["adaptive_wait_us"].min()),
            "maximum_adaptive_wait_us": float(valid_rows["adaptive_wait_us"].max()),
            "minimum_fixed_wait_us": float(valid_rows["fixed_wait_us"].min()),
            "maximum_fixed_wait_us": float(valid_rows["fixed_wait_us"].max()),
            "minimum_time_saved_us": float(valid_rows["time_saved_us"].min()),
            "maximum_time_saved_us": float(valid_rows["time_saved_us"].max()),
        }
    return frame, summary


def create_history_entry(
    features: Mapping[str, Any],
    prediction: Mapping[str, Any],
    actual_jitter_ps: float | None = None,
    source: str = "manual",
) -> dict[str, Any]:
    """Create a JSON/CSV-friendly session history row."""
    actual = (
        float(actual_jitter_ps)
        if actual_jitter_ps is not None and np.isfinite(float(actual_jitter_ps))
        else None
    )
    predicted = prediction.get("predicted_jitter_ps")
    error = float(predicted) - actual if actual is not None and predicted is not None else None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "die_id": features.get("die_id"),
        **{name: features.get(name) for name in FEATURE_COLUMNS},
        "predicted_jitter_ps": predicted,
        "guard_band_ps": prediction.get("guard_band_ps"),
        "upper_prediction_bound_ps": prediction.get("upper_bound_ps"),
        "actual_jitter_ps": actual,
        "signed_error_ps": error,
        "absolute_error_ps": abs(error) if error is not None else None,
        "validation_status": prediction.get("validation_status", "VALID"),
        "recommendation": prediction.get("recommendation", "RUN FULL JITTER TEST"),
        "reason": prediction.get("reason", ""),
    }


def append_history_entry(
    history: Sequence[Mapping[str, Any]],
    entry: Mapping[str, Any],
    maximum_entries: int = 5000,
) -> list[dict[str, Any]]:
    """Append a serializable prediction record and bound per-session memory use."""
    if maximum_entries <= 0:
        raise ValueError("maximum_entries must be positive.")
    return [dict(item) for item in [*history, entry][-maximum_entries:]]


def append_history_entries(
    history: Sequence[Mapping[str, Any]],
    entries: Sequence[Mapping[str, Any]],
    maximum_entries: int = 5000,
) -> list[dict[str, Any]]:
    """Append many batch records in one bounded operation."""
    if maximum_entries <= 0:
        raise ValueError("maximum_entries must be positive.")
    combined = [*history, *entries]
    return [dict(item) for item in combined[-maximum_entries:]]


def prediction_from_batch_row(row: pd.Series) -> dict[str, Any]:
    """Adapt a batch result row to the manual-prediction history schema."""
    return {
        "predicted_jitter_ps": (
            None if pd.isna(row["predicted_jitter_ps"]) else row["predicted_jitter_ps"]
        ),
        "guard_band_ps": row.get("guard_band_ps"),
        "upper_bound_ps": (
            None
            if pd.isna(row["upper_prediction_bound_ps"])
            else row["upper_prediction_bound_ps"]
        ),
        "validation_status": row["validation_status"],
        "recommendation": row["recommendation"],
        "reason": row["recommendation_reason"],
    }


def safe_metric_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Pick report-ready, actual saved evaluation values."""
    return {
        "model": metrics.get("model", "unknown"),
        "test_rows": metrics.get("test_rows"),
        "test_mae_ps": metrics.get("test_mae_ps"),
        "test_rmse_ps": metrics.get("test_rmse_ps"),
        "test_r2": metrics.get("test_r2"),
        "guard_band_ps": metrics.get("guard_band_ps"),
        "known_test_failures_among_skip_candidates": metrics.get(
            "test_failures_among_skip_candidates"
        ),
        "evaluation_note": metrics.get("evaluation_note"),
    }
