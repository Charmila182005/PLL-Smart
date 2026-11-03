"""Train, calibrate, evaluate, and persist the PLL-Smart jitter model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_utils import (
    FEATURE_COLUMNS,
    ILLUSTRATIVE_CONFORMAL_ALPHA,
    ILLUSTRATIVE_JITTER_USL_PS,
    ILLUSTRATIVE_SCREEN_THRESHOLD,
    JITTER_DATA_PATH,
    LOCKTIME_DATA_PATH,
    PARTITION_COUNTS,
    TARGET_COLUMN,
    analyze_locktime_dataset,
    conformal_upper_residual_quantile,
    load_jitter_partitions,
)


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "jitter_model.joblib"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def create_model() -> HistGradientBoostingRegressor:
    """Create the deterministic tabular regressor used in the MVP."""
    return HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_iter=180,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
    )


def fit_evaluate(
    partitions: Mapping[str, pd.DataFrame],
    alpha: float = ILLUSTRATIVE_CONFORMAL_ALPHA,
    model: Any | None = None,
) -> tuple[Any, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Fit on train, calibrate on calibration, and evaluate once on test."""
    train = partitions["train"]
    calibration = partitions["calibration"]
    test = partitions["test"]

    model = create_model() if model is None else model
    started = time.perf_counter()
    model.fit(train.loc[:, FEATURE_COLUMNS], train[TARGET_COLUMN])
    fit_seconds = time.perf_counter() - started

    calibration_eligible = calibration["leak_screen_norm"].lt(
        ILLUSTRATIVE_SCREEN_THRESHOLD
    )
    calibration_rows = calibration.loc[calibration_eligible]
    if calibration_rows.empty:
        raise ValueError("No screen-passing rows are available for calibration.")
    calibration_predictions = model.predict(
        calibration_rows.loc[:, FEATURE_COLUMNS]
    )
    calibration_residuals = np.abs(
        calibration_rows[TARGET_COLUMN].to_numpy(dtype=float)
        - calibration_predictions
    )
    guard_band = conformal_upper_residual_quantile(calibration_residuals, alpha)

    predictions = model.predict(test.loc[:, FEATURE_COLUMNS])
    actual = test[TARGET_COLUMN].to_numpy(dtype=float)
    train_mean = float(train[TARGET_COLUMN].mean())
    baseline_predictions = np.full(actual.shape, train_mean, dtype=float)
    feature_ranges = {
        name: (
            float(train[name].min()),
            float(train[name].max()),
        )
        for name in FEATURE_COLUMNS
    }

    in_training_range = np.ones(len(test), dtype=bool)
    for name in FEATURE_COLUMNS:
        lower, upper = feature_ranges[name]
        in_training_range &= test[name].between(lower, upper).to_numpy()
    screen_pass = test["leak_screen_norm"].lt(
        ILLUSTRATIVE_SCREEN_THRESHOLD
    ).to_numpy()
    upper_bounds = predictions + guard_band
    skip_candidates = (
        in_training_range
        & screen_pass
        & (upper_bounds < ILLUSTRATIVE_JITTER_USL_PS)
    )
    candidate_failures = int(
        np.count_nonzero(
            skip_candidates & (actual >= ILLUSTRATIVE_JITTER_USL_PS)
        )
    )

    metrics: dict[str, Any] = {
        "model": "HistGradientBoostingRegressor",
        "features": list(FEATURE_COLUMNS),
        "target": TARGET_COLUMN,
        "train_rows": int(len(train)),
        "calibration_rows": int(len(calibration)),
        "calibration_screen_passing_rows": int(len(calibration_rows)),
        "test_rows": int(len(test)),
        "conformal_alpha_illustrative": float(alpha),
        "guard_band_ps": float(guard_band),
        "illustrative_usl_ps": ILLUSTRATIVE_JITTER_USL_PS,
        "illustrative_screen_threshold": ILLUSTRATIVE_SCREEN_THRESHOLD,
        "test_mae_ps": float(mean_absolute_error(actual, predictions)),
        "test_rmse_ps": float(np.sqrt(mean_squared_error(actual, predictions))),
        "test_r2": float(r2_score(actual, predictions)),
        "baseline_test_mae_ps": float(
            mean_absolute_error(actual, baseline_predictions)
        ),
        "baseline_test_rmse_ps": float(
            np.sqrt(mean_squared_error(actual, baseline_predictions))
        ),
        "baseline_test_r2": float(r2_score(actual, baseline_predictions)),
        "fit_seconds": float(fit_seconds),
        "test_skip_candidate_count": int(skip_candidates.sum()),
        "test_skip_candidate_rate": float(skip_candidates.mean()),
        "test_failures_among_skip_candidates": candidate_failures,
        "evaluation_note": (
            "Synthetic-data holdout evaluation only; not production qualification "
            "or a DPPM guarantee."
        ),
    }

    artifact = {
        "model": model,
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "feature_ranges": feature_ranges,
        "guard_band_ps": float(guard_band),
        "conformal_alpha_illustrative": float(alpha),
        "illustrative_usl_ps": ILLUSTRATIVE_JITTER_USL_PS,
        "illustrative_screen_threshold": ILLUSTRATIVE_SCREEN_THRESHOLD,
        "calibration_screen_passing_rows": int(len(calibration_rows)),
        "train_rows": int(len(train)),
        "calibration_rows": int(len(calibration)),
        "test_rows": int(len(test)),
    }
    test_predictions = pd.DataFrame(
        {
            "die_id": test["die_id"].to_numpy(),
            "actual_jitter_ps": actual,
            "predicted_jitter_ps": predictions,
            "residual_ps": actual - predictions,
            "upper_prediction_bound_ps": upper_bounds,
            "within_training_ranges": in_training_range,
            "illustrative_screen_pass": screen_pass,
            "skip_candidate": skip_candidates,
        }
    )
    return model, metrics, test_predictions, artifact


def run_training(
    data_path: str | Path | None = None,
    model_path: str | Path = MODEL_PATH,
    outputs_dir: str | Path = OUTPUTS_DIR,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the full workflow and write model and evaluation artifacts."""
    partitions, diagnostics = load_jitter_partitions(
        data_path or JITTER_DATA_PATH,
        expected_counts=PARTITION_COUNTS,
    )
    _, metrics, test_predictions, artifact = fit_evaluate(partitions)
    metrics["data_quality"] = {
        key: value
        for key, value in diagnostics.items()
        if key != "invalid_rows"
    }

    model_path = Path(model_path)
    outputs_dir = Path(outputs_dir)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    calibration_artifact = {
        "guard_band_ps": artifact["guard_band_ps"],
        "conformal_alpha_illustrative": artifact["conformal_alpha_illustrative"],
        "calibration_screen_passing_rows": artifact[
            "calibration_screen_passing_rows"
        ],
        "illustrative_usl_ps": artifact["illustrative_usl_ps"],
        "illustrative_screen_threshold": artifact[
            "illustrative_screen_threshold"
        ],
        "note": (
            "Synthetic-data calibration artifact; not a production limit or "
            "DPPM guarantee."
        ),
    }
    (model_path.parent / "jitter_calibration.json").write_text(
        json.dumps(calibration_artifact, indent=2), encoding="utf-8"
    )
    (outputs_dir / "jitter_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    test_predictions.to_csv(outputs_dir / "test_predictions.csv", index=False)

    _, locktime_summary = analyze_locktime_dataset(LOCKTIME_DATA_PATH)
    (outputs_dir / "locktime_metrics.json").write_text(
        json.dumps(locktime_summary, indent=2), encoding="utf-8"
    )
    return metrics, locktime_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Path to jitter CSV.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--outputs", type=Path, default=OUTPUTS_DIR)
    args = parser.parse_args()

    metrics, locktime_summary = run_training(
        data_path=args.data,
        model_path=args.model,
        outputs_dir=args.outputs,
    )
    print(json.dumps({"jitter_metrics": metrics, "locktime": locktime_summary}, indent=2))


if __name__ == "__main__":
    main()