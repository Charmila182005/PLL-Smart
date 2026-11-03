"""Data loading, validation, and safe decision helpers for PLL-Smart."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
JITTER_DATA_PATH = PROJECT_ROOT / "data" / "synthetic_pll_jitter_dataset.csv"
LOCKTIME_DATA_PATH = PROJECT_ROOT / "data" / "synthetic_pll_locktime_dataset.csv"

FEATURE_COLUMNS = (
    "vco_band_code",
    "ring_osc_ghz",
    "chargepump_ua",
    "lock_time_us",
    "vdd_droop_norm",
    "wafer_radius_norm",
    "leak_screen_norm",
)
TARGET_COLUMN = "rms_jitter_ps"
PARTITION_COLUMN = "split"
PARTITION_COUNTS = {"train": 60_000, "calibration": 40_000, "test": 100_000}

# The source PDF and dataset describe these as illustrative, not approved limits.
ILLUSTRATIVE_JITTER_USL_PS = 2.2
ILLUSTRATIVE_SCREEN_THRESHOLD = 2.0
ILLUSTRATIVE_CONFORMAL_ALPHA = 0.05
LOCKTIME_MARGIN_US = 3.0
LOCKTIME_FIXED_WAIT_US = 60.0
GLOBAL_SAFETY_INHIBIT_REASON = (
    "Global safety inhibit: the original held-out evaluation admitted 52 measured "
    "failing dies among 96,121 illustrative skip candidates. Keep the full test enabled."
)

JITTER_REQUIRED_COLUMNS = (
    "die_id",
    *FEATURE_COLUMNS,
    TARGET_COLUMN,
    PARTITION_COLUMN,
)
LOCKTIME_REQUIRED_COLUMNS = (
    "die_id",
    "lock_time_us",
    "fixed_wait_us",
    "adaptive_wait_us",
)


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def load_jitter_partitions(
    path: str | Path = JITTER_DATA_PATH,
    expected_counts: Mapping[str, int] | None = PARTITION_COUNTS,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load supplied partitions as-is and exclude invalid model rows explicitly."""
    frame = pd.read_csv(path)
    _require_columns(frame, JITTER_REQUIRED_COLUMNS, "Jitter dataset")

    raw_counts = frame[PARTITION_COLUMN].value_counts().to_dict()
    if expected_counts is not None and raw_counts != dict(expected_counts):
        raise ValueError(
            f"Unexpected jitter partition sizes: {raw_counts}; "
            f"expected {dict(expected_counts)}"
        )
    if frame["die_id"].duplicated().any():
        raise ValueError("Jitter dataset contains duplicate die_id values.")

    numeric_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    numeric_values = frame[numeric_columns].to_numpy(dtype=float)
    non_finite = ~np.isfinite(numeric_values).all(axis=1)
    negative_lock_time = frame["lock_time_us"].lt(0).fillna(False)
    invalid = non_finite | negative_lock_time.to_numpy()

    invalid_rows = frame.loc[invalid, ["die_id", PARTITION_COLUMN]].copy()
    invalid_rows["reason"] = [
        "negative lock_time_us" if is_negative else "missing or non-finite numeric value"
        for is_negative, is_non_finite in zip(
            negative_lock_time.loc[invalid].to_numpy(),
            non_finite[invalid],
        )
    ]

    clean = frame.loc[~invalid].copy()
    partitions = {
        name: clean.loc[clean[PARTITION_COLUMN].eq(name)].copy().reset_index(drop=True)
        for name in ("train", "calibration", "test")
    }

    partition_ids = [set(partitions[name]["die_id"]) for name in partitions]
    if any(partition_ids[i] & partition_ids[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Die IDs overlap across train, calibration, and test partitions.")

    diagnostics = {
        "raw_partition_counts": {name: int(raw_counts.get(name, 0)) for name in PARTITION_COUNTS},
        "usable_partition_counts": {
            name: int(len(partitions[name])) for name in partitions
        },
        "invalid_row_count": int(invalid.sum()),
        "invalid_by_partition": {
            name: int(invalid_rows[PARTITION_COLUMN].eq(name).sum())
            for name in PARTITION_COUNTS
        },
        "invalid_rows": invalid_rows.reset_index(drop=True),
    }
    return partitions, diagnostics


def conformal_upper_residual_quantile(
    residuals: np.ndarray | pd.Series,
    alpha: float = ILLUSTRATIVE_CONFORMAL_ALPHA,
) -> float:
    """Return the finite-sample split-conformal quantile of absolute residuals."""
    values = np.asarray(residuals, dtype=float).reshape(-1)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1.")
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Calibration residuals must be non-empty and finite.")
    rank = int(np.ceil((values.size + 1) * (1 - alpha)))
    if rank > values.size:
        raise ValueError(
            "The calibration set is too small for a finite quantile at this alpha."
        )
    return float(np.partition(values, rank - 1)[rank - 1])


def _safety_inhibit_reason(value: Any = None) -> str:
    """Keep the global inhibit active; caller state may add context, never disable it."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return GLOBAL_SAFETY_INHIBIT_REASON


def decide_jitter_test(
    features: Mapping[str, Any],
    model: Any,
    artifact: Mapping[str, Any],
    global_safety_inhibit: Any = None,
) -> dict[str, Any]:
    """Score inputs, but always retain the full-test recommendation."""
    inhibit_reason = _safety_inhibit_reason(global_safety_inhibit)
    result: dict[str, Any] = {
        "recommendation": "RUN FULL JITTER TEST",
        "reason": inhibit_reason,
        "predicted_jitter_ps": None,
        "upper_bound_ps": None,
        "guard_band_ps": float(artifact["guard_band_ps"]),
        "usl_ps": float(artifact["illustrative_usl_ps"]),
        "validation_status": "INVALID",
        "out_of_training_range_features": [],
        "illustrative_candidate_skip": False,
        "safety_inhibit_active": True,
    }

    missing = [name for name in FEATURE_COLUMNS if name not in features or features[name] is None]
    if missing:
        result["reason"] = (
            f"Missing required feature(s): {', '.join(missing)}. "
            f"{inhibit_reason}"
        )
        return result

    values: dict[str, float] = {}
    for name in FEATURE_COLUMNS:
        try:
            value = float(features[name])
        except (TypeError, ValueError):
            result["reason"] = f"Feature {name} is not a valid number. {inhibit_reason}"
            return result
        if not np.isfinite(value):
            result["reason"] = f"Feature {name} is not finite. {inhibit_reason}"
            return result
        values[name] = value
    if values["lock_time_us"] < 0:
        result["reason"] = (
            "Negative lock_time_us is invalid; use the full test. "
            f"{inhibit_reason}"
        )
        return result

    out_of_range_features: list[str] = []
    for name in FEATURE_COLUMNS:
        lower, upper = artifact["feature_ranges"][name]
        if not lower <= values[name] <= upper:
            out_of_range_features.append(name)

    result["validation_status"] = (
        "OUT OF TRAINING RANGE" if out_of_range_features else "VALID"
    )
    result["out_of_training_range_features"] = out_of_range_features

    row = pd.DataFrame(
        [[values[name] for name in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )
    prediction = float(model.predict(row)[0])
    upper_bound = prediction + float(artifact["guard_band_ps"])
    result["predicted_jitter_ps"] = prediction
    result["upper_bound_ps"] = upper_bound
    if out_of_range_features:
        result["reason"] = (
            "Out-of-training-range feature(s): "
            + ", ".join(out_of_range_features)
            + f". Use the full test. {inhibit_reason}"
        )
    elif values["leak_screen_norm"] >= artifact["illustrative_screen_threshold"]:
        result["validation_status"] = "SCREEN FLAGGED"
        result["reason"] = (
            "Illustrative defect-screen condition did not pass "
            f"(leak_screen_norm must be < "
            f"{artifact['illustrative_screen_threshold']:.2f}). "
            f"{inhibit_reason}"
        )
    elif upper_bound < float(artifact["illustrative_usl_ps"]):
        result["illustrative_candidate_skip"] = True
        result["reason"] = (
            "Illustrative candidate criteria passed, but skipping is inhibited. "
            f"{inhibit_reason}"
        )
    else:
        result["reason"] = (
            "The upper prediction bound is not below the illustrative USL. "
            f"{inhibit_reason}"
        )
    return result


def adaptive_wait_us(
    lock_time_us: float,
    margin_us: float = LOCKTIME_MARGIN_US,
    fixed_wait_us: float = LOCKTIME_FIXED_WAIT_US,
) -> float:
    """Calculate the capped adaptive wait; reject physically invalid inputs."""
    values = (lock_time_us, margin_us, fixed_wait_us)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Lock time, margin, and fixed wait must be finite.")
    if lock_time_us < 0:
        raise ValueError("lock_time_us cannot be negative.")
    if margin_us < 0 or fixed_wait_us <= 0:
        raise ValueError("margin_us must be non-negative and fixed_wait_us positive.")
    return float(min(lock_time_us + margin_us, fixed_wait_us))


def analyze_locktime_dataset(
    path: str | Path = LOCKTIME_DATA_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate lock-time rows and recompute waits from the documented formula."""
    frame = pd.read_csv(path)
    _require_columns(frame, LOCKTIME_REQUIRED_COLUMNS, "Lock-time dataset")
    if frame["die_id"].duplicated().any():
        raise ValueError("Lock-time dataset contains duplicate die_id values.")
    for column in ("lock_time_us", "fixed_wait_us", "adaptive_wait_us"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    numeric = frame[["lock_time_us", "fixed_wait_us", "adaptive_wait_us"]].to_numpy(
        dtype=float
    )
    valid = np.isfinite(numeric).all(axis=1)
    valid &= frame["lock_time_us"].ge(0).to_numpy()
    valid &= frame["fixed_wait_us"].gt(0).to_numpy()
    invalid_count = int((~valid).sum())
    clean = frame.loc[valid].copy().reset_index(drop=True)
    clean["calculated_adaptive_wait_us"] = [
        adaptive_wait_us(lock_time, fixed_wait_us=fixed_wait)
        for lock_time, fixed_wait in zip(
            clean["lock_time_us"].to_numpy(),
            clean["fixed_wait_us"].to_numpy(),
        )
    ]
    formula_mismatches = int(
        (
            ~np.isclose(
                clean["adaptive_wait_us"].to_numpy(),
                clean["calculated_adaptive_wait_us"].to_numpy(),
                rtol=0,
                atol=1e-6,
            )
        ).sum()
    )
    summary = {
        "raw_rows": int(len(frame)),
        "usable_rows": int(len(clean)),
        "invalid_rows": invalid_count,
        "formula_mismatches": formula_mismatches,
        "mean_fixed_wait_us": float(clean["fixed_wait_us"].mean()),
        "mean_adaptive_wait_us": float(clean["calculated_adaptive_wait_us"].mean()),
        "mean_savings_us": float(
            clean["fixed_wait_us"].mean() - clean["calculated_adaptive_wait_us"].mean()
        ),
        "savings_percent": float(
            (
                1
                - clean["calculated_adaptive_wait_us"].mean()
                / clean["fixed_wait_us"].mean()
            )
            * 100
        ),
    }
    return clean, summary