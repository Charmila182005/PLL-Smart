from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_utils import (
    FEATURE_COLUMNS,
    GLOBAL_SAFETY_INHIBIT_REASON,
    ILLUSTRATIVE_JITTER_USL_PS,
    ILLUSTRATIVE_SCREEN_THRESHOLD,
    adaptive_wait_us,
    analyze_locktime_dataset,
    conformal_upper_residual_quantile,
    decide_jitter_test,
    load_jitter_partitions,
)
from train import fit_evaluate


class ConstantModel:
    def __init__(self, value: float = 1.0) -> None:
        self.value = value
        self.fit_targets: list[float] = []
        self.predict_lengths: list[int] = []

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "ConstantModel":
        self.fit_targets = target.astype(float).tolist()
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        self.predict_lengths.append(len(features))
        return np.full(len(features), self.value, dtype=float)


def valid_features(value: float = 1.0) -> dict[str, float]:
    return {name: value for name in FEATURE_COLUMNS}


def valid_artifact() -> dict[str, object]:
    return {
        "guard_band_ps": 0.1,
        "illustrative_usl_ps": ILLUSTRATIVE_JITTER_USL_PS,
        "illustrative_screen_threshold": ILLUSTRATIVE_SCREEN_THRESHOLD,
        "feature_ranges": {name: (0.0, 10.0) for name in FEATURE_COLUMNS},
    }


def test_model_features_exclude_target_derived_and_identifier_columns() -> None:
    assert FEATURE_COLUMNS == (
        "vco_band_code",
        "ring_osc_ghz",
        "chargepump_ua",
        "lock_time_us",
        "vdd_droop_norm",
        "wafer_radius_norm",
        "leak_screen_norm",
    )
    assert not {
        "die_id",
        "predicted_jitter_ps",
        "jitter_fail",
        "defect_injected",
        "usl_ps",
        "split",
    }.intersection(FEATURE_COLUMNS)


def test_supplied_partitions_remain_separate_and_keep_expected_sizes() -> None:
    partitions, diagnostics = load_jitter_partitions()
    assert diagnostics["raw_partition_counts"] == {
        "train": 60_000,
        "calibration": 40_000,
        "test": 100_000,
    }
    ids = {name: set(frame["die_id"]) for name, frame in partitions.items()}
    assert not (ids["train"] & ids["calibration"])
    assert not (ids["train"] & ids["test"])
    assert not (ids["calibration"] & ids["test"])


def test_negative_lock_time_row_is_reported_and_excluded() -> None:
    partitions, diagnostics = load_jitter_partitions()
    bad_rows = diagnostics["invalid_rows"]
    row = bad_rows.loc[bad_rows["die_id"].eq(128948)]
    assert len(row) == 1
    assert row.iloc[0]["reason"] == "negative lock_time_us"
    assert row.iloc[0]["split"] == "test"
    assert 128948 not in set(partitions["test"]["die_id"])
    assert diagnostics["usable_partition_counts"]["test"] == 99_999


def test_calibration_quantile_uses_only_calibration_rows() -> None:
    def make_frame(ids: list[int], targets: list[float]) -> pd.DataFrame:
        frame = pd.DataFrame(
            {name: [1.0] * len(ids) for name in FEATURE_COLUMNS}
        )
        frame["die_id"] = ids
        frame["rms_jitter_ps"] = targets
        frame["leak_screen_norm"] = 1.0
        return frame

    partitions = {
        "train": make_frame([1, 2], [1.0, 2.0]),
        "calibration": make_frame([3, 4], [10.0, 20.0]),
        "test": make_frame([5, 6], [500.0, 600.0]),
    }
    first_model = ConstantModel(0.0)
    _, first_metrics, _, _ = fit_evaluate(partitions, alpha=0.5, model=first_model)
    assert first_model.fit_targets == [1.0, 2.0]
    assert first_model.predict_lengths == [2, 2]
    assert first_metrics["guard_band_ps"] == 20.0

    changed_test = {
        **partitions,
        "test": partitions["test"].assign(rms_jitter_ps=[50_000.0, 60_000.0]),
    }
    _, changed_metrics, _, _ = fit_evaluate(
        changed_test, alpha=0.5, model=ConstantModel(0.0)
    )
    assert changed_metrics["guard_band_ps"] == first_metrics["guard_band_ps"]


def test_conformal_quantile_rejects_empty_or_invalid_calibration() -> None:
    with pytest.raises(ValueError):
        conformal_upper_residual_quantile(np.array([]))
    with pytest.raises(ValueError):
        conformal_upper_residual_quantile(np.array([1.0, np.nan]))


def test_adaptive_wait_formula_and_timeout_cap() -> None:
    assert adaptive_wait_us(10.0) == pytest.approx(13.0)
    assert adaptive_wait_us(100.0) == 60.0
    with pytest.raises(ValueError, match="cannot be negative"):
        adaptive_wait_us(-0.1)


def test_lock_time_dataset_formula_is_recomputed_and_invalid_rows_reported(
    tmp_path,
) -> None:
    path = tmp_path / "lock.csv"
    pd.DataFrame(
        [
            {"die_id": 1, "lock_time_us": 10.0, "fixed_wait_us": 60.0, "adaptive_wait_us": 13.0},
            {"die_id": 2, "lock_time_us": 100.0, "fixed_wait_us": 60.0, "adaptive_wait_us": 60.0},
            {"die_id": 3, "lock_time_us": -1.0, "fixed_wait_us": 60.0, "adaptive_wait_us": 2.0},
        ]
    ).to_csv(path, index=False)
    rows, summary = analyze_locktime_dataset(path)
    assert rows["calculated_adaptive_wait_us"].tolist() == [13.0, 60.0]
    assert summary["invalid_rows"] == 1
    assert summary["formula_mismatches"] == 0


def test_safe_fallback_for_missing_invalid_out_of_range_and_screen_flagged_inputs() -> None:
    artifact = valid_artifact()
    model = ConstantModel()

    missing = decide_jitter_test({}, model, artifact)
    assert missing["recommendation"] == "RUN FULL JITTER TEST"
    assert "Missing" in missing["reason"]

    negative_lock = valid_features()
    negative_lock["lock_time_us"] = -1.0
    result = decide_jitter_test(negative_lock, model, artifact)
    assert result["recommendation"] == "RUN FULL JITTER TEST"
    assert "Negative lock_time_us" in result["reason"]

    out_of_range = valid_features()
    out_of_range["ring_osc_ghz"] = 11.0
    result = decide_jitter_test(out_of_range, model, artifact)
    assert result["recommendation"] == "RUN FULL JITTER TEST"
    assert "Out-of-training-range" in result["reason"]

    screen_flagged = valid_features()
    screen_flagged["leak_screen_norm"] = ILLUSTRATIVE_SCREEN_THRESHOLD
    result = decide_jitter_test(screen_flagged, model, artifact)
    assert result["recommendation"] == "RUN FULL JITTER TEST"
    assert "defect-screen condition did not pass" in result["reason"]


def test_global_safety_inhibit_never_allows_skip() -> None:
    result = decide_jitter_test(valid_features(), ConstantModel(1.0), valid_artifact())
    assert result["recommendation"] == "RUN FULL JITTER TEST"
    assert result["safety_inhibit_active"]
    assert result["illustrative_candidate_skip"]
    assert GLOBAL_SAFETY_INHIBIT_REASON in result["reason"]
    assert result["upper_bound_ps"] < ILLUSTRATIVE_JITTER_USL_PS

    artifact = valid_artifact()
    artifact["guard_band_ps"] = 1.2
    result = decide_jitter_test(valid_features(), ConstantModel(1.0), artifact)
    assert result["recommendation"] == "RUN FULL JITTER TEST"
    assert result["upper_bound_ps"] == ILLUSTRATIVE_JITTER_USL_PS

    result = decide_jitter_test(
        valid_features(),
        ConstantModel(1.0),
        valid_artifact(),
        global_safety_inhibit="Explicit additional safety hold",
    )
    assert result["recommendation"] == "RUN FULL JITTER TEST"
    assert "Explicit additional safety hold" in result["reason"]


@pytest.mark.parametrize("inhibit_state", [None, False, 0, "", "   ", object()])
def test_missing_or_invalid_inhibit_state_fails_closed(inhibit_state: object) -> None:
    result = decide_jitter_test(
        valid_features(),
        ConstantModel(1.0),
        valid_artifact(),
        global_safety_inhibit=inhibit_state,
    )
    assert result["recommendation"] == "RUN FULL JITTER TEST"
    assert result["safety_inhibit_active"]
    assert GLOBAL_SAFETY_INHIBIT_REASON in result["reason"]
