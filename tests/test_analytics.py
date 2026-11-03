from __future__ import annotations

import csv
import json
from io import BytesIO

import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader
from sklearn.base import BaseEstimator, RegressorMixin

from analytics import (
    analyze_locktime_values,
    append_history_entry,
    append_history_entries,
    compare_actual_predictions,
    compare_feature_distributions,
    compare_what_if_predictions,
    compute_global_feature_importance,
    create_history_entry,
    explain_prediction,
    predict_batch,
    read_csv_upload,
    safe_csv_download,
    summarize_drift,
    validate_dataset,
)
from data_utils import FEATURE_COLUMNS, ILLUSTRATIVE_JITTER_USL_PS
from reporting import build_engineering_pdf_report, build_engineering_report_exports


class FixedModel:
    def __init__(self, value: float = 1.3) -> None:
        self.value = value
        self.calls = 0
        self.input_columns: list[str] | None = None

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        self.calls += 1
        self.input_columns = list(features.columns)
        return np.full(len(features), self.value, dtype=float)


class WeightedModel(RegressorMixin, BaseEstimator):
    def __init__(self) -> None:
        self.weights = np.arange(1, len(FEATURE_COLUMNS) + 1, dtype=float) / 100
        self.input_columns: list[str] | None = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "WeightedModel":
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        self.input_columns = list(features.columns)
        return features.to_numpy(dtype=float) @ self.weights + 0.6


def feature_ranges() -> dict[str, tuple[float, float]]:
    return {name: (0.0, 10.0) for name in FEATURE_COLUMNS}


def artifact() -> dict[str, object]:
    return {
        "feature_ranges": feature_ranges(),
        "guard_band_ps": 0.1,
        "illustrative_usl_ps": ILLUSTRATIVE_JITTER_USL_PS,
        "illustrative_screen_threshold": 2.0,
    }


def valid_frame(size: int = 3) -> pd.DataFrame:
    data = {name: np.full(size, 1.0) for name in FEATURE_COLUMNS}
    data["die_id"] = np.arange(101, 101 + size)
    return pd.DataFrame(data)


def test_read_csv_upload_accepts_valid_csv_without_saving_it() -> None:
    frame = read_csv_upload(b"die_id,lock_time_us\n1,4.2\n")
    assert frame.to_dict(orient="records") == [{"die_id": 1, "lock_time_us": 4.2}]


@pytest.mark.parametrize("contents", [b"", b"a,b\n1,2,3\n", b"a,b\n"])
def test_read_csv_upload_rejects_empty_or_malformed_csv(contents: bytes) -> None:
    with pytest.raises(ValueError):
        read_csv_upload(contents)


def test_csv_download_escapes_formula_leading_text_without_mutating_source() -> None:
    frame = pd.DataFrame(
        {
            "uploaded_text": ["=1+1", "  @SUM(A1:A2)", "+cmd", "-not a number", "safe"],
            "numeric_measurement": [-1.25, 2.0, 3.0, 4.0, 5.0],
        }
    )
    original = frame.copy(deep=True)

    exported = safe_csv_download(frame)
    restored = pd.read_csv(BytesIO(exported))

    assert restored["uploaded_text"].tolist() == [
        "'=1+1",
        "'  @SUM(A1:A2)",
        "'+cmd",
        "'-not a number",
        "safe",
    ]
    assert restored["numeric_measurement"].tolist() == [-1.25, 2.0, 3.0, 4.0, 5.0]
    pd.testing.assert_frame_equal(frame, original)


def test_validate_dataset_reports_missing_columns_values_types_and_duplicates() -> None:
    frame = valid_frame(4)
    frame.loc[0, "ring_osc_ghz"] = np.nan
    frame["chargepump_ua"] = frame["chargepump_ua"].astype(object)
    frame.loc[1, "chargepump_ua"] = "not-a-number"
    frame.loc[2, "die_id"] = frame.loc[3, "die_id"]
    frame = frame.drop(columns=["vdd_droop_norm"])

    validated, summary = validate_dataset(frame, feature_ranges())

    assert summary["rows"] == 4
    assert summary["columns"] == len(frame.columns)
    assert "vdd_droop_norm" in summary["missing_columns"]
    assert summary["missing_value_counts"]["ring_osc_ghz"] == 1
    assert summary["invalid_type_counts"]["chargepump_ua"] == 1
    assert summary["duplicate_row_count"] == 2
    assert summary["invalid_feature_row_count"] == 4
    assert not validated["feature_valid"].any()


def test_batch_prediction_scores_valid_rows_and_applies_fail_safe_recommendations() -> None:
    frame = valid_frame(4)
    frame["die_id"] = [1, 2, 3, 4]
    frame.loc[1, "vco_band_code"] = 11.0
    frame.loc[2, "leak_screen_norm"] = 2.5
    frame.loc[3, "lock_time_us"] = -1.0
    model = FixedModel()

    scored, summary, comparison = predict_batch(
        frame,
        model,
        artifact(),
        global_safety_inhibit="Retrospective test failures",
    )

    assert model.calls == 1
    assert summary["total_records"] == 4
    assert summary["successfully_predicted_records"] == 3
    assert summary["invalid_records"] == 1
    assert scored.loc[0, "illustrative_candidate_skip"]
    assert scored.loc[0, "recommendation"] == "RUN FULL JITTER TEST"
    assert "Retrospective" in scored.loc[0, "recommendation_reason"]
    assert scored.loc[1, "predicted_jitter_ps"] == pytest.approx(1.3)
    assert scored.loc[1, "validation_status"] == "OUT OF TRAINING RANGE"
    assert scored.loc[1, "recommendation"] == "RUN FULL JITTER TEST"
    assert scored.loc[2, "recommendation"] == "RUN FULL JITTER TEST"
    assert pd.isna(scored.loc[3, "predicted_jitter_ps"])
    assert comparison is None


def test_batch_candidate_without_inhibit_argument_still_runs_full_test() -> None:
    scored, _, _ = predict_batch(valid_frame(1), FixedModel(1.0), artifact())

    assert scored.loc[0, "illustrative_candidate_skip"]
    assert scored.loc[0, "recommendation"] == "RUN FULL JITTER TEST"
    assert scored.loc[0, "safety_inhibit_active"]
    assert "52 measured failing dies" in scored.loc[0, "recommendation_reason"]


def test_batch_missing_feature_column_marks_rows_invalid_without_predicting() -> None:
    frame = valid_frame().drop(columns=["ring_osc_ghz"])
    model = FixedModel()
    scored, summary, comparison = predict_batch(frame, model, artifact())
    assert summary["invalid_records"] == len(frame)
    assert summary["successfully_predicted_records"] == 0
    assert model.calls == 0
    assert comparison is None
    assert scored["recommendation"].eq("RUN FULL JITTER TEST").all()


@pytest.mark.parametrize("actual_column", ["rms_jitter_ps", "actual_jitter_ps"])
def test_batch_prediction_supports_both_actual_measurement_columns(
    actual_column: str,
) -> None:
    frame = valid_frame(3)
    frame[actual_column] = [1.0, 1.5, 2.0]
    original = frame.copy(deep=True)
    model = FixedModel(1.4)

    scored, summary, comparison = predict_batch(frame, model, artifact())

    assert summary["successfully_predicted_records"] == 3
    assert comparison is not None
    assert comparison["sufficient"]
    assert comparison["count"] == 3
    assert scored["rms_jitter_ps"].tolist() == [1.0, 1.5, 2.0]
    assert model.input_columns == list(FEATURE_COLUMNS)
    pd.testing.assert_frame_equal(frame, original)


def test_batch_prediction_prefers_rms_target_without_overwriting_alias() -> None:
    frame = valid_frame(2)
    frame["rms_jitter_ps"] = [1.0, 2.0]
    frame["actual_jitter_ps"] = [8.0, 9.0]
    model = FixedModel()

    scored, _, comparison = predict_batch(frame, model, artifact())

    assert comparison is not None
    assert comparison["mae_ps"] == pytest.approx(0.5)
    assert scored["rms_jitter_ps"].tolist() == [1.0, 2.0]
    assert scored["actual_jitter_ps"].tolist() == [8.0, 9.0]
    assert model.input_columns == list(FEATURE_COLUMNS)


def test_batch_prediction_without_actual_measurements_is_prediction_only() -> None:
    frame = valid_frame(3)
    model = FixedModel()

    scored, summary, comparison = predict_batch(frame, model, artifact())

    assert summary["successfully_predicted_records"] == 3
    assert comparison is None
    assert scored["predicted_jitter_ps"].notna().all()
    assert model.input_columns == list(FEATURE_COLUMNS)


def test_global_feature_importance_covers_only_the_seven_model_features() -> None:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        rng.uniform(0.0, 10.0, size=(80, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
    )
    frame["actual_jitter_ps"] = 99.0
    target = pd.Series(WeightedModel().predict(frame.loc[:, FEATURE_COLUMNS]))
    model = WeightedModel()

    importance = compute_global_feature_importance(
        model, frame, target, sample_size=40, n_repeats=2
    )

    assert set(importance["feature"]) == set(FEATURE_COLUMNS)
    assert len(importance) == len(FEATURE_COLUMNS)
    assert model.input_columns == list(FEATURE_COLUMNS)
    assert "actual_jitter_ps" not in model.input_columns


def test_local_explanation_matches_prediction_and_reports_all_feature_effects() -> None:
    features = {name: 1.0 for name in FEATURE_COLUMNS}
    features["vco_band_code"] = 5.0
    features["rms_jitter_ps"] = 100.0
    features["defect_injected"] = 1
    reference = {name: 1.0 for name in FEATURE_COLUMNS}
    model = WeightedModel()

    explanation = explain_prediction(features, model, artifact(), reference)

    assert explanation["valid"]
    assert explanation["prediction_ps"] == pytest.approx(
        model.predict(pd.DataFrame([[features[name] for name in FEATURE_COLUMNS]],
                                   columns=FEATURE_COLUMNS))[0]
    )
    assert explanation["reference_prediction_ps"] == pytest.approx(0.88)
    effects = explanation["effects"].set_index("feature")
    assert set(effects.index) == set(FEATURE_COLUMNS)
    assert effects.loc["vco_band_code", "effect_ps"] == pytest.approx(0.04)
    assert len(effects) == 7
    assert model.input_columns == list(FEATURE_COLUMNS)
    assert "non-additive" in explanation["method"]


def test_local_explanation_handles_invalid_and_out_of_range_inputs() -> None:
    model = FixedModel()
    reference = {name: 1.0 for name in FEATURE_COLUMNS}
    invalid_features = {name: 1.0 for name in FEATURE_COLUMNS}
    invalid_features["lock_time_us"] = -1.0

    invalid = explain_prediction(invalid_features, model, artifact(), reference)

    assert not invalid["valid"]
    assert invalid["prediction_ps"] is None
    assert model.calls == 0

    out_of_range_features = {name: 1.0 for name in FEATURE_COLUMNS}
    out_of_range_features["vco_band_code"] = 11.0
    out_of_range = explain_prediction(
        out_of_range_features, model, artifact(), reference
    )
    assert out_of_range["valid"]
    assert out_of_range["validation_status"] == "OUT OF TRAINING RANGE"
    assert out_of_range["out_of_training_range_features"] == ["vco_band_code"]
    assert out_of_range["prediction_ps"] == pytest.approx(1.3)


def test_what_if_comparison_reports_prediction_changes_and_keeps_full_test() -> None:
    baseline = {name: 1.0 for name in FEATURE_COLUMNS}
    modified = dict(baseline)
    modified["vco_band_code"] = 5.0
    model = WeightedModel()

    result = compare_what_if_predictions(baseline, modified, model, artifact())

    assert result["baseline"]["recommendation"] == "RUN FULL JITTER TEST"
    assert result["modified"]["recommendation"] == "RUN FULL JITTER TEST"
    assert result["prediction_delta_ps"] == pytest.approx(0.04)
    assert result["upper_bound_delta_ps"] == pytest.approx(0.04)
    assert not result["illustrative_candidate_changed"]
    assert "remained unchanged" in result["change_summary"]
    assert model.input_columns == list(FEATURE_COLUMNS)


def test_what_if_out_of_range_or_invalid_inputs_fail_safe() -> None:
    baseline = {name: 1.0 for name in FEATURE_COLUMNS}
    out_of_range = dict(baseline)
    out_of_range["vco_band_code"] = 11.0
    result = compare_what_if_predictions(
        baseline, out_of_range, WeightedModel(), artifact()
    )
    assert result["modified"]["validation_status"] == "OUT OF TRAINING RANGE"
    assert result["modified"]["recommendation"] == "RUN FULL JITTER TEST"
    assert result["validation_status_changed"]
    assert "status changed" in result["change_summary"]

    invalid = dict(baseline)
    invalid["lock_time_us"] = -1.0
    result = compare_what_if_predictions(
        baseline, invalid, WeightedModel(), artifact()
    )
    assert result["modified"]["validation_status"] == "INVALID"
    assert result["modified"]["predicted_jitter_ps"] is None
    assert result["modified"]["recommendation"] == "RUN FULL JITTER TEST"
    assert result["prediction_delta_ps"] is None


def test_what_if_detects_candidate_change_without_changing_full_test_recommendation() -> None:
    baseline = {name: 1.0 for name in FEATURE_COLUMNS}
    modified = dict(baseline)
    for feature in FEATURE_COLUMNS[:-1]:
        modified[feature] = 10.0

    result = compare_what_if_predictions(
        baseline, modified, WeightedModel(), artifact()
    )

    assert result["illustrative_candidate_changed"]
    assert result["baseline"]["recommendation"] == "RUN FULL JITTER TEST"
    assert result["modified"]["recommendation"] == "RUN FULL JITTER TEST"
    assert "eligibility" in result["change_summary"]


def test_manual_prediction_validation_and_actual_error_history() -> None:
    frame = valid_frame(1)
    features = frame.iloc[0].to_dict()
    model = FixedModel(1.5)
    from data_utils import decide_jitter_test

    result = decide_jitter_test(features, model, artifact())
    assert result["predicted_jitter_ps"] == pytest.approx(1.5)
    assert result["upper_bound_ps"] == pytest.approx(1.6)
    assert result["recommendation"] == "RUN FULL JITTER TEST"
    assert result["illustrative_candidate_skip"]

    history = create_history_entry(features, result, actual_jitter_ps=1.2)
    assert history["actual_jitter_ps"] == pytest.approx(1.2)
    assert history["signed_error_ps"] == pytest.approx(0.3)
    assert history["absolute_error_ps"] == pytest.approx(0.3)
    assert history["timestamp_utc"]
    assert history["recommendation"] == "RUN FULL JITTER TEST"
    assert append_history_entry([], history) == [history]
    bounded = append_history_entry([history], {"timestamp_utc": "later"}, maximum_entries=1)
    assert bounded == [{"timestamp_utc": "later"}]
    assert append_history_entries([], [history, history], maximum_entries=1) == [history]
    with pytest.raises(ValueError):
        append_history_entry([], history, maximum_entries=0)


def test_actual_comparison_metrics_and_largest_errors() -> None:
    frame = pd.DataFrame(
        {
            "rms_jitter_ps": [1.0, 2.0, 3.0],
            "predicted_jitter_ps": [1.1, 1.8, 3.4],
        }
    )
    result = compare_actual_predictions(frame)
    assert result is not None
    assert result["sufficient"]
    assert result["count"] == 3
    assert result["mae_ps"] == pytest.approx(0.2333333333)
    assert result["rmse_ps"] == pytest.approx(np.sqrt(0.07))
    assert len(result["largest_errors"]) == 3

    alias_only = pd.DataFrame(
        {
            "actual_jitter_ps": [1.0, 2.0, 3.0],
            "predicted_jitter_ps": [1.1, 1.8, 3.4],
        }
    )
    alias_original = alias_only.copy(deep=True)
    alias_result = compare_actual_predictions(alias_only)
    assert alias_result is not None
    assert alias_result["mae_ps"] == pytest.approx(result["mae_ps"])
    assert alias_result["table"]["rms_jitter_ps"].tolist() == [1.0, 2.0, 3.0]
    pd.testing.assert_frame_equal(alias_only, alias_original)

    no_actual = compare_actual_predictions(
        pd.DataFrame({"predicted_jitter_ps": [1.0, 2.0]})
    )
    assert no_actual is None
    insufficient = compare_actual_predictions(
        pd.DataFrame(
            {"rms_jitter_ps": [1.0], "predicted_jitter_ps": [1.2]}
        )
    )
    assert insufficient is not None
    assert not insufficient["sufficient"]


def test_drift_comparison_and_summary_detect_out_of_range_shift() -> None:
    baseline = pd.DataFrame(
        {name: np.linspace(0.0, 1.0, 100) for name in FEATURE_COLUMNS}
    )
    uploaded = pd.DataFrame(
        {name: np.linspace(0.0, 1.0, 100) for name in FEATURE_COLUMNS}
    )
    uploaded["vco_band_code"] = 50.0
    ranges = {name: (0.0, 1.0) for name in FEATURE_COLUMNS}

    result = compare_feature_distributions(uploaded, baseline, ranges)
    summary = summarize_drift(result)

    assert len(result) == len(FEATURE_COLUMNS)
    shifted = result.loc[result["feature"].eq("vco_band_code")].iloc[0]
    assert shifted["out_of_training_range_percent"] == 100.0
    assert shifted["drift_warning"] == "SHIFT WARNING"
    assert summary["warning_features"] >= 1
    assert "not proof" in summary["interpretation"]


def test_locktime_batch_invalid_inputs_statistics_and_cap() -> None:
    result, summary = analyze_locktime_values([10.0, 100.0, -1.0, "bad"])
    assert result.loc[0, "adaptive_wait_us"] == pytest.approx(13.0)
    assert result.loc[1, "adaptive_wait_us"] == pytest.approx(60.0)
    assert summary["valid_records"] == 2
    assert summary["invalid_records"] == 2
    assert summary["mean_adaptive_wait_us"] == pytest.approx(36.5)
    assert summary["minimum_adaptive_wait_us"] == pytest.approx(13.0)
    assert summary["maximum_adaptive_wait_us"] == pytest.approx(60.0)
    with pytest.raises(ValueError):
        analyze_locktime_values([1.0], fixed_wait_us=0)


def test_pdf_report_generation_returns_downloadable_pdf_bytes() -> None:
    content = build_engineering_pdf_report(
        dataset_info={"name": "sample.csv", "rows": 2, "columns": 7},
        input_measurements={"vco_band_code": 1.0},
        prediction_results={"recommendation": "RUN FULL JITTER TEST"},
        actual_comparison=None,
        model_performance={"test_mae_ps": 0.05, "test_r2": 0.8},
        locktime_analysis={"savings_percent": 70.75},
        warnings=["Synthetic data only."],
        recommendations=["Keep full testing enabled."],
    )
    assert content.startswith(b"%PDF-")
    assert content.rstrip().endswith(b"%%EOF")
    assert len(content) > 1000


def test_report_exports_reopen_with_shared_timestamp_and_complete_payload(tmp_path) -> None:
    timestamp = "2026-11-02T18:00:00+00:00"
    payload = {
        "dataset_info": {"name": "PLL <sample> & wafer.csv", "rows": 12, "invalid_rows": 1},
        "input_measurements": {
            "vco_band_code": 2,
            "formula_eq": "=1+1",
            "formula_plus": "+cmd",
            "formula_minus": "-cmd",
            "formula_at": "@SUM(A1:A2)",
            "unicode": "λ μ <tag>",
        },
        "prediction_results": {
            "predicted_jitter_ps": 1.25,
            "recommendation": "RUN FULL JITTER TEST",
        },
        "actual_comparison": None,
        "model_performance": {"test_mae_ps": 0.057, "test_r2": 0.85},
        "locktime_analysis": {"savings_percent": 70.75},
        "warnings": ["Safety warning: keep full testing enabled.", "Special <tag> & text"],
        "recommendations": ["Validate on representative production data."],
    }

    exports = build_engineering_report_exports(payload, generated_at_utc=timestamp)
    json_path = tmp_path / "engineering_report.json"
    csv_path = tmp_path / "engineering_report.csv"
    pdf_path = tmp_path / "engineering_report.pdf"
    json_path.write_bytes(exports["json"])
    csv_path.write_bytes(exports["csv"])
    pdf_path.write_bytes(exports["pdf"])

    parsed_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed_json["generated_at_utc"] == timestamp
    assert parsed_json["dataset_info"]["name"] == "PLL <sample> & wafer.csv"
    assert parsed_json["prediction_results"]["recommendation"] == "RUN FULL JITTER TEST"
    assert parsed_json["model_performance"]["test_mae_ps"] == pytest.approx(0.057)
    assert parsed_json["warnings"] == payload["warnings"]

    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        csv_rows = list(csv.DictReader(csv_file))
    assert list(csv_rows[0]) == ["generated_at_utc", "section", "metric", "value"]
    assert all(row["generated_at_utc"] == timestamp for row in csv_rows)
    assert any(row["section"] == "warnings" for row in csv_rows)
    assert any(row["section"] == "recommendations" for row in csv_rows)
    for metric, escaped in (
        ("formula_eq", "'=1+1"),
        ("formula_plus", "'+cmd"),
        ("formula_minus", "'-cmd"),
        ("formula_at", "'@SUM(A1:A2)"),
    ):
        formula_row = next(row for row in csv_rows if row["metric"] == metric)
        assert formula_row["value"] == escaped
    assert any(row["metric"] == "test_mae_ps" and row["value"] == "0.057" for row in csv_rows)

    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    extracted_pdf = "\n".join(pages)
    assert len(pages) >= 1
    assert timestamp in extracted_pdf
    for section in (
        "Dataset information",
        "Input measurements",
        "Prediction results",
        "Actual versus predicted comparison",
        "Original model performance",
        "Adaptive lock-time analysis",
        "Safety warnings",
        "Engineering recommendations",
        "Synthetic-data / non-production disclaimer",
    ):
        assert section in extracted_pdf
    assert "Page 1" in pages[0]
    assert "52 measured failing" in " ".join(extracted_pdf.split())
    assert r"\u03bb \u03bc <tag>" in extracted_pdf
    assert "PLL <sample> & wafer.csv" in extracted_pdf


def test_report_exports_handle_empty_optional_sections_and_multipage_content(tmp_path) -> None:
    timestamp = "2026-11-02T18:00:00+00:00"
    exports = build_engineering_report_exports(
        {"dataset_info": {}, "warnings": [], "recommendations": []},
        generated_at_utc=timestamp,
    )

    empty_json_path = tmp_path / "empty_report.json"
    empty_csv_path = tmp_path / "empty_report.csv"
    empty_pdf_path = tmp_path / "empty_report.pdf"
    empty_json_path.write_bytes(exports["json"])
    empty_csv_path.write_bytes(exports["csv"])
    empty_pdf_path.write_bytes(exports["pdf"])

    parsed_json = json.loads(empty_json_path.read_text(encoding="utf-8"))
    assert parsed_json["generated_at_utc"] == timestamp
    with empty_csv_path.open(encoding="utf-8", newline="") as csv_file:
        empty_csv_rows = list(csv.DictReader(csv_file))
    assert empty_csv_rows[0]["metric"] == "generated_at_utc"
    assert empty_csv_rows[0]["value"] == timestamp
    reader = PdfReader(str(empty_pdf_path))
    extracted_pages = [page.extract_text() or "" for page in reader.pages]
    assert "No data available for this section." in "\n".join(extracted_pages)
    assert "No measured actual jitter was provided." in "\n".join(extracted_pages)

    long_exports = build_engineering_report_exports(
        {
            "dataset_info": {"name": "multi-page sample", "rows": 100},
            "model_performance": {"detail": "Long report row. " * 1200 + "FULL-ROW-END"},
            "warnings": [f"Warning {index}: review this condition." for index in range(70)],
            "recommendations": ["Keep full testing enabled."],
        },
        generated_at_utc=timestamp,
    )
    long_pdf_path = tmp_path / "long_report.pdf"
    long_pdf_path.write_bytes(long_exports["pdf"])
    long_reader = PdfReader(str(long_pdf_path))
    long_pages = [page.extract_text() or "" for page in long_reader.pages]
    assert len(long_pages) > 1
    for page_number, text in enumerate(long_pages, start=1):
        assert f"Page {page_number}" in text
    assert "Warning 69: review this condition." in "\n".join(long_pages)
    assert "FULL-ROW-END" in "\n".join(long_pages)
    assert timestamp in "\n".join(long_pages)
