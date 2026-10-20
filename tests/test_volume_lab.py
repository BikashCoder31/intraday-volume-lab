"""Regression checks for the mistakes found in the supplied research notebook."""

import importlib
import importlib.util
import json

import numpy as np
import pandas as pd
import pytest


def load_lab():
    assert importlib.util.find_spec("volume_lab"), (
        "The maintained data pipeline is not implemented yet"
    )
    return importlib.import_module("volume_lab")


def ticks(day="2026-01-05", minutes=6, symbol="ALPHA"):
    """Hand-checkable counter: each complete minute adds exactly 60 shares."""
    start = pd.Timestamp(f"{day} 09:15:00", tz="Asia/Kolkata")
    rows = []
    for minute in range(minutes):
        for second, counter in ((0, minute * 60), (30, minute * 60 + 30)):
            rows.append(
                dict(
                    timestamp=start + pd.Timedelta(minutes=minute, seconds=second),
                    symbol=symbol,
                    ltp=100.0,
                    bid_price=99.0,
                    ask_price=101.0,
                    bid_size=100.0,
                    ask_size=100.0,
                    vol_traded_today=counter,
                )
            )
    # The final tick in each minute completes its 60-share interval.
    for minute in range(minutes):
        rows.append(
            dict(
                rows[0],
                timestamp=start + pd.Timedelta(minutes=minute, seconds=59),
                vol_traded_today=(minute + 1) * 60,
            )
        )
    return pd.DataFrame(rows).sort_values("timestamp", kind="stable").reset_index(drop=True)


def write_jsonl(path, frame, unit="s"):
    records = frame.copy()
    records["timestamp"] = records["timestamp"].map(
        lambda value: value.value // (10**9 if unit == "s" else 10**6)
    )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in records.to_dict("records")), encoding="utf-8"
    )


def sample_bundle(lab, dates, symbols=None, values=None):
    count = len(dates)
    values = np.asarray(values if values is not None else np.arange(count) + 1, dtype=np.float32)
    return lab.Samples(
        high=np.repeat(values[:, None, None], 10, axis=1).reshape(count, 2, 5),
        context=np.repeat(values[:, None], 6, axis=1),
        target=values,
        metadata=pd.DataFrame(
            {"timestamp": pd.to_datetime(dates, utc=True), "symbol": symbols or ["ALPHA"] * count}
        ),
    )


def test_volume_is_interval_shares_not_cumulative_counter():
    lab = load_lab()
    bars = lab.minute_bars(ticks())
    assert bars["volume"].tolist() == [60.0] * 6
    samples = lab.make_samples(bars, lookback=2)
    np.testing.assert_allclose(np.expm1(samples.target), [60.0] * 4, rtol=1e-6)
    np.testing.assert_allclose(samples.high[0, :, 1], [60.0, 60.0])


def test_unknown_first_counter_drops_the_partial_first_minute():
    lab = load_lab()
    frame = ticks()
    frame["vol_traded_today"] += 500
    bars = lab.minute_bars(frame)
    assert bars["timestamp"].min().strftime("%H:%M") == "09:16"
    assert bars["volume"].tolist() == [60.0] * 5


def test_daily_counter_reset_does_not_create_negative_volume():
    lab = load_lab()
    bars = lab.minute_bars(pd.concat([ticks(), ticks("2026-01-06")]))
    assert bars["volume"].tolist() == [60.0] * 12
    samples = lab.make_samples(bars, lookback=2)
    assert len(samples.target) == 8
    assert all(samples.metadata["timestamp"].dt.date == samples.metadata["feature_start"].dt.date)


def test_mid_session_counter_decrease_is_rejected():
    lab = load_lab()
    frame = ticks()
    frame.loc[frame.index[-1], "vol_traded_today"] = 5
    with pytest.raises(ValueError, match="counter"):
        lab.minute_bars(frame)


def test_feature_windows_do_not_bridge_missing_minutes():
    lab = load_lab()
    bars = lab.minute_bars(ticks(minutes=7))
    bars = bars[bars["timestamp"].dt.minute != 18]
    samples = lab.make_samples(bars, lookback=2)
    assert samples.metadata["timestamp"].dt.strftime("%H:%M").tolist() == ["09:17", "09:21"]


def test_raw_feed_gap_does_not_move_missing_volume_into_the_next_minute():
    lab = load_lab()
    frame = ticks(minutes=8)
    frame = frame[frame["timestamp"].dt.minute != 17]
    bars = lab.minute_bars(frame)
    assert bars["timestamp"].dt.strftime("%H:%M").tolist() == [
        "09:15",
        "09:16",
        "09:19",
        "09:20",
        "09:21",
        "09:22",
    ]
    assert bars["volume"].tolist() == [60.0] * 6
    samples = lab.make_samples(bars, lookback=2)
    assert samples.metadata["timestamp"].dt.strftime("%H:%M").tolist() == ["09:21", "09:22"]


def test_later_observations_cannot_change_earlier_features():
    lab = load_lab()
    bars = lab.minute_bars(ticks(minutes=8))
    before = lab.make_samples(bars, lookback=2)
    changed = bars.copy()
    changed.loc[changed["timestamp"].dt.minute >= 20, ["volume", "close"]] *= 100
    after = lab.make_samples(changed, lookback=2)
    earlier = before.metadata["timestamp"].dt.minute < 20
    np.testing.assert_array_equal(before.high[earlier], after.high[earlier])
    np.testing.assert_array_equal(before.context[earlier], after.context[earlier])
    assert (before.metadata["feature_end"] < before.metadata["timestamp"]).all()


def test_multiple_files_for_one_symbol_are_combined(tmp_path):
    lab = load_lab()
    write_jsonl(tmp_path / "ALPHA-NSE-day1.json", ticks())
    write_jsonl(tmp_path / "ALPHA-NSE-day2.jsonl", ticks("2026-01-06"))
    loaded = lab.load_ticks(tmp_path)
    assert len(loaded) == 36
    assert loaded["timestamp"].dt.date.nunique() == 2
    assert loaded["symbol"].unique().tolist() == ["ALPHA"]


def test_overlapping_input_files_do_not_double_count_ticks(tmp_path):
    lab = load_lab()
    write_jsonl(tmp_path / "a.jsonl", ticks())
    write_jsonl(tmp_path / "b.jsonl", ticks())
    loaded = lab.load_ticks(tmp_path)
    assert len(loaded) == 18
    assert lab.minute_bars(loaded)["volume"].tolist() == [60.0] * 6


def test_malformed_json_reports_file_and_line(tmp_path):
    lab = load_lab()
    (tmp_path / "broken.jsonl").write_text('{}\n{"invalid"\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"broken.jsonl.*2"):
        lab.load_ticks(tmp_path)


@pytest.mark.parametrize("unit", ["s", "ms"])
def test_timestamp_units_are_explicit_and_preserve_exchange_time(tmp_path, unit):
    lab = load_lab()
    write_jsonl(tmp_path / "ALPHA.jsonl", ticks(), unit=unit)
    loaded = lab.load_ticks(tmp_path, timestamp_unit=unit)
    assert loaded["timestamp"].min().isoformat() == "2026-01-05T09:15:00+05:30"


def test_wrong_timestamp_unit_produces_actionable_error(tmp_path):
    lab = load_lab()
    write_jsonl(tmp_path / "ALPHA.jsonl", ticks(), unit="ms")
    with pytest.raises(ValueError, match="timestamp"):
        lab.load_ticks(tmp_path, timestamp_unit="s")


def test_invalid_timezone_is_reported_as_an_input_error(tmp_path):
    lab = load_lab()
    write_jsonl(tmp_path / "ALPHA.jsonl", ticks())
    with pytest.raises(ValueError, match="timezone"):
        lab.load_ticks(tmp_path, exchange_timezone="Asia/KatmanduTypo")


def test_ingestion_settings_remain_available_for_run_provenance(tmp_path):
    lab = load_lab()
    write_jsonl(tmp_path / "ALPHA.jsonl", ticks(), unit="ms")
    samples = lab.make_samples(
        lab.minute_bars(lab.load_ticks(tmp_path, timestamp_unit="ms")), lookback=2
    )
    assert samples.metadata.attrs["ingestion"] == {
        "timestamp_unit": "ms",
        "exchange_timezone": "Asia/Kolkata",
        "session_policy": "local_calendar_date",
    }


def test_report_export_rejects_artifacts_from_another_run():
    from tools.export_report import validate_report_run

    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "outputs": [
                    {
                        "output_type": "display_data",
                        "data": {
                            "application/vnd.intraday-volume-lab.run+json": {
                                "run_id": "notebook-run"
                            }
                        },
                    }
                ],
            }
        ]
    }
    with pytest.raises(ValueError, match="same run"):
        validate_report_run(notebook, {"run_id": "overwritten-cli-run"})
    validate_report_run(notebook, {"run_id": "notebook-run"})


def test_report_export_rejects_cells_executed_out_of_order():
    from tools.export_report import validate_report_run

    notebook = {
        "cells": [
            {"cell_type": "code", "execution_count": 4, "outputs": []},
            {"cell_type": "code", "execution_count": 2, "outputs": []},
        ]
    }
    with pytest.raises(ValueError, match="in order"):
        validate_report_run(notebook, {"run_id": "run"})


@pytest.mark.parametrize(
    "field,value",
    [("bid_size", -1), ("ltp", float("nan")), ("ask_price", 98), ("vol_traded_today", -1)],
)
def test_invalid_market_values_are_rejected(tmp_path, field, value):
    lab = load_lab()
    frame = ticks()
    frame.loc[0, field] = value
    write_jsonl(tmp_path / "invalid.jsonl", frame)
    with pytest.raises(ValueError):
        lab.load_ticks(tmp_path)


def test_empty_dataset_has_actionable_error(tmp_path):
    lab = load_lab()
    with pytest.raises(ValueError, match="JSON"):
        lab.load_ticks(tmp_path)


def test_split_uses_whole_dates_across_all_symbols():
    lab = load_lab()
    dates = [f"2026-01-{day:02} 04:00" for day in range(5, 15) for _ in range(2)]
    samples = sample_bundle(lab, dates, ["ALPHA", "BETA"] * 10)
    splits = lab.chronological_split(samples.take(np.arange(19, -1, -1)))
    assert [len(s.target) for s in splits.values()] == [14, 2, 4]
    assert (
        splits["train"].metadata["timestamp"].max()
        < splits["validation"].metadata["timestamp"].min()
    )
    assert (
        splits["validation"].metadata["timestamp"].max()
        < splits["test"].metadata["timestamp"].min()
    )


def test_split_rejects_insufficient_sessions():
    lab = load_lab()
    with pytest.raises(ValueError, match="three"):
        lab.chronological_split(sample_bundle(lab, ["2026-01-05", "2026-01-06"]))


def test_scalers_fit_training_data_only():
    lab = load_lab()
    splits = {
        "train": sample_bundle(lab, ["2026-01-05", "2026-01-06"], values=[1, 3]),
        "validation": sample_bundle(lab, ["2026-01-07"], values=[1000]),
        "test": sample_bundle(lab, ["2026-01-08"], values=[2000]),
    }
    prepared = lab.prepare_data(splits)
    assert prepared["target_scaler"].mean_.tolist() == [2.0]
    assert prepared["high_scaler"].center_.tolist() == [2.0] * 5
    assert prepared["context_scaler"].center_.tolist() == [2.0] * 6


def test_unseen_evaluation_symbol_is_not_given_an_untrained_embedding():
    lab = load_lab()
    splits = {
        "train": sample_bundle(lab, ["2026-01-05"], ["ALPHA"]),
        "validation": sample_bundle(lab, ["2026-01-06"], ["BETA"]),
        "test": sample_bundle(lab, ["2026-01-07"], ["ALPHA"]),
    }
    with pytest.raises(ValueError, match="BETA"):
        lab.prepare_data(splits)


def test_metrics_report_share_units_and_handle_zero_volume():
    lab = load_lab()
    result = lab.regression_metrics(np.log1p([0.0, 100.0]), np.log1p([0.0, 50.0]))
    assert result["shares"]["mae"] == pytest.approx(25.0)
    assert result["shares"]["rmse"] == pytest.approx(35.3553390593)
    assert result["shares"]["smape_pct"] == pytest.approx(100 / 3)
    assert result["log_volume"]["mae"] == pytest.approx(0.3416474421)
    assert lab.regression_metrics([0.0, 0.0], [0.0, 0.0])["shares"]["smape_pct"] == 0


def test_non_finite_predictions_are_rejected():
    lab = load_lab()
    with pytest.raises(ValueError, match="finite"):
        lab.regression_metrics([1.0, 2.0], [float("nan"), 1.0])


def test_synthetic_data_is_reproducible_with_daily_counter_resets(tmp_path):
    lab = load_lab()
    first, second = tmp_path / "first", tmp_path / "second"
    lab.generate_demo_data(first, seed=7, sessions=3, minutes=20)
    lab.generate_demo_data(second, seed=7, sessions=3, minutes=20)
    assert {p.name: p.read_bytes() for p in first.glob("*.jsonl")} == {
        p.name: p.read_bytes() for p in second.glob("*.jsonl")
    }
    loaded = lab.load_ticks(first)
    assert sorted(loaded["symbol"].unique()) == ["ALPHA", "BETA"]
    bars = lab.minute_bars(loaded)
    assert len(bars) == 120
    assert (bars["volume"] > 0).all()
    assert (
        loaded.groupby(["symbol", loaded["timestamp"].dt.date])["vol_traded_today"]
        .first()
        .eq(0)
        .all()
    )
    with pytest.raises(FileExistsError):
        lab.generate_demo_data(first)


@pytest.mark.integration
def test_cpu_training_saves_a_reloadable_model_and_json_results(tmp_path):
    lab = load_lab()
    tf = pytest.importorskip("tensorflow")
    frames = [
        ticks(f"2026-01-{day:02}", minutes=24, symbol=symbol)
        for day in range(5, 9)
        for symbol in ("ALPHA", "BETA")
    ]
    samples = lab.make_samples(lab.minute_bars(pd.concat(frames)), lookback=3)
    result = lab.train_experiment(
        samples, tmp_path / "run", epochs=1, batch_size=16, data_label="Synthetic test fixture"
    )
    assert result["data_label"] == "Synthetic test fixture"
    assert result["sample_counts"]["test"] > 0
    assert np.isfinite(result["metrics"]["test"]["model"]["shares"]["mae"])
    assert (
        json.loads((tmp_path / "run" / "results.json").read_text())["target"]
        == "next-minute interval volume"
    )
    model = tf.keras.models.load_model(tmp_path / "run" / "model.keras")
    assert model.output_shape == (None, 1)
    assert (tmp_path / "run" / "preprocessing.pkl").exists()
