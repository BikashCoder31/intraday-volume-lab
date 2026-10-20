"""Intraday Volume Lab: maintained research utilities for Bikash Kushwaha's portfolio.

The notebook is the narrative entry point. Numerical transformations live here so
their behavior can be checked without importing TensorFlow or starting training.
See SOURCE_ACKNOWLEDGEMENTS.md for the supplied research source and contributions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfoNotFoundError

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

PRICE_COLUMNS = ["ltp", "bid_price", "ask_price"]
NUMERIC_COLUMNS = PRICE_COLUMNS + ["bid_size", "ask_size", "vol_traded_today"]
HIGH_FEATURES = ["close", "volume", "spread", "imbalance", "range"]
CONTEXT_FEATURES = [
    "mean_close",
    "std_close",
    "total_volume",
    "mean_spread",
    "mean_imbalance",
    "range",
]


@dataclass(frozen=True)
class Samples:
    """Aligned feature arrays, log-volume targets, and causal sample provenance."""

    high: np.ndarray
    context: np.ndarray
    target: np.ndarray
    metadata: pd.DataFrame

    def take(self, positions) -> Samples:
        positions = np.asarray(positions)
        return Samples(
            self.high[positions],
            self.context[positions],
            self.target[positions],
            self.metadata.iloc[positions].reset_index(drop=True),
        )


def load_ticks(
    folder: str | Path,
    *,
    timestamp_unit: Literal["s", "ms"] = "s",
    exchange_timezone: str = "Asia/Kolkata",
) -> pd.DataFrame:
    """Combine JSON-lines files, validating schema instead of silently losing rows."""
    if timestamp_unit not in {"s", "ms"}:
        raise ValueError("timestamp_unit must be 's' or 'ms'")
    folder = Path(folder)
    paths = sorted(
        path
        for path in folder.glob("*")
        if path.suffix.lower() in {".json", ".jsonl"} and path.is_file()
    )
    if not paths:
        raise ValueError(f"No JSON-lines input files found in {folder}")
    records, manifest = [], []
    for path in paths:
        manifest.append(
            {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path.name}, line {line_number}: invalid JSON") from error
                if not isinstance(record, dict):
                    raise ValueError(f"{path.name}, line {line_number}: expected a JSON object")
                record.setdefault("symbol", path.stem.split("-NSE", 1)[0])
                records.append(record)
    frame = pd.DataFrame.from_records(records)
    missing = set(["timestamp", "symbol", *NUMERIC_COLUMNS]) - set(frame.columns)
    if frame.empty or missing:
        raise ValueError(
            f"Input is empty or missing required columns: {', '.join(sorted(missing))}"
        )
    frame = frame[["timestamp", "symbol", *NUMERIC_COLUMNS]].copy()
    if not frame["symbol"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Every symbol must be a non-empty string")
    frame["symbol"] = frame["symbol"].str.strip()
    for column in NUMERIC_COLUMNS + ["timestamp"]:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (ValueError, TypeError) as error:
            raise ValueError(f"{column} must contain numeric values") from error
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain finite values")
    if (frame[PRICE_COLUMNS] <= 0).any().any():
        raise ValueError("Prices must be positive")
    if (frame[["bid_size", "ask_size", "vol_traded_today"]] < 0).any().any():
        raise ValueError("Sizes and cumulative counters must be non-negative")
    if (frame["ask_price"] < frame["bid_price"]).any():
        raise ValueError("Crossed quotes: ask_price must be at least bid_price")
    try:
        converted = pd.to_datetime(frame["timestamp"], unit=timestamp_unit, utc=True)
        if not converted.dt.year.between(2000, 2100).all():
            raise ValueError("timestamps must fall within 2000–2100; check the timestamp unit")
        frame["timestamp"] = converted.dt.tz_convert(exchange_timezone)
    except (ValueError, TypeError, OverflowError, ZoneInfoNotFoundError) as error:
        raise ValueError(
            f"Invalid timestamp values; check unit={timestamp_unit!r} and timezone={exchange_timezone!r}"
        ) from error
    frame = (
        frame.drop_duplicates()
        .sort_values(["symbol", "timestamp", "vol_traded_today"], kind="stable")
        .reset_index(drop=True)
    )
    frame.attrs["input_manifest"] = manifest
    frame.attrs["ingestion"] = {
        "timestamp_unit": timestamp_unit,
        "exchange_timezone": exchange_timezone,
        "session_policy": "local_calendar_date",
    }
    return frame


def minute_bars(ticks: pd.DataFrame) -> pd.DataFrame:
    """Turn session-local cumulative counters into complete interval-volume bars."""
    if ticks.empty:
        raise ValueError("No ticks to aggregate")
    if not isinstance(ticks["timestamp"].dtype, pd.DatetimeTZDtype):
        raise ValueError("Load ticks with an explicit timestamp unit and timezone first")
    ticks = ticks.sort_values(["symbol", "timestamp", "vol_traded_today"], kind="stable")
    parts = []
    for (symbol, _session), group in ticks.groupby(
        ["symbol", ticks["timestamp"].dt.date], sort=True
    ):
        group = group.copy().set_index("timestamp")
        increment = group["vol_traded_today"].diff()
        if group["vol_traded_today"].iloc[0] == 0:
            increment.iloc[0] = 0
        if (increment.dropna() < 0).any():
            raise ValueError(f"Cumulative counter decreased within a session for {symbol}")
        # A counter jump across an unobserved minute cannot be allocated to the
        # resumed minute. Discard that partial minute before building windows.
        minute_steps = group.index.to_series().dt.floor("min").diff()
        increment.loc[minute_steps > pd.Timedelta(minutes=1)] = np.nan
        group["increment"] = increment
        group["known"] = increment.notna()
        group["spread"] = group["ask_price"] - group["bid_price"]
        sizes = group["bid_size"] + group["ask_size"]
        group["imbalance"] = np.divide(
            group["bid_size"] - group["ask_size"],
            sizes,
            out=np.zeros(len(group)),
            where=sizes.to_numpy() != 0,
        )
        grouped = group.resample("1min")
        bars = grouped.agg(
            open=("ltp", "first"),
            high=("ltp", "max"),
            low=("ltp", "min"),
            close=("ltp", "last"),
            volume=("increment", "sum"),
            spread=("spread", "mean"),
            imbalance=("imbalance", "mean"),
            known=("known", "all"),
            observations=("ltp", "count"),
        )
        bars = (
            bars[(bars["observations"] > 0) & bars["known"]]
            .drop(columns=["known", "observations"])
            .dropna()
        )
        bars["symbol"] = symbol
        parts.append(bars.reset_index())
    result = (
        pd.concat(parts, ignore_index=True)
        .sort_values(["timestamp", "symbol"], kind="stable")
        .reset_index(drop=True)
    )
    if result.empty:
        raise ValueError("No complete one-minute bars remain after counter validation")
    result.attrs.update(ticks.attrs)
    return result


def make_samples(bars: pd.DataFrame, *, lookback: int = 15) -> Samples:
    """Build preceding-minute windows without spanning dates or missing bars."""
    if not isinstance(lookback, int) or lookback < 1:
        raise ValueError("lookback must be a positive integer")
    high, context, targets, metadata = [], [], [], []
    for (symbol, _session), group in bars.groupby(["symbol", bars["timestamp"].dt.date], sort=True):
        group = group.sort_values("timestamp").reset_index(drop=True)
        for index in range(lookback, len(group)):
            sequence = group.iloc[index - lookback : index + 1]
            if not sequence["timestamp"].diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all():
                continue
            window = sequence.iloc[:-1]
            high.append(
                np.column_stack(
                    [
                        window["close"],
                        window["volume"],
                        window["spread"],
                        window["imbalance"],
                        window["high"] - window["low"],
                    ]
                )
            )
            context.append(
                [
                    window["close"].mean(),
                    window["close"].std(ddof=0),
                    window["volume"].sum(),
                    window["spread"].mean(),
                    window["imbalance"].mean(),
                    window["high"].max() - window["low"].min(),
                ]
            )
            targets.append(np.log1p(sequence.iloc[-1]["volume"]))
            metadata.append(
                {
                    "timestamp": sequence.iloc[-1]["timestamp"],
                    "symbol": symbol,
                    "feature_start": window.iloc[0]["timestamp"],
                    "feature_end": window.iloc[-1]["timestamp"],
                }
            )
    if not targets:
        raise ValueError(
            f"No contiguous windows remain; provide at least {lookback + 1} complete minutes in a session"
        )
    provenance = pd.DataFrame(metadata)
    provenance.attrs.update(bars.attrs)
    provenance.attrs.setdefault(
        "ingestion",
        {
            "timestamp_unit": "datetime",
            "exchange_timezone": str(bars["timestamp"].dt.tz),
            "session_policy": "local_calendar_date",
        },
    )
    return Samples(
        np.asarray(high, dtype=np.float32),
        np.asarray(context, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        provenance,
    )


def chronological_split(
    samples: Samples, *, train_fraction: float = 0.7, validation_fraction: float = 0.15
) -> dict[str, Samples]:
    """Use common whole-date cutoffs for all instruments."""
    if not (
        0 < train_fraction < 1
        and 0 < validation_fraction < 1
        and train_fraction + validation_fraction < 1
    ):
        raise ValueError("Split fractions must be positive and leave a test partition")
    samples = samples.take(
        samples.metadata.sort_values(["timestamp", "symbol"], kind="stable").index.to_numpy()
    )
    sample_dates = samples.metadata["timestamp"].dt.date
    dates = sorted(sample_dates.unique())
    if len(dates) < 3:
        raise ValueError(
            "At least three session dates are required for train, validation, and test"
        )
    train_count = min(max(1, int(len(dates) * train_fraction)), len(dates) - 2)
    validation_count = min(
        max(1, int(len(dates) * validation_fraction)), len(dates) - train_count - 1
    )
    partitions = {
        "train": dates[:train_count],
        "validation": dates[train_count : train_count + validation_count],
        "test": dates[train_count + validation_count :],
    }
    return {
        name: samples.take(np.flatnonzero(sample_dates.isin(days)))
        for name, days in partitions.items()
    }


def prepare_data(splits: dict[str, Samples]) -> dict:
    """Fit scalers and the instrument vocabulary on the training partition only."""
    train = splits["train"]
    symbol_map = {
        symbol: index for index, symbol in enumerate(sorted(train.metadata["symbol"].unique()))
    }
    for name, partition in splits.items():
        unseen = set(partition.metadata["symbol"]) - set(symbol_map)
        if unseen:
            raise ValueError(
                f"Unseen symbols in {name}: {', '.join(sorted(unseen))}; provide training history"
            )
    high_scaler, context_scaler, target_scaler = RobustScaler(), RobustScaler(), StandardScaler()
    high_scaler.fit(train.high.reshape(-1, train.high.shape[-1]))
    context_scaler.fit(train.context)
    target_scaler.fit(train.target.reshape(-1, 1))
    inputs, targets = {}, {}
    for name, partition in splits.items():
        high = (
            high_scaler.transform(partition.high.reshape(-1, partition.high.shape[-1]))
            .reshape(partition.high.shape)
            .astype(np.float32)
        )
        context = context_scaler.transform(partition.context).astype(np.float32)
        ids = partition.metadata["symbol"].map(symbol_map).to_numpy(dtype=np.int32).reshape(-1, 1)
        inputs[name] = (high, context, ids)
        targets[name] = target_scaler.transform(partition.target.reshape(-1, 1)).astype(np.float32)
    return dict(
        inputs=inputs,
        targets=targets,
        high_scaler=high_scaler,
        context_scaler=context_scaler,
        target_scaler=target_scaler,
        symbol_map=symbol_map,
    )


def regression_metrics(true_log, predicted_log) -> dict:
    """Report log-volume and share-volume errors with a defined zero convention."""
    true_log, predicted_log = (
        np.asarray(true_log, dtype=float).reshape(-1),
        np.asarray(predicted_log, dtype=float).reshape(-1),
    )
    if true_log.size == 0 or true_log.shape != predicted_log.shape:
        raise ValueError("Provide equally sized, non-empty targets and predictions")
    if not (np.isfinite(true_log).all() and np.isfinite(predicted_log).all()):
        raise ValueError("Targets and predictions must be finite")
    if (true_log < 0).any():
        raise ValueError("Log-volume targets cannot be negative")

    def summarize(actual, predicted):
        error = actual - predicted
        denominator = np.abs(actual) + np.abs(predicted)
        percentage = np.divide(
            2 * np.abs(error), denominator, out=np.zeros_like(error), where=denominator != 0
        )
        variance = np.sum((actual - actual.mean()) ** 2)
        return {
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "r2": float(1 - np.sum(error**2) / variance) if variance > 0 else None,
            "smape_pct": float(percentage.mean() * 100),
        }

    try:
        with np.errstate(over="raise", invalid="raise"):
            return {
                "log_volume": summarize(true_log, predicted_log),
                "shares": summarize(np.expm1(true_log), np.expm1(np.maximum(predicted_log, 0))),
            }
    except FloatingPointError as error:
        raise ValueError("Predictions exceed the finite numerical range") from error


def generate_demo_data(
    folder: str | Path, *, seed: int = 42, sessions: int = 8, minutes: int = 60
) -> None:
    """Write reproducible, fictional tick records; never overwrite existing inputs."""
    folder = Path(folder)
    if sessions < 3 or minutes < 3:
        raise ValueError("Use at least three sessions and three minutes per session")
    if folder.exists() and any(
        path.suffix.lower() in {".json", ".jsonl"} for path in folder.iterdir()
    ):
        raise FileExistsError(
            f"Input files already exist in {folder}; choose an empty sample directory"
        )
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for day in pd.bdate_range("2026-01-05", periods=sessions):
        for symbol, base_price, base_volume in (("ALPHA", 100.0, 600), ("BETA", 180.0, 1100)):
            start = pd.Timestamp(f"{day.date()} 09:15:00", tz="Asia/Kolkata")
            counter, price = 0, base_price
            rows = []

            def record(stamp, value, price):
                return dict(
                    timestamp=int(stamp.timestamp()),
                    symbol=symbol,
                    ltp=round(price, 4),
                    bid_price=round(price - 0.02, 4),
                    ask_price=round(price + 0.02, 4),
                    bid_size=int(rng.integers(150, 450)),
                    ask_size=int(rng.integers(150, 450)),
                    vol_traded_today=value,
                )

            rows.append(record(start, 0, price))
            day_scale = rng.uniform(0.92, 1.08)
            for minute in range(minutes):
                position = minute / max(minutes - 1, 1)
                u_shape = 1 + 0.8 * (2 * position - 1) ** 2
                volume = max(1, int(base_volume * day_scale * u_shape * rng.lognormal(0, 0.08)))
                first = volume // 2
                for second, increment in ((10, first), (50, volume - first)):
                    counter += increment
                    price += rng.normal(0, 0.015)
                    rows.append(
                        record(start + pd.Timedelta(minutes=minute, seconds=second), counter, price)
                    )
            target = folder / f"{symbol}-{day.date()}.jsonl"
            target.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def build_model(lookback: int, n_symbols: int, *, seed: int = 42):
    """Retain the supplied CNN/context/embedding architecture for direct regression."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    layers = tf.keras.layers
    penalty = tf.keras.regularizers.l2(0.0005)
    high = layers.Input((lookback, 5), name="recent_bars")
    encoded = layers.Conv1D(64, 3, padding="same", activation="relu", kernel_regularizer=penalty)(
        high
    )
    encoded = layers.LayerNormalization()(encoded)
    encoded = layers.Conv1D(32, 3, padding="same", activation="relu", kernel_regularizer=penalty)(
        encoded
    )
    encoded = layers.LayerNormalization()(encoded)
    encoded = layers.Flatten()(encoded)
    context = layers.Input((6,), name="context")
    context_encoded = layers.Dense(32, activation="relu", kernel_regularizer=penalty)(context)
    context_encoded = layers.LayerNormalization()(context_encoded)
    stock = layers.Input((1,), dtype="int32", name="symbol_id")
    embedded = layers.Flatten()(layers.Embedding(n_symbols, 8)(stock))
    combined = layers.Concatenate()([encoded, context_encoded, embedded])
    combined = layers.Dropout(0.2)(combined)
    combined = layers.Dense(128, activation="relu", kernel_regularizer=penalty)(combined)
    combined = layers.LayerNormalization()(combined)
    combined = layers.Dropout(0.2)(combined)
    combined = layers.Dense(64, activation="relu", kernel_regularizer=penalty)(combined)
    output = layers.Dense(1, name="scaled_log_volume")(combined)
    model = tf.keras.Model([high, context, stock], output, name="IntradayVolumeLab")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.0),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_experiment(
    samples: Samples,
    output_dir: str | Path,
    *,
    epochs: int = 5,
    batch_size: int = 64,
    seed: int = 42,
    data_label: str,
    input_manifest: list[dict] | None = None,
) -> dict:
    """Run chronological evaluation, save preprocessing/model, and record provenance."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if not data_label.strip():
        raise ValueError("Supply a data label identifying synthetic or real input")
    # Keep small CPU experiments bounded; respect user-configured thread settings.
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    import tensorflow as tf

    splits = chronological_split(samples)
    prepared = prepare_data(splits)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.backend.clear_session()
    model = build_model(samples.high.shape[1], len(prepared["symbol_map"]), seed=seed)

    def dataset(name, shuffle=False):
        data = tf.data.Dataset.from_tensor_slices(
            (prepared["inputs"][name], prepared["targets"][name])
        )
        if shuffle:
            data = data.shuffle(len(splits[name].target), seed=seed)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = 1
        return data.batch(batch_size).with_options(options)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            output_dir / "best.keras", monitor="val_loss", save_best_only=True
        ),
    ]
    history = model.fit(
        dataset("train", shuffle=True),
        validation_data=dataset("validation"),
        epochs=epochs,
        callbacks=callbacks,
        shuffle=False,
        verbose=0,
    )
    predictions, metrics = {}, {}
    for name, partition in splits.items():
        scaled = model(prepared["inputs"][name], training=False).numpy()
        predicted = prepared["target_scaler"].inverse_transform(scaled).reshape(-1)
        predictions[name] = predicted
        persistence = np.log1p(partition.high[:, -1, 1])
        metrics[name] = {
            "model": regression_metrics(partition.target, predicted),
            "persistence": regression_metrics(partition.target, persistence),
        }
    model.save(output_dir / "model.keras")
    with (output_dir / "preprocessing.pkl").open("wb") as stream:
        pickle.dump(
            {
                name: prepared[name]
                for name in ["high_scaler", "context_scaler", "target_scaler", "symbol_map"]
            }
            | {
                "high_features": HIGH_FEATURES,
                "context_features": CONTEXT_FEATURES,
                "lookback": samples.high.shape[1],
            },
            stream,
        )
    predictions_frame = splits["test"].metadata.copy()
    predictions_frame["actual_shares"] = np.expm1(splits["test"].target.astype(float))
    predictions_frame["predicted_shares"] = np.expm1(
        np.maximum(predictions["test"].astype(float), 0)
    )
    predictions_frame["persistence_shares"] = splits["test"].high[:, -1, 1]
    predictions_frame.to_csv(output_dir / "test-predictions.csv", index=False)
    result = {
        "project": "Intraday Volume Lab",
        "maintainer": "Bikash Kushwaha",
        "run_id": uuid.uuid4().hex,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "ingestion": samples.metadata.attrs.get("ingestion", {}),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_label": data_label,
        "target": "next-minute interval volume",
        "target_transform": "log1p(shares), then training-only StandardScaler",
        "lookback_minutes": samples.high.shape[1],
        "seed": seed,
        "epochs_requested": epochs,
        "epochs_completed": len(history.history["loss"]),
        "batch_size": batch_size,
        "symbol_map": prepared["symbol_map"],
        "sample_counts": {name: len(part.target) for name, part in splits.items()},
        "date_ranges": {
            name: {
                "start": part.metadata["timestamp"].min().isoformat(),
                "end": part.metadata["timestamp"].max().isoformat(),
                "session_dates": part.metadata["timestamp"].dt.date.nunique(),
            }
            for name, part in splits.items()
        },
        "input_manifest": input_manifest or [],
        "versions": {
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in ["tensorflow", "keras", "numpy", "pandas", "scikit-learn", "matplotlib"]
            },
        },
        "history": {
            name: [float(value) for value in values] for name, values in history.history.items()
        },
        "metrics": metrics,
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    plot_results(result, predictions_frame, output_dir / "figures")
    return result


def plot_results(result: dict, predictions: pd.DataFrame, folder: Path) -> None:
    """Export scientific figures with units and the run's explicit data label."""
    from matplotlib.figure import Figure

    folder.mkdir(parents=True, exist_ok=True)
    teal, amber, ink = "#147d80", "#b86a21", "#26384a"

    def save(figure, name):
        figure.set_facecolor("white")
        figure.tight_layout()
        figure.savefig(
            folder / name,
            dpi=160,
            metadata={
                "Title": "Intraday Volume Lab",
                "Author": "Bikash Kushwaha",
                "Description": result["data_label"],
            },
        )

    figure = Figure(figsize=(9, 4.5))
    ax = figure.subplots()
    epochs = np.arange(1, len(result["history"]["loss"]) + 1)
    ax.plot(epochs, result["history"]["loss"], "o-", color=teal, label="Training")
    ax.plot(epochs, result["history"]["val_loss"], "o-", color=amber, label="Validation")
    ax.set(
        title=f"Training history\n{result['data_label']}",
        xlabel="Epoch",
        ylabel="MSE + regularization (scaled log-volume)",
    )
    ax.grid(alpha=0.15)
    ax.legend(frameon=False)
    save(figure, "learning-curve.png")

    figure = Figure(figsize=(11, 4.5))
    scatter, series = figure.subplots(1, 2)
    scatter.scatter(
        predictions["actual_shares"], predictions["predicted_shares"], color=teal, alpha=0.6, s=16
    )
    limits = [
        0,
        max(predictions["actual_shares"].max(), predictions["predicted_shares"].max()) * 1.05,
    ]
    scatter.plot(limits, limits, "--", color=ink, alpha=0.5, label="Ideal")
    scatter.set(
        xlabel="Actual interval volume (shares)",
        ylabel="Predicted interval volume (shares)",
        title="Held-out observations",
    )
    scatter.legend(frameon=False)
    first = predictions[predictions["symbol"] == predictions["symbol"].iloc[0]].head(70)
    series.plot(np.arange(len(first)), first["actual_shares"], color=ink, label="Actual")
    series.plot(np.arange(len(first)), first["predicted_shares"], color=teal, label="Model")
    series.plot(
        np.arange(len(first)),
        first["persistence_shares"],
        color=amber,
        alpha=0.7,
        label="Persistence",
    )
    series.set(
        title=f"{first['symbol'].iloc[0]} · first held-out observations",
        xlabel="Observation (chronological)",
        ylabel="Interval volume (shares)",
    )
    series.legend(frameon=False)
    for ax in (scatter, series):
        ax.grid(alpha=0.15)
    figure.suptitle(result["data_label"], color=ink, fontsize=11)
    save(figure, "predictions.png")

    figure = Figure(figsize=(9, 4.5))
    ax = figure.subplots()
    ax.hist(
        predictions["actual_shares"] - predictions["predicted_shares"],
        bins=24,
        color=teal,
        edgecolor="white",
    )
    ax.axvline(0, color=ink, linestyle="--", linewidth=1)
    ax.set(
        title=f"Test error distribution\n{result['data_label']}",
        xlabel="Actual − predicted volume (shares)",
        ylabel="Observations",
    )
    ax.grid(axis="y", alpha=0.15)
    save(figure, "error-distribution.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Intraday Volume Lab · Python research portfolio by Bikash Kushwaha"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    sample = subparsers.add_parser(
        "sample", help="Generate fictional demonstration ticks in an empty directory"
    )
    sample.add_argument("--output", type=Path, default=Path("data/sample"))
    sample.add_argument("--seed", type=int, default=42)
    train = subparsers.add_parser("train", help="Train and evaluate the direct volume regressor")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, default=Path("artifacts/demo"))
    train.add_argument("--timestamp-unit", choices=["s", "ms"], default="s")
    train.add_argument("--timezone", default="Asia/Kolkata")
    train.add_argument("--lookback", type=int, default=15)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--data-label",
        required=True,
        help="Identify synthetic demonstration data or your real dataset",
    )
    args = parser.parse_args()
    try:
        if args.command == "sample":
            generate_demo_data(args.output, seed=args.seed)
            print(f"Synthetic demonstration data saved to {args.output}")
        else:
            ticks = load_ticks(
                args.data, timestamp_unit=args.timestamp_unit, exchange_timezone=args.timezone
            )
            samples = make_samples(minute_bars(ticks), lookback=args.lookback)
            result = train_experiment(
                samples,
                args.output,
                epochs=args.epochs,
                seed=args.seed,
                data_label=args.data_label,
                input_manifest=ticks.attrs["input_manifest"],
            )
            print(
                json.dumps(
                    {
                        "data_label": result["data_label"],
                        "sample_counts": result["sample_counts"],
                        "test_share_metrics": result["metrics"]["test"],
                    },
                    indent=2,
                )
            )
            print(f"Results, model, preprocessing, and figures saved to {args.output}")
    except (ValueError, FileExistsError) as error:
        parser.exit(2, f"Input error: {error}\n")


if __name__ == "__main__":
    main()
