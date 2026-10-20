# Data contract and forecasting target

The bundled data is **synthetic**, generated for reproducibility with fictional symbols ALPHA and BETA. Each sample session contains a short 60-minute interval rather than a complete exchange trading day. It does not represent historical market activity.

## JSON-lines inputs

Place `.jsonl` or `.json` files containing one JSON object per line in one data directory. Multiple files for one symbol are combined. Exact duplicate records are removed; different snapshots with the same timestamp remain available. Records are ordered by symbol, timestamp, and cumulative counter.

| Field | Contract |
| --- | --- |
| `timestamp` | Numeric Unix epoch time. Specify `s` or `ms`; units are never guessed. Supported example dates are within 2000–2100. |
| `symbol` | Non-empty instrument name. If omitted, the loader uses the filename before `-NSE`, or its stem. Explicit symbols are recommended. |
| `ltp` | Positive finite last traded price. |
| `bid_price`, `ask_price` | Positive finite prices with ask greater than or equal to bid. |
| `bid_size`, `ask_size` | Non-negative finite sizes. |
| `vol_traded_today` | Non-negative cumulative shares traded in the local trading session. |

The default exchange timezone is **Asia/Kolkata**. Epoch timestamps are parsed as UTC and converted to that timezone before session grouping. For another market, supply its timezone explicitly.

Malformed JSON, absent columns, non-finite values, crossed quotes, invalid timestamps, and negative counters/sizes produce actionable errors. Data is not silently repaired or skipped. Blank lines are ignored.

## Cumulative counter policy

1. Separate each symbol into local calendar-date sessions.
2. Compute consecutive cumulative-counter differences within a session.
3. If the first observed counter is zero, use it as a known zero baseline. If it is positive, the first increment is unknown, and the initial partial minute is discarded.
4. A counter decrease within a session raises an error. A new date starts a new counter sequence, so normal daily resets do not create negative volume.
5. Aggregate increments into one-minute intervals. A minute with an unknown increment is discarded. Empty minutes are not filled or bridged. The first observed minute after a missing-minute gap is also discarded, because its counter jump includes unallocated volume.

This policy assumes the cumulative counter is monotonic between observed ticks and resets by local calendar date. Overnight sessions and intraday feed resets require a different session policy before using this pipeline. A missing feed can hide trades inside a cumulative difference; exact tick-feed completeness cannot be established from this schema alone.

## Features and target

The target is **the next one-minute interval's share volume**, transformed with `log1p` for training. The default lookback is the 15 immediately preceding complete one-minute bars. A window cannot cross a missing minute or a session boundary.

Sequential features are close price, interval volume, mean spread, mean bid/ask imbalance, and high–low range. Context features are mean close, population standard deviation of close, total lookback volume, mean spread, mean imbalance, and the window's high–low range. The final branch embeds the instrument ID.

These six context summaries are not a CMEM estimator. There is no five-day CMEM baseline or residual reconstruction in this edition.

## Chronological evaluation

Samples retain target timestamps and symbols. Whole local dates are assigned in chronological order: approximately 70% train, 15% validation, and the remainder test. At least three dates are required. With only eight synthetic dates this becomes five training dates, one validation date, and two test dates.

All instruments use the same date boundaries. Evaluation symbols must appear in training. Scalers fit the training partition only. Random shuffling occurs only inside the already selected training partition.

Results report metrics separately in log-volume and original share units. Negative predicted log values are clipped to zero only when converting predictions to non-negative shares. The share-unit SMAPE convention assigns zero contribution to a pair of zero values. R² is null for a constant target rather than presented as a meaningful score.

The comparison baseline predicts the last observed minute's volume. It is a persistence baseline, not CMEM. Beating it on synthetic data would not establish performance on a real market.

## Memory and reproducibility

Inputs and feature arrays are held in memory. Float32 arrays limit feature memory use, but the loader is designed for bounded experiments rather than unlimited tick histories. For large data, partition by symbol/session and measure memory before extending storage.

Each run records its seed, library versions, lookback, date boundaries, sample counts, input filenames and hashes when supplied by the notebook/CLI, and data label. Exact floating-point reproducibility can still vary by platform. Keep private data under the ignored `data/private/` folder.
