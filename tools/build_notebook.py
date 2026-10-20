"""Generate the maintained narrative notebook from readable, reviewable cells."""

import subprocess
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
markdown, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    markdown("""# Intraday Volume Lab

### A reproducible Python research portfolio

**Portfolio edition maintained and improved by Bikash Kushwaha**

This notebook forecasts the next minute's trading volume using recent price, volume, and order-book features. It combines a convolutional encoder, a context branch, and instrument embeddings.

**The demonstration uses synthetic data and fictional symbols.** It demonstrates the workflow and its checks; it is not evidence of real-market forecasting performance. The model is compared with the simple baseline of repeating the last observed minute's volume.
"""),
    markdown("""## 1. Experiment configuration

The target is interval volume in shares. A 15-minute history predicts the immediately following minute. A fixed seed, explicit timezone, file hashes, and whole-date evaluation boundaries make the assumptions inspectable.
"""),
    code("""from pathlib import Path

import pandas as pd
from IPython.display import Image, display

from volume_lab import chronological_split, load_ticks, make_samples, minute_bars, train_experiment

ROOT = Path.cwd()
if not (ROOT / "volume_lab.py").is_file():
    raise RuntimeError("Start this notebook from the project root.")
DATA = ROOT / "data" / "sample"
ARTIFACTS = ROOT / "artifacts" / "demo"
DATA_LABEL = "Synthetic demonstration · fictional symbols"
LOOKBACK = 15
EPOCHS = 5
SEED = 42

display(pd.DataFrame({
    "Setting": ["Data", "Target", "Lookback", "Exchange timezone", "Epochs", "Seed"],
    "Value": [DATA_LABEL, "Next-minute interval volume (shares)", "15 preceding complete minutes", "Asia/Kolkata", EPOCHS, SEED],
}))
"""),
    markdown("""## 2. Validate ticks and recover interval volume

The bundled sample contains two fictional instruments over eight short sessions. It uses numeric epoch seconds and a cumulative volume counter that resets daily. The loader combines files, removes exact duplicate records, and rejects invalid input.

Volume is computed from counter increments within each instrument/session. An unknown initial counter makes the first minute partial, so that minute is discarded. A counter decrease within a session is an error. Missing minutes are not filled.
"""),
    code("""ticks = load_ticks(DATA, timestamp_unit="s", exchange_timezone="Asia/Kolkata")
bars = minute_bars(ticks)
display(pd.DataFrame({
    "Input": ["Tick records", "Complete minute bars", "Symbols", "Local session dates"],
    "Count": [len(ticks), len(bars), bars["symbol"].nunique(), bars["timestamp"].dt.date.nunique()],
}))
display(bars[["timestamp", "symbol", "close", "volume", "spread", "imbalance"]].head(6))
"""),
    markdown("""## 3. Causal features and chronological validation

Each window uses five sequential features and six aggregate context features from completed bars. Windows cannot cross a gap or a session boundary. All instruments share the same date cutoffs. Scalers and the instrument vocabulary fit the training partition only.

The eight-date example assigns five dates to training, one to validation, and two to testing. Shuffling is allowed only after the training partition has been selected.
"""),
    code("""samples = make_samples(bars, lookback=LOOKBACK)
splits = chronological_split(samples)
partition_table = pd.DataFrame([
    {"Partition": name.title(), "Samples": len(part.target),
     "First target": part.metadata["timestamp"].min().isoformat(),
     "Last target": part.metadata["timestamp"].max().isoformat(),
     "Session dates": part.metadata["timestamp"].dt.date.nunique()}
    for name, part in splits.items()
])
display(partition_table)
print(f"Sequential tensor: {samples.high.shape}; context tensor: {samples.context.shape}")
assert (samples.metadata["feature_end"] < samples.metadata["timestamp"]).all()
assert splits["train"].metadata["timestamp"].max() < splits["validation"].metadata["timestamp"].min()
assert splits["validation"].metadata["timestamp"].max() < splits["test"].metadata["timestamp"].min()
print("Causality and partition-boundary checks passed.")
"""),
    markdown("""## 4. Three-branch neural regressor

The sequential branch has Conv1D layers with 64 and 32 filters, layer normalization, and flattening. A 32-unit dense branch encodes the context. An 8-dimensional embedding represents each instrument. Their concatenation feeds 128- and 64-unit dense layers with dropout and L2 regularization.

Training uses Adam, mean squared error on standardized log-volume, early stopping, learning-rate reduction, and best-model checkpointing. This direct regressor does not implement CMEM residual estimation.
"""),
    code("""result = train_experiment(
    samples, ARTIFACTS, epochs=EPOCHS, seed=SEED,
    data_label=DATA_LABEL, input_manifest=ticks.attrs["input_manifest"],
)
display(pd.DataFrame([
    {"Library": name, "Version": version} for name, version in result["versions"].items()
]))
print(f"Completed {result['epochs_completed']} epochs. Model and preprocessing saved locally.")
display({
    "application/vnd.intraday-volume-lab.run+json": {"run_id": result["run_id"]},
    "text/plain": f"Run reference: {result['run_id']}",
}, raw=True)
"""),
    markdown("""## 5. Compare against persistence

The table reports errors in **original share units**. The persistence baseline predicts the last observed minute's volume. It is a demanding reference for this smooth synthetic series; a neural model need not beat it, especially after a short demonstration run.

The results JSON also reports log-volume metrics separately. No archived research scores are substituted for the run below.
"""),
    code("""rows = []
for partition, models in result["metrics"].items():
    for model_name, metrics in models.items():
        rows.append({"Partition": partition.title(), "Predictor": model_name.title(),
                     "MAE (shares)": metrics["shares"]["mae"],
                     "RMSE (shares)": metrics["shares"]["rmse"],
                     "R²": metrics["shares"]["r2"],
                     "SMAPE (%)": metrics["shares"]["smape_pct"]})
metric_table = pd.DataFrame(rows)
display(metric_table.round(3))
test_metrics = result["metrics"]["test"]
model_mae = test_metrics["model"]["shares"]["mae"]
baseline_mae = test_metrics["persistence"]["shares"]["mae"]
print(f"Held-out MAE: model {model_mae:.2f} shares; persistence {baseline_mae:.2f} shares.")
print("Synthetic-data workflow check only; no real-market performance claim.")
"""),
    code("""for filename in ("learning-curve.png", "predictions.png", "error-distribution.png"):
    display(Image(filename=str(ARTIFACTS / "figures" / filename)))
"""),
    markdown("""## 6. Reproducibility and limitations

The maintained edition includes regression checks for interval-volume arithmetic, partial minutes, daily resets, duplicate files, timestamp units, missing bars, causal windows, common date boundaries, training-only scalers, finite metrics, and model persistence.

The sample data is deliberately small and synthetic. Evaluation does not establish market performance, execution gains, or live readiness. Calendar-date sessions suit this sample; overnight sessions or feed resets require a different session policy. Inputs remain in memory, so larger datasets need memory profiling and partitioning.

To reproduce this report, install the pinned development requirements, run the test suite, then execute this notebook from the project root. Its results file records the inputs' hashes, seed, library versions, date ranges, and metric units. Full real-data reproduction requires an authorized dataset that was not included in the supplied archive.
"""),
    markdown("""## Source acknowledgement

This portfolio edition is maintained and improved by **Bikash Kushwaha**. It consolidates the supplied research workflow into a reproducible, tested demonstration.

The architecture adapts the supplied notebook's CNN/context/embedding regressor. [DeepLOB](https://arxiv.org/abs/1808.03668) and [CMEM research](https://academic.oup.com/jfec/article-abstract/9/3/489/840918) provide scientific context. This edition does not claim a complete reproduction of either paper.
"""),
]
notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "title": "Intraday Volume Lab",
        "authors": [{"name": "Bikash Kushwaha", "role": "Portfolio edition maintainer"}],
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
)
nbf.write(notebook, ROOT / "Intraday_Volume_Lab.ipynb")
subprocess.run(
    [sys.executable, "-m", "ruff", "format", str(ROOT / "Intraday_Volume_Lab.ipynb")], check=True
)
print("Generated Intraday_Volume_Lab.ipynb with one documented training flow.")
