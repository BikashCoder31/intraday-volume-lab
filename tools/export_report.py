"""Export a consistent executed notebook and result snapshot for static hosting."""

import hashlib
import html
import json
import re
import shutil
from pathlib import Path

RUN_MIME = "application/vnd.intraday-volume-lab.run+json"


def validate_report_run(notebook: dict, results: dict) -> None:
    """Reject partial/reordered execution and notebook/artifact run mismatches."""
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    counts = [cell.get("execution_count") for cell in cells]
    if not cells or any(not isinstance(count, int) or count < 1 for count in counts):
        raise ValueError("Execute all notebook code cells successfully in order before export.")
    if counts != sorted(set(counts)) or any(
        output["output_type"] == "error" for cell in cells for output in cell.get("outputs", [])
    ):
        raise ValueError("Execute all notebook code cells successfully in order before export.")
    run_ids = [
        output["data"][RUN_MIME]["run_id"]
        for cell in cells
        for output in cell.get("outputs", [])
        if RUN_MIME in output.get("data", {})
    ]
    if run_ids != [results.get("run_id")]:
        raise ValueError(
            "Notebook outputs and artifacts must come from the same run. Rerun the notebook before export."
        )


def export_report(root: Path) -> None:
    import nbformat
    from nbconvert import HTMLExporter

    notebook = nbformat.read(root / "Intraday_Volume_Lab.ipynb", as_version=4)
    results = json.loads((root / "artifacts" / "demo" / "results.json").read_text(encoding="utf-8"))
    validate_report_run(notebook, results)
    if (
        results.get("implementation_sha256")
        != hashlib.sha256((root / "volume_lab.py").read_bytes()).hexdigest()
    ):
        raise ValueError(
            "The implementation changed since this run. Rerun the notebook before export."
        )
    # Keep diagnostic warnings, while making captured output portable for publication.
    prefixes = {str(root.resolve()), root.resolve().as_posix()}
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            if isinstance(output.get("text"), str):
                for prefix in prefixes:
                    output.text = output.text.replace(prefix, "[project]")
            for mime, value in output.get("data", {}).items():
                if mime.startswith("text/") and isinstance(value, str):
                    for prefix in prefixes:
                        value = value.replace(prefix, "[project]")
                    output.data[mime] = value
    nbformat.write(notebook, root / "Intraday_Volume_Lab.ipynb")
    exporter = HTMLExporter(
        template_name="lab",
        exclude_input=True,
        exclude_output_prompt=True,
        embed_images=True,
        mathjax_url="",
    )
    body, _ = exporter.from_notebook_node(notebook)
    # This static report has no widgets; omit external loaders for offline viewing.
    body = re.sub(r'<script\b[^>]*\bsrc=["\'][^"\']+["\'][^>]*>\s*</script>', "", body, flags=re.I)
    description = "A reproducible Python volume-forecasting research portfolio by Bikash Kushwaha. Synthetic demonstration with chronological evaluation and explicit data assumptions."
    metadata = f'<meta name="author" content="Bikash Kushwaha">\n<meta name="description" content="{html.escape(description, quote=True)}">\n<meta property="og:title" content="Intraday Volume Lab">\n<meta property="og:description" content="{html.escape(description, quote=True)}">\n'
    body = body.replace(
        "</head>",
        metadata
        + '<link rel="icon" href="icon.svg" type="image/svg+xml">\n'
        + '<link rel="stylesheet" href="report.css">\n</head>',
        1,
    )
    site = root / "site"
    site.mkdir(exist_ok=True)
    (site / "index.html").write_text(body, encoding="utf-8")
    (site / "results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False), encoding="utf-8"
    )
    (site / "robots.txt").write_text("User-agent: *\nAllow: /\n", encoding="utf-8")
    figure_dir = root / "docs" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for path in (root / "artifacts" / "demo" / "figures").glob("*.png"):
        shutil.copy2(path, figure_dir / path.name)
    snapshot = root / "docs" / "results"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "synthetic-demo.json").write_text(
        json.dumps(results, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        f"Exported static report for {results['data_label']}; figures and result snapshot refreshed."
    )


if __name__ == "__main__":
    export_report(Path(__file__).resolve().parents[1])
