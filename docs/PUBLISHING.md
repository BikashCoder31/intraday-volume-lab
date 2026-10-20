# GitHub, Vercel, and documentation screenshots

The finalized project repository is [BikashCoder31/intraday-volume-lab](https://github.com/BikashCoder31/intraday-volume-lab), and the live report is [intraday-volume-lab.vercel.app](https://intraday-volume-lab.vercel.app). GitHub is connected to the Vercel project, with `main` as the production branch and `site` as its root directory. The following guide documents publication, future updates, and screenshot capture.

## 1. Review the local edition

1. Open `README.md` and the executed notebook.
2. Run `python -m pytest`, `python -m ruff check .`, and `python -m ruff format --check .` using the project virtual environment.
3. Read `SOURCE_ACKNOWLEDGEMENTS.md` and the data assumptions in `docs/DATA.md`.
4. Preview the included report:

```powershell
python -m http.server 8000 --bind 127.0.0.1 --directory site
```

Open `http://127.0.0.1:8000`, check the overview, data table, date boundaries, metrics, and charts, then stop the server with Ctrl+C. The report is a static result of the documented synthetic experiment, not a live model service.

## 2. Publish the finalized folder to GitHub

The project root is the folder containing `volume_lab.py`, `README.md`, and `.gitignore`. Do not initialize the surrounding workspace or upload the original archive/audit folder.

1. In GitHub, create an empty repository named `intraday-volume-lab`. Leave initial README, `.gitignore`, and license options unselected when pushing the existing local folder.
2. Open PowerShell in this project root. Check whether it already has Git history:

```powershell
git status
```

3. If it is a new, untracked folder, initialize it and review the proposed commit:

```powershell
git init -b main
git add .
git status
git diff --cached --stat
git diff --cached
```

The virtual environment, caches, raw/private data, model files, pickle files, and local `artifacts/` outputs are ignored. Confirm that the staged list contains the maintained notebook, code, sample data, documentation, screenshots, static report, and dependency files.

4. Commit the reviewed edition:

```powershell
git commit -m "Prepare Intraday Volume Lab portfolio edition"
```

If Git needs identity configuration, set your preferred author name and your own GitHub email or private noreply address. No email address has been invented for you.

5. Copy the HTTPS repository URL from GitHub and replace the placeholder:

```powershell
git remote add origin "<YOUR_GITHUB_REPOSITORY_HTTPS_URL>"
git remote -v
git push -u origin main
```

If a remote already exists, inspect it before changing it. Complete GitHub's browser/credential-manager sign-in when requested.

6. Open the repository. Verify README images, notebook rendering, the acknowledgement link, and the CI result. Add a concise repository description and topics such as `python`, `tensorflow`, `time-series`, and `portfolio`. Replace the README's repository placeholder with the real URL.

The procedure follows [GitHub's official local-code guide](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github?platform=windows). The included [CI workflow](https://github.com/BikashCoder31/intraday-volume-lab/actions) passed on Ubuntu with Python 3.12 after publication.

## 3. Refresh the report when results change

Execute all notebook cells in order using the project's interpreter. A terminal route is:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name intraday-volume-lab --display-name "Python 3 (Intraday Volume Lab)"
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=intraday-volume-lab --ExecutePreprocessor.timeout=240 Intraday_Volume_Lab.ipynb
.\.venv\Scripts\python.exe tools/export_report.py
```

The exporter hides input code for readable presentation, embeds the figure pixels, and refreshes `site/results.json`, `docs/results/synthetic-demo.json`, and `docs/figures/`. It replaces the local project-directory prefix in captured text outputs with `[project]`, retaining diagnostic warnings. It verifies successful execution order, matching run IDs, and the implementation hash. If a separate CLI run overwrites `artifacts/demo`, execute the notebook again before exporting so tables, figures, and JSON describe the same run.

The [nbconvert documentation](https://nbconvert.readthedocs.io/en/latest/usage.html) describes notebook execution and static HTML export. Keep private data paths and outputs out of a public report.

## 4. Deploy the static report on Vercel

1. Sign in to Vercel and select **Add New → Project**.
2. Import the finalized GitHub repository and verify the intended account/team.
3. Set **Root Directory** to **`site`**. Confirm that `index.html` is directly inside it.
4. Use **Framework Preset: Other**. The included `site/vercel.json` sets an empty install/build command and serves the selected root.
5. Confirm that no notebook execution or training command is configured as the deployment build.
6. Deploy and open the returned URL. Check the introduction, figures, anchor navigation, and a narrow browser width.
7. Add the actual URL to the repository About section and README, replacing the live-demo placeholder. Capture the deployed page for documentation, then commit that documentation update.

Vercel's [build configuration guide](https://vercel.com/docs/builds/configure-a-build) documents static projects, root/output directories, and skipping builds. This deployment serves the pre-exported HTML; it does not provide a Python notebook kernel, retraining endpoint, or live inference API.

For future updates, execute and export the notebook locally, review the changed figures/results, and push the updated files to `main`. The connected Vercel project deploys that branch automatically. The root `.vercelignore` also limits manual CLI uploads to the six files in `site/`.

## 5. Capture and organize screenshots

The `docs/screenshots/` folder contains captures of the verified live report. Recapture the published page if its URL or layout changes.

1. Use a consistent desktop browser size, approximately 1440 × 1000, at 100% zoom.
2. Capture the report overview, input validation/date-boundary tables, and held-out evaluation. Include one narrow view, approximately 390 pixels wide.
3. On Windows, use **Win+Shift+S** for a rectangular capture, or your browser's full-page capture command.
4. Name PNGs clearly:

```text
docs/screenshots/
  01-report-overview.png
  02-data-and-boundaries.png
  03-evaluation.png
  04-report-mobile.png
```

5. Prefer exported Matplotlib figures from `docs/figures/` when documenting a chart. Keep synthetic-data labels and meaningful axis units visible.
6. Avoid personal account details, tokens, private Drive paths, unrelated tabs, and unverified historical results in the frame.
7. Embed actual files with descriptive alternative text:

```markdown
![Chronological evaluation on the synthetic example](docs/screenshots/03-evaluation.png)
```

The example path above is relative to the root README. Preview it on GitHub to check readability and path casing.

## 6. Final handoff details

Record the repository URL, published commit, Vercel URL, environment versions, run ID, input hashes, checks performed, and dataset limitations. The checked demonstration uses synthetic inputs; real-market results require a separate documented experiment.
