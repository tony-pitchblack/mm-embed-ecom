# Agent / contributor setup

## First-time initialization

Run from the repository root.

### 1. Environment (uv + `requirements.txt`)

There is no `pyproject.toml`; install dependencies with `uv pip` into a local venv:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Secrets and environment

Copy `.env-default` to `.env` (or merge: keep your existing `HF_TOKEN` and add the `MLFLOW_*` lines from the default). Set `HF_TOKEN` to a Hugging Face token with read access to the dataset ([tokens](https://huggingface.co/settings/tokens)).

The default `.env` also defines MLflow: `MLFLOW_HOST`, `MLFLOW_PORT`, `MLFLOW_TRACKING_URI`, `MLFLOW_BACKEND_STORE_URI` (SQLite at `logs/mlflow/database.db`), and `MLFLOW_ARTIFACT_ROOT` (under `logs/mlflow/mlartifacts`). Training scripts read `MLFLOW_TRACKING_URI` when set.

### 3. MLflow server (optional, for experiment tracking)

After [§1](#1-environment-uv--requirementstxt) is installed, you can run the local tracking server from the repo root:

```bash
source .venv/bin/activate
bash scripts/mlflow_server.sh
```

The script loads `.env` and runs `mlflow server` with the configured backend and artifact store. The SQLite file is created or reused at `logs/mlflow/database.db` once the server starts; after you have used MLflow, that database should be non-empty (metadata tables, and `runs` rows when experiments have been logged).

To confirm the server: open `http://localhost:5000` (or the host/port in `.env`). To point training and evals at it, set `MLFLOW_TRACKING_URI` in `.env` and ensure the server is running, or set it only in the shell for a one-off run.

### 4. Hub dataset (`data/` and `logs/`)

The Hub repo `tony-pitchblack/mm-embed-ecom` (type **dataset**) stores artifacts under top-level `data/` and `logs/`. Download **to the repository root** so those folders stay siblings at the project top level (do **not** use `--local-dir data`, or you will get `data/logs/` nested incorrectly).

```bash
source .venv/bin/activate
set -a && source .env && set +a
rm -rf data logs
hf download tony-pitchblack/mm-embed-ecom --repo-type dataset --local-dir "$(pwd)"
```

`hf` is provided by the `huggingface_hub` package installed from `requirements.txt`.

After this step you should have `./data/` (e.g. embeddings / model inputs) and `./logs/` (e.g. MLflow, training logs) as separate directories next to the Python package layout.
