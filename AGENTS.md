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

### 2. Secrets

Copy `.env-default` to `.env` and set `HF_TOKEN` to a Hugging Face token with read access to the dataset ([tokens](https://huggingface.co/settings/tokens)).

### 3. Hub dataset (`data/` and `logs/`)

The Hub repo `tony-pitchblack/mm-embed-ecom` (type **dataset**) stores artifacts under top-level `data/` and `logs/`. Download **to the repository root** so those folders stay siblings at the project top level (do **not** use `--local-dir data`, or you will get `data/logs/` nested incorrectly).

```bash
source .venv/bin/activate
set -a && source .env && set +a
rm -rf data logs
hf download tony-pitchblack/mm-embed-ecom --repo-type dataset --local-dir "$(pwd)"
```

`hf` is provided by the `huggingface_hub` package installed from `requirements.txt`.

After this step you should have `./data/` (e.g. embeddings / model inputs) and `./logs/` (e.g. MLflow, training logs) as separate directories next to the Python package layout.
