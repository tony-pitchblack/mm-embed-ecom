---
name: evaluation-launcher
description: Launch ColBERT rerank evaluations in tmux with MLflow readiness checks. Use when running evals/colbert_rerank.py, evaluating a trained MLflow run, or starting evaluation jobs that should run in a separate tmux session.
---

# Evaluation Launcher

Use this skill to launch project evaluation runs reliably in separate tmux sessions.

## Rules

- Run from repository root: `/workspace/mm-embed-ecom`.
- Activate the project virtualenv before scripts: `source .venv/bin/activate`.
- Launch evaluation in a dedicated tmux session.
- Ensure MLflow is reachable before evaluation.
- If MLflow is unreachable, launch it in a separate tmux session via `scripts/mlflow_server.sh`.
- Use the ColBERT rerank evaluator at `evals/colbert_rerank.py`.

## Workflow

1. Resolve `MLFLOW_TRACKING_URI`, defaulting to `http://127.0.0.1:5000`.
2. Check MLflow reachability with the experiments search API.
3. Start or restart the `mlflow_server` tmux session if needed.
4. Wait until MLflow responds, then fail fast if it remains unreachable.
5. Launch evaluation in a new tmux session.
6. Report the tmux session names and monitoring commands.

## Commands

Resolve the tracking endpoint:

```bash
cd /workspace/mm-embed-ecom
source .venv/bin/activate
set -a && source .env && set +a
TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
```

Check MLflow:

```bash
curl -sf -X POST -H "Content-Type: application/json" \
  -d '{"max_results": 1}' "${TRACKING_URI%/}/api/2.0/mlflow/experiments/search" >/dev/null
```

Start MLflow if unreachable:

```bash
tmux new-session -d -s mlflow_server "cd /workspace/mm-embed-ecom && source .venv/bin/activate && bash scripts/mlflow_server.sh"
tmux set-window-option -t "mlflow_server:0" remain-on-exit on
```

If `mlflow_server` already exists but the endpoint is still unreachable, restart it:

```bash
tmux kill-session -t mlflow_server
tmux new-session -d -s mlflow_server "cd /workspace/mm-embed-ecom && source .venv/bin/activate && bash scripts/mlflow_server.sh"
tmux set-window-option -t "mlflow_server:0" remain-on-exit on
```

Wait for MLflow:

```bash
for i in $(seq 1 30); do
  curl -sf -X POST -H "Content-Type: application/json" \
    -d '{"max_results": 1}' "${TRACKING_URI%/}/api/2.0/mlflow/experiments/search" >/dev/null && break
  sleep 2
done
curl -sf -X POST -H "Content-Type: application/json" \
  -d '{"max_results": 1}' "${TRACKING_URI%/}/api/2.0/mlflow/experiments/search" >/dev/null
```

Launch evaluation:

```bash
EVAL_SESSION="eval_$(date +%Y%m%d_%H%M%S)"
tmux new-session -d -s "${EVAL_SESSION}" "cd /workspace/mm-embed-ecom && source .venv/bin/activate && python -m evals.colbert_rerank --mlflow-run-id <RUN_ID> --config <RERANK_CONFIG_PATH>"
tmux set-window-option -t "${EVAL_SESSION}:0" remain-on-exit on
```

Example:

```bash
EVAL_SESSION="eval_$(date +%Y%m%d_%H%M%S)"
tmux new-session -d -s "${EVAL_SESSION}" "cd /workspace/mm-embed-ecom && source .venv/bin/activate && python -m evals.colbert_rerank --mlflow-run-id cc1e6cc3a91e472495b8aa21039ba0e0 --config configs/colbert_rerank/default.yaml"
tmux set-window-option -t "${EVAL_SESSION}:0" remain-on-exit on
```

## Monitoring

```bash
tmux ls
tmux attach -t mlflow_server
tmux attach -t <EVAL_SESSION>
```

## Final Response

When finishing a launch, report:

- Whether MLflow was already reachable or was started/restarted.
- MLflow tmux session name: `mlflow_server`.
- Evaluation tmux session name.
- Exact evaluator path: `evals/colbert_rerank.py`.
- Exact run id and rerank config path used.
