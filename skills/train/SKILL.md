---
name: train
description: Launch project training runs in tmux with MLflow safety checks. Use when starting `train.py`, running configs under `configs/`, or when the user asks to launch/relaunch training and ensure MLflow tracking is up.
---

# Training Launcher

Use this skill to launch training reliably in separate tmux sessions.

## Rules

- Always run from repository root: `/workspace/mm-embed-ecom`.
- Always activate the project virtualenv before scripts: `source .venv/bin/activate`.
- Always launch training in a dedicated tmux session.
- Ensure MLflow is reachable before training.
- If MLflow is not reachable, launch it in a separate tmux session via `scripts/mlflow_server.sh`.

## Workflow

1. Resolve tracking endpoint.
2. Check MLflow reachability.
3. Start/restart MLflow tmux session if needed.
4. Wait until MLflow responds.
5. Launch training in a new tmux session.
6. Report tmux session name(s) and how to monitor logs.

## Commands

### 1) Resolve tracking URI

```bash
cd /workspace/mm-embed-ecom
source .venv/bin/activate
set -a && source .env && set +a
TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
```

### 2) Check MLflow reachability

```bash
curl -sf -X POST -H "Content-Type: application/json" \
  -d '{"max_results": 1}' "${TRACKING_URI%/}/api/2.0/mlflow/experiments/search" >/dev/null
```

### 3) If unreachable: launch MLflow in tmux (per-session remain-on-exit)

Use a dedicated session name:

```bash
tmux new-session -d -s mlflow_server "cd /workspace/mm-embed-ecom && source .venv/bin/activate && bash scripts/mlflow_server.sh"
tmux set-window-option -t "mlflow_server:0" remain-on-exit on
```

If `mlflow_server` already exists but endpoint is still unreachable, restart it:

```bash
tmux kill-session -t mlflow_server
tmux new-session -d -s mlflow_server "cd /workspace/mm-embed-ecom && source .venv/bin/activate && bash scripts/mlflow_server.sh"
tmux set-window-option -t "mlflow_server:0" remain-on-exit on
```

### 4) Wait until MLflow is up

```bash
for i in $(seq 1 30); do
  curl -sf -X POST -H "Content-Type: application/json" \
    -d '{"max_results": 1}' "${TRACKING_URI%/}/api/2.0/mlflow/experiments/search" >/dev/null && break
  sleep 2
done
curl -sf -X POST -H "Content-Type: application/json" \
  -d '{"max_results": 1}' "${TRACKING_URI%/}/api/2.0/mlflow/experiments/search" >/dev/null
```

Fail fast if still unreachable.

### 5) Launch training in its own tmux session (per-session remain-on-exit)

Use a unique session name per run:

```bash
TRAIN_SESSION="train_$(date +%Y%m%d_%H%M%S)"
tmux new-session -d -s "${TRAIN_SESSION}" "cd /workspace/mm-embed-ecom && source .venv/bin/activate && python train.py --config <CONFIG_PATH>"
tmux set-window-option -t "${TRAIN_SESSION}:0" remain-on-exit on
```

Example:

```bash
TRAIN_SESSION="train_$(date +%Y%m%d_%H%M%S)"
tmux new-session -d -s "${TRAIN_SESSION}" "cd /workspace/mm-embed-ecom && source .venv/bin/activate && python train.py --config configs/siamese_clip_colbert/50_pairs.yml"
tmux set-window-option -t "${TRAIN_SESSION}:0" remain-on-exit on
```

This keeps each session's pane visible when the command exits (success or error), without setting a global tmux option.

## Monitoring

- List sessions:

```bash
tmux ls
```

- Attach to MLflow:

```bash
tmux attach -t mlflow_server
```

- Attach to training:

```bash
tmux attach -t <TRAIN_SESSION>
```

## Response Checklist

When finishing a launch, report:

- Whether MLflow was already reachable or was started/restarted.
- MLflow tmux session name (`mlflow_server`).
- Training tmux session name.
- Exact config path used.
