"""Plot Recall/MRR/NDCG@k across MLflow runs.

Each --run spec is `RUN_ID:LABEL:COLOR:MARKER:LINESTYLE`.
Metric keys are read as `{prefix}{name}_at_{k}{suffix}`.
"""
from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient


METRIC_NAMES = ("recall", "mrr", "ndcg")


def parse_run_spec(spec: str) -> Dict[str, str]:
    parts = spec.split(":")
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            f"--run expects RUN_ID:LABEL:COLOR:MARKER:LINESTYLE, got {spec!r}"
        )
    return {
        "run_id": parts[0],
        "label": parts[1],
        "color": parts[2],
        "marker": parts[3],
        "linestyle": parts[4],
    }


def collect_metrics(
    client: MlflowClient,
    run_id: str,
    prefix: str,
    suffix: str,
) -> Tuple[List[int], Dict[str, Dict[int, float]]]:
    metrics = client.get_run(run_id).data.metrics
    pat = re.compile(
        rf"^{re.escape(prefix)}(recall|mrr|ndcg)_at_(\d+){re.escape(suffix)}$"
    )
    out: Dict[str, Dict[int, float]] = {m: {} for m in METRIC_NAMES}
    ks: set[int] = set()
    for key, value in metrics.items():
        m = pat.match(key)
        if not m:
            continue
        name, k = m.group(1), int(m.group(2))
        out[name][k] = float(value)
        ks.add(k)
    return sorted(ks), out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=parse_run_spec)
    parser.add_argument("--metric-prefix", default="test_full/")
    parser.add_argument("--metric-suffix", default="_colbert")
    parser.add_argument("--title", default="Pairwise metrics comparison")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--annotate", action="store_true")
    args = parser.parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    series = []
    all_ks: set[int] = set()
    for spec in args.run:
        ks, metrics = collect_metrics(
            client, spec["run_id"], args.metric_prefix, args.metric_suffix
        )
        if not ks:
            raise SystemExit(f"No matching metrics for run {spec['run_id']}")
        all_ks.update(ks)
        series.append({"spec": spec, "ks": ks, "metrics": metrics})

    ks_sorted = sorted(all_ks)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, name in zip(axes, METRIC_NAMES):
        for s in series:
            spec = s["spec"]
            xs = [k for k in ks_sorted if k in s["metrics"][name]]
            ys = [s["metrics"][name][k] for k in xs]
            ax.plot(
                xs, ys,
                label=spec["label"],
                color=spec["color"],
                marker=spec["marker"],
                linestyle=spec["linestyle"],
                linewidth=2,
                markersize=8,
            )
            if args.annotate and len(series) > 1 and len(xs) > 0:
                base = series[0]
                for k, y in zip(xs, ys):
                    if s is base or k not in base["metrics"][name]:
                        continue
                    delta = y - base["metrics"][name][k]
                    ax.annotate(
                        f"{delta:+.4f}",
                        xy=(k, y),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=8,
                        color=spec["color"],
                    )
        ax.set_title(name.upper())
        ax.set_xlabel("k")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Score")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(series),
               bbox_to_anchor=(0.5, 0.98))
    fig.suptitle(args.title, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
