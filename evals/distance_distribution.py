import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from evals.colbert_rerank import (
    _as_sku_list,
    _encode_multi_vectors,
    _encode_single_vectors,
    _get_checkpoint_path,
    _load_cfg_from_run,
    _processed_root,
)
from models.siamese_clip import Tokenizers, get_transform
from models.siamese_clip_colbert import SiameseRuCLIPColBERT


CACHE_ROOT = Path("data/cache/embeddings")


def _cache_paths(run_id: str, title_v: int, desc_v: int, image_v: int) -> Tuple[Path, Path]:
    run_dir = CACHE_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / "single.pt", run_dir / f"multi_t{title_v}_d{desc_v}_i{image_v}.pt"


def _load_or_build_single(path, model, sku_list, source_indexed, images_dir,
                          tokenizers, transform, batch_size, device):
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    cache = _encode_single_vectors(
        model, sku_list, source_indexed, images_dir, tokenizers, transform, batch_size, device
    )
    torch.save(cache, path)
    return cache


def _load_or_build_multi(path, model, sku_list, source_indexed, images_dir,
                         tokenizers, transform, batch_size, device,
                         title_v, desc_v, image_v):
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    cache = _encode_multi_vectors(
        model, sku_list, source_indexed, images_dir, tokenizers, transform,
        batch_size, device, title_v, desc_v, image_v,
    )
    torch.save(cache, path)
    return cache


def _collect_per_query(
    grouped_queries: List[Tuple[str, set]],
    catalog_skus: List,
    single_cache: Dict,
    multi_cache: Dict,
    model: SiameseRuCLIPColBERT,
    device: str,
    k_max: int,
):
    cat_idx = {sku: i for i, sku in enumerate(catalog_skus)}
    catalog_mat = torch.stack([single_cache[sku] for sku in catalog_skus]).to(device)
    d_name = torch.stack([multi_cache[sku]["name"] for sku in catalog_skus]).to(device)
    d_desc = torch.stack([multi_cache[sku]["desc"] for sku in catalog_skus]).to(device)
    d_img = torch.stack([multi_cache[sku]["img"] for sku in catalog_skus]).to(device)

    base_curves, rer_curves = [], []
    base_ranks, rer_ranks = [], []
    top1_base, top1_rer = [], []

    with torch.no_grad():
        for q, relevant in tqdm(grouped_queries, desc="per_query_scores"):
            q_vec = single_cache[q].to(device)
            cos = catalog_mat @ q_vec
            if q in cat_idx:
                cos[cat_idx[q]] = float("-inf")
            top_scores, top_idx = torch.topk(cos, k=k_max, largest=True, sorted=True)
            shortlist_idx = top_idx.tolist()
            shortlist = [catalog_skus[i] for i in shortlist_idx]
            s1 = top_scores.cpu().numpy()

            qn = multi_cache[q]["name"].to(device)
            qd = multi_cache[q]["desc"].to(device)
            qi = multi_cache[q]["img"].to(device)
            s2 = model.colbert_score(
                qn, qd, qi,
                d_name[top_idx], d_desc[top_idx], d_img[top_idx],
            ).cpu().numpy()

            def _norm(x):
                lo, hi = float(x.min()), float(x.max())
                return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

            base_curves.append(_norm(s1))
            rer_curves.append(_norm(np.sort(s2)[::-1]))

            order_rer = np.argsort(-s2)
            shortlist_rer = [shortlist[i] for i in order_rer]
            for rank, sku in enumerate(shortlist, start=1):
                if sku in relevant:
                    base_ranks.append(rank)
            for rank, sku in enumerate(shortlist_rer, start=1):
                if sku in relevant:
                    rer_ranks.append(rank)
            top1_base.append(1 if shortlist[0] in relevant else 0)
            top1_rer.append(1 if shortlist_rer[0] in relevant else 0)

    return {
        "base_curves": np.stack(base_curves),
        "rer_curves": np.stack(rer_curves),
        "base_ranks": np.array(base_ranks, dtype=np.int32),
        "rer_ranks": np.array(rer_ranks, dtype=np.int32),
        "top1_base": float(np.mean(top1_base)),
        "top1_rer": float(np.mean(top1_rer)),
    }


def _plot(data: Dict, k_max: int, title: str, out_path: Path):
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    mean_base = data["base_curves"].mean(axis=0)
    mean_rer = data["rer_curves"].mean(axis=0)
    x = np.arange(1, k_max + 1)
    ax[0, 0].plot(x, mean_base, label="Bi-Encoder", color="#1f77b4")
    ax[0, 0].plot(x, mean_rer, label="ColBERT", color="#ff7f0e")
    ax[0, 0].set(title="Mean normalized score vs rank (shortlist)",
                 xlabel="rank", ylabel="score (min-max per query)")
    ax[0, 0].grid(alpha=0.3)
    ax[0, 0].legend()

    bins = np.arange(1, k_max + 2)
    ax[0, 1].hist(data["base_ranks"], bins=bins, alpha=0.55,
                  label="Bi-Encoder", color="#1f77b4", density=True)
    ax[0, 1].hist(data["rer_ranks"], bins=bins, alpha=0.55,
                  label="ColBERT", color="#ff7f0e", density=True)
    ax[0, 1].set(title="Rank distribution of relevant items",
                 xlabel="rank", ylabel="density")
    ax[0, 1].grid(alpha=0.3)
    ax[0, 1].legend()

    for name, ranks, color in [
        ("Bi-Encoder", data["base_ranks"], "#1f77b4"),
        ("ColBERT", data["rer_ranks"], "#ff7f0e"),
    ]:
        if len(ranks) == 0:
            continue
        r = np.sort(ranks)
        ax[1, 0].plot(r, np.arange(1, len(r) + 1) / len(r), label=name, color=color)
    ax[1, 0].set(title="CDF of relevant-item ranks",
                 xlabel="rank ≤ x", ylabel="fraction of positives")
    ax[1, 0].grid(alpha=0.3)
    ax[1, 0].legend()

    ax[1, 1].bar(["Bi-Encoder", "ColBERT"],
                 [data["top1_base"], data["top1_rer"]],
                 color=["#1f77b4", "#ff7f0e"])
    ax[1, 1].set(title="P(top-1 is relevant)", ylim=(0, max(0.05, 1.2 * max(data["top1_base"], data["top1_rer"]))))
    for i, v in enumerate([data["top1_base"], data["top1_rer"]]):
        ax[1, 1].text(i, v, f"{v:.3f}", ha="center", va="bottom")
    ax[1, 1].grid(alpha=0.3, axis="y")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow-run-id", required=True)
    parser.add_argument("--config", default="configs/distance_distribution/default.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_cfg = _load_cfg_from_run(args.mlflow_run_id)
    if "data_path" not in train_cfg:
        raise KeyError("Could not reconstruct train config from run params: missing data_path")

    tracking = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking:
        host = os.environ.get("MLFLOW_HOST", "127.0.0.1")
        port = os.environ.get("MLFLOW_PORT", "5000")
        tracking = f"http://{host}:{port}"
    mlflow.set_tracking_uri(tracking)

    device = str(cfg.get("device", train_cfg.get("device", "cuda"))).lower()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    data_path = Path(train_cfg["data_path"])
    source_df = pd.read_csv(data_path / train_cfg["source_table"])
    source_indexed = source_df.set_index("sku")
    images_dir = str(data_path / train_cfg["img_dataset_name"])

    proc = _processed_root(train_cfg)
    split_info = f"test={train_cfg['test_ratio']}_val={train_cfg['val_ratio']}"
    test_df = pd.read_parquet(proc / "pairwise-mapping-splits" / split_info / "test.parquet")

    title_v = int(cfg["title_vectors"])
    desc_v = int(cfg["desc_vectors"])
    image_v = int(cfg["image_vectors"])
    batch_size = int(cfg.get("batch_size", 128))
    k_max = int(cfg.get("k_max", 100))

    tokenizers = Tokenizers(train_cfg["name_model_name"], train_cfg["description_model_name"])
    transform = get_transform()

    model = SiameseRuCLIPColBERT(
        device=device,
        name_model_name=train_cfg["name_model_name"],
        description_model_name=train_cfg["description_model_name"],
        preload_model_name=None,
        models_dir=None,
        dropout=train_cfg.get("dropout"),
    )
    checkpoint_path = _get_checkpoint_path(args.mlflow_run_id)
    state_dict = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    catalog_skus_all = [sku for sku in source_df["sku"].tolist() if sku in source_indexed.index]

    single_path, multi_path = _cache_paths(args.mlflow_run_id, title_v, desc_v, image_v)
    single_cache = _load_or_build_single(
        single_path, model, catalog_skus_all, source_indexed, images_dir,
        tokenizers, transform, batch_size, device,
    )
    multi_cache = _load_or_build_multi(
        multi_path, model, catalog_skus_all, source_indexed, images_dir,
        tokenizers, transform, batch_size, device, title_v, desc_v, image_v,
    )

    catalog_skus = [sku for sku in catalog_skus_all if sku in single_cache and sku in multi_cache]

    grouped = test_df.groupby("sku_query", sort=False).first().reset_index()
    grouped_queries = []
    for _, row in grouped.iterrows():
        q = row["sku_query"]
        if q not in single_cache or q not in multi_cache:
            continue
        relevant = set(_as_sku_list(row["sku_pos"]))
        relevant.discard(q)
        if not relevant:
            continue
        grouped_queries.append((q, relevant))

    max_queries = cfg.get("max_queries")
    if max_queries:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(grouped_queries), size=min(int(max_queries), len(grouped_queries)), replace=False)
        grouped_queries = [grouped_queries[i] for i in sorted(idx.tolist())]

    data = _collect_per_query(
        grouped_queries, catalog_skus, single_cache, multi_cache, model, device, k_max,
    )

    out_dir = Path(cfg.get("output_dir", "logs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"distance_distribution_{args.mlflow_run_id[:8]}.png"
    title = (f"Peaked vs flat diagnostics (run {args.mlflow_run_id[:8]}, "
             f"n_queries={len(grouped_queries)}, k_max={k_max})")
    _plot(data, k_max, title, out_path)

    def _q(arr, q):
        return float(np.quantile(arr, q)) if len(arr) else float("nan")

    summary = {
        "n_queries": len(grouped_queries),
        "n_catalog": len(catalog_skus),
        "k_max": k_max,
        "top1_relevant_base": data["top1_base"],
        "top1_relevant_rer": data["top1_rer"],
        "relevant_rank_base_median": _q(data["base_ranks"], 0.5),
        "relevant_rank_rer_median": _q(data["rer_ranks"], 0.5),
        "relevant_rank_base_p25": _q(data["base_ranks"], 0.25),
        "relevant_rank_rer_p25": _q(data["rer_ranks"], 0.25),
        "score_curve_base_rank1_mean": float(data["base_curves"][:, 0].mean()),
        "score_curve_rer_rank1_mean": float(data["rer_curves"][:, 0].mean()),
        "score_curve_base_rank10_mean": float(data["base_curves"][:, min(9, k_max - 1)].mean()),
        "score_curve_rer_rank10_mean": float(data["rer_curves"][:, min(9, k_max - 1)].mean()),
        "plot_path": str(out_path),
    }
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    main()
