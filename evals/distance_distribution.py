import argparse
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from evals._cache import (
    build_siamese_colbert,
    cache_paths,
    load_or_build_multi,
    load_or_build_single,
)
from evals.colbert_rerank import (
    _as_sku_list,
    _get_checkpoint_path,
    _load_cfg_from_run,
    _processed_root,
)
from models.siamese_clip import Tokenizers, get_transform
from models.siamese_clip_colbert import SiameseRuCLIPColBERTWithHead


SERIES_CACHE_DIR = Path("data/cache/distdist")
VALID_MODES = ("stage1", "stage1_colbert_rerank", "colbert")


def _norm_curve(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def _queries_hash(queries: List[Tuple[Any, set]]) -> str:
    blob = "|".join(f"{q}:{','.join(sorted(map(str, rel)))}" for q, rel in queries)
    return hashlib.md5(blob.encode()).hexdigest()[:8]


def _series_cache_path(
    run_id: str, mode: str, k_max: int, t: int, d: int, i: int, qhash: str
) -> Path:
    SERIES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return SERIES_CACHE_DIR / f"{run_id[:12]}_{mode}_k{k_max}_t{t}_d{d}_i{i}_q{qhash}.pt"


def _build_run_resources(
    run_id: str,
    rerank_cfg: dict,
    device: str,
    needs_single: bool,
    needs_multi: bool,
    force: bool,
):
    train_cfg = _load_cfg_from_run(run_id)
    data_path = Path(train_cfg["data_path"])
    source_df = pd.read_csv(data_path / train_cfg["source_table"])
    source_indexed = source_df.set_index("sku")
    images_dir = str(data_path / train_cfg["img_dataset_name"])

    tokenizers = Tokenizers(train_cfg["name_model_name"], train_cfg["description_model_name"])
    transform = get_transform()

    title_v = int(rerank_cfg["title_vectors"])
    desc_v = int(rerank_cfg["desc_vectors"])
    image_v = int(rerank_cfg["image_vectors"])
    batch_size = int(rerank_cfg.get("batch_size", 128))
    catalog_skus_all = [sku for sku in source_df["sku"].tolist() if sku in source_indexed.index]

    single_path, multi_path = cache_paths(run_id, title_v, desc_v, image_v)
    needs_model = (needs_single and (force or not single_path.exists())) or (
        needs_multi and (force or not multi_path.exists())
    )
    model = None
    if needs_model:
        model = build_siamese_colbert(train_cfg, device, _get_checkpoint_path(run_id)).to(device)
        model.eval()

    single_cache = None
    multi_cache = None
    if needs_single:
        single_cache = load_or_build_single(
            single_path, model, catalog_skus_all, source_indexed, images_dir,
            tokenizers, transform, batch_size, device, force=force,
        )
    if needs_multi:
        multi_cache = load_or_build_multi(
            multi_path, model, catalog_skus_all, source_indexed, images_dir,
            tokenizers, transform, batch_size, device,
            title_v, desc_v, image_v, force=force,
        )

    proc = _processed_root(train_cfg)
    split_info = f"test={train_cfg['test_ratio']}_val={train_cfg['val_ratio']}"
    test_df = pd.read_parquet(proc / "pairwise-mapping-splits" / split_info / "test.parquet")
    return {
        "train_cfg": train_cfg,
        "single_cache": single_cache,
        "multi_cache": multi_cache,
        "catalog_skus_all": catalog_skus_all,
        "test_df": test_df,
    }


def _select_queries(test_df: pd.DataFrame, valid_skus: set, max_queries):
    grouped = test_df.groupby("sku_query", sort=False).first().reset_index()
    out = []
    for _, row in grouped.iterrows():
        q = row["sku_query"]
        if q not in valid_skus:
            continue
        relevant = set(_as_sku_list(row["sku_pos"]))
        relevant.discard(q)
        if not relevant:
            continue
        out.append((q, relevant))
    if max_queries:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(out), size=min(int(max_queries), len(out)), replace=False)
        out = [out[i] for i in sorted(idx.tolist())]
    return out


def _stack_multi(multi_cache, skus, device):
    d_name = torch.stack([multi_cache[s]["name"] for s in skus]).to(device)
    d_desc = torch.stack([multi_cache[s]["desc"] for s in skus]).to(device)
    d_img = torch.stack([multi_cache[s]["img"] for s in skus]).to(device)
    return d_name, d_desc, d_img


def _colbert_scores_query_vs_docs(q, d_name, d_desc, d_img, t, d, i, device):
    q_n = q["name"].to(device).unsqueeze(0)
    q_d = q["desc"].to(device).unsqueeze(0)
    q_i = q["img"].to(device).unsqueeze(0)
    li = SiameseRuCLIPColBERTWithHead.late_interaction
    name_score = li(q_n, d_name) / float(max(t, 1))
    desc_score = li(q_d, d_desc) / float(max(d, 1))
    img_score = li(q_i, d_img) / float(max(i, 1))
    return (name_score + desc_score + img_score) / 3.0


def _collect_stage1(catalog_skus, single_cache, queries, k_max, device):
    cat_idx = {s: i for i, s in enumerate(catalog_skus)}
    catalog_mat = torch.stack([single_cache[s] for s in catalog_skus]).to(device)
    curves, ranks, top1 = [], [], []
    for q, relevant in tqdm(queries, desc="stage1"):
        q_vec = single_cache[q].to(device)
        cos = catalog_mat @ q_vec
        if q in cat_idx:
            cos[cat_idx[q]] = float("-inf")
        top_scores, top_idx = torch.topk(cos, k=k_max, largest=True, sorted=True)
        shortlist = [catalog_skus[i] for i in top_idx.tolist()]
        curves.append(_norm_curve(top_scores.cpu().numpy()))
        for r, sku in enumerate(shortlist, start=1):
            if sku in relevant:
                ranks.append(r)
        top1.append(1 if shortlist[0] in relevant else 0)
    return _pack(curves, ranks, top1)


def _collect_stage1_rerank(
    catalog_skus, single_cache, multi_cache, queries, k_max, device, t, d, i
):
    cat_idx = {s: i for i, s in enumerate(catalog_skus)}
    catalog_mat = torch.stack([single_cache[s] for s in catalog_skus]).to(device)
    d_name, d_desc, d_img = _stack_multi(multi_cache, catalog_skus, device)
    curves, ranks, top1 = [], [], []
    for q, relevant in tqdm(queries, desc="stage1_colbert_rerank"):
        q_vec = single_cache[q].to(device)
        cos = catalog_mat @ q_vec
        if q in cat_idx:
            cos[cat_idx[q]] = float("-inf")
        _, top_idx = torch.topk(cos, k=k_max, largest=True, sorted=True)
        shortlist = [catalog_skus[i] for i in top_idx.tolist()]
        s2 = _colbert_scores_query_vs_docs(
            multi_cache[q],
            d_name.index_select(0, top_idx),
            d_desc.index_select(0, top_idx),
            d_img.index_select(0, top_idx),
            t, d, i, device,
        ).cpu().numpy()
        order = np.argsort(-s2)
        reranked = [shortlist[k] for k in order]
        curves.append(_norm_curve(np.sort(s2)[::-1]))
        for r, sku in enumerate(reranked, start=1):
            if sku in relevant:
                ranks.append(r)
        top1.append(1 if reranked[0] in relevant else 0)
    return _pack(curves, ranks, top1)


def _collect_colbert(catalog_skus, multi_cache, queries, k_max, device, t, d, i):
    cat_idx = {s: i for i, s in enumerate(catalog_skus)}
    d_name, d_desc, d_img = _stack_multi(multi_cache, catalog_skus, device)
    curves, ranks, top1 = [], [], []
    for q, relevant in tqdm(queries, desc="colbert"):
        scores = _colbert_scores_query_vs_docs(
            multi_cache[q], d_name, d_desc, d_img, t, d, i, device
        )
        if q in cat_idx:
            scores[cat_idx[q]] = float("-inf")
        top_scores, top_idx = torch.topk(scores, k=k_max, largest=True, sorted=True)
        shortlist = [catalog_skus[k] for k in top_idx.tolist()]
        curves.append(_norm_curve(top_scores.cpu().numpy()))
        for r, sku in enumerate(shortlist, start=1):
            if sku in relevant:
                ranks.append(r)
        top1.append(1 if shortlist[0] in relevant else 0)
    return _pack(curves, ranks, top1)


def _pack(curves, ranks, top1):
    return {
        "curves": np.stack(curves) if curves else np.zeros((0, 0)),
        "ranks": np.array(ranks, dtype=np.int32),
        "top1": float(np.mean(top1)) if top1 else 0.0,
    }


def compute_or_load_series(
    series_cfg: dict, rerank_cfg: dict, device: str, force: bool, run_resources_cache: dict
):
    run_id = series_cfg["run_id"]
    mode = series_cfg["mode"]
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode: {mode}; valid: {VALID_MODES}")
    k_max = int(rerank_cfg["k_max"])
    t = int(rerank_cfg["title_vectors"])
    d = int(rerank_cfg["desc_vectors"])
    i = int(rerank_cfg["image_vectors"])
    max_queries = rerank_cfg.get("max_queries")

    needs_single = mode in ("stage1", "stage1_colbert_rerank")
    needs_multi = mode in ("stage1_colbert_rerank", "colbert")
    rk = (run_id, needs_single, needs_multi)
    if rk not in run_resources_cache:
        run_resources_cache[rk] = _build_run_resources(
            run_id, rerank_cfg, device, needs_single, needs_multi, force
        )
    res = run_resources_cache[rk]

    catalog_skus = list(res["catalog_skus_all"])
    if needs_single:
        catalog_skus = [s for s in catalog_skus if s in res["single_cache"]]
    if needs_multi:
        catalog_skus = [s for s in catalog_skus if s in res["multi_cache"]]
    valid = set(catalog_skus)
    queries = _select_queries(res["test_df"], valid, max_queries)
    qhash = _queries_hash(queries)

    cache_path = _series_cache_path(run_id, mode, k_max, t, d, i, qhash)
    if cache_path.exists() and not force:
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        data["n_queries"] = len(queries)
        data["n_catalog"] = len(catalog_skus)
        return data

    with torch.no_grad():
        if mode == "stage1":
            data = _collect_stage1(catalog_skus, res["single_cache"], queries, k_max, device)
        elif mode == "stage1_colbert_rerank":
            data = _collect_stage1_rerank(
                catalog_skus, res["single_cache"], res["multi_cache"], queries, k_max, device, t, d, i
            )
        else:
            data = _collect_colbert(
                catalog_skus, res["multi_cache"], queries, k_max, device, t, d, i
            )
    data["n_queries"] = len(queries)
    data["n_catalog"] = len(catalog_skus)
    torch.save({k: v for k, v in data.items() if k in ("curves", "ranks", "top1")}, cache_path)
    return data


def _plot_multi(series: List[Tuple[dict, dict]], k_max: int, title: str, out_path: Path):
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    x = np.arange(1, k_max + 1)
    for s_cfg, data in series:
        if data["curves"].size == 0:
            continue
        ax[0, 0].plot(x, data["curves"].mean(axis=0), label=s_cfg["name"], color=s_cfg["color"])
    ax[0, 0].set(title="Mean normalized score vs rank (shortlist)",
                 xlabel="rank", ylabel="score (min-max per query)")
    ax[0, 0].grid(alpha=0.3)
    ax[0, 0].legend()

    bins = np.arange(1, k_max + 2)
    for s_cfg, data in series:
        if len(data["ranks"]) == 0:
            continue
        ax[0, 1].hist(data["ranks"], bins=bins, alpha=0.45,
                      label=s_cfg["name"], color=s_cfg["color"], density=True)
    ax[0, 1].set(title="Rank distribution of relevant items",
                 xlabel="rank", ylabel="density")
    ax[0, 1].grid(alpha=0.3)
    ax[0, 1].legend()

    for s_cfg, data in series:
        ranks = data["ranks"]
        if len(ranks) == 0:
            continue
        r = np.sort(ranks)
        ax[1, 0].plot(r, np.arange(1, len(r) + 1) / len(r),
                      label=s_cfg["name"], color=s_cfg["color"])
    ax[1, 0].set(title="CDF of relevant-item ranks",
                 xlabel="rank ≤ x", ylabel="fraction of positives")
    ax[1, 0].grid(alpha=0.3)
    ax[1, 0].legend()

    names = [s_cfg["name"] for s_cfg, _ in series]
    top1_vals = [data["top1"] for _, data in series]
    colors = [s_cfg["color"] for s_cfg, _ in series]
    ax[1, 1].bar(names, top1_vals, color=colors)
    ax[1, 1].set(title="P(top-1 is relevant)",
                 ylim=(0, max(0.05, 1.2 * (max(top1_vals) if top1_vals else 0.0))))
    for j, v in enumerate(top1_vals):
        ax[1, 1].text(j, v, f"{v:.3f}", ha="center", va="bottom")
    ax[1, 1].tick_params(axis="x", labelrotation=15)
    ax[1, 1].grid(alpha=0.3, axis="y")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/distance_distribution/3way.yaml")
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tracking = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking:
        host = os.environ.get("MLFLOW_HOST", "127.0.0.1")
        port = os.environ.get("MLFLOW_PORT", "5000")
        tracking = f"http://{host}:{port}"
    mlflow.set_tracking_uri(tracking)

    device = str(cfg.get("device", "cuda")).lower()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    series_cfgs = cfg.get("series")
    if not series_cfgs:
        raise ValueError("config must include a non-empty 'series' list")

    run_resources_cache: Dict[Any, Any] = {}
    series_results = []
    for s in series_cfgs:
        data = compute_or_load_series(s, cfg, device, args.force_recompute, run_resources_cache)
        series_results.append((s, data))

    out_dir = Path(cfg.get("output_dir", "logs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = cfg.get("out_name", "distance_distribution_multi.png")
    out_path = out_dir / out_name
    nq = max((d["n_queries"] for _, d in series_results), default=0)
    title = f"Peaked vs flat diagnostics (n_queries={nq}, k_max={int(cfg['k_max'])})"
    _plot_multi(series_results, int(cfg["k_max"]), title, out_path)

    def _q(arr, q):
        return float(np.quantile(arr, q)) if len(arr) else float("nan")

    summary = {"plot_path": str(out_path), "series": []}
    for s_cfg, d in series_results:
        summary["series"].append({
            "name": s_cfg["name"],
            "run_id": s_cfg["run_id"],
            "mode": s_cfg["mode"],
            "n_queries": d["n_queries"],
            "n_catalog": d["n_catalog"],
            "top1_relevant": d["top1"],
            "rank_median": _q(d["ranks"], 0.5),
            "rank_p25": _q(d["ranks"], 0.25),
            "score_curve_rank1_mean": float(d["curves"][:, 0].mean()) if d["curves"].size else float("nan"),
            "score_curve_rank10_mean": float(d["curves"][:, min(9, int(cfg["k_max"]) - 1)].mean()) if d["curves"].size else float("nan"),
        })
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    main()
