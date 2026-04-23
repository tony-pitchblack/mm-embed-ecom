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
    single_path = run_dir / "single.pt"
    multi_path = run_dir / f"multi_t{title_v}_d{desc_v}_i{image_v}.pt"
    return single_path, multi_path


def _load_or_build_single(
    path: Path, model, sku_list, source_indexed, images_dir, tokenizers, transform, batch_size, device
) -> Dict:
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    cache = _encode_single_vectors(
        model, sku_list, source_indexed, images_dir, tokenizers, transform, batch_size, device
    )
    torch.save(cache, path)
    return cache


def _load_or_build_multi(
    path: Path,
    model,
    sku_list,
    source_indexed,
    images_dir,
    tokenizers,
    transform,
    batch_size,
    device,
    title_v,
    desc_v,
    image_v,
) -> Dict:
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    cache = _encode_multi_vectors(
        model,
        sku_list,
        source_indexed,
        images_dir,
        tokenizers,
        transform,
        batch_size,
        device,
        title_v,
        desc_v,
        image_v,
    )
    torch.save(cache, path)
    return cache


def _single_cosine_distances(
    query_skus: List, catalog_skus: List, single_cache: Dict, device: str
) -> np.ndarray:
    cat_idx = {sku: i for i, sku in enumerate(catalog_skus)}
    catalog_mat = torch.stack([single_cache[sku] for sku in catalog_skus]).to(device)
    out = []
    with torch.no_grad():
        for q in tqdm(query_skus, desc="single_distances"):
            q_vec = single_cache[q].to(device)
            cos = catalog_mat @ q_vec
            dist = (1.0 - cos).cpu().numpy()
            mask = np.ones(len(catalog_skus), dtype=bool)
            mask[cat_idx[q]] = False
            out.append(dist[mask])
    return np.concatenate(out)


def _colbert_cosine_distances(
    query_skus: List,
    catalog_skus: List,
    multi_cache: Dict,
    model: SiameseRuCLIPColBERT,
    title_v: int,
    desc_v: int,
    image_v: int,
    device: str,
) -> np.ndarray:
    cat_idx = {sku: i for i, sku in enumerate(catalog_skus)}
    d_name = torch.stack([multi_cache[sku]["name"] for sku in catalog_skus]).to(device)
    d_desc = torch.stack([multi_cache[sku]["desc"] for sku in catalog_skus]).to(device)
    d_img = torch.stack([multi_cache[sku]["img"] for sku in catalog_skus]).to(device)
    n_total = float(title_v + desc_v + image_v)
    out = []
    with torch.no_grad():
        for q in tqdm(query_skus, desc="colbert_distances"):
            qn = multi_cache[q]["name"].to(device)
            qd = multi_cache[q]["desc"].to(device)
            qi = multi_cache[q]["img"].to(device)
            scores = model.colbert_score(qn, qd, qi, d_name, d_desc, d_img)
            dist = (1.0 - scores / n_total).cpu().numpy()
            mask = np.ones(len(catalog_skus), dtype=bool)
            mask[cat_idx[q]] = False
            out.append(dist[mask])
    return np.concatenate(out)


def _plot(
    single_dists: np.ndarray,
    colbert_dists: np.ndarray,
    bins: int,
    title: str,
    out_path: Path,
):
    lo = float(min(single_dists.min(), colbert_dists.min()))
    hi = float(max(single_dists.max(), colbert_dists.max()))
    edges = np.linspace(lo, hi, bins + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(single_dists, bins=edges, alpha=0.55, label="Siamese Bi-Encoder", density=True, color="#1f77b4")
    ax.hist(colbert_dists, bins=edges, alpha=0.55, label="ColBERT late-interaction", density=True, color="#ff7f0e")
    ax.axvline(single_dists.mean(), color="#1f77b4", linestyle="--", linewidth=1)
    ax.axvline(colbert_dists.mean(), color="#ff7f0e", linestyle="--", linewidth=1)
    ax.set_xlabel("cosine distance (1 - cos)")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
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
    query_skus = [sku for sku in test_df["sku_query"].drop_duplicates().tolist() if sku in single_cache and sku in multi_cache]
    max_queries = cfg.get("max_queries")
    if max_queries:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(query_skus), size=min(int(max_queries), len(query_skus)), replace=False)
        query_skus = [query_skus[i] for i in sorted(idx.tolist())]

    single_dists = _single_cosine_distances(query_skus, catalog_skus, single_cache, device)
    colbert_dists = _colbert_cosine_distances(
        query_skus, catalog_skus, multi_cache, model, title_v, desc_v, image_v, device
    )

    out_dir = Path(cfg.get("output_dir", "logs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"distance_distribution_{args.mlflow_run_id[:8]}.png"
    title = f"Query→catalog distance distribution (run {args.mlflow_run_id[:8]}, n_queries={len(query_skus)})"
    _plot(single_dists, colbert_dists, int(cfg.get("bins", 60)), title, out_path)

    summary = {
        "n_queries": len(query_skus),
        "n_catalog": len(catalog_skus),
        "single_mean": float(single_dists.mean()),
        "single_std": float(single_dists.std()),
        "colbert_mean": float(colbert_dists.mean()),
        "colbert_std": float(colbert_dists.std()),
        "plot_path": str(out_path),
    }
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    main()
