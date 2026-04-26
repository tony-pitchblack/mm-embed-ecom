import argparse
import ast
import os
from pathlib import Path
from typing import Dict, Iterable, List, Set

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from mlflow.tracking import MlflowClient
from tqdm import tqdm

from evals._cache import (
    build_siamese_colbert,
    cache_paths,
    load_or_build_multi,
    load_or_build_single,
)
from models.siamese_clip import Tokenizers, get_transform


def _convert_scalar(raw: str):
    if raw is None:
        return None
    val = raw.strip()
    low = val.lower()
    if low == "none" or low == "null":
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return ast.literal_eval(val)
    except Exception:
        return val


def _load_cfg_from_run(run_id: str) -> dict:
    client = MlflowClient()
    run = client.get_run(run_id)
    return {k: _convert_scalar(v) for k, v in run.data.params.items()}


def _processed_root(cfg: dict) -> Path:
    return Path(cfg["data_path"]) / Path(cfg["source_table"]).parent


def _as_sku_list(x) -> List:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (tuple, set)):
        return list(x)
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return []
    return []


def _metric_update(metrics, ranked_skus: List, relevant: Set, ks: Iterable[int]):
    rel_ranks = [idx + 1 for idx, sku in enumerate(ranked_skus) if sku in relevant]
    for k in ks:
        topk = ranked_skus[:k]
        hits = sum(1 for sku in topk if sku in relevant)
        metrics["recall"][k].append(float(hits / max(len(relevant), 1)))
        rr = 0.0
        if rel_ranks and rel_ranks[0] <= k:
            rr = 1.0 / float(rel_ranks[0])
        metrics["mrr"][k].append(rr)
        if hits == 0:
            metrics["ndcg"][k].append(0.0)
            continue
        rel = np.array([1.0 if sku in relevant else 0.0 for sku in topk], dtype=np.float32)
        dcg = float((rel / np.log2(np.arange(2, len(rel) + 2))).sum()) if len(rel) > 0 else 0.0
        ideal_len = min(len(relevant), k)
        ideal = np.ones(ideal_len, dtype=np.float32)
        idcg = float((ideal / np.log2(np.arange(2, ideal_len + 2))).sum()) if ideal_len > 0 else 0.0
        metrics["ndcg"][k].append(float(dcg / idcg) if idcg > 0 else 0.0)


def _metric_finalize(metrics, ks: Iterable[int]) -> Dict[str, float]:
    out = {}
    for name in ("recall", "mrr", "ndcg"):
        for k in ks:
            vals = metrics[name][k]
            out[f"{name}_at_{k}"] = float(np.mean(vals)) if vals else 0.0
    return out


def _normalized_colbert_scores(model, q, d_name, d_desc, d_img, n_title: int, n_desc: int, n_img: int):
    name_score = model.late_interaction(q["name"].to(d_name.device).unsqueeze(0), d_name) / float(
        max(n_title, 1)
    )
    desc_score = model.late_interaction(q["desc"].to(d_desc.device).unsqueeze(0), d_desc) / float(
        max(n_desc, 1)
    )
    img_score = model.late_interaction(q["img"].to(d_img.device).unsqueeze(0), d_img) / float(
        max(n_img, 1)
    )
    return (name_score + desc_score + img_score) / 3.0


def _get_checkpoint_path(run_id: str) -> str:
    model_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model")
    model_path = Path(model_dir)
    candidates = sorted(model_path.rglob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint .pt found under artifact path: {model_path}")
    return str(candidates[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow-run-id", required=True)
    parser.add_argument("--config", default="configs/colbert_rerank/default.yaml")
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        rerank_cfg = yaml.safe_load(f)

    tracking = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking:
        host = os.environ.get("MLFLOW_HOST", "127.0.0.1")
        port = os.environ.get("MLFLOW_PORT", "5000")
        tracking = f"http://{host}:{port}"
    mlflow.set_tracking_uri(tracking)
    train_cfg = _load_cfg_from_run(args.mlflow_run_id)
    if "data_path" not in train_cfg:
        raise KeyError("Could not reconstruct train config from run params: missing data_path")

    checkpoint_path = _get_checkpoint_path(args.mlflow_run_id)
    data_path = Path(train_cfg["data_path"])
    source_df = pd.read_csv(data_path / train_cfg["source_table"])
    source_indexed = source_df.set_index("sku")
    images_dir = str(data_path / train_cfg["img_dataset_name"])

    proc = _processed_root(train_cfg)
    split_info = f"test={train_cfg['test_ratio']}_val={train_cfg['val_ratio']}"
    test_path = proc / "pairwise-mapping-splits" / split_info / "test.parquet"
    test_df = pd.read_parquet(test_path)

    device = str(rerank_cfg.get("device", train_cfg.get("device", "cuda"))).lower()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    tokenizers = Tokenizers(train_cfg["name_model_name"], train_cfg["description_model_name"])
    transform = get_transform()

    model = build_siamese_colbert(train_cfg, device, checkpoint_path)
    model = model.to(device)
    model.eval()

    ks = sorted(int(k) for k in rerank_cfg["k"])
    k_max = max(ks)
    batch_size = int(rerank_cfg.get("batch_size", 128))
    title_vectors = int(rerank_cfg["title_vectors"])
    desc_vectors = int(rerank_cfg["desc_vectors"])
    image_vectors = int(rerank_cfg["image_vectors"])

    catalog_skus = [sku for sku in source_df["sku"].tolist() if sku in source_indexed.index]
    single_path, multi_path = cache_paths(
        args.mlflow_run_id, title_vectors, desc_vectors, image_vectors
    )
    single_cache = load_or_build_single(
        single_path,
        model,
        catalog_skus,
        source_indexed,
        images_dir,
        tokenizers,
        transform,
        batch_size,
        device,
        force=args.force_recompute,
    )
    catalog_skus = [sku for sku in catalog_skus if sku in single_cache]
    catalog_mat = torch.stack([single_cache[sku] for sku in catalog_skus]).to(device)

    grouped = test_df.groupby("sku_query", sort=False).first().reset_index()
    baseline_acc = {m: {k: [] for k in ks} for m in ("recall", "mrr", "ndcg")}
    rerank_acc = {m: {k: [] for k in ks} for m in ("recall", "mrr", "ndcg")}
    shortlist_by_query = {}
    relevant_by_query = {}

    with torch.no_grad():
        for _, row in tqdm(grouped.iterrows(), total=len(grouped), desc="stage1_retrieval"):
            query_sku = row["sku_query"]
            if query_sku not in single_cache:
                continue
            relevant = set(_as_sku_list(row["sku_pos"]))
            relevant.discard(query_sku)
            if not relevant:
                continue
            q_vec = single_cache[query_sku].to(device)
            dists = torch.norm(catalog_mat - q_vec.unsqueeze(0), dim=1)
            order = torch.argsort(dists, descending=False).tolist()
            ranked = [catalog_skus[i] for i in order if catalog_skus[i] != query_sku]
            ranked = ranked[:k_max]
            if not ranked:
                continue
            shortlist_by_query[query_sku] = ranked
            relevant_by_query[query_sku] = relevant
            _metric_update(baseline_acc, ranked, relevant, ks)

    baseline_metrics = _metric_finalize(baseline_acc, ks)

    needed_skus = set(shortlist_by_query.keys())
    for cand_list in shortlist_by_query.values():
        needed_skus.update(cand_list)
    multi_cache = load_or_build_multi(
        multi_path,
        model,
        sorted(needed_skus),
        source_indexed,
        images_dir,
        tokenizers,
        transform,
        batch_size,
        device,
        title_vectors,
        desc_vectors,
        image_vectors,
        force=args.force_recompute,
    )

    with torch.no_grad():
        for query_sku, candidates in tqdm(shortlist_by_query.items(), desc="stage2_rerank"):
            if query_sku not in multi_cache:
                continue
            available = [sku for sku in candidates if sku in multi_cache]
            if not available:
                continue
            q = multi_cache[query_sku]
            d_name = torch.stack([multi_cache[sku]["name"] for sku in available]).to(device)
            d_desc = torch.stack([multi_cache[sku]["desc"] for sku in available]).to(device)
            d_img = torch.stack([multi_cache[sku]["img"] for sku in available]).to(device)
            scores = _normalized_colbert_scores(
                model,
                q,
                d_name,
                d_desc,
                d_img,
                title_vectors,
                desc_vectors,
                image_vectors,
            )
            order = torch.argsort(scores, descending=True).tolist()
            reranked = [available[i] for i in order]
            _metric_update(rerank_acc, reranked, relevant_by_query[query_sku], ks)

    rerank_metrics = _metric_finalize(rerank_acc, ks)

    with mlflow.start_run(run_id=args.mlflow_run_id):
        mlflow.log_params(
            {
                "colbert_k": ",".join(str(x) for x in ks),
                "colbert_title_vectors": title_vectors,
                "colbert_desc_vectors": desc_vectors,
                "colbert_image_vectors": image_vectors,
            }
        )
        for metric_name, value in baseline_metrics.items():
            mlflow.log_metric(f"test_full/{metric_name}", value)
        for metric_name, value in rerank_metrics.items():
            mlflow.log_metric(f"test_full/{metric_name}_colbert", value)


if __name__ == "__main__":
    main()
