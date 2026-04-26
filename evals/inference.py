import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import pandas as pd
import torch
import yaml
from mlflow.tracking import MlflowClient

from dataset import split_query_groups
from evals._cache import build_siamese_colbert, cache_paths, load_or_build_multi, load_or_build_single
from evals.colbert_rerank import (
    _convert_scalar,
    _get_checkpoint_path,
    _normalized_colbert_pair_scores,
)
from models.siamese_clip import Tokenizers, get_transform
from models.siamese_clip_colbert import SiameseRuCLIPColBERTWithHead


OUTPUT_BY_MODE = {
    "final_emb": "result.json",
    "colbert_rerank": "result_colbert_rerank.json",
    "colbert": "result_colbert.json",
}
OZON_URL_TEMPLATE = "https://www.ozon.ru/context/detail/id/{sku}/"


def _tracking_uri() -> str:
    tracking = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking:
        return tracking
    host = os.environ.get("MLFLOW_HOST", "127.0.0.1")
    port = os.environ.get("MLFLOW_PORT", "5000")
    return f"http://{host}:{port}"


def _load_cfg_from_tracking(run_id: str) -> dict:
    run = MlflowClient().get_run(run_id)
    return {k: _convert_scalar(v) for k, v in run.data.params.items()}


def _load_cfg_from_sqlite(run_id: str, db_path: Path = Path("logs/mlflow/database.db")) -> dict:
    if not db_path.is_file():
        return {}
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "select key, value from params where run_uuid = ?",
            (run_id,),
        ).fetchall()
    return {k: _convert_scalar(v) for k, v in rows}


def _load_cfg(run_id: str, data_run_id: Optional[str]) -> dict:
    cfg = {}
    try:
        cfg = _load_cfg_from_tracking(run_id)
    except Exception:
        cfg = _load_cfg_from_sqlite(run_id)
    if "data_path" in cfg:
        return cfg
    if data_run_id:
        try:
            cfg = _load_cfg_from_tracking(data_run_id)
        except Exception:
            cfg = _load_cfg_from_sqlite(data_run_id)
    if "data_path" not in cfg:
        raise KeyError("Could not reconstruct data config from MLflow params.")
    return cfg


def _processed_root(cfg: dict) -> Path:
    return Path(cfg["data_path"]) / Path(cfg["source_table"]).parent


def _load_test_df(cfg: dict) -> pd.DataFrame:
    proc = _processed_root(cfg)
    split_info = f"test={cfg['test_ratio']}_val={cfg['val_ratio']}"
    test_path = proc / "pairwise-mapping-splits" / split_info / "test.parquet"
    if test_path.is_file():
        return pd.read_parquet(test_path)
    mapping = pd.read_parquet(Path(cfg["data_path"]) / cfg["pairwise_mapping_file"])
    splits = split_query_groups(
        mapping,
        test_size=float(cfg["test_ratio"]),
        val_size=float(cfg["val_ratio"]),
        random_state=int(cfg["random_seed"]),
    )
    return splits["test"]


def _resolve_device(cfg: dict, rerank_cfg: dict) -> str:
    device = str(rerank_cfg.get("device", cfg.get("device", "cuda"))).lower()
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _load_model_if_needed(
    run_id: str,
    cfg: dict,
    device: str,
    single_path: Path,
    multi_path: Path,
    mode: str,
    force: bool,
):
    needs_single = mode in {"final_emb", "colbert_rerank"}
    needs_multi = mode in {"colbert_rerank", "colbert"}
    if not force and (not needs_single or single_path.is_file()) and (not needs_multi or multi_path.is_file()):
        return None
    checkpoint_path = _get_checkpoint_path(run_id)
    model = build_siamese_colbert(cfg, device, checkpoint_path)
    model = model.to(device)
    model.eval()
    return model


def _product(row) -> dict:
    sku = int(row.name)
    return {
        "id": sku,
        "url": str(row.get("url") or OZON_URL_TEMPLATE.format(sku=sku)),
        "image_url": str(row.get("image_url") or row.get("thumb") or ""),
    }


def _rank_final(single_cache, catalog_skus: List, catalog_mat, query_sku, limit: int, device: str):
    q_vec = single_cache[query_sku].to(device)
    dists = torch.norm(catalog_mat - q_vec.unsqueeze(0), dim=1)
    order = torch.argsort(dists, descending=False).tolist()
    return [(catalog_skus[i], float(dists[i].item())) for i in order if catalog_skus[i] != query_sku][:limit]


def _rank_colbert(model, multi_cache, catalog_skus: List, d_name, d_desc, d_img, query_sku, cfg: dict):
    scores = _normalized_colbert_pair_scores(
        model,
        multi_cache[query_sku],
        d_name,
        d_desc,
        d_img,
        int(cfg["title_vectors"]),
        int(cfg["desc_vectors"]),
        int(cfg["image_vectors"]),
    )
    order = torch.argsort(scores, descending=True).tolist()
    return [(catalog_skus[i], float(scores[i].item())) for i in order if catalog_skus[i] != query_sku]


def _write_results(path: Path, run_id: str, mode: str, top_k: int, results: Dict[str, dict]) -> None:
    payload = {
        "_comment": f"Ozon product URL template: {OZON_URL_TEMPLATE}",
        "mlflow_run_id": run_id,
        "mode": mode,
        "top_k": top_k,
        "queries": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_mode(args, mode: str, cfg: dict, rerank_cfg: dict) -> Path:
    data_path = Path(cfg["data_path"])
    source_df = pd.read_csv(data_path / cfg["source_table"])
    source_indexed = source_df.set_index("sku")
    test_df = _load_test_df(cfg)
    grouped = test_df.groupby("sku_query", sort=False).first().reset_index()

    device = _resolve_device(cfg, rerank_cfg)
    title_vectors = int(rerank_cfg["title_vectors"])
    desc_vectors = int(rerank_cfg["desc_vectors"])
    image_vectors = int(rerank_cfg["image_vectors"])
    single_path, multi_path = cache_paths(args.mlflow_run_id, title_vectors, desc_vectors, image_vectors)
    model = _load_model_if_needed(
        args.mlflow_run_id,
        cfg,
        device,
        single_path,
        multi_path,
        mode,
        args.force_recompute,
    )

    tokenizers = None
    transform = None
    if model is not None:
        tokenizers = Tokenizers(cfg["name_model_name"], cfg["description_model_name"])
        transform = get_transform()

    catalog_skus = [sku for sku in source_df["sku"].tolist() if sku in source_indexed.index]
    single_cache = None
    catalog_mat = None
    if mode in {"final_emb", "colbert_rerank"}:
        single_cache = load_or_build_single(
            single_path,
            model,
            catalog_skus,
            source_indexed,
            str(data_path / cfg["img_dataset_name"]),
            tokenizers,
            transform,
            int(rerank_cfg.get("batch_size", 128)),
            device,
            force=args.force_recompute,
        )
        catalog_skus = [sku for sku in catalog_skus if sku in single_cache]
        catalog_mat = torch.stack([single_cache[sku] for sku in catalog_skus]).to(device)

    multi_cache = None
    d_name = d_desc = d_img = None
    if mode in {"colbert_rerank", "colbert"}:
        multi_cache = load_or_build_multi(
            multi_path,
            model,
            catalog_skus,
            source_indexed,
            str(data_path / cfg["img_dataset_name"]),
            tokenizers,
            transform,
            int(rerank_cfg.get("batch_size", 128)),
            device,
            title_vectors,
            desc_vectors,
            image_vectors,
            force=args.force_recompute,
        )
        catalog_skus = [sku for sku in catalog_skus if sku in multi_cache]
        d_name = torch.stack([multi_cache[sku]["name"] for sku in catalog_skus]).to(device)
        d_desc = torch.stack([multi_cache[sku]["desc"] for sku in catalog_skus]).to(device)
        d_img = torch.stack([multi_cache[sku]["img"] for sku in catalog_skus]).to(device)

    catalog_pos = {sku: i for i, sku in enumerate(catalog_skus)}
    scorer = model or SiameseRuCLIPColBERTWithHead
    results = {}
    pool_k = max(args.top_k, args.rerank_pool_k or max(int(k) for k in rerank_cfg.get("k", [100])))

    with torch.no_grad():
        for _, row in grouped.iterrows():
            query_sku = row["sku_query"]
            if query_sku not in source_indexed.index or query_sku not in catalog_pos:
                continue
            if mode in {"final_emb", "colbert_rerank"} and query_sku not in single_cache:
                continue
            if mode in {"colbert_rerank", "colbert"} and query_sku not in multi_cache:
                continue

            if mode == "final_emb":
                ranked = _rank_final(single_cache, catalog_skus, catalog_mat, query_sku, args.top_k, device)
            elif mode == "colbert":
                ranked = _rank_colbert(
                    scorer, multi_cache, catalog_skus, d_name, d_desc, d_img, query_sku, rerank_cfg
                )[: args.top_k]
            else:
                stage1 = _rank_final(single_cache, catalog_skus, catalog_mat, query_sku, pool_k, device)
                stage1_skus = [sku for sku, _ in stage1]
                idx_t = torch.tensor([catalog_pos[sku] for sku in stage1_skus], device=device)
                reranked = _rank_colbert(
                    scorer,
                    multi_cache,
                    stage1_skus,
                    d_name.index_select(0, idx_t),
                    d_desc.index_select(0, idx_t),
                    d_img.index_select(0, idx_t),
                    query_sku,
                    rerank_cfg,
                )
                ranked = reranked[: args.top_k]

            query = _product(source_indexed.loc[query_sku])
            candidates = []
            for sku, score in ranked:
                item = _product(source_indexed.loc[sku])
                item["score"] = score
                candidates.append(item)
            results[str(int(query_sku))] = {**query, "candidates": candidates}

    out_path = Path("logs/inference") / args.mlflow_run_id / OUTPUT_BY_MODE[mode]
    _write_results(out_path, args.mlflow_run_id, mode, args.top_k, results)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow-run-id", required=True)
    parser.add_argument("--data-run-id")
    parser.add_argument("--config", default="configs/colbert_rerank/default.yaml")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank-pool-k", type=int)
    parser.add_argument("--mode", choices=["final_emb", "colbert_rerank", "colbert", "all"], default="all")
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()

    mlflow.set_tracking_uri(_tracking_uri())
    with open(args.config, "r", encoding="utf-8") as f:
        rerank_cfg = yaml.safe_load(f)
    cfg = _load_cfg(args.mlflow_run_id, args.data_run_id)
    modes = list(OUTPUT_BY_MODE) if args.mode == "all" else [args.mode]
    for mode in modes:
        run_mode(args, mode, cfg, rerank_cfg)


if __name__ == "__main__":
    main()
