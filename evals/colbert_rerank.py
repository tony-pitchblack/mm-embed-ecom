import argparse
import ast
import os
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import cv2
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from mlflow.tracking import MlflowClient
from PIL import Image
from tqdm import tqdm

from models.siamese_clip import Tokenizers, get_transform
from models.siamese_clip_colbert import SiameseRuCLIPColBERT


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


def _load_image(image_path: str, transform):
    img = cv2.imread(image_path)
    if img is None:
        return torch.zeros(3, 224, 224)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return transform(img)


def _build_inputs_for_skus(
    sku_batch: List,
    source_indexed: pd.DataFrame,
    images_dir: str,
    tokenizers: Tokenizers,
    transform,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = []
    names = []
    descriptions = []
    for sku in sku_batch:
        row = source_indexed.loc[sku]
        images.append(_load_image(os.path.join(images_dir, row["image_name"]), transform))
        names.append(str(row["name"]))
        descriptions.append(str(row["description"]))
    return (
        torch.stack(images),
        tokenizers.tokenize_name(names),
        tokenizers.tokenize_description(descriptions),
    )


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


def _get_checkpoint_path(run_id: str) -> str:
    model_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model")
    model_path = Path(model_dir)
    candidates = sorted(model_path.rglob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint .pt found under artifact path: {model_path}")
    return str(candidates[-1])


def _encode_single_vectors(
    model: SiameseRuCLIPColBERT,
    sku_list: List,
    source_indexed: pd.DataFrame,
    images_dir: str,
    tokenizers: Tokenizers,
    transform,
    batch_size: int,
    device: str,
) -> Dict:
    embeddings = {}
    with torch.no_grad():
        for start in tqdm(range(0, len(sku_list), batch_size), desc="encode_single"):
            batch_skus = sku_list[start : start + batch_size]
            im, name, desc = _build_inputs_for_skus(
                batch_skus, source_indexed, images_dir, tokenizers, transform
            )
            im = im.to(device, non_blocking=True)
            name = name.to(device, non_blocking=True)
            desc = desc.to(device, non_blocking=True)
            emb = model.get_final_embedding(im, name, desc)
            emb = F.normalize(emb, dim=-1).cpu()
            for sku, vec in zip(batch_skus, emb):
                embeddings[sku] = vec
    return embeddings


def _encode_multi_vectors(
    model: SiameseRuCLIPColBERT,
    sku_list: List,
    source_indexed: pd.DataFrame,
    images_dir: str,
    tokenizers: Tokenizers,
    transform,
    batch_size: int,
    device: str,
    title_vectors: int,
    desc_vectors: int,
    image_vectors: int,
):
    cache = {}
    with torch.no_grad():
        for start in tqdm(range(0, len(sku_list), batch_size), desc="encode_multi"):
            batch_skus = sku_list[start : start + batch_size]
            im, name, desc = _build_inputs_for_skus(
                batch_skus, source_indexed, images_dir, tokenizers, transform
            )
            im = im.to(device, non_blocking=True)
            name = name.to(device, non_blocking=True)
            desc = desc.to(device, non_blocking=True)
            img_m = model.encode_image_multi(im, image_vectors).cpu()
            name_m = model.encode_name_multi(name, title_vectors).cpu()
            desc_m = model.encode_description_multi(desc, desc_vectors).cpu()
            for i, sku in enumerate(batch_skus):
                cache[sku] = {
                    "img": img_m[i],
                    "name": name_m[i],
                    "desc": desc_m[i],
                }
    return cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow-run-id", required=True)
    parser.add_argument("--config", default="configs/colbert_rerank/default.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        rerank_cfg = yaml.safe_load(f)

    train_cfg = _load_cfg_from_run(args.mlflow_run_id)
    if "data_path" not in train_cfg:
        raise KeyError("Could not reconstruct train config from run params: missing data_path")

    tracking = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking:
        host = os.environ.get("MLFLOW_HOST", "127.0.0.1")
        port = os.environ.get("MLFLOW_PORT", "5000")
        tracking = f"http://{host}:{port}"
    mlflow.set_tracking_uri(tracking)

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

    model = SiameseRuCLIPColBERT(
        device=device,
        name_model_name=train_cfg["name_model_name"],
        description_model_name=train_cfg["description_model_name"],
        preload_model_name=None,
        models_dir=None,
        dropout=train_cfg.get("dropout"),
    )
    state_dict = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    ks = sorted(int(k) for k in rerank_cfg["k"])
    k_max = max(ks)
    batch_size = int(rerank_cfg.get("batch_size", 128))
    title_vectors = int(rerank_cfg["title_vectors"])
    desc_vectors = int(rerank_cfg["desc_vectors"])
    image_vectors = int(rerank_cfg["image_vectors"])

    catalog_skus = [sku for sku in source_df["sku"].tolist() if sku in source_indexed.index]
    single_cache = _encode_single_vectors(
        model,
        catalog_skus,
        source_indexed,
        images_dir,
        tokenizers,
        transform,
        batch_size,
        device,
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
    multi_cache = _encode_multi_vectors(
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
            scores = model.colbert_score(
                q["name"].to(device),
                q["desc"].to(device),
                q["img"].to(device),
                d_name,
                d_desc,
                d_img,
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
