import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from dotenv import load_dotenv
from sklearn.metrics import f1_score, fbeta_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PairwiseDataset
from logging_config import configure_logging
from models.siamese_clip import ContrastiveLoss, SiameseRuCLIP, Tokenizers, get_transform
from models.siamese_clip_colbert import SiameseRuCLIPColBERT

logger = logging.getLogger(__name__)


def sku_to_model_inputs(sku_list, source_df, images_dir, tokenizers, transform):
    import cv2
    from PIL import Image

    sku_data = source_df.loc[source_df["sku"].isin(sku_list)].set_index("sku")
    images = []
    names = []
    descriptions = []
    for sku in sku_list:
        if sku in sku_data.index:
            row = sku_data.loc[sku]
            img_path = os.path.join(images_dir, row["image_name"])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = transform(img)
            images.append(img)
            names.append(str(row["name"]))
            descriptions.append(str(row["description"]))
        else:
            images.append(torch.zeros(3, 224, 224))
            names.append("Unknown Product")
            descriptions.append("No description available")
    images = torch.stack(images)
    name_tokens = tokenizers.tokenize_name(names)
    desc_tokens = tokenizers.tokenize_description(descriptions)
    return images, name_tokens, desc_tokens


def evaluation(
    model,
    criterion,
    data_loader,
    epoch,
    device,
    split_name,
    threshold,
    margin,
    steps,
    source_df,
    images_dir,
    precompute_pairs,
    mlflow_active,
    name_model_name,
    description_model_name,
):
    assert split_name in ("val", "test")
    model.eval()
    if len(data_loader) == 0:
        empty_rank = {
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "recall_at_100": 0.0,
            "mrr_at_5": 0.0,
            "mrr_at_10": 0.0,
            "mrr_at_100": 0.0,
            "ndcg_at_5": 0.0,
            "ndcg_at_10": 0.0,
            "ndcg_at_100": 0.0,
        }
        return {
            "precision": 0.0,
            "npv": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
            "balanced_accuracy": 0.0,
            "f1_score": 0.0,
            "loss": 0.0,
            "threshold": threshold or 0.0,
            **empty_rank,
        }
    total_loss = 0.0
    all_d, all_lbl = [], []
    all_query = []
    if not precompute_pairs:
        tokenizers = Tokenizers(name_model_name, description_model_name)
        transform = get_transform()
    else:
        tokenizers = None
        transform = None
    dataset_pairs = getattr(data_loader.dataset, "pairs", None)
    pair_offset = 0
    use_amp = device == "cuda"
    amp_dtype = (
        torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    )
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"eval_{split_name}"):
            if precompute_pairs:
                im1 = batch["image_first"].to(device, non_blocking=True)
                n1 = batch["name_first"].to(device, non_blocking=True)
                d1 = batch["desc_first"].to(device, non_blocking=True)
                im2 = batch["image_second"].to(device, non_blocking=True)
                n2 = batch["name_second"].to(device, non_blocking=True)
                d2 = batch["desc_second"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                batch_size = int(labels.shape[0])
                if dataset_pairs is not None:
                    batch_query = [
                        dataset_pairs[i]["query_sku"] for i in range(pair_offset, pair_offset + batch_size)
                    ]
                    all_query.extend(batch_query)
                pair_offset += batch_size
            else:
                sku_first = batch["sku_first"]
                sku_second = batch["sku_second"]
                labels = batch["label"].to(device, non_blocking=True)
                all_query.extend([int(x) for x in batch["query_sku"]])
                im1, n1, d1 = sku_to_model_inputs(
                    sku_first.tolist(), source_df, images_dir, tokenizers, transform
                )
                im2, n2, d2 = sku_to_model_inputs(
                    sku_second.tolist(), source_df, images_dir, tokenizers, transform
                )
                im1 = im1.to(device, non_blocking=True)
                n1 = n1.to(device, non_blocking=True)
                d1 = d1.to(device, non_blocking=True)
                im2 = im2.to(device, non_blocking=True)
                n2 = n2.to(device, non_blocking=True)
                d2 = d2.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out1, out2 = model(im1, n1, d1, im2, n2, d2)
                loss = criterion(out1, out2, labels)
            total_loss += loss.item()
            all_d.append(F.pairwise_distance(out1.float(), out2.float()).cpu())
            all_lbl.append(labels.cpu())
    distances = torch.cat(all_d)
    labels = torch.cat(all_lbl)
    avg_loss = total_loss / len(data_loader)
    if threshold is None:
        grid = np.linspace(0.0, margin, steps)
        best_val, best_thr = -1.0, 0.0
        y_true = (labels.numpy() == 0).astype(int)
        for t in grid:
            y_pred = (distances.numpy() < t).astype(int)
            val = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)
            if val > best_val:
                best_val, best_thr = val, t
        threshold = best_thr
    else:
        best_thr = threshold
    preds = (distances < threshold).long()
    pos_mask = labels == 0
    neg_mask = labels == 1
    recall = preds[pos_mask].float().mean().item() if pos_mask.any() else 0.0
    specificity = (preds[neg_mask] == 0).float().mean().item() if neg_mask.any() else 0.0
    balanced_accuracy = (recall + specificity) / 2.0
    y_bin = (labels.numpy() == 0).astype(int)
    pred_bin = preds.numpy()
    f1_score_val = f1_score(y_bin, pred_bin, zero_division=0)
    tp = ((preds == 1) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 1)).sum().item()
    fn = ((preds == 0) & (labels == 0)).sum().item()
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    rank_metrics = {
        "recall_at_5": 0.0,
        "recall_at_10": 0.0,
        "recall_at_100": 0.0,
        "mrr_at_5": 0.0,
        "mrr_at_10": 0.0,
        "mrr_at_100": 0.0,
        "ndcg_at_5": 0.0,
        "ndcg_at_10": 0.0,
        "ndcg_at_100": 0.0,
    }
    if all_query:
        dist_np = distances.numpy()
        lbl_np = labels.numpy()
        query_np = np.array(all_query)
        ks = (5, 10, 100)
        recalls_by_k = {k: [] for k in ks}
        mrr_by_k = {k: [] for k in ks}
        ndcg_by_k = {k: [] for k in ks}
        for query_id in np.unique(query_np):
            mask = query_np == query_id
            query_dist = dist_np[mask]
            query_lbl = lbl_np[mask]
            total_relevant = int((query_lbl == 0).sum())
            if total_relevant <= 0:
                continue
            order = np.argsort(query_dist)
            ranked_lbl = query_lbl[order]
            relevance = (ranked_lbl == 0).astype(np.int32)
            positive_idx = np.where(relevance == 1)[0]
            for k in ks:
                topk_rel = relevance[:k]
                hits_k = int(topk_rel.sum())
                recalls_by_k[k].append(float(hits_k / total_relevant))
                rr_k = 0.0
                if positive_idx.size > 0:
                    first_rank = int(positive_idx[0]) + 1
                    if first_rank <= k:
                        rr_k = 1.0 / float(first_rank)
                mrr_by_k[k].append(rr_k)
                if hits_k == 0:
                    ndcg_by_k[k].append(0.0)
                else:
                    gains = topk_rel / np.log2(np.arange(2, len(topk_rel) + 2))
                    dcg_k = float(gains.sum())
                    ideal_len = min(total_relevant, k)
                    ideal_rel = np.ones(ideal_len, dtype=np.float32)
                    idcg_k = float((ideal_rel / np.log2(np.arange(2, ideal_len + 2))).sum())
                    ndcg_by_k[k].append(float(dcg_k / idcg_k) if idcg_k > 0 else 0.0)
        for k in ks:
            if recalls_by_k[k]:
                rank_metrics[f"recall_at_{k}"] = float(np.mean(recalls_by_k[k]))
            if mrr_by_k[k]:
                rank_metrics[f"mrr_at_{k}"] = float(np.mean(mrr_by_k[k]))
            if ndcg_by_k[k]:
                rank_metrics[f"ndcg_at_{k}"] = float(np.mean(ndcg_by_k[k]))
    if mlflow_active:
        prefix = f"{split_name}/"
        step = int(epoch) if isinstance(epoch, int) else 0
        kw = dict(step=step)
        mlflow.log_metric(f"{prefix}threshold", float(best_thr), **kw)
        mlflow.log_metric(f"{prefix}loss", avg_loss, **kw)
        mlflow.log_metric(f"{prefix}precision", precision, **kw)
        mlflow.log_metric(f"{prefix}npv", npv, **kw)
        mlflow.log_metric(f"{prefix}recall", recall, **kw)
        mlflow.log_metric(f"{prefix}specificity", specificity, **kw)
        mlflow.log_metric(f"{prefix}balanced_accuracy", balanced_accuracy, **kw)
        mlflow.log_metric(f"{prefix}f1_score", f1_score_val, **kw)
        mlflow.log_metric(f"{prefix}recall_at_5", rank_metrics["recall_at_5"], **kw)
        mlflow.log_metric(f"{prefix}recall_at_10", rank_metrics["recall_at_10"], **kw)
        mlflow.log_metric(f"{prefix}recall_at_100", rank_metrics["recall_at_100"], **kw)
        mlflow.log_metric(f"{prefix}mrr_at_5", rank_metrics["mrr_at_5"], **kw)
        mlflow.log_metric(f"{prefix}mrr_at_10", rank_metrics["mrr_at_10"], **kw)
        mlflow.log_metric(f"{prefix}mrr_at_100", rank_metrics["mrr_at_100"], **kw)
        mlflow.log_metric(f"{prefix}ndcg_at_5", rank_metrics["ndcg_at_5"], **kw)
        mlflow.log_metric(f"{prefix}ndcg_at_10", rank_metrics["ndcg_at_10"], **kw)
        mlflow.log_metric(f"{prefix}ndcg_at_100", rank_metrics["ndcg_at_100"], **kw)
    return {
        "precision": precision,
        "npv": npv,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "f1_score": f1_score_val,
        "loss": avg_loss,
        "threshold": best_thr,
        **rank_metrics,
    }


def _unwrap_model(m):
    return m.module if isinstance(m, torch.nn.DataParallel) else m


def colbert_pair_score_and_distance(
    model: SiameseRuCLIPColBERT,
    im1: torch.Tensor,
    n1: torch.Tensor,
    d1: torch.Tensor,
    im2: torch.Tensor,
    n2: torch.Tensor,
    d2: torch.Tensor,
    n_title: int,
    n_desc: int,
    n_img: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    m = _unwrap_model(model)
    a = m.encode_multivectors(im1, n1, d1, n_title, n_desc, n_img)
    k = m.encode_multivectors(im2, n2, d2, n_title, n_desc, n_img)
    name_score = m.late_interaction(a["name"], k["name"]) / float(max(n_title, 1))
    desc_score = m.late_interaction(a["desc"], k["desc"]) / float(max(n_desc, 1))
    img_score = m.late_interaction(a["img"], k["img"]) / float(max(n_img, 1))
    score = (name_score + desc_score + img_score) / 3.0
    distance = 1.0 - score
    return score, distance


def colbert_contrastive_loss(
    model: SiameseRuCLIPColBERT,
    im1: torch.Tensor,
    n1: torch.Tensor,
    d1: torch.Tensor,
    im2: torch.Tensor,
    n2: torch.Tensor,
    d2: torch.Tensor,
    label: torch.Tensor,
    n_title: int,
    n_desc: int,
    n_img: int,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    score, distance = colbert_pair_score_and_distance(
        model, im1, n1, d1, im2, n2, d2, n_title, n_desc, n_img
    )
    labels = label.view(-1).float()
    pos = (1.0 - labels) * torch.pow(distance, 2)
    neg = labels * torch.pow(torch.clamp(margin - distance, min=0.0), 2)
    loss = torch.mean(pos + neg)
    return loss, score, distance


def evaluation_colbert(
    model: SiameseRuCLIPColBERT,
    data_loader,
    epoch,
    device: str,
    split_name: str,
    n_title: int,
    n_desc: int,
    n_img: int,
    margin: float,
    mlflow_active: bool,
) -> dict:
    assert split_name in ("val", "test")
    model.eval()
    m = _unwrap_model(model)
    if len(data_loader) == 0:
        return {
            "loss": 0.0,
            "threshold": 0.0,
            "precision": 0.0,
            "npv": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
            "balanced_accuracy": 0.0,
            "f1_score": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "recall_at_100": 0.0,
            "mrr_at_5": 0.0,
            "mrr_at_10": 0.0,
            "mrr_at_100": 0.0,
            "ndcg_at_5": 0.0,
            "ndcg_at_10": 0.0,
            "ndcg_at_100": 0.0,
        }
    total_loss = 0.0
    all_scores, all_lbl, all_dist = [], [], []
    all_query = []
    use_amp = device == "cuda"
    amp_dtype = (
        torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    )
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"eval_{split_name}_colbert"):
            im1 = batch["image_first"].to(device, non_blocking=True)
            n1 = batch["name_first"].to(device, non_blocking=True)
            d1 = batch["desc_first"].to(device, non_blocking=True)
            im2 = batch["image_second"].to(device, non_blocking=True)
            n2 = batch["name_second"].to(device, non_blocking=True)
            d2 = batch["desc_second"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            q = batch["query_sku"]
            if isinstance(q, torch.Tensor):
                all_query.extend(q.detach().cpu().tolist())
            else:
                all_query.extend(list(q))
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                loss, pair_score, pair_dist = colbert_contrastive_loss(
                    model,
                    im1,
                    n1,
                    d1,
                    im2,
                    n2,
                    d2,
                    labels,
                    n_title,
                    n_desc,
                    n_img,
                    margin,
                )
            total_loss += loss.item()
            all_scores.append(pair_score.detach().float().cpu())
            all_dist.append(pair_dist.detach().float().cpu())
            all_lbl.append(labels.detach().cpu())
    avg_loss = total_loss / max(len(data_loader), 1)
    distances = torch.cat(all_dist)
    labels = torch.cat(all_lbl)
    grid = np.linspace(0.0, margin, 200)
    best_val, best_thr = -1.0, 0.0
    y_true = (labels.numpy() == 0).astype(int)
    for t in grid:
        y_pred = (distances.numpy() < t).astype(int)
        val = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)
        if val > best_val:
            best_val, best_thr = val, t
    preds = (distances < best_thr).long()
    pos_mask = labels == 0
    neg_mask = labels == 1
    recall = preds[pos_mask].float().mean().item() if pos_mask.any() else 0.0
    specificity = (preds[neg_mask] == 0).float().mean().item() if neg_mask.any() else 0.0
    balanced_accuracy = (recall + specificity) / 2.0
    y_bin = (labels.numpy() == 0).astype(int)
    pred_bin = preds.numpy()
    f1_score_val = f1_score(y_bin, pred_bin, zero_division=0)
    tp = ((preds == 1) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 1)).sum().item()
    fn = ((preds == 0) & (labels == 0)).sum().item()
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    rank_metrics = {
        "recall_at_5": 0.0,
        "recall_at_10": 0.0,
        "recall_at_100": 0.0,
        "mrr_at_5": 0.0,
        "mrr_at_10": 0.0,
        "mrr_at_100": 0.0,
        "ndcg_at_5": 0.0,
        "ndcg_at_10": 0.0,
        "ndcg_at_100": 0.0,
    }
    if all_query:
        score_np = torch.cat(all_scores).numpy()
        lbl_np = torch.cat(all_lbl).numpy()
        query_np = np.array(all_query)
        ks = (5, 10, 100)
        recalls_by_k = {k: [] for k in ks}
        mrr_by_k = {k: [] for k in ks}
        ndcg_by_k = {k: [] for k in ks}
        for query_id in np.unique(query_np):
            mask = query_np == query_id
            query_scores = score_np[mask]
            query_lbl = lbl_np[mask]
            total_relevant = int((query_lbl == 0).sum())
            if total_relevant <= 0:
                continue
            order = np.argsort(-query_scores)
            ranked_lbl = query_lbl[order]
            relevance = (ranked_lbl == 0).astype(np.int32)
            positive_idx = np.where(relevance == 1)[0]
            for k in ks:
                topk_rel = relevance[:k]
                hits_k = int(topk_rel.sum())
                recalls_by_k[k].append(float(hits_k / total_relevant))
                rr_k = 0.0
                if positive_idx.size > 0:
                    first_rank = int(positive_idx[0]) + 1
                    if first_rank <= k:
                        rr_k = 1.0 / float(first_rank)
                mrr_by_k[k].append(rr_k)
                if hits_k == 0:
                    ndcg_by_k[k].append(0.0)
                else:
                    gains = topk_rel / np.log2(np.arange(2, len(topk_rel) + 2))
                    dcg_k = float(gains.sum())
                    ideal_len = min(total_relevant, k)
                    ideal_rel = np.ones(ideal_len, dtype=np.float32)
                    idcg_k = float((ideal_rel / np.log2(np.arange(2, ideal_len + 2))).sum())
                    ndcg_by_k[k].append(float(dcg_k / idcg_k) if idcg_k > 0 else 0.0)
        for k in ks:
            if recalls_by_k[k]:
                rank_metrics[f"recall_at_{k}"] = float(np.mean(recalls_by_k[k]))
            if mrr_by_k[k]:
                rank_metrics[f"mrr_at_{k}"] = float(np.mean(mrr_by_k[k]))
            if ndcg_by_k[k]:
                rank_metrics[f"ndcg_at_{k}"] = float(np.mean(ndcg_by_k[k]))
    if mlflow_active:
        prefix = f"{split_name}/"
        step = int(epoch) if isinstance(epoch, int) else 0
        kw = dict(step=step)
        mlflow.log_metric(f"{prefix}loss", avg_loss, **kw)
        mlflow.log_metric(f"{prefix}threshold", float(best_thr), **kw)
        mlflow.log_metric(f"{prefix}precision", precision, **kw)
        mlflow.log_metric(f"{prefix}npv", npv, **kw)
        mlflow.log_metric(f"{prefix}recall", recall, **kw)
        mlflow.log_metric(f"{prefix}specificity", specificity, **kw)
        mlflow.log_metric(f"{prefix}balanced_accuracy", balanced_accuracy, **kw)
        mlflow.log_metric(f"{prefix}f1_score", f1_score_val, **kw)
        mlflow.log_metric(f"{prefix}recall_at_5", rank_metrics["recall_at_5"], **kw)
        mlflow.log_metric(f"{prefix}recall_at_10", rank_metrics["recall_at_10"], **kw)
        mlflow.log_metric(f"{prefix}recall_at_100", rank_metrics["recall_at_100"], **kw)
        mlflow.log_metric(f"{prefix}mrr_at_5", rank_metrics["mrr_at_5"], **kw)
        mlflow.log_metric(f"{prefix}mrr_at_10", rank_metrics["mrr_at_10"], **kw)
        mlflow.log_metric(f"{prefix}mrr_at_100", rank_metrics["mrr_at_100"], **kw)
        mlflow.log_metric(f"{prefix}ndcg_at_5", rank_metrics["ndcg_at_5"], **kw)
        mlflow.log_metric(f"{prefix}ndcg_at_10", rank_metrics["ndcg_at_10"], **kw)
        mlflow.log_metric(f"{prefix}ndcg_at_100", rank_metrics["ndcg_at_100"], **kw)
    return {
        "loss": avg_loss,
        "threshold": float(best_thr),
        "precision": precision,
        "npv": npv,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "f1_score": f1_score_val,
        **rank_metrics,
    }


def train_with_threshold_tracking(
    model,
    optimizer,
    criterion,
    epochs_num,
    train_loader,
    valid_loader,
    device,
    models_dir,
    source_df,
    images_dir,
    precompute_pairs,
    train_stats,
    scheduler_patience,
    contrastive_margin,
    ckpt_ppq,
    ckpt_pnr,
    ckpt_hsr,
    mlflow_active,
    name_model_name,
    description_model_name,
    best_ckpt_metric,
    early_stop_patience,
    contrastive_threshold,
):
    model.to(device)
    train_losses, val_losses = [], []
    best_valid_metric, best_threshold = float("-inf"), None
    best_weights = None
    best_val_loss = float("inf")
    epochs_since_loss_improved = 0
    if not precompute_pairs:
        tokenizers = Tokenizers(name_model_name, description_model_name)
        transform = get_transform()
    else:
        tokenizers = None
        transform = None
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.1,
        patience=scheduler_patience,
        threshold=1e-4,
        threshold_mode="rel",
    )
    use_amp = device == "cuda"
    amp_dtype = (
        torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    probe_steps = int(os.environ.get("PROBE_STEPS", "0") or "0")
    probe_step_count = 0
    if models_dir:
        Path(models_dir).mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs_num + 1):
        model.train()
        total_train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"train_ep{epoch}"):
            if precompute_pairs:
                im1 = batch["image_first"].to(device, non_blocking=True)
                n1 = batch["name_first"].to(device, non_blocking=True)
                d1 = batch["desc_first"].to(device, non_blocking=True)
                im2 = batch["image_second"].to(device, non_blocking=True)
                n2 = batch["name_second"].to(device, non_blocking=True)
                d2 = batch["desc_second"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
            else:
                sku_first = batch["sku_first"]
                sku_second = batch["sku_second"]
                labels = batch["label"].to(device, non_blocking=True)
                im1, n1, d1 = sku_to_model_inputs(
                    sku_first.tolist(), source_df, images_dir, tokenizers, transform
                )
                im2, n2, d2 = sku_to_model_inputs(
                    sku_second.tolist(), source_df, images_dir, tokenizers, transform
                )
                im1 = im1.to(device, non_blocking=True)
                n1 = n1.to(device, non_blocking=True)
                d1 = d1.to(device, non_blocking=True)
                im2 = im2.to(device, non_blocking=True)
                n2 = n2.to(device, non_blocking=True)
                d2 = d2.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out1, out2 = model(im1, n1, d1, im2, n2, d2)
                loss = criterion(out1, out2, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()
            if probe_steps > 0:
                probe_step_count += 1
                if probe_step_count >= probe_steps:
                    if device == "cuda":
                        torch.cuda.synchronize()
                        peak_mib = torch.cuda.max_memory_allocated() // (1024 * 1024)
                        reserved_mib = torch.cuda.max_memory_reserved() // (1024 * 1024)
                    else:
                        peak_mib = 0
                        reserved_mib = 0
                    print(
                        f"PROBE_RESULT peak_mib={peak_mib} reserved_mib={reserved_mib} "
                        f"bs={im1.size(0)}",
                        flush=True,
                    )
                    sys.exit(0)
        train_losses.append(total_train_loss / max(len(train_loader), 1))
        if valid_loader is not None:
            val_metrics = evaluation(
                model,
                criterion,
                valid_loader,
                epoch,
                device=device,
                split_name="val",
                threshold=contrastive_threshold,
                margin=contrastive_margin,
                steps=200,
                source_df=source_df,
                images_dir=images_dir,
                precompute_pairs=precompute_pairs,
                mlflow_active=mlflow_active,
                name_model_name=name_model_name,
                description_model_name=description_model_name,
            )
            val_losses.append(val_metrics["loss"])
            if best_ckpt_metric not in val_metrics:
                raise KeyError(
                    f"best_ckpt_metric={best_ckpt_metric!r} not in val metrics: "
                    f"{sorted(val_metrics)}"
                )
            cur_metric = val_metrics[best_ckpt_metric]
            scheduler.step(cur_metric)
            if cur_metric > best_valid_metric:
                best_valid_metric = cur_metric
                best_threshold = val_metrics["threshold"]
                sd = (
                    model.module.state_dict()
                    if isinstance(model, torch.nn.DataParallel)
                    else model.state_dict()
                )
                best_weights = {k: v.detach().cpu().clone() for k, v in sd.items()}
                if models_dir:
                    metric_tag = best_ckpt_metric.replace("/", "-")
                    checkpoint_filename = (
                        f"siamese_contrastive_soft-neg_val_ep{epoch}_{metric_tag}-{cur_metric:.3f}_"
                        f"f1-{val_metrics['f1_score']:.3f}_precision-{val_metrics['precision']:.3f}_"
                        f"recall-{val_metrics['recall']:.3f}_specificity-{val_metrics['specificity']:.3f}_"
                        f"pos{train_stats['positives']}_hard{train_stats['hard_negatives']}_soft{train_stats['soft_negatives']}_"
                        f"ppq{ckpt_ppq}_pnr{ckpt_pnr}_hsr{ckpt_hsr}_thr{val_metrics['threshold']:.3f}.pt"
                    )
                    torch.save(best_weights, Path(models_dir) / checkpoint_filename)
            if val_metrics["loss"] < best_val_loss - 1e-4:
                best_val_loss = val_metrics["loss"]
                epochs_since_loss_improved = 0
            else:
                epochs_since_loss_improved += 1
                if early_stop_patience is not None and epochs_since_loss_improved >= early_stop_patience:
                    logger.info(
                        "early stop: val/loss did not improve for %d epochs (best=%.4f)",
                        early_stop_patience,
                        best_val_loss,
                    )
                    break
    return train_losses, val_losses, best_valid_metric, best_weights, best_threshold


def train_with_colbert_contrastive(
    model,
    optimizer,
    epochs_num,
    train_loader,
    valid_loader,
    device,
    models_dir,
    train_stats,
    scheduler_patience,
    ckpt_ppq,
    ckpt_pnr,
    ckpt_hsr,
    mlflow_active,
    best_ckpt_metric,
    early_stop_patience,
    n_title,
    n_desc,
    n_img,
    contrastive_margin,
):
    model.to(device)
    train_losses, val_losses = [], []
    maximize = best_ckpt_metric in (
        "precision",
        "npv",
        "recall",
        "specificity",
        "balanced_accuracy",
        "f1_score",
        "recall_at_5",
        "recall_at_10",
        "recall_at_100",
        "mrr_at_5",
        "mrr_at_10",
        "mrr_at_100",
        "ndcg_at_5",
        "ndcg_at_10",
        "ndcg_at_100",
    )
    best_valid_metric = float("-inf") if maximize else float("inf")
    best_threshold = None
    best_weights = None
    best_val_loss = float("inf")
    epochs_since_loss_improved = 0
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max" if maximize else "min",
        factor=0.1,
        patience=scheduler_patience,
        threshold=1e-4,
        threshold_mode="rel",
    )
    use_amp = device == "cuda"
    amp_dtype = (
        torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    probe_steps = int(os.environ.get("PROBE_STEPS", "0") or "0")
    probe_step_count = 0
    if models_dir:
        Path(models_dir).mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs_num + 1):
        model.train()
        total_train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"train_ep{epoch}_colbert"):
            im1 = batch["image_first"].to(device, non_blocking=True)
            n1 = batch["name_first"].to(device, non_blocking=True)
            d1 = batch["desc_first"].to(device, non_blocking=True)
            im2 = batch["image_second"].to(device, non_blocking=True)
            n2 = batch["name_second"].to(device, non_blocking=True)
            d2 = batch["desc_second"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                loss, _, _ = colbert_contrastive_loss(
                    model,
                    im1,
                    n1,
                    d1,
                    im2,
                    n2,
                    d2,
                    labels,
                    n_title,
                    n_desc,
                    n_img,
                    contrastive_margin,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()
            if probe_steps > 0:
                probe_step_count += 1
                if probe_step_count >= probe_steps:
                    if device == "cuda":
                        torch.cuda.synchronize()
                        peak_mib = torch.cuda.max_memory_allocated() // (1024 * 1024)
                        reserved_mib = torch.cuda.max_memory_reserved() // (1024 * 1024)
                    else:
                        peak_mib = 0
                        reserved_mib = 0
                    print(
                        f"PROBE_RESULT peak_mib={peak_mib} reserved_mib={reserved_mib} "
                        f"bs={im1.size(0)}",
                        flush=True,
                    )
                    sys.exit(0)
        train_losses.append(total_train_loss / max(len(train_loader), 1))
        if valid_loader is not None:
            val_metrics = evaluation_colbert(
                model,
                valid_loader,
                epoch,
                device=device,
                split_name="val",
                n_title=n_title,
                n_desc=n_desc,
                n_img=n_img,
                margin=contrastive_margin,
                mlflow_active=mlflow_active,
            )
            val_losses.append(val_metrics["loss"])
            if best_ckpt_metric not in val_metrics:
                raise KeyError(
                    f"best_ckpt_metric={best_ckpt_metric!r} not in val metrics: "
                    f"{sorted(val_metrics)}"
                )
            cur_metric = val_metrics[best_ckpt_metric]
            scheduler.step(cur_metric)
            improved = cur_metric < best_valid_metric if not maximize else cur_metric > best_valid_metric
            if improved:
                best_valid_metric = cur_metric
                best_threshold = val_metrics.get("threshold")
                sd = (
                    model.module.state_dict()
                    if isinstance(model, torch.nn.DataParallel)
                    else model.state_dict()
                )
                best_weights = {k: v.detach().cpu().clone() for k, v in sd.items()}
                if models_dir:
                    metric_tag = best_ckpt_metric.replace("/", "-")
                    checkpoint_filename = (
                        f"siamese_colbert_contrastive_val_ep{epoch}_{metric_tag}-{cur_metric:.3f}_"
                        f"f1-{val_metrics['f1_score']:.3f}_precision-{val_metrics['precision']:.3f}_"
                        f"recall-{val_metrics['recall']:.3f}_specificity-{val_metrics['specificity']:.3f}_"
                        f"pos{train_stats['positives']}_hard{train_stats['hard_negatives']}_soft{train_stats['soft_negatives']}_"
                        f"ppq{ckpt_ppq}_pnr{ckpt_pnr}_hsr{ckpt_hsr}_thr{val_metrics['threshold']:.3f}.pt"
                    )
                    torch.save(best_weights, Path(models_dir) / checkpoint_filename)
            if val_metrics["loss"] < best_val_loss - 1e-4:
                best_val_loss = val_metrics["loss"]
                epochs_since_loss_improved = 0
            else:
                epochs_since_loss_improved += 1
                if early_stop_patience is not None and epochs_since_loss_improved >= early_stop_patience:
                    logger.info(
                        "early stop: val/loss did not improve for %d epochs (best=%.4f)",
                        early_stop_patience,
                        best_val_loss,
                    )
                    break
    return train_losses, val_losses, best_valid_metric, best_weights, best_threshold


def get_optimal_num_workers():
    import psutil

    num_gpus = torch.cuda.device_count()
    num_cpus = psutil.cpu_count(logical=False) or 2
    if num_gpus > 1:
        workers = min(4, max(num_cpus // num_gpus, 1))
    else:
        workers = min(8, num_cpus)
    return max(2, workers)


def _processed_root(cfg: dict) -> Path:
    return Path(cfg["data_path"]) / Path(cfg["source_table"]).parent


def main():
    load_dotenv()
    configure_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    logger.debug("start, config=%s", os.path.abspath(args.config))
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    logger.debug("yaml loaded")
    tracking = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking:
        host = os.environ.get("MLFLOW_HOST", "127.0.0.1")
        port = os.environ.get("MLFLOW_PORT", "5000")
        tracking = f"http://{host}:{port}"
    mlflow_active = bool(os.environ.get("MLFLOW_TRACKING_URI", "").strip())
    if mlflow_active:
        mlflow.set_tracking_uri(tracking)
        mlflow.set_experiment(cfg["mlflow_experiment"])
        logger.debug("mlflow on, experiment=%s", cfg["mlflow_experiment"])
    else:
        logger.debug("mlflow off (set MLFLOW_TRACKING_URI to enable)")
    proc = _processed_root(cfg)
    split_info = f"test={cfg['test_ratio']}_val={cfg['val_ratio']}"
    split_dir = proc / "pairwise-mapping-splits" / split_info
    logger.debug("split_dir=%s", split_dir)
    for s in ("train", "val", "test"):
        if not (split_dir / f"{s}.parquet").is_file():
            raise FileNotFoundError(f"Missing {split_dir / f'{s}.parquet'}; run prepare_data.py first")
    splits = {}
    for n in ("train", "val", "test"):
        p = split_dir / f"{n}.parquet"
        logger.debug("loading parquet %s … %s", n, p)
        splits[n] = pd.read_parquet(p)
        logger.debug("loaded %s: %d rows", n, len(splits[n]))
    data_path = Path(cfg["data_path"])
    src_csv = data_path / cfg["source_table"]
    logger.debug("loading source csv … %s", src_csv)
    source_df = pd.read_csv(src_csv)
    logger.debug("loaded source: %d rows, %d columns", len(source_df), source_df.shape[1])
    images_dir = str(data_path / cfg["img_dataset_name"])
    results_root = data_path / cfg["results_dir"]
    models_dir_preload = str(results_root)
    device = str(cfg.get("device", "cuda")).lower()
    if device != "cuda":
        raise ValueError("Only CUDA is supported. Set device: cuda in config.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available on this machine.")
    logger.debug(
        "cuda: %d device(s), %s",
        torch.cuda.device_count(),
        torch.cuda.get_device_name(0),
    )
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    num_workers = get_optimal_num_workers()
    pin = True
    logger.debug("DataLoader num_workers=%d, pin_memory=%s", num_workers, pin)
    persistent = num_workers > 0
    limits = {
        "train": cfg["limit_train_pos_pairs_per_query"],
        "val": cfg.get("limit_val_pos_pairs_per_query"),
        "test": cfg.get("limit_test_pos_pairs_per_query"),
    }
    loaders = {}
    train_stats = None
    for split_name in ("train", "val", "test"):
        logger.debug("PairwiseDataset + DataLoader: split=%r …", split_name)
        ds = PairwiseDataset(
            splits[split_name],
            source_df,
            images_dir,
            max_pos_pairs_per_query=limits[split_name],
            pos_neg_ratio=cfg["pos_neg_ratio"],
            hard_soft_ratio=cfg["hard_soft_ratio"],
            random_seed=cfg["random_seed"],
            precompute=True,
            name_model_name=cfg["name_model_name"],
            description_model_name=cfg["description_model_name"],
            pair_gen_split_label=split_name,
        )
        st = ds.get_batch_stats()
        logger.debug(
            "dataset %s ready: %d pairs, queries=%d, batch_size=%d",
            split_name,
            st["total_pairs"],
            st["queries"],
            cfg["batch_size_per_device"],
        )
        if split_name == "train":
            train_stats = st
        dl_kw = dict(
            batch_size=cfg["batch_size_per_device"],
            shuffle=split_name == "train",
            num_workers=num_workers,
            pin_memory=pin,
            drop_last=False,
        )
        if persistent:
            dl_kw["persistent_workers"] = True
            dl_kw["prefetch_factor"] = 4
        loaders[split_name] = DataLoader(ds, **dl_kw)
        logger.debug("DataLoader %s: %d batches/epoch", split_name, len(loaders[split_name]))
    num_gpus = torch.cuda.device_count()
    mkey = str(cfg.get("model", "siamese_clip"))
    if mkey == "siamese_clip_colbert":
        temp_cfg = cfg.get("infonce_temperature", "auto")
        if isinstance(temp_cfg, str) and temp_cfg.lower() == "auto":
            infonce_temperature = 0.5
        else:
            infonce_temperature = float(temp_cfg)
        logger.debug("building SiameseRuCLIPColBERT (download/load weights if needed) …")
        model = SiameseRuCLIPColBERT(
            device,
            cfg["name_model_name"],
            cfg["description_model_name"],
            cfg.get("preload_model_name"),
            models_dir_preload if cfg.get("preload_model_name") else None,
            cfg.get("dropout"),
            int(cfg.get("proj_dim", 128)),
            bool(cfg.get("use_projection_heads", True)),
            cfg.get("freeze_patterns"),
            cfg.get("unfreeze_patterns"),
            infonce_temperature,
        )
        n_title = int(cfg["title_vectors"])
        n_desc = int(cfg["desc_vectors"])
        n_img = int(cfg["image_vectors"])
        m0 = model
        n_train = sum(p.numel() for p in m0.parameters() if p.requires_grad)
        logger.info("ColBERT trainable parameters: %d", n_train)
    else:
        logger.debug("building SiameseRuCLIP (download/load weights if needed) …")
        model = SiameseRuCLIP(
            device,
            cfg["name_model_name"],
            cfg["description_model_name"],
            cfg.get("preload_model_name"),
            models_dir_preload,
            cfg.get("dropout"),
        )
        n_title, n_desc, n_img = 0, 0, 0
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    if mkey == "siamese_clip_colbert":
        core = model.module if isinstance(model, torch.nn.DataParallel) else model
        trainable = [p for p in core.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        criterion = None
    else:
        criterion = ContrastiveLoss(margin=cfg["contrastive_margin"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    logger.debug("model built, type=%s", mkey)
    tmp_ckpt = results_root / "tmp"
    tmp_ckpt.mkdir(parents=True, exist_ok=True)
    run_ctx = mlflow.start_run() if mlflow_active else None
    try:
        if mlflow_active:
            mlflow.log_params({k: str(v) for k, v in cfg.items() if isinstance(v, (int, float, str))})
        if mkey == "siamese_clip_colbert":
            best_ckpt_metric = cfg.get("best_ckpt_metric", "ndcg_at_10")
            early_stop_patience = cfg.get("early_stop_patience")
            thr_cfg = "auto"
            contrastive_threshold = None
        else:
            best_ckpt_metric = cfg.get("best_ckpt_metric", "ndcg_at_10")
            early_stop_patience = cfg.get("early_stop_patience")
            thr_cfg = cfg.get("contrastive_threshold", "auto")
            if isinstance(thr_cfg, str) and thr_cfg.lower() == "auto":
                contrastive_threshold = None
            else:
                contrastive_threshold = float(thr_cfg)
        logger.debug(
            "entering training: model=%s, epochs=%d, best_ckpt_metric=%s, early_stop_patience=%s, contrastive_threshold=%s",
            mkey,
            cfg["epochs"],
            best_ckpt_metric,
            early_stop_patience,
            "auto" if contrastive_threshold is None else contrastive_threshold,
        )
        if mkey == "siamese_clip_colbert":
            train_losses, val_losses, best_metric_val, best_weights, best_threshold = (
                train_with_colbert_contrastive(
                    model,
                    optimizer,
                    cfg["epochs"],
                    loaders["train"],
                    loaders["val"],
                    device,
                    str(tmp_ckpt),
                    train_stats,
                    cfg["scheduler_patience"],
                    cfg["limit_train_pos_pairs_per_query"],
                    cfg["pos_neg_ratio"],
                    cfg["hard_soft_ratio"],
                    mlflow_active,
                    best_ckpt_metric,
                    early_stop_patience,
                    n_title,
                    n_desc,
                    n_img,
                    cfg["contrastive_margin"],
                )
            )
        else:
            train_losses, val_losses, best_metric_val, best_weights, best_threshold = (
                train_with_threshold_tracking(
                    model,
                    optimizer,
                    criterion,
                    cfg["epochs"],
                    loaders["train"],
                    loaders["val"],
                    device,
                    str(tmp_ckpt),
                    source_df,
                    images_dir,
                    True,
                    train_stats,
                    cfg["scheduler_patience"],
                    cfg["contrastive_margin"],
                    cfg["limit_train_pos_pairs_per_query"],
                    cfg["pos_neg_ratio"],
                    cfg["hard_soft_ratio"],
                    mlflow_active,
                    cfg["name_model_name"],
                    cfg["description_model_name"],
                    best_ckpt_metric,
                    early_stop_patience,
                    contrastive_threshold,
                )
            )
        if best_weights is None:
            raise RuntimeError("No validation improvement; best_weights is None")
        if len(train_losses) >= 2 and mlflow_active:
            fig, ax = plt.subplots()
            ax.plot(range(2, len(train_losses) + 1), train_losses[1:], label="train")
            ax.plot(range(2, len(val_losses) + 1), val_losses[1:], label="val")
            ax.legend()
            mlflow.log_figure(fig, "loss_by_epoch.png")
            plt.close(fig)
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(best_weights)
        else:
            model.load_state_dict(best_weights)
        final_eval_step = max(1, len(train_losses))
        if mkey == "siamese_clip_colbert":
            test_metrics = evaluation_colbert(
                model,
                loaders["test"],
                final_eval_step,
                device=device,
                split_name="test",
                n_title=n_title,
                n_desc=n_desc,
                n_img=n_img,
                margin=cfg["contrastive_margin"],
                mlflow_active=mlflow_active,
            )
            metric_tag = best_ckpt_metric.replace("/", "-")
            filename = (
                f"siamese_colbert_contrastive_test_ep{cfg['epochs']}_{metric_tag}-{test_metrics[best_ckpt_metric]:.3f}_"
                f"f1-{test_metrics['f1_score']:.3f}_precision-{test_metrics['precision']:.3f}_"
                f"recall-{test_metrics['recall']:.3f}_specificity-{test_metrics['specificity']:.3f}_"
                f"pos{train_stats['positives']}_hard{train_stats['hard_negatives']}_soft{train_stats['soft_negatives']}_"
                f"ppq{cfg['limit_train_pos_pairs_per_query']}_pnr{cfg['pos_neg_ratio']}_hsr{cfg['hard_soft_ratio']}_"
                f"thr{test_metrics['threshold']:.3f}.pt"
            )
        else:
            test_metrics = evaluation(
                model,
                criterion,
                loaders["test"],
                final_eval_step,
                device=device,
                split_name="test",
                threshold=best_threshold,
                margin=cfg["contrastive_margin"],
                steps=200,
                source_df=source_df,
                images_dir=images_dir,
                precompute_pairs=True,
                mlflow_active=mlflow_active,
                name_model_name=cfg["name_model_name"],
                description_model_name=cfg["description_model_name"],
            )
            metric_tag = best_ckpt_metric.replace("/", "-")
            filename = (
                f"siamese_contrastive_soft-neg_test_ep{cfg['epochs']}_{metric_tag}-{test_metrics[best_ckpt_metric]:.3f}_"
                f"f1-{test_metrics['f1_score']:.3f}_precision-{test_metrics['precision']:.3f}_"
                f"recall-{test_metrics['recall']:.3f}_specificity-{test_metrics['specificity']:.3f}_"
                f"pos{train_stats['positives']}_hard{train_stats['hard_negatives']}_soft{train_stats['soft_negatives']}_"
                f"ppq{cfg['limit_train_pos_pairs_per_query']}_pnr{cfg['pos_neg_ratio']}_hsr{cfg['hard_soft_ratio']}_"
                f"thr{best_threshold:.3f}.pt"
            )
        final_path = results_root / filename
        final_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_weights, final_path)
        if mlflow_active:
            mlflow.log_artifact(str(final_path), artifact_path="model")
            mlflow.set_tag("checkpoint_filename", final_path.name)
    finally:
        if run_ctx is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
