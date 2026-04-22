import argparse
import os
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from dotenv import load_dotenv
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PairwiseDataset
from models.siamese_clip import ContrastiveLoss, SiameseRuCLIP, Tokenizers, get_transform


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
    metric,
    source_df,
    images_dir,
    precompute_pairs,
    mlflow_active,
    name_model_name,
    description_model_name,
):
    assert metric in ("f1", "pos_acc")
    assert split_name in ("val", "test")
    model.eval()
    if len(data_loader) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, threshold or 0.0
    total_loss = 0.0
    all_d, all_lbl = [], []
    if not precompute_pairs:
        tokenizers = Tokenizers(name_model_name, description_model_name)
        transform = get_transform()
    else:
        tokenizers = None
        transform = None
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"eval_{split_name}"):
            if precompute_pairs:
                im1 = batch["image_first"].to(device)
                n1 = batch["name_first"].to(device)
                d1 = batch["desc_first"].to(device)
                im2 = batch["image_second"].to(device)
                n2 = batch["name_second"].to(device)
                d2 = batch["desc_second"].to(device)
                labels = batch["label"].to(device)
            else:
                sku_first = batch["sku_first"]
                sku_second = batch["sku_second"]
                labels = batch["label"].to(device)
                im1, n1, d1 = sku_to_model_inputs(
                    sku_first.tolist(), source_df, images_dir, tokenizers, transform
                )
                im2, n2, d2 = sku_to_model_inputs(
                    sku_second.tolist(), source_df, images_dir, tokenizers, transform
                )
                im1, n1, d1 = im1.to(device), n1.to(device), d1.to(device)
                im2, n2, d2 = im2.to(device), n2.to(device), d2.to(device)
            out1, out2 = model(im1, n1, d1, im2, n2, d2)
            total_loss += criterion(out1, out2, labels).item()
            all_d.append(F.pairwise_distance(out1, out2).cpu())
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
            if metric == "f1":
                val = f1_score(y_true, y_pred, zero_division=0)
            else:
                pos_mask = y_true == 1
                val = (y_pred[pos_mask] == 1).mean() if pos_mask.sum() > 0 else 0.0
            if val > best_val:
                best_val, best_thr = val, t
        threshold = best_thr
    else:
        best_thr = threshold
    preds = (distances < threshold).long()
    pos_mask = labels == 0
    neg_mask = labels == 1
    pos_acc = preds[pos_mask].float().mean().item() if pos_mask.any() else 0.0
    neg_acc = (preds[neg_mask] == 0).float().mean().item() if neg_mask.any() else 0.0
    avg_acc = (pos_acc + neg_acc) / 2.0
    f1 = f1_score((labels.numpy() == 0).astype(int), preds.numpy(), zero_division=0)
    if mlflow_active and split_name == "val":
        step = int(epoch) if isinstance(epoch, int) else 0
        if metric == "f1":
            mlflow.log_metric("valid_f1_score", f1, step=step)
        else:
            mlflow.log_metric("valid_pos_accuracy", pos_acc, step=step)
    return pos_acc, neg_acc, avg_acc, f1, avg_loss, best_thr


def train_with_threshold_tracking(
    model,
    optimizer,
    criterion,
    epochs_num,
    train_loader,
    valid_loader,
    device,
    models_dir,
    metric,
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
):
    model.to(device)
    train_losses, val_losses = [], []
    best_valid_metric, best_threshold = float("-inf"), None
    best_weights = None
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
    if models_dir:
        Path(models_dir).mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs_num + 1):
        model.train()
        total_train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"train_ep{epoch}"):
            if precompute_pairs:
                im1 = batch["image_first"].to(device)
                n1 = batch["name_first"].to(device)
                d1 = batch["desc_first"].to(device)
                im2 = batch["image_second"].to(device)
                n2 = batch["name_second"].to(device)
                d2 = batch["desc_second"].to(device)
                labels = batch["label"].to(device)
            else:
                sku_first = batch["sku_first"]
                sku_second = batch["sku_second"]
                labels = batch["label"].to(device)
                im1, n1, d1 = sku_to_model_inputs(
                    sku_first.tolist(), source_df, images_dir, tokenizers, transform
                )
                im2, n2, d2 = sku_to_model_inputs(
                    sku_second.tolist(), source_df, images_dir, tokenizers, transform
                )
                im1, n1, d1 = im1.to(device), n1.to(device), d1.to(device)
                im2, n2, d2 = im2.to(device), n2.to(device), d2.to(device)
            optimizer.zero_grad()
            out1, out2 = model(im1, n1, d1, im2, n2, d2)
            loss = criterion(out1, out2, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        train_losses.append(total_train_loss / max(len(train_loader), 1))
        if valid_loader is not None:
            pos_acc, neg_acc, avg_acc, f1_val, val_loss, val_thr = evaluation(
                model,
                criterion,
                valid_loader,
                epoch,
                device=device,
                split_name="val",
                threshold=None,
                margin=contrastive_margin,
                steps=200,
                metric=metric,
                source_df=source_df,
                images_dir=images_dir,
                precompute_pairs=precompute_pairs,
                mlflow_active=mlflow_active,
                name_model_name=name_model_name,
                description_model_name=description_model_name,
            )
            val_losses.append(val_loss)
            cur_metric = pos_acc if metric == "pos_acc" else f1_val
            scheduler.step(cur_metric)
            if models_dir:
                checkpoint_filename = (
                    f"siamese_contrastive_soft-neg_val_ep{epoch}_f1-{f1_val:.3f}_pacc-{pos_acc:.3f}_nacc-{neg_acc:.3f}_"
                    f"pos{train_stats['positives']}_hard{train_stats['hard_negatives']}_soft{train_stats['soft_negatives']}_"
                    f"ppq{ckpt_ppq}_pnr{ckpt_pnr}_hsr{ckpt_hsr}_thr{val_thr:.3f}.pt"
                )
                sd = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                torch.save(sd, Path(models_dir) / checkpoint_filename)
            if cur_metric > best_valid_metric:
                best_valid_metric = cur_metric
                best_threshold = val_thr
                best_weights = deepcopy(
                    model.module.state_dict()
                    if isinstance(model, torch.nn.DataParallel)
                    else model.state_dict()
                )
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tracking = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking:
        host = os.environ.get("MLFLOW_HOST", "127.0.0.1")
        port = os.environ.get("MLFLOW_PORT", "5000")
        tracking = f"http://{host}:{port}"
    mlflow_active = bool(os.environ.get("MLFLOW_TRACKING_URI", "").strip())
    if mlflow_active:
        mlflow.set_tracking_uri(tracking)
        mlflow.set_experiment(cfg["mlflow_experiment"])
    proc = _processed_root(cfg)
    split_info = f"test={cfg['test_ratio']}_val={cfg['val_ratio']}"
    split_dir = proc / "pairwise-mapping-splits" / split_info
    for s in ("train", "val", "test"):
        if not (split_dir / f"{s}.parquet").is_file():
            raise FileNotFoundError(f"Missing {split_dir / f'{s}.parquet'}; run prepare_data.py first")
    splits = {n: pd.read_parquet(split_dir / f"{n}.parquet") for n in ("train", "val", "test")}
    data_path = Path(cfg["data_path"])
    source_df = pd.read_csv(data_path / cfg["source_table"])
    images_dir = str(data_path / cfg["img_dataset_name"])
    results_root = data_path / cfg["results_dir"]
    models_dir_preload = str(results_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = get_optimal_num_workers() if device == "cuda" else 0
    pin = device == "cuda"
    persistent = num_workers > 0
    limits = {
        "train": cfg["limit_train_pos_pairs_per_query"],
        "val": cfg.get("limit_val_pos_pairs_per_query"),
        "test": cfg.get("limit_test_pos_pairs_per_query"),
    }
    loaders = {}
    train_stats = None
    for split_name in ("train", "val", "test"):
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
        )
        st = ds.get_batch_stats()
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
    num_gpus = torch.cuda.device_count()
    model = SiameseRuCLIP(
        device,
        cfg["name_model_name"],
        cfg["description_model_name"],
        cfg.get("preload_model_name"),
        models_dir_preload,
        cfg.get("dropout"),
    )
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    criterion = ContrastiveLoss(margin=cfg["contrastive_margin"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    tmp_ckpt = results_root / "tmp"
    tmp_ckpt.mkdir(parents=True, exist_ok=True)
    run_ctx = mlflow.start_run() if mlflow_active else None
    try:
        if mlflow_active:
            mlflow.log_params({k: str(v) for k, v in cfg.items() if isinstance(v, (int, float, str))})
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
                cfg["best_ckpt_metric"],
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
        test_pos_acc, test_neg_acc, test_acc, test_f1, test_loss, _ = evaluation(
            model,
            criterion,
            loaders["test"],
            0,
            device=device,
            split_name="test",
            threshold=best_threshold,
            margin=cfg["contrastive_margin"],
            steps=200,
            metric=cfg["best_ckpt_metric"],
            source_df=source_df,
            images_dir=images_dir,
            precompute_pairs=True,
            mlflow_active=mlflow_active,
            name_model_name=cfg["name_model_name"],
            description_model_name=cfg["description_model_name"],
        )
        if mlflow_active:
            mlflow.log_metric("test_pos_accuracy", test_pos_acc)
            mlflow.log_metric("test_neg_accuracy", test_neg_acc)
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.log_metric("test_f1_score", test_f1)
        filename = (
            f"siamese_contrastive_soft-neg_test_ep{cfg['epochs']}_f1-{test_f1:.3f}_pacc-{test_pos_acc:.3f}_nacc-{test_neg_acc:.3f}_"
            f"pos{train_stats['positives']}_hard{train_stats['hard_negatives']}_soft{train_stats['soft_negatives']}_"
            f"ppq{cfg['limit_train_pos_pairs_per_query']}_pnr{cfg['pos_neg_ratio']}_hsr{cfg['hard_soft_ratio']}_"
            f"thr{best_threshold:.3f}.pt"
        )
        final_path = results_root / filename
        final_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_weights, final_path)
    finally:
        if run_ctx is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
