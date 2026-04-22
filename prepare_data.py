import argparse
import os
import zipfile
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download

from dataset import PairwiseDataset, split_query_groups


def _processed_root(cfg: dict) -> Path:
    return Path(cfg["data_path"]) / Path(cfg["source_table"]).parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    load_dotenv()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    token = os.environ.get("HF_TOKEN") or None
    if token:
        login(token=token)
    data_path = Path(cfg["data_path"])
    data_path.mkdir(parents=True, exist_ok=True)
    need_fetch = not (data_path / cfg["source_table"]).is_file() or not (
        data_path / cfg["pairwise_mapping_file"]
    ).is_file()
    preload_rel = Path("train_results") / cfg["preload_model_name"] if cfg.get("preload_model_name") else None
    if cfg.get("preload_model_name") and preload_rel is not None:
        need_fetch = need_fetch or not (data_path / preload_rel).is_file()
    zip_path = data_path / f"{cfg['img_dataset_name']}.zip"
    img_dir = data_path / cfg["img_dataset_name"]
    need_fetch = need_fetch or (not img_dir.is_dir() and not zip_path.is_file())
    if need_fetch:
        allow = [cfg["source_table"], cfg["pairwise_mapping_file"]]
        if cfg.get("preload_model_name"):
            allow.append(f"train_results/{cfg['preload_model_name']}")
        allow.append(f"{cfg['img_dataset_name']}.zip")
        snapshot_download(
            repo_id=cfg["repo_id"],
            repo_type="dataset",
            local_dir=str(data_path),
            allow_patterns=allow,
        )
    if zip_path.is_file() and (not img_dir.is_dir() or not any(img_dir.iterdir())):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_path)
    source_df = pd.read_csv(data_path / cfg["source_table"])
    pairwise_mapping_df = pd.read_parquet(data_path / cfg["pairwise_mapping_file"])
    proc = _processed_root(cfg)
    proc.mkdir(parents=True, exist_ok=True)
    pm = pairwise_mapping_df
    lq = cfg.get("limit_queries")
    if lq is not None:
        sampled = pm["sku_query"].drop_duplicates().sample(n=int(lq), random_state=cfg["random_seed"])
        pm = pm[pm["sku_query"].isin(sampled)]
    splits = split_query_groups(
        pm,
        test_size=cfg["test_ratio"],
        val_size=cfg["val_ratio"],
        random_state=cfg["random_seed"],
    )
    split_info = f"test={cfg['test_ratio']}_val={cfg['val_ratio']}"
    split_dir = proc / "pairwise-mapping-splits" / split_info
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.to_parquet(split_dir / f"{name}.parquet", index=False)
    images_dir = str(data_path / cfg["img_dataset_name"])
    for split in ("val", "test"):
        lim = cfg.get(f"limit_{split}_pos_pairs_per_query")
        ds = PairwiseDataset(
            splits[split],
            source_df,
            images_dir,
            max_pos_pairs_per_query=lim,
            pos_neg_ratio=cfg["pos_neg_ratio"],
            hard_soft_ratio=cfg["hard_soft_ratio"],
            random_seed=cfg["random_seed"],
            precompute=False,
        )
        pw = ds.to_pairwise_dataframe()
        params = [
            ("num-rows", len(pw)),
            ("limit-pos", lim),
            ("pos-neg", cfg["pos_neg_ratio"]),
            ("hard-soft", cfg["hard_soft_ratio"]),
            ("seed", cfg["random_seed"]),
        ]
        param_str = "_".join(f"{k}={v}" for k, v in params)
        out = proc / "pairwise-rendered" / split / param_str / "pairs.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pw.to_parquet(out, index=False)


if __name__ == "__main__":
    main()
