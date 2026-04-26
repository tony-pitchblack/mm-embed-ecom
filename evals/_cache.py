import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from models.siamese_clip import Tokenizers, get_transform
from models.siamese_clip_colbert import SiameseRuCLIPColBERT

CACHE_ROOT = Path("data/cache/embeddings")


def cache_paths(
    run_id: str, title_v: int, desc_v: int, image_v: int
) -> Tuple[Path, Path]:
    run_dir = CACHE_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / "single.pt", run_dir / f"multi_t{title_v}_d{desc_v}_i{image_v}.pt"


def _results_dir_for_preload(cfg: dict) -> Optional[str]:
    rp = cfg.get("data_path")
    if not rp:
        return None
    rd = cfg.get("results_dir", "train_results")
    if isinstance(rd, str):
        return str(Path(rp) / rd)
    return str(Path(rp) / "train_results")


def build_siamese_colbert(
    train_cfg: dict, device: str, checkpoint_path: str
) -> SiameseRuCLIPColBERT:
    proj_dim = int(train_cfg.get("proj_dim", 128))
    use_ph = bool(train_cfg.get("use_projection_heads", True))
    fp = train_cfg.get("freeze_patterns")
    up = train_cfg.get("unfreeze_patterns")
    if isinstance(fp, str):
        fp = None
    if isinstance(up, str):
        up = None
    m_dir = _results_dir_for_preload(train_cfg) if train_cfg.get("preload_model_name") else None
    model = SiameseRuCLIPColBERT(
        device=device,
        name_model_name=train_cfg["name_model_name"],
        description_model_name=train_cfg["description_model_name"],
        preload_model_name=train_cfg.get("preload_model_name"),
        models_dir=m_dir,
        dropout=train_cfg.get("dropout"),
        proj_dim=proj_dim,
        use_projection_heads=use_ph,
        freeze_patterns=fp,
        unfreeze_patterns=up,
    )
    mkey = str(train_cfg.get("model", "siamese_clip"))
    strict = mkey == "siamese_clip_colbert" and use_ph
    sd = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=True)
    res = model.load_state_dict(sd, strict=strict)
    if not strict and (res.missing_keys or res.unexpected_keys):
        warnings.warn(
            f"load_state_dict strict=False: "
            f"missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}",
            stacklevel=2,
        )
    return model


def _load_image(image_path: str, transform) -> torch.Tensor:
    img = cv2.imread(image_path)
    if img is None:
        return torch.zeros(3, 224, 224)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return transform(img)


def build_inputs_for_skus(
    sku_batch: List,
    source_indexed,
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


def encode_single_vectors(
    model: SiameseRuCLIPColBERT,
    sku_list: List,
    source_indexed,
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
            im, name, desc = build_inputs_for_skus(
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


def encode_multi_vectors(
    model: SiameseRuCLIPColBERT,
    sku_list: List,
    source_indexed,
    images_dir: str,
    tokenizers: Tokenizers,
    transform,
    batch_size: int,
    device: str,
    title_vectors: int,
    desc_vectors: int,
    image_vectors: int,
) -> Dict:
    cache: Dict[Any, Any] = {}
    with torch.no_grad():
        for start in tqdm(range(0, len(sku_list), batch_size), desc="encode_multi"):
            batch_skus = sku_list[start : start + batch_size]
            im, name, desc = build_inputs_for_skus(
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


def load_or_build_single(
    path: Path,
    model: SiameseRuCLIPColBERT,
    sku_list: List,
    source_indexed,
    images_dir: str,
    tokenizers: Tokenizers,
    transform,
    batch_size: int,
    device: str,
    force: bool = False,
) -> Dict:
    if force and path.exists():
        path.unlink()
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    cache = encode_single_vectors(
        model, sku_list, source_indexed, images_dir, tokenizers, transform, batch_size, device
    )
    torch.save(cache, path)
    return cache


def load_or_build_multi(
    path: Path,
    model: SiameseRuCLIPColBERT,
    sku_list: List,
    source_indexed,
    images_dir: str,
    tokenizers: Tokenizers,
    transform,
    batch_size: int,
    device: str,
    title_v: int,
    desc_v: int,
    image_v: int,
    force: bool = False,
) -> Dict:
    if force and path.exists():
        path.unlink()
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    cache = encode_multi_vectors(
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
