import re
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.siamese_clip import SiameseRuCLIP


def _grid_for_vectors(num_vectors: int) -> Tuple[int, int]:
    if num_vectors <= 0:
        raise ValueError("num_vectors must be > 0")
    best_h, best_w = 1, num_vectors
    best_gap = abs(best_w - best_h)
    for h in range(1, int(num_vectors**0.5) + 1):
        if num_vectors % h == 0:
            w = num_vectors // h
            gap = abs(w - h)
            if gap < best_gap:
                best_h, best_w = h, w
                best_gap = gap
    return best_h, best_w


def _compile_patterns(pats: Optional[List[str]]):
    if not pats:
        return []
    return [re.compile(p) for p in pats]


class SiameseRuCLIPColBERTWithHead(SiameseRuCLIP):
    """ColBERT multi-vector model that retains the inherited single-vector head.

    Use as a wrapper around regular Siamese checkpoints to evaluate them with
    ColBERT-style reranking on top of final-embedding retrieval. The ColBERT
    projection heads here are untrained when loaded from a regular checkpoint.
    """

    def __init__(
        self,
        device: str,
        name_model_name: str,
        description_model_name: str,
        preload_model_name: Optional[str] = None,
        models_dir: Optional[str] = None,
        dropout: Optional[float] = None,
        proj_dim: int = 128,
        use_projection_heads: bool = True,
        freeze_patterns: Optional[List[str]] = None,
        unfreeze_patterns: Optional[List[str]] = None,
        infonce_temperature: float = 0.5,
    ):
        super().__init__(
            device, name_model_name, description_model_name, preload_model_name, models_dir, dropout
        )
        device_t = torch.device(device)
        vision_dim = self.ruclip.visual.num_features
        name_in = self.ruclip.transformer.config.hidden_size
        desc_in = self.description_transformer.config.hidden_size
        self.proj_dim = proj_dim
        self.use_projection_heads = use_projection_heads
        self._freeze_pats = _compile_patterns(freeze_patterns)
        self._unfreeze_pats = _compile_patterns(unfreeze_patterns)
        if use_projection_heads:
            self.name_proj = nn.Linear(name_in, proj_dim, bias=True)
            self.desc_proj = nn.Linear(desc_in, proj_dim, bias=True)
            self.image_proj = nn.Conv2d(vision_dim, proj_dim, kernel_size=1, bias=True)
        else:
            self.name_proj = None
            self.desc_proj = None
            self.image_proj = None
        temperature = float(infonce_temperature)
        if temperature <= 0.0:
            raise ValueError("infonce_temperature must be > 0")
        self._colbert_logit_scale = nn.Parameter(
            torch.tensor(float(np.log(1.0 / temperature)), device=device_t)
        )
        self.apply_freeze_config()

    def apply_freeze_config(self) -> None:
        for _, p in self.named_parameters():
            p.requires_grad = True
        if not self._freeze_pats and not self._unfreeze_pats:
            return
        for name, p in self.named_parameters():
            frozen = any(rx.search(name) for rx in self._freeze_pats) if self._freeze_pats else False
            if self._unfreeze_pats and any(rx.search(name) for rx in self._unfreeze_pats):
                p.requires_grad = True
            elif frozen:
                p.requires_grad = False

    @property
    def colbert_logit_scale_param(self) -> nn.Parameter:
        return self._colbert_logit_scale

    @staticmethod
    def _chunk_mean_pool(
        hidden_states: Tensor, attention_mask: Tensor, num_vectors: int, normalize: bool
    ) -> Tensor:
        if num_vectors <= 0:
            raise ValueError("num_vectors must be > 0")
        bsz, seq_len, hid = hidden_states.shape
        pooled = hidden_states.new_zeros((bsz, num_vectors, hid))
        mask = attention_mask.bool()
        for b in range(bsz):
            valid_idx = torch.where(mask[b])[0]
            if valid_idx.numel() == 0:
                valid_hidden = hidden_states[b : b + 1, :1, :].squeeze(0)
            else:
                valid_hidden = hidden_states[b, valid_idx, :]
            valid_len = valid_hidden.size(0)
            edges = torch.linspace(
                0, valid_len, steps=num_vectors + 1, device=hidden_states.device
            ).long()
            for i in range(num_vectors):
                start = int(edges[i].item())
                end = int(edges[i + 1].item())
                if end <= start:
                    end = min(start + 1, valid_len)
                    start = max(0, end - 1)
                pooled[b, i, :] = valid_hidden[start:end, :].mean(dim=0)
        if normalize:
            return F.normalize(pooled, dim=-1)
        return pooled

    def encode_image_multi(self, image: Tensor, num_vectors: int) -> Tensor:
        h, w = _grid_for_vectors(num_vectors)
        feats = self.ruclip.visual.forward_features(image.type(self.ruclip.dtype))
        if self.use_projection_heads and self.image_proj is not None:
            feats = self.image_proj(feats)
        pooled = F.adaptive_avg_pool2d(feats, (h, w))
        vecs = pooled.flatten(2).transpose(1, 2).contiguous()
        return F.normalize(vecs, dim=-1)

    def encode_name_multi(self, name: Tensor, num_vectors: int) -> Tensor:
        out = self.ruclip.transformer(input_ids=name[:, 0, :], attention_mask=name[:, 1, :])
        hidden = out.last_hidden_state
        mask = name[:, 1, :]
        pooled = self._chunk_mean_pool(hidden, mask, num_vectors, normalize=False)
        if self.use_projection_heads and self.name_proj is not None:
            pooled = self.name_proj(pooled)
        else:
            pooled = self.ruclip.final_ln(pooled)
        return F.normalize(pooled, dim=-1)

    def encode_description_multi(self, desc: Tensor, num_vectors: int) -> Tensor:
        out = self.description_transformer(input_ids=desc[:, 0, :], attention_mask=desc[:, 1, :])
        hidden = out.last_hidden_state
        mask = desc[:, 1, :]
        pooled = self._chunk_mean_pool(hidden, mask, num_vectors, normalize=False)
        if self.use_projection_heads and self.desc_proj is not None:
            pooled = self.desc_proj(pooled)
        return F.normalize(pooled, dim=-1)

    @staticmethod
    def late_interaction(query_vectors: Tensor, doc_vectors: Tensor) -> Tensor:
        if query_vectors.dim() == 2:
            query_vectors = query_vectors.unsqueeze(0)
        if doc_vectors.dim() == 2:
            doc_vectors = doc_vectors.unsqueeze(0)
        bq, _, d = query_vectors.shape
        _, _, d2 = doc_vectors.shape
        if d != d2:
            raise ValueError("query and doc last dim must match")
        if bq == 1:
            sim = torch.einsum("bqd,ckd->bcqk", query_vectors, doc_vectors)
            return sim.max(dim=-1).values.sum(dim=-1).squeeze(0)
        sim = torch.einsum("iqd,jkd->ijqk", query_vectors, doc_vectors)
        return sim.max(dim=-1).values.sum(dim=-1)

    def colbert_score(
        self,
        query_name: Tensor,
        query_desc: Tensor,
        query_img: Tensor,
        doc_name: Tensor,
        doc_desc: Tensor,
        doc_img: Tensor,
    ) -> Tensor:
        return (
            self.late_interaction(query_name, doc_name)
            + self.late_interaction(query_desc, doc_desc)
            + self.late_interaction(query_img, doc_img)
        )

    def encode_multivectors(
        self, im: Tensor, name: Tensor, desc: Tensor, n_title: int, n_desc: int, n_img: int
    ) -> dict:
        return {
            "name": self.encode_name_multi(name, n_title),
            "desc": self.encode_description_multi(desc, n_desc),
            "img": self.encode_image_multi(im, n_img),
        }


class SiameseRuCLIPColBERT(SiameseRuCLIPColBERTWithHead):
    """Pure ColBERT multi-vector model. Does NOT support single-vector final embedding.

    Use for models trained with the ColBERT objective.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self, "head"):
            del self.head
        self.hidden_dim = None

    def get_final_embedding(self, *args, **kwargs):
        raise NotImplementedError(
            "SiameseRuCLIPColBERT does not support get_final_embedding; "
            "use SiameseRuCLIPColBERTWithHead for models with a trained bi-encoder head."
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "SiameseRuCLIPColBERT does not support pairwise forward; "
            "use encode_multivectors + late_interaction."
        )
