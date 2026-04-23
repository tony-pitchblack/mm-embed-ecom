from typing import Tuple

import torch
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


class SiameseRuCLIPColBERT(SiameseRuCLIP):
    @staticmethod
    def _chunk_mean_pool(hidden_states: Tensor, attention_mask: Tensor, num_vectors: int) -> Tensor:
        if num_vectors <= 0:
            raise ValueError("num_vectors must be > 0")
        bsz, seq_len, hid = hidden_states.shape
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        masked_hidden = hidden_states * mask
        edges = torch.linspace(0, seq_len, num_vectors + 1, device=hidden_states.device).long()
        chunks = []
        for i in range(num_vectors):
            start = int(edges[i].item())
            end = int(edges[i + 1].item())
            if end <= start:
                end = min(start + 1, seq_len)
                start = max(0, end - 1)
            chunk_hidden = masked_hidden[:, start:end, :]
            chunk_mask = mask[:, start:end, :]
            denom = chunk_mask.sum(dim=1).clamp_min(1.0)
            chunks.append(chunk_hidden.sum(dim=1) / denom)
        pooled = torch.stack(chunks, dim=1).view(bsz, num_vectors, hid)
        return F.normalize(pooled, dim=-1)

    def encode_image_multi(self, image: Tensor, num_vectors: int) -> Tensor:
        h, w = _grid_for_vectors(num_vectors)
        feats = self.ruclip.visual.forward_features(image.type(self.ruclip.dtype))
        pooled = F.adaptive_avg_pool2d(feats, (h, w))
        vecs = pooled.flatten(2).transpose(1, 2).contiguous()
        return F.normalize(vecs, dim=-1)

    def encode_name_multi(self, name: Tensor, num_vectors: int) -> Tensor:
        out = self.ruclip.transformer(input_ids=name[:, 0, :], attention_mask=name[:, 1, :])
        hidden = out.last_hidden_state
        pooled = self._chunk_mean_pool(hidden, name[:, 1, :], num_vectors)
        projected = self.ruclip.final_ln(pooled)
        return F.normalize(projected, dim=-1)

    def encode_description_multi(self, desc: Tensor, num_vectors: int) -> Tensor:
        out = self.description_transformer(input_ids=desc[:, 0, :], attention_mask=desc[:, 1, :])
        hidden = out.last_hidden_state
        return self._chunk_mean_pool(hidden, desc[:, 1, :], num_vectors)

    @staticmethod
    def late_interaction(query_vectors: Tensor, doc_vectors: Tensor) -> Tensor:
        if query_vectors.dim() == 2:
            query_vectors = query_vectors.unsqueeze(0)
        if doc_vectors.dim() == 2:
            doc_vectors = doc_vectors.unsqueeze(0)
        if query_vectors.size(0) == 1:
            sim = torch.einsum("bqd,nkd->bnqk", query_vectors, doc_vectors)
            return sim.max(dim=-1).values.sum(dim=-1).squeeze(0)
        if query_vectors.size(0) == doc_vectors.size(0):
            sim = torch.einsum("bqd,bkd->bqk", query_vectors, doc_vectors)
            return sim.max(dim=-1).values.sum(dim=-1)
        raise ValueError("query/doc batch sizes must match or query batch size must be 1")

    def colbert_score(
        self,
        query_name: Tensor,
        query_desc: Tensor,
        query_img: Tensor,
        doc_name: Tensor,
        doc_desc: Tensor,
        doc_img: Tensor,
    ) -> Tensor:
        name_score = self.late_interaction(query_name, doc_name)
        desc_score = self.late_interaction(query_desc, doc_desc)
        img_score = self.late_interaction(query_img, doc_img)
        return name_score + desc_score + img_score
