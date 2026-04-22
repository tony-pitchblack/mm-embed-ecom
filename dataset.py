import os
import random
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from models.siamese_clip import Tokenizers, get_transform


def _as_sku_list(x) -> List:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (tuple, set)):
        return list(x)
    return []


def split_query_groups(
    mapping_df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.05,
    random_state: int = 42,
    min_positives_for_3way: int = 6,
):
    rng = np.random.default_rng(random_state)
    split_rows = []

    for _, row in mapping_df.iterrows():
        q = row["sku_query"]
        pos_without_query = list(set(row["sku_pos"]) - {q})
        hard_neg = set(row["sku_hard_neg"]) - {q}
        soft_neg = set(row["sku_soft_neg"]) - {q}
        total_positives = len(pos_without_query)

        def split_list(lst, test_frac, val_frac=None):
            lst = np.array(list(lst))
            n = len(lst)
            if val_frac is None:
                n_test = int(np.ceil(test_frac * n))
                idx = rng.permutation(n)
                test_idx = idx[:n_test]
                train_idx = idx[n_test:]
                return lst[train_idx].tolist(), [], lst[test_idx].tolist()
            n_test = int(np.ceil(test_frac * n))
            n_val = int(np.ceil(val_frac * n))
            idx = rng.permutation(n)
            test_idx = idx[:n_test]
            val_idx = idx[n_test : n_test + n_val]
            train_idx = idx[n_test + n_val :]
            return lst[train_idx].tolist(), lst[val_idx].tolist(), lst[test_idx].tolist()

        if total_positives >= 6:
            pos_train, pos_val, pos_test = split_list(pos_without_query, test_size, val_size)
            hard_train, hard_val, hard_test = split_list(hard_neg, test_size, val_size)
            soft_train, soft_val, soft_test = split_list(soft_neg, test_size, val_size)
            pos_test.append(q)
            splits_to_create = ["train", "val", "test"]
            pos_lists = [pos_train, pos_val, pos_test]
            hard_lists = [hard_train, hard_val, hard_test]
            soft_lists = [soft_train, soft_val, soft_test]
        elif total_positives == 5:
            rng.shuffle(pos_without_query)
            pos1, pos2, pos3, pos4, pos5 = pos_without_query[:5]
            hard_train, hard_val, hard_test = split_list(hard_neg, test_size, val_size)
            soft_train, soft_val, soft_test = split_list(soft_neg, test_size, val_size)
            splits_to_create = ["train", "val", "test"]
            pos_lists = [[pos1, pos2], [pos3, pos4], [q, pos5]]
            hard_lists = [hard_train, hard_val, hard_test]
            soft_lists = [soft_train, soft_val, soft_test]
        elif total_positives == 4:
            rng.shuffle(pos_without_query)
            pos1, pos2, pos3, pos4 = pos_without_query[:4]
            hard_train, hard_val, hard_test = split_list(hard_neg, test_size, val_size)
            soft_train, soft_val, soft_test = split_list(soft_neg, test_size, val_size)
            splits_to_create = ["train", "val", "test"]
            pos_lists = [[pos1, pos2], [pos3, pos4], [q, pos3, pos4]]
            hard_lists = [hard_train, hard_val, hard_test]
            soft_lists = [soft_train, soft_val, soft_test]
        elif total_positives == 3:
            rng.shuffle(pos_without_query)
            pos1, pos2, pos3 = pos_without_query[:3]
            hard_train, hard_val, hard_test = split_list(hard_neg, test_size, val_size)
            soft_train, soft_val, soft_test = split_list(soft_neg, test_size, val_size)
            splits_to_create = ["train", "val", "test"]
            pos_lists = [[pos1, pos2], [pos2, pos3], [q, pos3]]
            hard_lists = [hard_train, hard_val, hard_test]
            soft_lists = [soft_train, soft_val, soft_test]
        elif total_positives == 2:
            rng.shuffle(pos_without_query)
            pos1, pos2 = pos_without_query[:2]
            hard_train, hard_val, hard_test = split_list(hard_neg, test_size, val_size)
            soft_train, soft_val, soft_test = split_list(soft_neg, test_size, val_size)
            splits_to_create = ["train", "val", "test"]
            pos_lists = [[pos1, pos2], [q, pos1], [q, pos2]]
            hard_lists = [hard_train, hard_val, hard_test]
            soft_lists = [soft_train, soft_val, soft_test]
        elif total_positives == 1:
            pos = pos_without_query[0]
            hard_train, hard_val, hard_test = split_list(hard_neg, test_size, val_size)
            soft_train, soft_val, soft_test = split_list(soft_neg, test_size, val_size)
            splits_to_create = ["train", "val", "test"]
            pos_lists = [[q, pos], [q, pos], [q, pos]]
            hard_lists = [hard_train, hard_val, hard_test]
            soft_lists = [soft_train, soft_val, soft_test]
        else:
            continue

        for split_name, pos_list, hard_list, soft_list in zip(
            splits_to_create, pos_lists, hard_lists, soft_lists
        ):
            split_rows.append(
                {
                    "sku_query": q,
                    "split": split_name,
                    "sku_pos": pos_list,
                    "sku_hard_neg": hard_list,
                    "sku_soft_neg": soft_list,
                }
            )

    split_df = pd.DataFrame(split_rows)
    split_dict = {
        split: split_df[split_df["split"] == split].reset_index(drop=True)
        for split in ["train", "val", "test"]
        if split in split_df["split"].values
    }
    return split_dict


class PairwiseDataset(Dataset):
    def __init__(
        self,
        split_df: pd.DataFrame,
        source_df: pd.DataFrame,
        images_dir: str,
        max_pos_pairs_per_query: Optional[int] = 3,
        pos_neg_ratio: float = 2.0,
        hard_soft_ratio: float = 0.7,
        random_seed: int = 42,
        precompute: bool = True,
        name_model_name: Optional[str] = None,
        description_model_name: Optional[str] = None,
    ):
        self.split_df = split_df.reset_index(drop=True)
        self.source_df_tabular = source_df
        self.source_df = source_df.set_index("sku") if source_df is not None else None
        self.images_dir = images_dir
        self.max_pos_pairs_per_query = max_pos_pairs_per_query
        self.pos_neg_ratio = pos_neg_ratio
        self.hard_soft_ratio = hard_soft_ratio
        self.random_seed = random_seed
        self.precompute = precompute
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.pairs = self._generate_pairs()
        self.sku_cache = None
        if precompute:
            if not name_model_name or not description_model_name:
                raise ValueError("precompute=True requires name_model_name and description_model_name")
            self.tokenizers = Tokenizers(name_model_name, description_model_name)
            self.transform = get_transform()
            self.sku_cache = self._preload_sku_data(source_df)

    def _generate_positive_pairs_fixed(self, query_sku, pos_skus: List) -> List[Dict[str, Any]]:
        if not pos_skus or (
            self.max_pos_pairs_per_query is not None and self.max_pos_pairs_per_query <= 0
        ):
            return []
        all_possible_pairs = []
        for pos1 in pos_skus:
            for pos2 in pos_skus:
                if pos1 != pos2:
                    all_possible_pairs.append((pos1, pos2))
        if not all_possible_pairs:
            if len(set(pos_skus)) == 1:
                single_sku = pos_skus[0]
                all_possible_pairs = [(single_sku, single_sku)]
            else:
                return []
        if self.max_pos_pairs_per_query is None:
            selected_pairs = all_possible_pairs
        elif len(all_possible_pairs) >= self.max_pos_pairs_per_query:
            selected_pairs = random.sample(all_possible_pairs, self.max_pos_pairs_per_query)
        else:
            selected_pairs = random.choices(all_possible_pairs, k=self.max_pos_pairs_per_query)
        pairs = []
        for pos1, pos2 in selected_pairs:
            pairs.append(
                {
                    "sku_first": pos1,
                    "sku_second": pos2,
                    "label": 0,
                    "query_sku": query_sku,
                    "pair_type": "positive",
                }
            )
        return pairs

    def _generate_hard_negative_pairs(
        self, query_sku, pos_skus: List, hard_neg_skus: List, num_hard_neg: int
    ) -> List[Dict[str, Any]]:
        if num_hard_neg <= 0 or not pos_skus or not hard_neg_skus:
            return []
        pairs = []
        for _ in range(num_hard_neg):
            pos_sku = random.choice(pos_skus)
            hard_sku = random.choice(hard_neg_skus)
            pairs.append(
                {
                    "sku_first": pos_sku,
                    "sku_second": hard_sku,
                    "label": 1,
                    "query_sku": query_sku,
                    "pair_type": "hard_negative",
                }
            )
        return pairs

    def _generate_soft_negative_pairs(
        self, query_sku, pos_skus: List, soft_neg_skus: List, num_soft_neg: int
    ) -> List[Dict[str, Any]]:
        if num_soft_neg <= 0 or not pos_skus or not soft_neg_skus:
            return []
        pairs = []
        for _ in range(num_soft_neg):
            pos_sku = random.choice(pos_skus)
            soft_sku = random.choice(soft_neg_skus)
            pairs.append(
                {
                    "sku_first": pos_sku,
                    "sku_second": soft_sku,
                    "label": 1,
                    "query_sku": query_sku,
                    "pair_type": "soft_negative",
                }
            )
        return pairs

    def _generate_pairs(self) -> List[Dict[str, Any]]:
        all_pairs = []
        for _, row in self.split_df.iterrows():
            query_sku = row["sku_query"]
            pos_skus = _as_sku_list(row["sku_pos"])
            hard_neg_skus = _as_sku_list(row["sku_hard_neg"])
            soft_neg_skus = _as_sku_list(row["sku_soft_neg"])
            positive_pairs = self._generate_positive_pairs_fixed(query_sku, pos_skus)
            actual_pos_count = len(positive_pairs)
            total_negatives_needed = int(actual_pos_count * self.pos_neg_ratio)
            num_hard_neg = int(total_negatives_needed * self.hard_soft_ratio)
            num_soft_neg = total_negatives_needed - num_hard_neg
            all_pairs.extend(positive_pairs)
            all_pairs.extend(
                self._generate_hard_negative_pairs(query_sku, pos_skus, hard_neg_skus, num_hard_neg)
            )
            all_pairs.extend(
                self._generate_soft_negative_pairs(query_sku, pos_skus, soft_neg_skus, num_soft_neg)
            )
        return all_pairs

    def _preload_sku_data(self, source_df: pd.DataFrame) -> Dict[Any, Dict[str, torch.Tensor]]:
        sku_cache: Dict[Any, Dict[str, torch.Tensor]] = {}
        unique_skus = set()
        for pair in self.pairs:
            unique_skus.add(pair["sku_first"])
            unique_skus.add(pair["sku_second"])
        source_indexed = source_df.set_index("sku")
        for sku in tqdm(unique_skus, desc="preload_skus"):
            if sku not in source_indexed.index:
                continue
            row = source_indexed.loc[sku]
            name_tokens = self.tokenizers.tokenize_name([str(row["name"])])
            desc_tokens = self.tokenizers.tokenize_description([str(row["description"])])
            img_path = os.path.join(self.images_dir, row["image_name"])
            try:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = self.transform(img)
            except Exception:
                img = torch.zeros(3, 224, 224)
            sku_cache[sku] = {
                "image": img,
                "name_tokens": name_tokens[0],
                "desc_tokens": desc_tokens[0],
            }
        return sku_cache

    def get_batch_stats(self) -> Dict[str, Any]:
        stats = {
            "total_pairs": len(self.pairs),
            "positives": sum(1 for p in self.pairs if p["pair_type"] == "positive"),
            "hard_negatives": sum(1 for p in self.pairs if p["pair_type"] == "hard_negative"),
            "soft_negatives": sum(1 for p in self.pairs if p["pair_type"] == "soft_negative"),
            "queries": len(self.split_df),
        }
        stats["total_negatives"] = stats["hard_negatives"] + stats["soft_negatives"]
        if stats["positives"] > 0:
            stats["actual_neg_pos_ratio"] = stats["total_negatives"] / stats["positives"]
        else:
            stats["actual_neg_pos_ratio"] = 0.0
        if stats["total_negatives"] > 0:
            stats["actual_hard_soft_ratio"] = stats["hard_negatives"] / stats["total_negatives"]
        else:
            stats["actual_hard_soft_ratio"] = 0.0
        stats["avg_pos_per_query"] = stats["positives"] / max(stats["queries"], 1)
        stats["avg_neg_per_query"] = stats["total_negatives"] / max(stats["queries"], 1)
        stats["avg_pairs_per_query"] = stats["total_pairs"] / max(stats["queries"], 1)
        return stats

    def print_detailed_stats(self):
        s = self.get_batch_stats()
        print(
            f"pairs={s['total_pairs']} pos={s['positives']} hard={s['hard_negatives']} "
            f"soft={s['soft_negatives']} queries={s['queries']}"
        )

    def to_pairwise_dataframe(self) -> pd.DataFrame:
        if self.source_df is None:
            raise ValueError("source_df is required to generate pairwise dataframe")
        pairwise_rows = []
        pair_type_mapping = {
            "positive": "pos",
            "hard_negative": "hard_neg",
            "soft_negative": "soft_neg",
        }
        pair_type_to_label = {"positive": 0, "hard_negative": 1, "soft_negative": 1}
        for pair in tqdm(self.pairs, desc="to_pairwise_dataframe"):
            sku_first = pair["sku_first"]
            sku_second = pair["sku_second"]
            query_sku = pair["query_sku"]
            pair_type = pair["pair_type"]
            mapped_pair_type = pair_type_mapping.get(pair_type, pair_type)
            label = pair_type_to_label.get(pair_type, 1)
            if sku_first in self.source_df.index and sku_second in self.source_df.index:
                row_first = self.source_df.loc[sku_first]
                row_second = self.source_df.loc[sku_second]
                pairwise_row: Dict[str, Any] = {}
                for col in self.source_df.columns:
                    pairwise_row[f"{col}_first"] = row_first[col]
                for col in self.source_df.columns:
                    pairwise_row[f"{col}_second"] = row_second[col]
                pairwise_row["pair_type"] = mapped_pair_type
                pairwise_row["sku_query"] = query_sku
                pairwise_row["label"] = 1 - label
                pairwise_row["sku_first"] = sku_first
                pairwise_row["sku_second"] = sku_second
                pairwise_rows.append(pairwise_row)
        return pd.DataFrame(pairwise_rows)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        if self.precompute:
            assert self.sku_cache is not None
            sku_first = pair["sku_first"]
            sku_second = pair["sku_second"]
            data_first = self.sku_cache[sku_first]
            data_second = self.sku_cache[sku_second]
            return {
                "image_first": data_first["image"],
                "name_first": data_first["name_tokens"],
                "desc_first": data_first["desc_tokens"],
                "image_second": data_second["image"],
                "name_second": data_second["name_tokens"],
                "desc_second": data_second["desc_tokens"],
                "label": torch.tensor(pair["label"], dtype=torch.float32),
                "pair_type": pair["pair_type"],
            }
        return {
            "sku_first": torch.tensor(pair["sku_first"]),
            "sku_second": torch.tensor(pair["sku_second"]),
            "label": torch.tensor(pair["label"], dtype=torch.float32),
            "query_sku": pair["query_sku"],
            "pair_type": pair["pair_type"],
        }
