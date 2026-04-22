import os
from typing import Optional

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer


def _convert_image_to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def get_transform():
    return transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class RuCLIPtiny(nn.Module):
    def __init__(self, name_model_name: str):
        super().__init__()
        self.visual = timm.create_model(
            "convnext_tiny", pretrained=False, num_classes=0, in_chans=3
        )
        self.transformer = AutoModel.from_pretrained(name_model_name)
        name_model_output_shape = self.transformer.config.hidden_size
        self.final_ln = torch.nn.Linear(name_model_output_shape, 768)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @property
    def dtype(self):
        return self.visual.stem[0].weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, input_ids, attention_mask):
        x = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        x = x.last_hidden_state[:, 0, :]
        x = self.final_ln(x)
        return x

    def forward(self, image, input_ids, attention_mask):
        image_features = self.encode_image(image)
        text_features = self.encode_text(input_ids, attention_mask)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class Tokenizers:
    def __init__(self, name_model_name: str, description_model_name: str):
        self.name_tokenizer = AutoTokenizer.from_pretrained(name_model_name)
        self.desc_tokenizer = AutoTokenizer.from_pretrained(description_model_name)

    def tokenize_name(self, texts, max_len: int = 77):
        tokenized = self.name_tokenizer(
            texts,
            truncation=True,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        return torch.stack([tokenized["input_ids"], tokenized["attention_mask"]], dim=1)

    def tokenize_description(self, texts, max_len: int = 77):
        tokenized = self.desc_tokenizer(
            texts,
            truncation=True,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        return torch.stack([tokenized["input_ids"], tokenized["attention_mask"]], dim=1)


class SiameseRuCLIP(nn.Module):
    def __init__(
        self,
        device: str,
        name_model_name: str,
        description_model_name: str,
        preload_model_name: Optional[str] = None,
        models_dir: Optional[str] = None,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        device_t = torch.device(device)
        self.ruclip = RuCLIPtiny(name_model_name)
        if preload_model_name is not None and models_dir is not None:
            std = torch.load(
                os.path.join(models_dir, preload_model_name),
                weights_only=True,
                map_location=device_t,
            )
            self.ruclip.load_state_dict(std)
            self.ruclip.eval()
        self.ruclip = self.ruclip.to(device_t)
        self.description_transformer = AutoModel.from_pretrained(description_model_name)
        self.description_transformer = self.description_transformer.to(device_t)
        vision_dim = self.ruclip.visual.num_features
        name_dim = self.ruclip.final_ln.out_features
        desc_dim = self.description_transformer.config.hidden_size
        self.hidden_dim = vision_dim + name_dim + desc_dim
        self.dropout = dropout
        layers = [
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            *([nn.Dropout(self.dropout)] if self.dropout is not None else []),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
        ]
        self.head = nn.Sequential(*layers).to(device_t)

    def encode_image(self, image):
        return self.ruclip.encode_image(image)

    def encode_name(self, name):
        return self.ruclip.encode_text(name[:, 0, :], name[:, 1, :])

    def encode_description(self, desc):
        last_hidden_states = self.description_transformer(
            input_ids=desc[:, 0, :], attention_mask=desc[:, 1, :]
        ).last_hidden_state
        attention_mask = desc[:, 1, :]
        return average_pool(last_hidden_states, attention_mask)

    def get_final_embedding(self, im, name, desc):
        image_emb = self.encode_image(im)
        name_emb = self.encode_name(name)
        desc_emb = self.encode_description(desc)
        combined_emb = torch.cat([image_emb, name_emb, desc_emb], dim=1)
        return self.head(combined_emb)

    def forward(self, im1, name1, desc1, im2, name2, desc2):
        b = im1.size(0)
        im = torch.cat([im1, im2], dim=0)
        name = torch.cat([name1, name2], dim=0)
        desc = torch.cat([desc1, desc2], dim=0)
        out = self.get_final_embedding(im, name, desc)
        return out[:b], out[b:]


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        pos = (1 - label) * torch.pow(euclidean_distance, 2)
        neg = label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return torch.mean(pos + neg)
