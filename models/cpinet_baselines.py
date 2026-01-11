# models/cpinet_baselines.py
from __future__ import annotations
from typing import Dict, Tuple, List

import torch
import torch.nn as nn

from .sardensenet_v1 import SARDenseNetV1
from .cpin_blocks import GroupBilinearPooling


class _Head(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.cls = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls(self.embed(x))


class BackboneNet(nn.Module):
    """
    Single-pol backbone baseline used for Table 7.
    """
    branch_names = ["main"]

    def __init__(self, num_classes: int, stem_ch: int = 24, embed_dim: int = 1024, dropout: float = 0.2):
        super().__init__()
        self.backbone = SARDenseNetV1(stem_ch=stem_ch)
        ch_db5 = self.backbone.out_channels["db5"]
        self.head = _Head(in_dim=ch_db5, embed_dim=embed_dim, num_classes=num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        feats = self.backbone(x)
        db5 = feats["db5"]
        pooled = torch.mean(db5, dim=(2, 3))
        logits = self.head(pooled)
        return logits, {"main": logits}


class GBCNN(nn.Module):
    """
    Dual-pol GBCNN-style baseline using group bilinear pooling on DB5 features.
    """
    branch_names = ["main"]

    def __init__(
        self,
        num_classes: int,
        stem_ch: int = 24,
        gbp_groups: int = 3,
        embed_dim: int = 1024,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.bb_vh = SARDenseNetV1(stem_ch=stem_ch)
        self.bb_vv = SARDenseNetV1(stem_ch=stem_ch)
        ch_db5 = self.bb_vh.out_channels["db5"]
        assert ch_db5 % gbp_groups == 0, "db5 channels must be divisible by gbp_groups"
        d = ch_db5 // gbp_groups
        gbp_dim = (gbp_groups * (gbp_groups + 1) // 2) * d * d
        self.gbp = GroupBilinearPooling(groups=gbp_groups)
        self.head = _Head(in_dim=gbp_dim, embed_dim=embed_dim, num_classes=num_classes, dropout=dropout)

    def forward(self, vh: torch.Tensor, vv: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        f_vh = self.bb_vh(vh)["db5"]
        f_vv = self.bb_vv(vv)["db5"]
        pooled = self.gbp(f_vh, f_vv)
        logits = self.head(pooled)
        return logits, {"main": logits}
