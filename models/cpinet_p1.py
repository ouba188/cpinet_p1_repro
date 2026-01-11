
# models/cpinet_p1.py
from __future__ import annotations
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sardensenet_v1 import SARDenseNetV1
from .cpin_blocks import IMDFF, MOSE, FBCPooling, GroupBilinearPooling, SEBlock, CBAM


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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.cls(self.embed(z))


class CPINetP1(nn.Module):
    """
    CPINet P1 reproduction:
      - SAR-DenseNet-v1 backbone (VH/VV)
      - IMDFF (Eq.1-2) -> DB3', DB4', DB5', FMS (DB34 fusion)
      - MO-SE augmentation on DB5'
      - FBC pooling for:
          DB34 cross, MO-SE cross, DB5-VH self, DB5-VV self
      - 4 heads -> 4 logits; inference assembling: mean of logits (configurable)
    """
    branch_names = ["db34", "mose", "db5_vh", "db5_vv"]

    def __init__(
        self,
        num_classes: int,
        stem_ch: int = 24,
        fuse_ch: int = 128,
        fbc_k: int = 2048,
        fbc_lam: float = 1e-3,
        embed_dim: int = 1024,
        gbp_groups: int = 3,
        dropout: float = 0.0,
        assemble: str = "logit_mean",   # logit_mean | prob_mean
        fusion: str = "imdff",          # imdff | msdff
        attention: str = "mose",        # mose | se | cbam | none
        pooling: str = "fbc",           # fbc | gbp
        share_head: bool = True,
        db34_source: str = "fms",       # fms | db4p
    ):
        super().__init__()
        self.num_classes = num_classes
        self.assemble = assemble
        self.fusion = fusion
        self.attention = attention
        self.pooling = pooling
        self.share_head = share_head
        self.db34_source = db34_source

        self.bb_vh = SARDenseNetV1(stem_ch=stem_ch)
        self.bb_vv = SARDenseNetV1(stem_ch=stem_ch)

        use_fms = fusion == "imdff"
        use_db5_fusion = fusion == "imdff"
        self.imdff_vh = IMDFF(self.bb_vh.out_channels, fuse_ch=fuse_ch, use_fms=use_fms, use_db5_fusion=use_db5_fusion)
        self.imdff_vv = IMDFF(self.bb_vv.out_channels, fuse_ch=fuse_ch, use_fms=use_fms, use_db5_fusion=use_db5_fusion)

        ch_db5 = self.bb_vh.out_channels["db5"]
        if attention == "mose":
            self.att_block = MOSE(ch=ch_db5, gbp_groups=gbp_groups)
        elif attention == "se":
            self.att_block = SEBlock(ch=ch_db5)
        elif attention == "cbam":
            self.att_block = CBAM(ch=ch_db5)
        else:
            self.att_block = None

        self.pool_db34, dim_db34 = self._build_pool(in_ch=fuse_ch, k=fbc_k, lam=fbc_lam, groups=gbp_groups)
        self.pool_mose, dim_mose = self._build_pool(in_ch=ch_db5, k=fbc_k, lam=fbc_lam, groups=gbp_groups)
        self.pool_db5, dim_db5 = self._build_pool(in_ch=ch_db5, k=fbc_k, lam=fbc_lam, groups=gbp_groups)

        # Heads
        if share_head:
            if not (dim_db34 == dim_mose == dim_db5):
                raise ValueError("share_head=True requires identical pooled dimensions across branches.")
            self.shared_head = _Head(in_dim=dim_db5, embed_dim=embed_dim, num_classes=num_classes, dropout=dropout)
            self.head_db34 = self.shared_head
            self.head_mose = self.shared_head
            self.head_db5_vh = self.shared_head
            self.head_db5_vv = self.shared_head
        else:
            self.head_db34 = _Head(in_dim=dim_db34, embed_dim=embed_dim, num_classes=num_classes, dropout=dropout)
            self.head_mose = _Head(in_dim=dim_mose, embed_dim=embed_dim, num_classes=num_classes, dropout=dropout)
            self.head_db5_vh = _Head(in_dim=dim_db5, embed_dim=embed_dim, num_classes=num_classes, dropout=dropout)
            self.head_db5_vv = _Head(in_dim=dim_db5, embed_dim=embed_dim, num_classes=num_classes, dropout=dropout)

    def _build_pool(self, in_ch: int, k: int, lam: float, groups: int) -> Tuple[nn.Module, int]:
        if self.pooling == "gbp":
            assert in_ch % groups == 0, f"in_ch({in_ch}) must be divisible by groups({groups})"
            d = in_ch // groups
            dim = (groups * (groups + 1) // 2) * d * d
            return GroupBilinearPooling(groups=groups), dim
        return FBCPooling(in_ch=in_ch, k=k, lam=lam), k

    def gradnorm_shared_parameters(self) -> List[torch.nn.Parameter]:
        # Use the last DenseBlock parameters as shared reference (both pols are shared topology but not weights).
        # GradNorm needs any shared trunk parameter set; choose DB5 of both towers.
        params = []
        params += list(self.bb_vh.db5.parameters())
        params += list(self.bb_vv.db5.parameters())
        return params

    def forward(self, vh: torch.Tensor, vv: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        feats_vh = self.bb_vh(vh)
        feats_vv = self.bb_vv(vv)

        out_vh = self.imdff_vh(feats_vh)
        out_vv = self.imdff_vv(feats_vv)

        # DB34 (FMS cross or DB4' if MSDFF ablation)
        if self.db34_source == "db4p":
            db34_vh = out_vh["db4p"]
            db34_vv = out_vv["db4p"]
        else:
            db34_vh = out_vh["fms"]
            db34_vv = out_vv["fms"]
        z_db34 = self.pool_db34(db34_vh, db34_vv)
        logits_db34 = self.head_db34(z_db34)

        # MO-SE/SE/CBAM augmented DB5' -> cross pooling
        if self.attention == "mose":
            vh_aug, vv_aug = self.att_block(out_vh["db5p"], out_vv["db5p"])
        elif self.att_block is None:
            vh_aug, vv_aug = out_vh["db5p"], out_vv["db5p"]
        else:
            vh_aug = self.att_block(out_vh["db5p"])
            vv_aug = self.att_block(out_vv["db5p"])
        z_mose = self.pool_mose(vh_aug, vv_aug)
        logits_mose = self.head_mose(z_mose)

        # DB5 self
        z_db5_vh = self.pool_db5(out_vh["db5p"], out_vh["db5p"])
        z_db5_vv = self.pool_db5(out_vv["db5p"], out_vv["db5p"])
        logits_db5_vh = self.head_db5_vh(z_db5_vh)
        logits_db5_vv = self.head_db5_vv(z_db5_vv)

        aux = {
            "db34": logits_db34,
            "mose": logits_mose,
            "db5_vh": logits_db5_vh,
            "db5_vv": logits_db5_vv,
        }

        if self.assemble == "prob_mean":
            probs = [F.softmax(aux[k], dim=1) for k in self.branch_names]
            prob = sum(probs) / len(probs)
            logits = torch.log(prob + 1e-12)
        else:  # logit_mean
            logits = sum([aux[k] for k in self.branch_names]) / len(self.branch_names)

        return logits, aux
