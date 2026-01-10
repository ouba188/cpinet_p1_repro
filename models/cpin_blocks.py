
# models/cpin_blocks.py
from __future__ import annotations
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resize_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Resize x spatially to ref using safe ops (avgpool for down, bilinear for up)."""
    H, W = ref.shape[-2:]
    h, w = x.shape[-2:]
    if h == H and w == W:
        return x
    # downsample: use adaptive avgpool to avoid aliasing
    if h > H or w > W:
        return F.adaptive_avg_pool2d(x, output_size=(H, W))
    # upsample
    return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)


class Conv1x1BNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class IMDFF(nn.Module):
    """
    Improved Multiscale Deep Feature Fusion (IMDFF) per CPINet:
      Eq(1): F'_DB3 = F_DB3 + f1x1(Concat[DS/US of DB1,DB2,DB4,DB5] to DB3 size)
      Eq(2): FMS = f1x1(Concat[MaxPool2(F'_DB3), F'_DB4])  (DB34 fusion)
    We implement fused maps for DB3/DB4/DB5 plus FMS.
    """
    def __init__(self, chs: Dict[str, int], fuse_ch: int = 128):
        super().__init__()
        self.chs = chs
        # DB3 fusion: concat of 4 resized feature maps -> fuse_ch -> residual add to DB3
        in3 = chs["db1"] + chs["db2"] + chs["db4"] + chs["db5"]
        self.fuse3 = Conv1x1BNReLU(in3, fuse_ch)
        self.proj3 = nn.Conv2d(fuse_ch, chs["db3"], kernel_size=1, bias=False)

        # DB4 fusion
        in4 = chs["db1"] + chs["db2"] + chs["db3"] + chs["db5"]
        self.fuse4 = Conv1x1BNReLU(in4, fuse_ch)
        self.proj4 = nn.Conv2d(fuse_ch, chs["db4"], kernel_size=1, bias=False)

        # DB5 fusion (for MO-SE / DB5 self-aggregation)
        in5 = chs["db1"] + chs["db2"] + chs["db3"] + chs["db4"]
        self.fuse5 = Conv1x1BNReLU(in5, fuse_ch)
        self.proj5 = nn.Conv2d(fuse_ch, chs["db5"], kernel_size=1, bias=False)

        # Eq(2): FMS from DB3' (maxpool2) + DB4'
        self.fms_fuse = Conv1x1BNReLU(chs["db3"] + chs["db4"], fuse_ch)
        self.fms_proj = nn.Conv2d(fuse_ch, fuse_ch, kernel_size=1, bias=False)

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        db1, db2, db3, db4, db5 = feats["db1"], feats["db2"], feats["db3"], feats["db4"], feats["db5"]

        # DB3'
        cat3 = torch.cat([_resize_to(db1, db3), _resize_to(db2, db3), _resize_to(db4, db3), _resize_to(db5, db3)], dim=1)
        r3 = self.proj3(self.fuse3(cat3))
        db3p = db3 + r3

        # DB4'
        cat4 = torch.cat([_resize_to(db1, db4), _resize_to(db2, db4), _resize_to(db3, db4), _resize_to(db5, db4)], dim=1)
        r4 = self.proj4(self.fuse4(cat4))
        db4p = db4 + r4

        # DB5'
        cat5 = torch.cat([_resize_to(db1, db5), _resize_to(db2, db5), _resize_to(db3, db5), _resize_to(db4, db5)], dim=1)
        r5 = self.proj5(self.fuse5(cat5))
        db5p = db5 + r5

        # FMS: maxpool2(DB3') -> concat with DB4' -> f1x1
        db3p_ds = F.max_pool2d(db3p, kernel_size=2, stride=2)
        db3p_ds = _resize_to(db3p_ds, db4p)  # robust if odd sizes
        fms = self.fms_proj(self.fms_fuse(torch.cat([db3p_ds, db4p], dim=1)))

        return {"db3p": db3p, "db4p": db4p, "db5p": db5p, "fms": fms}


class GroupBilinearPooling(nn.Module):
    """
    GBP per MS-GBCNN / CPINet MO-SE squeeze (Eq.5 in MS-GBCNN paper):
      Split channels into G groups of d=C/G, compute bilinear pooling for each pair i<=j,
      vectorize and concatenate -> z_gbp.
    """
    def __init__(self, groups: int = 3, eps: float = 1e-6):
        super().__init__()
        self.groups = groups
        self.eps = eps

    def forward(self, x1: torch.Tensor, x2: torch.Tensor | None = None) -> torch.Tensor:
        if x2 is None:
            x2 = x1
        B, C, H, W = x1.shape
        G = self.groups
        assert C % G == 0, f"channels({C}) must be divisible by groups({G})"
        d = C // G
        N = H * W

        a = x1.view(B, G, d, N)
        b = x2.view(B, G, d, N)

        outs = []
        for i in range(G):
            for j in range(i, G):
                ai = a[:, i]                 # (B,d,N)
                bj = b[:, j]                 # (B,d,N)
                # bilinear pooling: (B,d,d)
                m = torch.bmm(ai, bj.transpose(1, 2)) / float(N)
                outs.append(m.reshape(B, d * d))
        z = torch.cat(outs, dim=1)          # (B, d*d*G*(G+1)/2)
        # mild normalization
        z = torch.sign(z) * torch.sqrt(torch.abs(z) + self.eps)
        z = F.normalize(z, p=2, dim=1)
        return z


class FBCPooling(nn.Module):
    """
    Factorized Bilinear Coding pooling (Eq.6-7 in CPINet):
      c' = P( U^T f1 ⊙ V^T f2 ),  c = sign(c') ⊙ max(|c'| - λ/2, 0)
      codes fused by max pooling over spatial locations.
    We use rank r=1 (as paper indicates), so P becomes identity.
    """
    def __init__(self, in_ch: int, k: int = 2048, lam: float = 1e-3, eps: float = 1e-6):
        super().__init__()
        self.k = k
        self.lam = lam
        self.eps = eps
        # U, V are implemented as 1x1 conv producing k channels
        self.U = nn.Conv2d(in_ch, k, kernel_size=1, bias=False)
        self.V = nn.Conv2d(in_ch, k, kernel_size=1, bias=False)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        # ensure same spatial size
        if f1.shape[-2:] != f2.shape[-2:]:
            f2 = F.interpolate(f2, size=f1.shape[-2:], mode="bilinear", align_corners=False)

        u = self.U(f1)          # (B,k,H,W)
        v = self.V(f2)          # (B,k,H,W)
        c_prime = u * v         # rank-1 factorized bilinear, P=I

        # suppress low values (soft-threshold): sign(x) * relu(|x| - lam/2)
        thr = self.lam * 0.5
        c = torch.sign(c_prime) * F.relu(torch.abs(c_prime) - thr)

        # max pool over spatial locations -> (B,k)
        z = torch.amax(c, dim=(2, 3))

        # signed sqrt + l2
        z = torch.sign(z) * torch.sqrt(torch.abs(z) + self.eps)
        z = F.normalize(z, p=2, dim=1)
        return z


class MOSE(nn.Module):
    """
    Mixed-Order SE (MO-SE) per CPINet:
      - first-order squeeze: GAP
      - second-order squeeze: GBP (single-pol self pooling)
      - excitation: Eq(3) with BN before activations
      - cross-pol augmentation: from pol-A get two gated vectors, then
        add + multiply to pol-B feature map (text after Eq.5 on CPINet p10).
    """
    def __init__(self, ch: int, gbp_groups: int = 3, reduction: int = 16):
        super().__init__()
        self.ch = ch
        self.gbp = GroupBilinearPooling(groups=gbp_groups)
        hid = max(8, ch // reduction)

        # GAP excitation MLP
        self.gap_fc1 = nn.Linear(ch, hid, bias=False)
        self.gap_bn1 = nn.BatchNorm1d(hid)
        self.gap_fc2 = nn.Linear(hid, ch, bias=False)
        self.gap_bn2 = nn.BatchNorm1d(ch)

        # GBP excitation MLP (input dim depends on groups)
        d = ch // gbp_groups
        gbp_dim = (gbp_groups * (gbp_groups + 1) // 2) * d * d
        self.gbp_fc1 = nn.Linear(gbp_dim, hid, bias=False)
        self.gbp_bn1 = nn.BatchNorm1d(hid)
        self.gbp_fc2 = nn.Linear(hid, ch, bias=False)
        self.gbp_bn2 = nn.BatchNorm1d(ch)

    def _excite(self, x: torch.Tensor, fc1: nn.Linear, bn1: nn.BatchNorm1d, fc2: nn.Linear, bn2: nn.BatchNorm1d) -> torch.Tensor:
        x = fc1(x)
        x = bn1(x)
        x = F.relu(x, inplace=True)
        x = fc2(x)
        x = bn2(x)
        x = torch.sigmoid(x)
        return x

    def forward(self, f_vh: torch.Tensor, f_vv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # squeeze VH
        gap_vh = torch.mean(f_vh, dim=(2, 3))                 # (B,C)
        gbp_vh = self.gbp(f_vh)                               # (B,gbp_dim) already normalized
        s_gap_vh = self._excite(gap_vh, self.gap_fc1, self.gap_bn1, self.gap_fc2, self.gap_bn2)
        s_gbp_vh = self._excite(gbp_vh, self.gbp_fc1, self.gbp_bn1, self.gbp_fc2, self.gbp_bn2)

        # squeeze VV
        gap_vv = torch.mean(f_vv, dim=(2, 3))
        gbp_vv = self.gbp(f_vv)
        s_gap_vv = self._excite(gap_vv, self.gap_fc1, self.gap_bn1, self.gap_fc2, self.gap_bn2)
        s_gbp_vv = self._excite(gbp_vv, self.gbp_fc1, self.gbp_bn1, self.gbp_fc2, self.gbp_bn2)

        # cross-pol augmentation: (F + s_gap) * s_gbp
        s_gap_vh4 = s_gap_vh[:, :, None, None]
        s_gbp_vh4 = s_gbp_vh[:, :, None, None]
        s_gap_vv4 = s_gap_vv[:, :, None, None]
        s_gbp_vv4 = s_gbp_vv[:, :, None, None]

        f_vh_aug = (f_vh + s_gap_vv4) * s_gbp_vv4
        f_vv_aug = (f_vv + s_gap_vh4) * s_gbp_vh4
        return f_vh_aug, f_vv_aug
