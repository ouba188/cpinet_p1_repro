# dualpol_dataset.py
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


@dataclass
class AugCfg:
    enable: bool = True
    hflip_p: float = 0.5
    vflip_p: float = 0.5
    rot90_p: float = 0.5               # 随机 90/180/270
    translate_p: float = 0.5
    translate_px: int = 5              # [-5, 5]
    gauss_p: float = 0.5
    gauss_std: float = 0.001           # 论文表里写的 std=0.001（在[0,1]尺度上非常小）


def _load_gray_png(path: str) -> torch.Tensor:
    # 返回: float32, shape [1, H, W], range [0,1]
    img = Image.open(path).convert("L")
    t = TF.to_tensor(img)  # [1,H,W] float32 in [0,1]
    return t


def _sync_augment(vv: torch.Tensor, vh: torch.Tensor, cfg: AugCfg) -> Tuple[torch.Tensor, torch.Tensor]:
    if not cfg.enable:
        return vv, vh

    # vv/vh: [1,H,W]
    # 水平翻转
    if torch.rand(1).item() < cfg.hflip_p:
        vv = torch.flip(vv, dims=[2])
        vh = torch.flip(vh, dims=[2])

    # 垂直翻转
    if torch.rand(1).item() < cfg.vflip_p:
        vv = torch.flip(vv, dims=[1])
        vh = torch.flip(vh, dims=[1])

    # 旋转（90/180/270）
    if torch.rand(1).item() < cfg.rot90_p:
        k = int(torch.randint(low=1, high=4, size=(1,)).item())  # 1,2,3
        vv = torch.rot90(vv, k=k, dims=[1, 2])
        vh = torch.rot90(vh, k=k, dims=[1, 2])

    # 平移（像素）
    if torch.rand(1).item() < cfg.translate_p:
        tx = int(torch.randint(-cfg.translate_px, cfg.translate_px + 1, (1,)).item())
        ty = int(torch.randint(-cfg.translate_px, cfg.translate_px + 1, (1,)).item())
        # 用 affine 做平移：translate=(tx,ty)
        vv = TF.affine(vv, angle=0.0, translate=[tx, ty], scale=1.0, shear=[0.0, 0.0], fill=0.0)
        vh = TF.affine(vh, angle=0.0, translate=[tx, ty], scale=1.0, shear=[0.0, 0.0], fill=0.0)

    # 高斯噪声
    if torch.rand(1).item() < cfg.gauss_p:
        noise_vv = torch.randn_like(vv) * cfg.gauss_std
        noise_vh = torch.randn_like(vh) * cfg.gauss_std
        vv = torch.clamp(vv + noise_vv, 0.0, 1.0)
        vh = torch.clamp(vh + noise_vh, 0.0, 1.0)

    return vv, vh


class DualPolCSVDataset(Dataset):
    """
    输入 CSV: vv_path, vh_path, label_name
    输出:
      x_vv: [1,H,W], x_vh: [1,H,W], y: int
    """
    def __init__(
        self,
        csv_path: str,
        label2id: Dict[str, int],
        aug: Optional[AugCfg] = None,
        img_size: Optional[int] = None,
    ):
        self.csv_path = str(csv_path)
        self.label2id = dict(label2id)
        self.aug = aug if aug is not None else AugCfg(enable=False)
        self.img_size = img_size

        self.items: List[Tuple[str, str, int]] = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                vv = row["vv_path"]
                vh = row["vh_path"]
                ln = row["label_name"]
                if ln not in self.label2id:
                    raise ValueError(f"label '{ln}' not in label2id keys={list(self.label2id.keys())}")
                self.items.append((vv, vh, self.label2id[ln]))

        if len(self.items) == 0:
            raise RuntimeError(f"Empty dataset: {self.csv_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        vv_path, vh_path, y = self.items[idx]
        x_vv = _load_gray_png(vv_path)
        x_vh = _load_gray_png(vh_path)

        if self.img_size is not None:
             x_vv = TF.resize(x_vv, [self.img_size, self.img_size], antialias=True)
             x_vh = TF.resize(x_vh, [self.img_size, self.img_size], antialias=True)

        x_vv, x_vh = _sync_augment(x_vv, x_vh, self.aug)
        return x_vv, x_vh, torch.tensor(y, dtype=torch.long)
