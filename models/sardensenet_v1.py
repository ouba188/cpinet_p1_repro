
# models/sardensenet_v1.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DenseLayer(nn.Module):
    def __init__(self, in_ch: int, growth: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_ch, growth, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(self.relu(self.bn(x)))
        return torch.cat([x, y], dim=1)


class _DenseBlock(nn.Module):
    def __init__(self, in_ch: int, num_layers: int, growth: int):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(num_layers):
            layers.append(_DenseLayer(ch, growth))
            ch = ch + growth
        self.block = nn.Sequential(*layers)
        self.out_ch = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Transition(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(self.conv(x)))
        return self.pool(x)


class SARDenseNetV1(nn.Module):
    """
    SAR-DenseNet-v1 (as described in GBCNN/MS-GBCNN/CPINet line of works):
      - 5 DenseBlocks, each with 3 layers
      - growth rates: [3, 6, 9, 12, 15]
      - transitions (1x1 conv + avgpool) after DB1-DB4
    Input: (B, 1, H, W)
    Output: dict with 'db1'..'db5'
    """
    def __init__(self, stem_ch: int = 24, num_layers_per_block: int = 3, growth_rates: List[int] | None = None):
        super().__init__()
        if growth_rates is None:
            growth_rates = [3, 6, 9, 12, 15]
        assert len(growth_rates) == 5

        self.stem = nn.Sequential(
            nn.Conv2d(1, stem_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.ReLU(inplace=True),
        )

        ch = stem_ch
        self.db1 = _DenseBlock(ch, num_layers_per_block, growth_rates[0]); ch = self.db1.out_ch
        self.t1  = _Transition(ch, ch);  # keep width, downsample
        self.db2 = _DenseBlock(ch, num_layers_per_block, growth_rates[1]); ch = self.db2.out_ch
        self.t2  = _Transition(ch, ch)
        self.db3 = _DenseBlock(ch, num_layers_per_block, growth_rates[2]); ch = self.db3.out_ch
        self.t3  = _Transition(ch, ch)
        self.db4 = _DenseBlock(ch, num_layers_per_block, growth_rates[3]); ch = self.db4.out_ch
        self.t4  = _Transition(ch, ch)
        self.db5 = _DenseBlock(ch, num_layers_per_block, growth_rates[4]); ch = self.db5.out_ch

        self.out_channels = {
            "db1": self.db1.out_ch,
            "db2": self.db2.out_ch,
            "db3": self.db3.out_ch,
            "db4": self.db4.out_ch,
            "db5": self.db5.out_ch,
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        f1 = self.db1(x)
        x = self.t1(f1)
        f2 = self.db2(x)
        x = self.t2(f2)
        f3 = self.db3(x)
        x = self.t3(f3)
        f4 = self.db4(x)
        x = self.t4(f4)
        f5 = self.db5(x)
        return {"db1": f1, "db2": f2, "db3": f3, "db4": f4, "db5": f5}
