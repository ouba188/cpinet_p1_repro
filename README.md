
# CPINet P1 Reproduction (Dual-Polarized SAR Ship Classification)

This package reproduces the **P1** stage of CPINet (Remote Sensing 2024):
- SAR-DenseNet-v1 backbone (two towers: VH / VV)
- IMDFF (Eq.1-2): multiscale fusion + DB34 fusion feature maps
- MO-SE: mixed-order SE augmentation (GAP + GBP) with cross-pol transfer
- FBC pooling (Eq.6-7): factorized bilinear coding + soft-threshold + spatial max
- 4 branches: DB34 / MO-SE / DB5-VH / DB5-VV
- Loss balancing: SUM or GradNorm (adaptive)

## 1) Prepare paired CSV
Use your existing folder structure (vv_root / vh_root with class folders).

```bash
python prepare_pairs_index.py \
  --vv_root /path/to/cls_vv \
  --vh_root /path/to/cls_vh \
  --out_dir pairs_csv
```

It will output `pairs_csv/train.csv`, `pairs_csv/val.csv`, `pairs_csv/test.csv`.

## 2) Train
```bash
python train_cpinet.py \
  --train_csv pairs_csv/train.csv \
  --val_csv pairs_csv/val.csv \
  --test_csv pairs_csv/test.csv \
  --out_dir runs/cpinet_p1 \
  --num_classes 3 \
  --imgsz 64 \
  --batch 64 \
  --loss_balance gradnorm \
  --fbc_k 2048 \
  --fbc_lam 1e-3
```

Switch to simple sum:
```bash
--loss_balance sum
```


## Repeated random splits (paper-style protocol)

Use `cv/` scripts to reproduce 5 random splits × 3 repeats, then report mean±std over split-wise medians.

See `cv/make_repeated_splits.py`, `cv/run_repeated_splits.py`, `cv/summarize_repeated.py`.
