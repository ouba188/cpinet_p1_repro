
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

### Paper datasets
Five-class dataset (OpenSARShip 5C):
```bash
python prepare_pairs_index.py \
  --vv_root /root/private_data/cpinet_pairs/vv \
  --vh_root /root/private_data/cpinet_pairs/vh \
  --out_dir pairs_csv_5c
```

Three-class dataset (OpenSARShip 3C):
```bash
python prepare_pairs_index.py \
  --vv_root /root/private_data/cls_vv \
  --vh_root /root/private_data/cls_vh \
  --out_dir pairs_csv_3c
```

## 2) Train (paper default)
```bash
python train_cpinet.py \
  --train_csv pairs_csv/train.csv \
  --val_csv pairs_csv/val.csv \
  --test_csv pairs_csv/test.csv \
  --out_dir runs/cpinet_p1 \
  --num_classes 3 \
  --imgsz 64 \
  --batch 32 \
  --loss_balance gradnorm \
  --optimizer sgd \
  --lr 0.01 \
  --momentum 0.9 \
  --wd 5e-4 \
  --dropout 0.2 \
  --fbc_k 2048 \
  --fbc_lam 1e-3
```

Switch to simple sum:
```bash
--loss_balance sum
```

## 3) Ablation configs (Section 4.5)
Below are CLI switches that map to the paper ablations:

| Ablation | CLI switches |
| --- | --- |
| w/ MSDFF | `--fusion msdff --db34_source db4p` |
| w/ FBC (vs GBP) | `--pooling fbc` (use `--pooling gbp --no_share_head` as baseline) |
| w/ IMDFF | `--fusion imdff --db34_source fms` |
| w/ CBAM | `--attention cbam` |
| w/ SE | `--attention se` |
| w/ MO-SE | `--attention mose` |
| w/ GradNorm | `--loss_balance gradnorm` (baseline: `--loss_balance sum`) |

Example (SE ablation):
```bash
python train_cpinet.py \
  --train_csv pairs_csv/train.csv \
  --val_csv pairs_csv/val.csv \
  --out_dir runs/ablation_se \
  --attention se \
  --loss_balance sum
```

## 4) Sensitivity curves (Section 4.4)
Run separate experiments with different `--fbc_k` and `--fbc_lam`, then plot:
```bash
python plot_paper_figures.py \
  --k_runs runs/k_1024 runs/k_2048 runs/k_4096 \
  --k_labels k=1024 k=2048 k=4096 \
  --lam_runs runs/lam_1e-2 runs/lam_1e-3 runs/lam_1e-4 \
  --lam_labels lam=0.01 lam=0.001 lam=0.0001 \
  --metric val_acc \
  --out_dir runs/plots
```

## 5) GradNorm curves (Section 4.5 Figure 7)
```bash
python plot_paper_figures.py \
  --gradnorm_run runs/cpinet_p1 \
  --out_dir runs/plots
```

## 6) Extra evaluation (Section 4.6)
Generate confusion matrix + Grad-CAM visualizations:
```bash
python eval_cpinet.py \
  --test_csv pairs_csv/test.csv \
  --checkpoint runs/cpinet_p1/best.pt \
  --out_dir runs/eval_cpinet \
  --num_classes 3
```
Single-pol evaluation (Table 7 style):
```bash
python eval_cpinet.py \
  --test_csv pairs_csv/test.csv \
  --checkpoint runs/cpinet_p1/best.pt \
  --out_dir runs/eval_vh \
  --num_classes 3 \
  --input_mode vh
```

## 7) Paper table alignment (runs for Tables 4–8)

### Table 4/5 (SOTA comparison, CPINet results for 3C/5C)
```bash
# 3-class CPINet
python train_cpinet.py \
  --train_csv pairs_csv_3c/train.csv \
  --val_csv pairs_csv_3c/val.csv \
  --test_csv pairs_csv_3c/test.csv \
  --out_dir runs/cpinet_3c \
  --num_classes 3

# 5-class CPINet
python train_cpinet.py \
  --train_csv pairs_csv_5c/train.csv \
  --val_csv pairs_csv_5c/val.csv \
  --test_csv pairs_csv_5c/test.csv \
  --out_dir runs/cpinet_5c \
  --num_classes 5
```

### Table 6 (ablation study, 3C)
```bash
# w/ MSDFF
python train_cpinet.py --train_csv pairs_csv_3c/train.csv --val_csv pairs_csv_3c/val.csv --out_dir runs/ablation_msdff --num_classes 3 --fusion msdff --db34_source db4p --loss_balance sum
# w/ FBC (use GBP baseline if needed)
python train_cpinet.py --train_csv pairs_csv_3c/train.csv --val_csv pairs_csv_3c/val.csv --out_dir runs/ablation_fbc --num_classes 3 --pooling fbc --loss_balance sum
# w/ IMDFF
python train_cpinet.py --train_csv pairs_csv_3c/train.csv --val_csv pairs_csv_3c/val.csv --out_dir runs/ablation_imdff --num_classes 3 --fusion imdff --db34_source fms --loss_balance sum
# w/ CBAM
python train_cpinet.py --train_csv pairs_csv_3c/train.csv --val_csv pairs_csv_3c/val.csv --out_dir runs/ablation_cbam --num_classes 3 --attention cbam --loss_balance sum
# w/ SE
python train_cpinet.py --train_csv pairs_csv_3c/train.csv --val_csv pairs_csv_3c/val.csv --out_dir runs/ablation_se --num_classes 3 --attention se --loss_balance sum
# w/ MO-SE
python train_cpinet.py --train_csv pairs_csv_3c/train.csv --val_csv pairs_csv_3c/val.csv --out_dir runs/ablation_mose --num_classes 3 --attention mose --loss_balance sum
# w/ GradNorm
python train_cpinet.py --train_csv pairs_csv_3c/train.csv --val_csv pairs_csv_3c/val.csv --out_dir runs/ablation_gradnorm --num_classes 3 --loss_balance gradnorm
```

### Table 7 (single-pol vs dual-pol effectiveness)
```bash
# Backbone single-pol (VH)
python train_cpinet.py --train_csv pairs_csv_3c/train.csv --val_csv pairs_csv_3c/val.csv --test_csv pairs_csv_3c/test.csv --out_dir runs/backbone_vh --num_classes 3 --model backbone --input_mode vh

# Backbone single-pol (VV)
python train_cpinet.py --train_csv pairs_csv_3c/train.csv --val_csv pairs_csv_3c/val.csv --test_csv pairs_csv_3c/test.csv --out_dir runs/backbone_vv --num_classes 3 --model backbone --input_mode vv

# GBCNN dual-pol
python train_cpinet.py --train_csv pairs_csv_3c/train.csv --val_csv pairs_csv_3c/val.csv --test_csv pairs_csv_3c/test.csv --out_dir runs/gbcnn --num_classes 3 --model gbcnn

# CPINet dual-pol
python train_cpinet.py --train_csv pairs_csv_3c/train.csv --val_csv pairs_csv_3c/val.csv --test_csv pairs_csv_3c/test.csv --out_dir runs/cpinet_3c --num_classes 3
```

### Table 8 / Section 4.6 (confusion matrix + Grad-CAM)
```bash
python eval_cpinet.py --test_csv pairs_csv_3c/test.csv --checkpoint runs/cpinet_3c/best.pt --out_dir runs/eval_3c --num_classes 3
python eval_cpinet.py --test_csv pairs_csv_5c/test.csv --checkpoint runs/cpinet_5c/best.pt --out_dir runs/eval_5c --num_classes 5
```


## Repeated random splits (paper-style protocol)

Use `cv/` scripts to reproduce 5 random splits × 3 repeats, then report mean±std over split-wise medians.

See `cv/make_repeated_splits.py`, `cv/run_repeated_splits.py`, `cv/summarize_repeated.py`.
