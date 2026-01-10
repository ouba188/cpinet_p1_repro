
# train_cpinet.py (P1: CPINet reproduction)
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dualpol_dataset import DualPolCSVDataset, AugCfg
from models.cpinet_p1 import CPINetP1
from utils_metrics import top1_accuracy, confusion_matrix, per_class_prf


def build_label2id(csv_paths):
    """Build label2id from one or more CSVs. CSV must contain 'label_name' column."""
    import csv
    labels = []
    for p in csv_paths:
        if p is None:
            continue
        with open(p, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                labels.append(row["label_name"])
    uniq = sorted(set(labels))
    return {name: i for i, name in enumerate(uniq)}




def ce_loss(logits: torch.Tensor, y: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    return F.cross_entropy(logits, y, label_smoothing=label_smoothing)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: str, num_classes: int) -> Dict[str, float]:
    model.eval()
    ys, ps = [], []
    total_loss = 0.0
    n = 0
    for x_vv, x_vh, y in loader:
        x_vv = x_vv.to(device)
        x_vh = x_vh.to(device)
        y = y.to(device)
        logits, aux = model(x_vh, x_vv)
        loss = F.cross_entropy(logits, y)
        total_loss += float(loss.item()) * y.size(0)
        n += y.size(0)

        pred = logits.argmax(dim=1).detach().cpu().numpy()
        ys.append(y.detach().cpu().numpy())
        ps.append(pred)

    y_true = np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=np.int64)
    y_pred = np.concatenate(ps, axis=0) if ps else np.zeros((0,), dtype=np.int64)
    cm = confusion_matrix(num_classes, y_true, y_pred)
    prf = per_class_prf(cm)
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0

    return {
        "loss": total_loss / max(1, n),
        "acc": acc,
        "macro_f1": float(np.mean(prf["f1"])) if cm.sum() else 0.0,
    }


def train_one_epoch(
    model: CPINetP1,
    loader: DataLoader,
    device: str,
    opt_model: torch.optim.Optimizer,
    loss_balance: str,
    gradnorm_alpha: float,
    opt_w: Optional[torch.optim.Optimizer],
    w: Optional[torch.nn.Parameter],
    init_losses: Optional[torch.Tensor],
    label_smoothing: float,
) -> Tuple[float, Dict[str, float], Optional[torch.Tensor]]:
    model.train()
    total = 0.0
    n = 0
    logs_sum: Dict[str, float] = {k: 0.0 for k in ["loss_total"] + [f"loss_{b}" for b in model.branch_names]}

    for x_vv, x_vh, y in loader:
        x_vv = x_vv.to(device)
        x_vh = x_vh.to(device)
        y = y.to(device)

        logits, aux = model(x_vh, x_vv)
        losses = torch.stack([ce_loss(aux[b], y, label_smoothing) for b in model.branch_names])  # (4,)

        if loss_balance == "sum":
            loss_total = losses.sum()
            opt_model.zero_grad(set_to_none=True)
            loss_total.backward()
            opt_model.step()
        else:
            # GradNorm (Adaptive loss balancing) on 4 CE losses
            assert w is not None and opt_w is not None
            # init losses on first batch
            if init_losses is None:
                init_losses = losses.detach()

            # weighted loss for model update
            weighted = (w * losses).sum()

            # compute gradient norms for each task wrt shared params
            shared_params = model.gradnorm_shared_parameters()
            g_list = []
            for i in range(len(model.branch_names)):
                g = torch.autograd.grad(w[i] * losses[i], shared_params, retain_graph=True, create_graph=True)
                # global L2 norm across shared params
                g_norm = torch.norm(torch.stack([gi.norm(p=2) for gi in g]), p=2)
                g_list.append(g_norm)
            g_list = torch.stack(g_list)  # (4,)
            g_avg = g_list.mean().detach()

            # inverse training rate
            with torch.no_grad():
                loss_ratio = losses.detach() / (init_losses + 1e-12)
                r = loss_ratio / (loss_ratio.mean() + 1e-12)

            # target gradient norms
            target = g_avg * (r ** gradnorm_alpha)

            # GradNorm loss to update w
            gn_loss = torch.sum(torch.abs(g_list - target))

            # update w
            opt_w.zero_grad(set_to_none=True)
            gn_loss.backward(retain_graph=True)
            
            # update model with weighted sum (do NOT include gn_loss)
            opt_model.zero_grad(set_to_none=True)
            weighted.backward()
            opt_model.step()

            # update w step AFTER model update to avoid inplace error
            opt_w.step()
            # renormalize w to keep sum = num_tasks
            with torch.no_grad():
                w.data.clamp_(min=1e-6)
                w.data = w.data * (len(model.branch_names) / w.data.sum())

            loss_total = weighted.detach()

        bs = y.size(0)
        total += float(loss_total.item()) * bs
        n += bs

        logs_sum["loss_total"] += float(loss_total.item()) * bs
        for i, b in enumerate(model.branch_names):
            logs_sum[f"loss_{b}"] += float(losses[i].item()) * bs

    logs = {k: v / max(1, n) for k, v in logs_sum.items()}
    return total / max(1, n), logs, init_losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--val_csv", type=str, required=True)
    ap.add_argument("--test_csv", type=str, required=False, default=None)
    ap.add_argument("--out_dir", type=str, default="runs/cpinet_p1")
    ap.add_argument("--run_dir", type=str, default=None, help="Optional explicit run directory (for repeated splits).")
    ap.add_argument("--save_json", action="store_true", help="Save final metrics to metrics.json in run_dir.")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=64)
    ap.add_argument("--num_classes", type=int, default=3)

    ap.add_argument("--loss_balance", type=str, default="gradnorm", choices=["sum", "gradnorm"])
    ap.add_argument("--gradnorm_alpha", type=float, default=1.5)
    ap.add_argument("--label_smoothing", type=float, default=0.0)

    ap.add_argument("--fbc_k", type=int, default=2048)
    ap.add_argument("--fbc_lam", type=float, default=1e-3)
    ap.add_argument('--fbc_l1', dest='fbc_lam', type=float, help='Alias of --fbc_lam')
    
    ap.add_argument("--fuse_ch", type=int, default=128)
    ap.add_argument("--embed_dim", type=int, default=1024)
    ap.add_argument("--gbp_groups", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--assemble", type=str, default="logit_mean", choices=["logit_mean", "prob_mean"])

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_aug", action="store_true")

    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    run_dir = Path(args.run_dir) if args.run_dir else Path(args.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build consistent label map for this run (important for repeated splits)
    import json
    csvs = [args.train_csv, args.val_csv]
    if args.test_csv:
        csvs.append(args.test_csv)
    label2id = build_label2id(csvs)
    (run_dir / "label2id.json").write_text(
        json.dumps(label2id, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    if len(label2id) != args.num_classes:
        print(f"[Warn] args.num_classes={args.num_classes}, but label2id has {len(label2id)} classes: {list(label2id.keys())}")

    aug = AugCfg(enable=not args.no_aug)
    ds_tr = DualPolCSVDataset(args.train_csv, label2id, img_size=args.imgsz, aug=aug)
    ds_va = DualPolCSVDataset(args.val_csv,   label2id, img_size=args.imgsz, aug=AugCfg(enable=False))
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)

    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = CPINetP1(
        num_classes=args.num_classes,
        fuse_ch=args.fuse_ch,
        fbc_k=args.fbc_k,
        fbc_lam=args.fbc_lam,
        embed_dim=args.embed_dim,
        gbp_groups=args.gbp_groups,
        dropout=args.dropout,
        assemble=args.assemble,
    ).to(device)

    opt_model = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # GradNorm weights
    w = None
    opt_w = None
    init_losses = None
    if args.loss_balance == "gradnorm":
        w = torch.nn.Parameter(torch.ones(len(model.branch_names), device=device))
        opt_w = torch.optim.Adam([w], lr=1e-3)  # small lr for weights

    best = -1.0
    best_path = run_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_logs, init_losses = train_one_epoch(
            model=model,
            loader=dl_tr,
            device=device,
            opt_model=opt_model,
            loss_balance=args.loss_balance,
            gradnorm_alpha=args.gradnorm_alpha,
            opt_w=opt_w,
            w=w,
            init_losses=init_losses,
            label_smoothing=args.label_smoothing,
        )

        va = evaluate(model, dl_va, device=device, num_classes=args.num_classes)

        # save
        if va["acc"] > best:
            best = va["acc"]
            torch.save({"model": model.state_dict(), "acc": best, "epoch": epoch}, best_path)

        # simple log to stdout
        w_str = ""
        if w is not None:
            w_str = " w=" + ",".join([f"{float(x):.3f}" for x in w.detach().cpu().tolist()])
        print(f"[{epoch:03d}/{args.epochs}] tr_loss={tr_loss:.4f} va_loss={va['loss']:.4f} va_acc={va['acc']:.4f} va_macro_f1={va['macro_f1']:.4f}{w_str}")

    print(f"Best val acc={best:.4f}  checkpoint={best_path}")

    # optional test
    if args.test_csv:
        ds_te = DualPolCSVDataset(args.test_csv, label2id, img_size=args.imgsz, aug=AugCfg(enable=False))
        dl_te = DataLoader(ds_te, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        te = evaluate(model, dl_te, device=device, num_classes=args.num_classes)
        print(f"Test: loss={te['loss']:.4f} acc={te['acc']:.4f} macro_f1={te['macro_f1']:.4f}")
        if args.save_json:
            import json
            metrics = {
                "seed": int(args.seed),
                "best_val_acc": float(best),
                "test": {k: (float(v) if isinstance(v,(int,float)) else v) for k,v in te.items()},
                "args": vars(args),
            }
            with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()