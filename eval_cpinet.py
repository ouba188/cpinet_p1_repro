# eval_cpinet.py
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dualpol_dataset import DualPolCSVDataset, AugCfg
from models.cpinet_p1 import CPINetP1
from models.cpinet_baselines import BackboneNet, GBCNN
from utils_metrics import confusion_matrix, per_class_prf


def load_label2id(path: Path) -> Dict[str, int]:
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def build_model_from_args(args: argparse.Namespace, ckpt_args: Dict[str, object]) -> torch.nn.Module:
    model_type = ckpt_args.get("model", args.model)
    if model_type == "backbone":
        model = BackboneNet(
            num_classes=args.num_classes,
            embed_dim=int(ckpt_args.get("embed_dim", args.embed_dim)),
            dropout=float(ckpt_args.get("dropout", args.dropout)),
        )
        model.input_mode = ckpt_args.get("input_mode", args.input_mode)
        return model
    if model_type == "gbcnn":
        model = GBCNN(
            num_classes=args.num_classes,
            gbp_groups=int(ckpt_args.get("gbp_groups", args.gbp_groups)),
            embed_dim=int(ckpt_args.get("embed_dim", args.embed_dim)),
            dropout=float(ckpt_args.get("dropout", args.dropout)),
        )
        model.input_mode = "dual"
        return model
    model = CPINetP1(
        num_classes=args.num_classes,
        fuse_ch=int(ckpt_args.get("fuse_ch", args.fuse_ch)),
        fbc_k=int(ckpt_args.get("fbc_k", args.fbc_k)),
        fbc_lam=float(ckpt_args.get("fbc_lam", args.fbc_lam)),
        embed_dim=int(ckpt_args.get("embed_dim", args.embed_dim)),
        gbp_groups=int(ckpt_args.get("gbp_groups", args.gbp_groups)),
        dropout=float(ckpt_args.get("dropout", args.dropout)),
        assemble=str(ckpt_args.get("assemble", args.assemble)),
        fusion=str(ckpt_args.get("fusion", args.fusion)),
        attention=str(ckpt_args.get("attention", args.attention)),
        pooling=str(ckpt_args.get("pooling", args.pooling)),
        share_head=bool(ckpt_args.get("share_head", True)),
        db34_source=str(ckpt_args.get("db34_source", args.db34_source)),
    )
    model.input_mode = ckpt_args.get("input_mode", args.input_mode)
    return model


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def close(self) -> None:
        for h in self.hook_handles:
            h.remove()

    def __call__(self, logits: torch.Tensor, class_idx: torch.Tensor) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        score = logits.gather(1, class_idx[:, None]).sum()
        score.backward(retain_graph=True)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1)
        cam = torch.relu(cam)
        cam = cam - cam.min(dim=1, keepdim=True)[0]
        cam = cam / (cam.max(dim=1, keepdim=True)[0] + 1e-12)
        return cam


def plot_confusion(cm: np.ndarray, labels: Tuple[str, ...], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Greys")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="tab:green" if i == j else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def overlay_cam(img: np.ndarray, cam: np.ndarray) -> np.ndarray:
    cmap = plt.get_cmap("jet")
    heat = cmap(cam)[:, :, :3]
    overlay = 0.6 * img[:, :, None] + 0.4 * heat
    return np.clip(overlay, 0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--label2id", type=str, default=None, help="Optional label2id.json path.")
    ap.add_argument("--out_dir", type=str, default="runs/eval")
    ap.add_argument("--num_classes", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--imgsz", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--model", type=str, default="cpinet", choices=["cpinet", "backbone", "gbcnn"])
    ap.add_argument("--input_mode", type=str, default="dual", choices=["dual", "vh", "vv"])
    ap.add_argument("--cam_layer", type=str, default="vh", choices=["vh", "vv"])
    ap.add_argument("--cam_samples", type=int, default=8)
    ap.add_argument("--fuse_ch", type=int, default=128)
    ap.add_argument("--fbc_k", type=int, default=2048)
    ap.add_argument("--fbc_lam", type=float, default=1e-3)
    ap.add_argument("--embed_dim", type=int, default=1024)
    ap.add_argument("--gbp_groups", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--assemble", type=str, default="logit_mean")
    ap.add_argument("--fusion", type=str, default="imdff")
    ap.add_argument("--attention", type=str, default="mose")
    ap.add_argument("--pooling", type=str, default="fbc")
    ap.add_argument("--db34_source", type=str, default="fms")

    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    model = build_model_from_args(args, ckpt_args).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    label2id_path = Path(args.label2id) if args.label2id else Path(args.checkpoint).parent / "label2id.json"
    label2id = load_label2id(label2id_path)
    id2label = {v: k for k, v in label2id.items()}

    ds = DualPolCSVDataset(args.test_csv, label2id, img_size=args.imgsz, aug=AugCfg(enable=False))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False)

    ys, ps = [], []
    for x_vv, x_vh, y in dl:
        x_vv = x_vv.to(device)
        x_vh = x_vh.to(device)
        y = y.to(device)
        if getattr(model, "input_mode", "dual") == "vh":
            logits, _ = model(x_vh)
        elif getattr(model, "input_mode", "dual") == "vv":
            logits, _ = model(x_vv)
        else:
            logits, _ = model(x_vh, x_vv)
        pred = logits.argmax(dim=1).detach().cpu().numpy()
        ys.append(y.detach().cpu().numpy())
        ps.append(pred)

    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(ps, axis=0)
    cm = confusion_matrix(args.num_classes, y_true, y_pred)
    prf = per_class_prf(cm)
    oa = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    macro_p = float(np.mean(prf["precision"])) if cm.sum() else 0.0
    macro_r = float(np.mean(prf["recall"])) if cm.sum() else 0.0
    macro_f1 = float(np.mean(prf["f1"])) if cm.sum() else 0.0
    np.save(out_dir / "confusion.npy", cm)
    labels = tuple(id2label[i] for i in range(args.num_classes))
    plot_confusion(cm, labels, out_dir / "confusion_matrix.png")
    np.savez(out_dir / "per_class_prf.npz", **prf)
    metrics = {
        "oa": oa,
        "macro_p": macro_p,
        "macro_r": macro_r,
        "macro_f1": macro_f1,
        "num_classes": args.num_classes,
        "input_mode": getattr(model, "input_mode", "dual"),
        "model": ckpt_args.get("model", args.model),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    if isinstance(model, BackboneNet):
        cam_layer = model.backbone.db5
    elif isinstance(model, GBCNN):
        cam_layer = model.bb_vh.db5 if args.cam_layer == "vh" else model.bb_vv.db5
    else:
        cam_layer = model.bb_vh.db5 if args.cam_layer == "vh" else model.bb_vv.db5
    grad_cam = GradCAM(model, cam_layer)
    cam_out = out_dir / "gradcam"
    cam_out.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(ds, batch_size=1, shuffle=False)
    for idx, (x_vv, x_vh, y) in enumerate(loader):
        if idx >= args.cam_samples:
            break
        x_vv = x_vv.to(device)
        x_vh = x_vh.to(device)
        if getattr(model, "input_mode", "dual") == "vh":
            logits, _ = model(x_vh)
        elif getattr(model, "input_mode", "dual") == "vv":
            logits, _ = model(x_vv)
        else:
            logits, _ = model(x_vh, x_vv)
        pred = logits.argmax(dim=1)
        cam = grad_cam(logits, pred)[0].cpu().numpy()
        img = x_vh[0, 0].cpu().numpy() if args.cam_layer == "vh" else x_vv[0, 0].cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-12)
        overlay = overlay_cam(img, cam)
        plt.imsave(cam_out / f"cam_{idx:03d}.png", overlay)

    grad_cam.close()


if __name__ == "__main__":
    main()
