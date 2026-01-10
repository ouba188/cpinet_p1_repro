# prepare_pairs_index.py
import argparse
from pathlib import Path
import csv

def swap_pol(name: str, src: str, dst: str) -> str:
    # 安全替换：优先替换 "__vh__" / "__vv__" 这种明确片段
    key_src = f"_{src}__"
    key_dst = f"_{dst}__"
    if key_src in name:
        return name.replace(key_src, key_dst)
    # 兜底：末尾 "_vh.png" / "_vv.png"
    if name.endswith(f"_{src}.png"):
        return name[:-len(f"_{src}.png")] + f"_{dst}.png"
    return name  # 不认识就原样返回，让上层报 missing

def build_split(vv_root: Path, vh_root: Path, split: str, out_csv: Path):
    rows = []
    missing = 0
    for cls_dir in sorted((vv_root / split).iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name
        for vv_path in sorted(cls_dir.glob("*.png")):
            vh_name = swap_pol(vv_path.name, "vv", "vh")
            vh_path = vh_root / split / cls_name / vh_name
            if not vh_path.exists():
                missing += 1
                continue
            rows.append((str(vv_path), str(vh_path), cls_name))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["vv_path", "vh_path", "label_name"])
        w.writerows(rows)

    print(f"[{split}] pairs={len(rows)} missing_vh={missing} -> {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vv_root", type=str, required=True,
                    help="e.g. /root/private_data/cls_vv")
    ap.add_argument("--vh_root", type=str, required=True,
                    help="e.g. /root/private_data/cls_vh")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="where to save train/val/test csv")
    args = ap.parse_args()

    vv_root = Path(args.vv_root)
    vh_root = Path(args.vh_root)
    out_dir = Path(args.out_dir)

    for split in ["train", "val", "test"]:
        build_split(vv_root, vh_root, split, out_dir / f"{split}.csv")

if __name__ == "__main__":
    main()
