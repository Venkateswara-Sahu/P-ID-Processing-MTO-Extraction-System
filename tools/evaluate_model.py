"""
Per-class evaluation of a P&ID detection model.

With 229 classes the aggregate mAP hides everything that matters. This
tool reports the number of classes that were actually evaluated, the
classes that score zero, and the class-health table that shows which
classes are worth retraining on.

Usage:
    python tools/evaluate_model.py --model models/best.pt --data <dataset_dir>
    python tools/evaluate_model.py --model models/best.pt --data <dataset_dir> \
        --split test --imgsz 640 --report eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_dataset_yaml(dataset_dir: Path) -> tuple[dict, list[str]]:
    """
    Build an absolute-path data.yaml and return (config, class_names).

    Roboflow's data.yaml uses paths like '../train/images', which assume
    a nested download layout (project/pid-1/). A flat download puts the
    splits directly under the dataset root, so those paths resolve to
    the wrong place. Both layouts are handled: the split directory is
    used when it actually exists.
    """
    src = dataset_dir / "data.yaml"
    if not src.exists():
        found = list(dataset_dir.rglob("data.yaml"))
        if not found:
            raise FileNotFoundError(f"no data.yaml under {dataset_dir}")
        src = found[0]

    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    base = src.parent

    for key, split in (("train", "train"), ("val", "valid"), ("test", "test")):
        value = cfg.get(key)
        if not value:
            continue
        if Path(value).is_absolute():
            continue
        # Prefer the flat layout when the split directory is a direct
        # child of the yaml's directory.
        flat = base / split / "images"
        if flat.exists():
            cfg[key] = str(flat.resolve())
        else:
            cfg[key] = str((base / value).resolve())

    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names, key=int)]
    return cfg, names


def check_class_compatibility(model, names: list[str]) -> dict:
    """
    Verify that the model's class list matches the dataset's.

    A mismatch is silent and catastrophic: YOLO predicts class index N,
    the dataset's label for index N is a different symbol, and every
    metric comes out as garbage that still looks like a number. The
    repaired dataset renumbers its classes, so an old checkpoint is
    incompatible with it until it is retrained.
    """
    model_names = [model.names[i] for i in range(len(model.names))]
    if model_names == names:
        return {"compatible": True, "reason": "identical, in the same order"}

    same_set = set(model_names) == set(names)
    if same_set:
        return {
            "compatible": False,
            "reason": "same class names but in a DIFFERENT ORDER - "
                      "every class index points at the wrong symbol",
        }
    only_model = sorted(set(model_names) - set(names))[:10]
    only_data = sorted(set(names) - set(model_names))[:10]
    return {
        "compatible": False,
        "reason": "different vocabularies",
        "only_in_model": only_model,
        "only_in_dataset": only_data,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "best.pt")
    ap.add_argument("--data", type=Path, required=True,
                    help="dataset directory containing data.yaml")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--top", type=int, default=25,
                    help="how many worst/best classes to print")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="evaluate anyway, even if the class lists differ")
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    cfg, names = load_dataset_yaml(args.data)
    out_yaml = Path(args.report).with_suffix(".data.yaml") if args.report else None
    if out_yaml is None:
        out_yaml = ROOT / ".eval_data.yaml"
    out_yaml.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")

    model = YOLO(str(args.model))
    print("=" * 78)
    print(f"MODEL  {args.model}")
    print(f"DATA   {args.data}   split={args.split}   imgsz={args.imgsz}")
    print(f"CLASSES IN VOCABULARY: {len(names)}")
    print("=" * 78)

    compat = check_class_compatibility(model, names)
    if not compat["compatible"]:
        print(f"\n  !! CLASS LISTS DO NOT MATCH: {compat['reason']}")
        if compat.get("only_in_model"):
            print(f"     only in model   : {compat['only_in_model']}")
        if compat.get("only_in_dataset"):
            print(f"     only in dataset : {compat['only_in_dataset']}")
        print("\n  The metrics below would be GARBAGE: the model predicts")
        print("  class index N, the dataset's label for index N is a")
        print("  different symbol. Retrain on this dataset, or point --data")
        print("  at the dataset the checkpoint was trained on.")
        if not args.allow_mismatch:
            print("\n  Aborting. Pass --allow-mismatch to evaluate anyway.")
            return None
    else:
        print(f"\n  class lists match: {compat['reason']}")

    metrics = model.val(
        data=str(out_yaml), split=args.split, imgsz=args.imgsz,
        batch=args.batch, device=args.device, plots=False, verbose=False,
    )
    box = metrics.box

    precision = float(box.mp)
    recall = float(box.mr)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    per_class = []
    for j, class_index in enumerate(box.ap_class_index):
        per_class.append({
            "id": int(class_index),
            "name": names[int(class_index)],
            "map50": float(box.p[j]),
        })
    per_class.sort(key=lambda r: r["map50"])

    evaluated = {r["id"] for r in per_class}
    never_evaluated = [
        {"id": i, "name": names[i]} for i in range(len(names)) if i not in evaluated
    ]
    zero_map = [r for r in per_class if r["map50"] == 0.0]

    print(f"\n  mAP50        {float(box.map50):.4f}")
    print(f"  mAP50-95     {float(box.map):.4f}")
    print(f"  precision    {precision:.4f}")
    print(f"  recall       {recall:.4f}")
    print(f"  f1           {f1:.4f}")
    print(f"\n  classes in vocabulary : {len(names)}")
    print(f"  classes EVALUATED      : {len(per_class)}"
          f"   ({100.0 * len(per_class) / len(names):.1f}% of the vocabulary)")
    print(f"  classes never evaluated: {len(never_evaluated)}")
    print(f"  classes with map50 == 0: {len(zero_map)}")

    print(f"\n  --- {args.top} WORST CLASSES ---")
    for row in per_class[:args.top]:
        print(f"    {row['map50']:.3f}  {row['name']!r}")
    print(f"\n  --- {args.top} BEST CLASSES ---")
    for row in per_class[-args.top:][::-1]:
        print(f"    {row['map50']:.3f}  {row['name']!r}")

    if never_evaluated:
        print(f"\n  --- CLASSES NEVER EVALUATED (absent from {args.split}) ---")
        for row in never_evaluated[:args.top]:
            print(f"    {row['name']!r}")

    report = {
        "model": str(args.model),
        "data": str(args.data),
        "split": args.split,
        "imgsz": args.imgsz,
        "class_lists_compatible": compat["compatible"],
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "vocabulary_size": len(names),
        "classes_evaluated": len(per_class),
        "classes_never_evaluated": len(never_evaluated),
        "classes_with_zero_map": len(zero_map),
        "per_class": per_class,
        "never_evaluated": never_evaluated,
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"\n  report -> {args.report}")
    return report


if __name__ == "__main__":
    main()
