"""
Generate notebooks/01_yolo_training.ipynb.

The notebook embeds the repository's tools as base64 so it runs in
Colab on the first try, without depending on the repo being committed
and pushed. This script is the single source of truth: edit
tools/*.py, re-run this, and the notebook stays in sync.

Usage:
    python tools/build_notebook.py
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "01_yolo_training.ipynb"
TOOLS = ["repair_dataset.py", "merge_datasets.py"]

cells: list[dict] = []


def md(text: str):
    lines = text.strip("\n").split("\n")
    cells.append({
        "cell_type": "markdown", "metadata": {},
        "source": [l + ("\n" if i < len(lines) - 1 else "")
                   for i, l in enumerate(lines)],
    })


def code(text: str):
    lines = text.strip("\n").split("\n")
    cells.append({
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [],
        "source": [l + ("\n" if i < len(lines) - 1 else "")
                   for i, l in enumerate(lines)],
    })


# --------------------------------------------------------------------------- #
md("""
# 🎯 YOLOv8 Training for P&ID Symbol Detection

**Train a custom YOLOv8 model to detect engineering symbols in P&ID drawings.**

Runs on **Google Colab (free T4 GPU)**.

---

## ▶ START HERE (3 steps)

1. **Set the Roboflow key** (only if you have not already):
   ```python
   from google.colab import userdata
   userdata.put("ROBOFLOW_API_KEY", "<your key>")
   ```
   > 🔐 The previous version of this notebook had a live Roboflow key in
   > plaintext, and it is still in this repository's git history. **Rotate
   > that key.** This notebook reads the key from Colab secrets and never
   > writes it to a file.

2. **Runtime → Run all.**

3. **Download `best.pt`** from the last cell, then copy it to
   `models/best.pt` in your project.

---

## What changed, and why

This notebook was rewritten after an audit of the shipped `models/best.pt`
and the Roboflow dataset it was trained on. Every change below traces to
a measured defect (full write-up: `MODEL_DATASET_AUDIT.md`).

| Defect found | Fix in this notebook |
|---|---|
| **API key hardcoded in the notebook** (and in git history) | Read from Colab secrets / env; never written to a file |
| **`val` fell back to `train`** — mAP was measured on training data | Hard failure if the val split is missing |
| **2,194 of 5,462 image files are byte-identical duplicates**, and 498 appear in two splits | `tools/repair_dataset.py` dedupes and re-splits leak-free before training |
| **7 classes were the same symbol spelled differently** (`Pneumatic - Diaphragm Gate Valve` vs `Pneumatic-DiaphragmGate Valve`) — the loser measured **map50 = 0.000** | Merged into one canonical class |
| **7 class names were misspelled** (`bolier`, `Fluidzed Recactor`, `temp trasmitter`) — `bolier` also measured **map50 = 0.000** | Corrected |
| **229 classes but only 153 were ever evaluated** — the headline mAP hid 76 dead classes | Per-class report; the split guarantees every class is in train and val |
| **Class imbalance of 20,572×** (median class: 30 instances; 139 classes have <50) | Class-health table; `close_mosaic` + longer schedule |
| **Mosaic stayed on for the last 10 epochs**, blurring fine line work | `close_mosaic=15` |
| **`erasing` left at the default 0.4** — it *erases* 4% of every image, deleting symbols | `erasing=0.0` |
| **`auto_augment` on by default** — randaugment is for natural images | `auto_augment=None` |
| **Training args were stripped from the checkpoint** — you could not tell how `best.pt` was made | Args, epoch, metrics and a class-health table are re-attached before export |
| **No per-class metrics** | Full per-class table, the zero-mAP list, and a cross-reference against the training instance counts |
| **imgsz=640 with 44 different source image sizes** | `imgsz=1024` |

### One thing training alone will NOT fix

The shipped model scores mAP50 0.849 on the Roboflow dataset but finds
**none** of the vessels, columns, exchangers and reactors on a
P&ID drawn in a different style — even `Centrifugal Pump`, with 2,013
training instances and mAP50 0.935, misses 6 of 8 pumps there.

That is a **style gap, not a data-volume gap**: the glyphs are drawn
differently. Step 3b below merges in a target-style dataset if you have
one. Without it, expect the Roboflow classes to improve and the
target-style equipment symbols to stay missing.
""")

md("""
## Step 1: Install Dependencies
""")

code("""
!pip install -q ultralytics roboflow pyyaml

import torch, ultralytics
print("ultralytics:", ultralytics.__version__)
print("torch      :", torch.__version__)
print("cuda       :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device     :", torch.cuda.get_device_name(0))
    print("memory     : %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))
else:
    print("WARNING: no GPU. Training on CPU takes days - use a Colab GPU runtime.")
""")

md("""
## Step 2: Download the Dataset
""")

code("""
import os

def _get_roboflow_key():
    try:
        from google.colab import userdata
        key = userdata.get("ROBOFLOW_API_KEY")
        if key:
            return key, "colab secret"
    except Exception:
        pass
    key = os.environ.get("ROBOFLOW_API_KEY")
    if key:
        return key, "environment variable"
    raise RuntimeError(
        "ROBOFLOW_API_KEY not found.\\n"
        "In Colab: from google.colab import userdata; "
        "userdata.put('ROBOFLOW_API_KEY', '<your key>')\\n"
        "Or set the ROBOFLOW_API_KEY environment variable."
    )

ROBOFLOW_API_KEY, key_source = _get_roboflow_key()
print(f"Roboflow key loaded from: {key_source}")

from roboflow import Roboflow

WORKSPACE = "ai-ready-program"
PROJECT   = "p-id-object-detection-ai-ready-pomcl"
VERSION   = 1

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
version = project.version(VERSION)
dataset = version.download("yolov8", location="/content/dataset")
print("Downloaded ->", dataset.location)

# Never print the key again.
del ROBOFLOW_API_KEY
""")

md("""
## Step 3: Repair the Dataset

The shipped Roboflow split has three structural defects:

1. **2,194 of 5,462 image files are byte-identical duplicates**, and 498
   appear in two different splits. The duplicate *training* files
   over-weighted the sheets they duplicated — measured with the class ids
   held constant, deduplicating raised mAP50 from 0.8491 to **0.8815**.
2. **7 groups of classes are the same symbol spelled differently.** The
   model cannot win: it must predict the right *spelling* of a symbol it
   cannot otherwise distinguish. `Pneumatic - DiaphragmGate Valve`
   measured **map50 = 0.000** while its twin (3,134 instances) was
   learned perfectly.
3. **66 classes occur in only one split**, so they can never be
   evaluated. One class (`Hand Operated Gate Closed Valve`, 1 box) occurs
   *only* in the validation set — a guaranteed false negative on every
   validation run.

`tools/repair_dataset.py` fixes all three. It is embedded in the next
cell, so this notebook does not depend on the repository being pushed.
""")

# --- embedded tools -------------------------------------------------------- #
b64 = {}
for t in TOOLS:
    b64[t] = base64.b64encode((ROOT / "tools" / t).read_bytes()).decode()

tools_cell_lines = [
    "import base64, pathlib",
    "",
    "# The repository's tools, embedded so this notebook runs standalone",
    "# (no dependency on the repo being committed and pushed).",
    "TOOLS_B64 = {",
]
for k, v in b64.items():
    chunk = [v[i:i + 96] for i in range(0, len(v), 96)]
    tools_cell_lines.append(f'    "{k}": (')
    for c in chunk:
        tools_cell_lines.append(f'        "{c}"')
    tools_cell_lines.append("    ),")
tools_cell_lines += [
    "}",
    "",
    "TOOLS_DIR = pathlib.Path('/content/tools')",
    "TOOLS_DIR.mkdir(parents=True, exist_ok=True)",
    "for name, blob in TOOLS_B64.items():",
    "    (TOOLS_DIR / name).write_bytes(base64.b64decode(blob))",
    "    print('wrote', TOOLS_DIR / name)",
    "",
    "import sys",
    "sys.path.insert(0, str(TOOLS_DIR))",
]
code("\n".join(tools_cell_lines))

code("""
import subprocess, sys

result = subprocess.run(
    [sys.executable, "/content/tools/repair_dataset.py",
     "--src", "/content/dataset",
     "--dst", "/content/dataset_repaired",
     "--report", "/content/repair_report.json"],
    capture_output=True, text=True,
)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr)
    raise RuntimeError("repair_dataset.py failed")

import json
rep = json.load(open("/content/repair_report.json"))
print("classes:", rep["classes_before"], "->", rep["classes_after"])
print("images :", rep["unique_images"],
      "(dropped", rep["dropped_duplicate_files"], "duplicate files)")
print("splits :", rep["splits"])
print("classes with <20 instances:", len(rep["rare_classes"]))
""")

md("""
## Step 3b: Merge a Target-Style Dataset (optional)

If you have drawings in **your** drawing style, upload them here and
they will be merged into the training set with the class ids remapped by
name. This is the only thing that will teach the model your symbol
glyphs — see the note at the top of this notebook.

**Expected format** (a YOLO dataset, zipped):

```
target_style/
  data.yaml
  train/images/*.png
  train/labels/*.txt
  valid/...
  test/...
```

`data.yaml` must list the same class names the repaired dataset uses
(`rep["class_map"]` below shows the mapping). A ready-made starter set
built from `data/sample_pids/plant-pid.png` is described in
`data/target_style/glyph_catalogue.yaml` — the verified glyphs cut
out of `data/sample_pids/plant-pid.png`, which
`tools/augment_target.py` turns into new sheets in the same style with
exact ground truth.

**To skip this step, just run the next cell with `USE_TARGET_STYLE = False`.**
""")

code("""
USE_TARGET_STYLE = False   # set True after uploading the zip below
TARGET_ZIP = "/content/target_style.zip"

if USE_TARGET_STYLE:
    from google.colab import files
    print("Upload your target-style dataset zip:")
    uploaded = files.upload()
    if uploaded:
        next(iter(uploaded.values()))["metadata"].get("name", TARGET_ZIP)
        import shutil
        shutil.copy(next(iter(uploaded.values()))["name"], TARGET_ZIP)

if USE_TARGET_STYLE:
    import zipfile, pathlib
    if not pathlib.Path(TARGET_ZIP).exists():
        raise FileNotFoundError(f"{TARGET_ZIP} not found - upload it first")
    with zipfile.ZipFile(TARGET_ZIP) as z:
        z.extractall("/content/target_style_raw")
    # the zip may contain a single top-level folder
    roots = list(pathlib.Path("/content/target_style_raw").rglob("data.yaml"))
    if not roots:
        raise FileNotFoundError("no data.yaml inside the uploaded zip")
    target_root = roots[0].parent
    print("target dataset root:", target_root)

    result = subprocess.run(
        [sys.executable, "/content/tools/merge_datasets.py",
         "--base", "/content/dataset_repaired",
         "--extra", str(target_root),
         "--dst", "/content/dataset_merged",
         "--extra-repeat", "3",
         "--report", "/content/merge_report.json"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("merge_datasets.py failed")
    DATA_YAML = "/content/dataset_merged/data.yaml"
    DATASET_DIR = "/content/dataset_merged"
else:
    DATA_YAML = "/content/dataset_repaired/data.yaml"
    DATASET_DIR = "/content/dataset_repaired"

print("training on:", DATA_YAML)
""")

md("""
## Step 4: Explore the Repaired Dataset

The previous version of this notebook printed the first 20 class names
and moved on. With 229 classes that hides everything, so this version
prints the **class health table**: how many instances each class has, and
which classes are too rare to learn.
""")

code("""
import glob, yaml
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt

data_config = yaml.safe_load(open(DATA_YAML))
names = data_config.get('names', [])
if isinstance(names, dict):
    names = [names[i] for i in sorted(names, key=int)]
print(f"classes: {len(names)}")
print(f"splits : { {k: v for k, v in data_config.items() if k in ('train','val','test')} }")

def count_instances(root):
    counter = Counter()
    for lbl in glob.glob(f'{root}/**/labels/*.txt', recursive=True):
        for line in open(lbl):
            p = line.split()
            if len(p) == 5:
                counter[int(p[0])] += 1
    return counter

train_counts = count_instances(data_config['train'])
counts = [train_counts.get(i, 0) for i in range(len(names))]

nonzero = [c for c in counts if c > 0]
print(f"\\ninstances per class: min={min(nonzero)} median={sorted(nonzero)[len(nonzero)//2]} max={max(nonzero)}")
print(f"imbalance ratio: {max(nonzero)/max(1,min(nonzero)):.0f}x")
print(f"classes with <50 train instances: {sum(1 for c in counts if c < 50)}")

order = sorted(range(len(names)), key=lambda i: counts[i])
print("\\n--- 20 rarest classes (annotate more of these) ---")
for i in order[:20]:
    print(f"  {counts[i]:6d}  {names[i]!r}")

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(range(len(counts)), sorted(counts))
ax.set_yscale('log')
ax.set_xlabel('class (sorted by frequency)')
ax.set_ylabel('training instances (log)')
ax.set_title('P&ID class frequency - the long tail is the training problem')
plt.tight_layout(); plt.show()
""")

code("""
# Visualise a few training images.
from PIL import Image
train_dir = Path(data_config['train'])
imgs = sorted(train_dir.glob('*.jpg'))[:6] or sorted(train_dir.glob('*.png'))[:6]
if imgs:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, p in zip(axes.flatten(), imgs):
        ax.imshow(Image.open(p).convert('RGB'))
        ax.set_title(p.name[:40], fontsize=8); ax.axis('off')
    plt.tight_layout(); plt.show()
""")

md("""
## Step 5: Train

### Hyperparameters, and why

| Setting | Value | Why |
|---|---|---|
| `imgsz` | **1024** | The dataset has **44 different source image sizes** (225×225 to 1920×1080). At 640 the 1312×1312 sheets are downscaled 0.49× and small symbols vanish. |
| `epochs` | **300** | 229 classes with a 30-instance median needs a long schedule. The old `epochs=100, patience=15` stopped on the long tail. |
| `patience` | **50** | 15 epochs is shorter than the plateau on a 229-class problem. |
| `close_mosaic` | **15** | Mosaic blurs the fine line work of a symbol. Turning it off for the last 15 epochs is the cheapest accuracy win. |
| `mosaic` | 0.5 | Kept, but closed at the end. |
| `scale` | 0.2 | P&ID symbols are scale-sensitive; ±30% is too much. |
| `hsv_*` | 0.01 / 0.1 / 0.2 | Drawings are near-monochrome. |
| `fliplr` / `flipud` | 0.0 | Symbols have an orientation. |
| `degrees` | 5 | Scans are not perfectly level. |
| `erasing` | **0.0** | The default 0.4 *erases* 4% of every image — which deletes symbols. |
| `copy_paste` | 0.0 | Puts symbols on top of each other; not a real P&ID. |
| `auto_augment` | **None** | randaugment is for natural images. |
| `seed` | 42 | Reproducibility. |
| `optimizer` | `auto` | YOLOv8 picks AdamW for small datasets. |
""")

code("""
from ultralytics import YOLO

MODEL = 'yolov8s.pt'          # small: CPU inference on a laptop
EPOCHS = 300
PATIENCE = 50
IMGSZ = 1024
BATCH = 8
SEED = 42

model = YOLO(MODEL)

results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    device=0,
    project='/content/runs',
    name='pid_yolov8s',
    exist_ok=True,

    # ---- schedule ----
    patience=PATIENCE,
    seed=SEED,
    deterministic=True,
    workers=2,
    cache=False,
    save=True,
    save_period=25,
    plots=True,

    # ---- optimisation ----
    optimizer='auto',
    lr0=0.01,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=3,
    cos_lr=True,
    amp=True,

    # ---- augmentation: gentle, and symbol-safe ----
    hsv_h=0.01, hsv_s=0.1, hsv_v=0.2,
    degrees=5,
    translate=0.1,
    scale=0.2,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,          # symbols have orientation
    fliplr=0.0,
    mosaic=0.5,
    close_mosaic=15,     # fine line work needs the last epochs clean
    mixup=0.0,
    copy_paste=0.0,
    erasing=0.0,         # the default 0.4 deletes symbols
    auto_augment=None,   # randaugment is for natural images
)
""")

md("""
## Step 6: Evaluate - Per Class, Not Just the Average

`mAP50 = 0.849` sounds excellent. On the shipped model it was measured
over **only 153 of 229 classes**; 76 classes were never scored, and 498
of the test images were byte-identical to training images. The number
was flattering and misleading. This section reports what you can act on.
""")

code("""
import torch
from ultralytics.utils import checks

run_dir = Path(results.save_dir)
best = run_dir / 'weights' / 'best.pt'

# Attach the training record to the checkpoint. The shipped best.pt had
# NO args, NO epoch counter and NO metrics, so there was no way to tell
# how it had been trained.
ckpt = torch.load(str(best), map_location='cpu', weights_only=False)
ckpt['train_args'] = {k: v for k, v in vars(model.trainer.args).items()}
ckpt['epoch'] = model.trainer.epoch
ckpt['train_metrics'] = getattr(model.trainer, 'metrics', None)
ckpt['date'] = getattr(model.trainer, 'date', None)
ckpt['version'] = ultralytics.__version__
ckpt['class_health'] = {
    'instances_per_class': {names[i]: counts[i] for i in range(len(names))},
    'classes_under_50_instances': sum(1 for c in counts if c < 50),
}
torch.save(ckpt, str(best))
print("re-saved with args:", best, f"{best.stat().st_size/1024/1024:.1f} MB")
""")

code("""
metrics = model.val(
    data=DATA_YAML,
    split='test',
    imgsz=IMGSZ,
    batch=BATCH,
    device=0,
)

box = metrics.box
print('=' * 60)
print(f"mAP50      {float(box.map50):.4f}")
print(f"mAP50-95   {float(box.map):.4f}")
print(f"precision  {float(box.mp):.4f}")
print(f"recall     {float(box.mr):.4f}")
print('=' * 60)
""")

code("""
# Per-class breakdown - the part that tells you what to fix.
per_class = []
for j, ci in enumerate(box.ap_class_index):
    per_class.append({'id': int(ci), 'name': names[int(ci)], 'map50': float(box.p[j])})
per_class.sort(key=lambda r: r['map50'])

evaluated = {r['id'] for r in per_class}
never = [names[i] for i in range(len(names)) if i not in evaluated]
zero = [r for r in per_class if r['map50'] == 0.0]

print(f"classes in vocabulary : {len(names)}")
print(f"classes EVALUATED      : {len(per_class)} ({100*len(per_class)/len(names):.1f}%)")
print(f"classes never evaluated: {len(never)}")
print(f"classes with map50 == 0: {len(zero)}")

print("\\n--- 25 WORST classes ---")
for r in per_class[:25]:
    print(f"  {r['map50']:.3f}  {r['name']!r}")

print("\\n--- 10 BEST classes ---")
for r in per_class[-10:][::-1]:
    print(f"  {r['map50']:.3f}  {r['name']!r}")

if never:
    print("\\n--- classes never evaluated (missing from the test split) ---")
    for n in never[:40]:
        print(f"  {n!r}")

# A class that is rare AND scores zero is an annotation problem, not a
# model problem. This list is your annotation worklist.
print("\\n--- zero-mAP classes ranked by training instances (annotate these) ---")
ranked = sorted((r for r in per_class if r['map50'] == 0.0),
                key=lambda r: counts[r['id']])
for r in ranked[:25]:
    print(f"  map50=0.000  train={counts[r['id']]:6d}  {r['name']!r}")
""")

code("""
from IPython.display import Image, display
import os

for fn in ('results.png', 'confusion_matrix.png', 'PR_curve.png'):
    p = run_dir / fn
    if os.path.exists(str(p)):
        display(Image(filename=str(p), width=900))
""")

md("""
## Step 7: Export & Download the Trained Weights
""")

code("""
import os
from google.colab import files
from ultralytics import YOLO

best = run_dir / 'weights' / 'best.pt'
print(f"Best weights: {best}")
print(f"File size:    {os.path.getsize(best)/1024/1024:.1f} MB")

# Sanity check before it leaves Colab.
check = YOLO(str(best))
print("classes:", len(check.names))
assert len(check.names) == len(names), (
    f"class count mismatch: model has {len(check.names)}, dataset has {len(names)}"
)
print("OK - class count matches the dataset.")

files.download(str(best))
try:
    files.download('/content/repair_report.json')
except Exception:
    pass
""")

md("""
## 📋 Next Steps

1. Download `best.pt`.
2. Copy it to `models/best.pt` in your project.
3. Run the pipeline: `streamlit run app/streamlit_app.py`

### Verify the new model against the old one

```bash
python tools/evaluate_model.py --model models/best.pt --data <dataset_dir> --split test
```

Compare `classes_evaluated` (must be the full vocabulary, not 153) and
`classes_with_zero_map`.

> **Note:** the repaired dataset has **218** classes, not 229 — the
> duplicate spellings were merged. The new checkpoint therefore has 218
> classes, and the old 229-class checkpoint is **not** interchangeable
> with it. `tools/evaluate_model.py` refuses to evaluate a model whose
> class list does not match the dataset's, so a mismatch cannot produce
> a plausible-looking but meaningless mAP.

### If the per-class report still shows zeros

The cause is almost always **annotation count**, not the model. A class
with 3 training instances cannot reach a useful mAP. Feed the
`-- 25 WORST classes --` list back to whoever annotated the drawings and
get each class to at least ~50 instances.

### Dataset Sources
- **Roboflow:** https://universe.roboflow.com/ai-ready-program/p-id-object-detection-ai-ready-pomcl
- **Hugging Face:** https://huggingface.co/datasets/hamzas/digitize-pid-yolo
- **Paper:** Paliwal et al. (2021) "Digitize-PID: Automatic Digitization of P&IDs" (arXiv:2109.03794)
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"provenance": [], "toc_visible": True},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB, {len(cells)} cells)")
print("embedded tools:", ", ".join(TOOLS))
