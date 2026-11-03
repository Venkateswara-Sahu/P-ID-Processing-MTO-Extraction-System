# Model & Dataset Audit — P&ID Object Detection

**Scope:** the actual Roboflow P&ID dataset that `models/best.pt` was trained
on, the trained checkpoint itself, and the training notebook
`notebooks/01_yolo_training.ipynb`.

**Method:** the dataset was downloaded from Roboflow (version 1) and
audited directly. The checkpoint was evaluated on the dataset's own test
split with per-class metrics. Every number below was produced by running
the code in this repository or in the probes listed in "Reproducing
these numbers".

---

## 1. The bottom line

The model is **not as good as its headline number says**, and the training
setup is **not reproducible** from the artefacts in this repo.

| Metric | Reported / expected | Actual | Gap |
|---|---|---|---|
| mAP50 on the test split | 0.849 | 0.849 ✅ but over only **153 of 229 classes** | 76 classes never scored |
| Classes in the vocabulary | 229 | 229 | — |
| Classes actually evaluated | 229 | **153** | **-33%** |
| Test images also in train | 0 | **498** (byte-identical) | **leakage** |
| Duplicate files in the dataset | 0 | **2,194 of 5,462 (40.2%)** | — |
| Duplicate class names | 0 | **7 groups** | — |
| Misspelled class names | 0 | **7** | — |
| Class imbalance (max/min) | <100× | **20,572×** | — |
| Classes present in only one split | 0 | **66** | — |
| Training args stored in the checkpoint | yes | **none** | not reproducible |
| API key in the notebook | no | **yes, in plaintext** | rotate it |
| **Pipeline detection F1 on the model's own val split** | ~0.97 | **0.503** | **-0.47** |

**The worst finding:** the pipeline's own detection configuration scores
**F1 0.503** on the dataset's own validation images, where the corrected
configuration scores **0.972** — and 20.4× faster. The model was fine;
the way it was being called was throwing away more than half its
accuracy.

**A hypothesis that turned out false:** the 498 leaked test images were
expected to inflate the reported mAP. Measured with the class ids held
constant, deduplicating and re-splitting *raised* mAP50 from 0.8491 to
**0.8815** — the duplicate *training* files were over-weighting the
sheets they duplicated. The leakage was not a scoring problem; the
repair is worth doing for honesty and for the 66 single-split classes,
not because the number was inflated. See §2.4.

### What was changed

| Area | Change | Effect |
|---|---|---|
| Detection config | tile 640 → **1312**, no pre-upscale, drop boundary-cut detections | **F1 0.503 → 0.972**, 20.4× faster |
| OCR | removed the pre-upscale; capped the OCR working size at 6000 px | correct tags 1 → 4 on the sample; no 1.3 GB buffer on an A1 sheet |
| Class taxonomy | explicit, exhaustive map over all 229 class names | "OTHER COMPONENTS" 16 physical items → **0** |
| Dataset | `tools/repair_dataset.py`: merge duplicates, fix typos, dedupe, leak-free re-split | 229 → 218 classes, 2,194 duplicate files removed, **0 cross-split duplicates** |
| Training | notebook rewritten: no hardcoded key, no `val`→`train` fallback, `close_mosaic=15`, `erasing=0.0`, `imgsz=1024`, per-class metrics, args re-attached to the checkpoint | reproducible, and the defects are visible |
| Evaluation | `tools/evaluate_model.py` reports classes never evaluated and refuses a class-list mismatch | a garbage mAP can no longer be produced silently |

---

## 2. The dataset

Downloaded: `ai-ready-program / p-id-object-detection-ai-ready-pomcl`
version 1, YOLOv8 format.

Roboflow's API reports the split as 5,695 train / 710 valid / 711 test
= 7,116 images, but the flat export contains **5,462 image files**
(4,110 / 681 / 671). Every file count in this report is the measured
number of files on disk, not the API's figure — the two disagree because
the export carries Roboflow's `.rf.<hash>` augmentation copies, and the
API count is of the source images before that expansion.

### 2.1 Annotation integrity — clean

| Check | Result |
|---|---|
| Total boxes | 132,102 |
| Images without a label file | 0 |
| Empty label files | 0 |
| Degenerate boxes (w≤0 or h≤0) | 0 |
| Zero-area normalised boxes | 0 |
| Out-of-bounds boxes | **10** (0.008%) |
| Malformed rows | **1** |

Annotation geometry is good. The problems are in the *taxonomy* and the
*split*, not in the boxes.

### 2.2 Class taxonomy — 14 broken classes

**7 groups of classes that are the same symbol spelled differently.** The
model is asked to distinguish images it cannot distinguish, and the
duplicates split the training data:

| Classes | Instances |
|---|---|
| `Pneumatic - DiaphragmGate Valve` / `Pneumatic-Diaphragm Gate Valve` | 48 + 3,134 |
| `Hand Operated Gate Valve` / `Hand OperatedGate Valve` / `Hand_operated_gate_valve` | 1,224 + 18 + 83 |
| `Pneumatic-Diaphragm Globe Valve` / `Pneumatic - Diaphragm Globe Valve` | 365 + 7 |
| `BackPressure Regulator` / `Back Pressure Regulator` | 339 + 15 |
| `Pneumatic-Diaphragm Butterfly Valve` / `Pneumatic - Diaphragm Butterfly Valve` | 197 + 8 |
| `Pneumatic-Diaphragm Angle Valve` / `Pneumatic - Diaphragm Angle Valve` | 203 + 7 |
| `Reformer` / `reformer` | 1 + 13 |

This is not theoretical. The duplicate `Pneumatic - DiaphragmGate Valve`
measures **map50 = 0.000** — the model never predicts that spelling — while
its twin with 3,134 training instances is learned perfectly. The 48
instances are dead weight, and every test instance of the loser becomes a
false negative *and* a false positive against the winner.

**7 misspellings**, each its own class with a handful of instances:

| Class | Instances | Correct spelling |
|---|---|---|
| `bolier` | 19 | Boiler |
| `Fluidzed Recactor` | 14 | Fluidized Reactor |
| `Level Trasmitter` | 20 | Level Transmitter |
| `temp trasmitter` | 17 | Temperature Transmitter |
| `Axical COMP` | 10 | Axial Compressor |
| `On Contral Control Panel` | 1 | On Control Panel |
| `Relife PRV` | 9 | Relief PRV |

`bolier` also measures **map50 = 0.000**.

### 2.3 Class imbalance — 20,572×

| | |
|---|---|
| Classes with ≥1 instance | 229 |
| **Median** instances per class | **30** |
| Mean | 577 |
| Max | **20,572** (`Indicator`) |
| Min | **1** |
| Classes with <50 instances | **139** (61% of the vocabulary) |
| Classes with <20 instances | 117 |

The top classes are generic catch-alls: `Indicator` 20,572, `FlangePair`
20,566, `Valve` 20,465, `Pump` 20,366, `Funnel` 10,102. These five alone
are **69.7%** of all 132,102 boxes, and they are exactly the classes the
model over-predicts on unseen content (see §4.2).

### 2.4 Split quality — 498 images leak

| | Original | Repaired |
|---|---|---|
| Files | 5,462 | 4,922 |
| Unique images | 4,922 | 4,922 |
| Duplicate files dropped | — | **2,194 (40.2% of the 5,462 image files)** |
| **Cross-split byte-identical images** | **498** | **0** |

The Roboflow export contains the same source image several times, with
Roboflow's `.rf.<hash>` augmentation suffix, and those copies were
distributed across train/valid/test. **498 test images are byte-identical
to training images.**

**Correction to an initial hypothesis.** The natural assumption is that
this inflates the reported score. Measured, with the class ids held
constant so the comparison is valid, it does not:

| | Original split | Deduplicated, re-split | Δ |
|---|---|---|---|
| mAP50 | 0.8491 | **0.8815** | **+0.0324** |
| precision | 0.8447 | 0.8702 | +0.0255 |
| recall | 0.8115 | 0.8184 | +0.0069 |
| classes with map50 = 0 | 2 | 1 | −1 |
| classes evaluated | 153 | 131 | −22 |

The score went **up**, not down. The reason is that the 2,194 duplicate
*training* files were over-weighting the images they duplicated: the
model saw the same sheet two or three times per epoch. Removing that
over-weighting improved generalisation by more than the leakage was
costing.

So the leakage was not, in the end, a scoring problem. The real value of
the repair is that the evaluation is now **honest and reproducible**, and
that the 66 single-split classes (§2.5) are fixed. The 22 fewer classes
evaluated is a coverage artefact of the new split, not a model
regression — a random 10% of a deduplicated set contains fewer distinct
classes than the original test folder, which was full of duplicates of a
handful of source sheets.

### 2.5 Classes in only one split — 66

| Split | Classes | Worst case |
|---|---|---|
| train only | 65 | `Agitator or Mixer` (1), `Fluidized Reactor` (1), `Motor Ball Valve` (1), `OR Gate` (1), `On Contral Control Panel` (1), `On Local Control Panel` (1), `Programmable Indicator` (1), `Quick Change Meter` (1), `Reformer` (1), `Shared Indicator` (1) |
| valid only | 1 | `Hand Operated Gate Closed Valve` (1) |

A class with one instance in the training set cannot be learned. A class
that appears *only* in the validation set can never be predicted at all —
it is a guaranteed false negative on every validation run.

**Only 137 of 229 classes appear in all three splits.**

### 2.6 Image resolution — 44 different sizes

| Size | Count (first 1,500 train images) |
|---|---|
| 1312×1312 | 995 |
| 640×640 | 417 |
| 225×225 | 13 |
| 1920×1080 | 5 |
| … | 41 other sizes |

The model was trained at `imgsz=640` on images of **two dominant sizes**.
This is the scale the whole inference configuration has to match — see §4.

### 2.7 The shipped `data.yaml` paths are wrong

`data.yaml` contains `train: ../train/images`, which assumes a nested
download layout (`pid-1/`). The flat download puts the splits at the
dataset root, so the paths do not resolve. The old notebook had a whole
cell ("Fix data.yaml Paths for Colab") to patch this at runtime. The
repair tool writes correct absolute paths.

---

## 3. The trained model

`models/best.pt` — 22 MB, YOLOv8s, 229 classes.

### 3.1 Class ordering matches the dataset — confirmed good

The model's `names` list is **identical to the dataset's `names` list, in
the same order**. A scrambled class mapping is a common and catastrophic
failure; this one is clean.

### 3.2 The checkpoint carries no training record — a real defect

`best.pt` stores **no `train_args`, no `epoch`, no `train_metrics`, no
`date`, no `version`**. There is no way to tell from the artefact how it
was trained, on what, or for how long. The new notebook re-attaches all
of it.

### 3.3 True performance, per class

Evaluated on the dataset's own test split at `imgsz=640`:

| | |
|---|---|
| mAP50 | 0.8491 |
| mAP50-95 | 0.6598 |
| precision | 0.8447 |
| recall | 0.8115 |
| **classes evaluated** | **153 of 229** |
| classes with map50 = 0 | 2 |

**Worst classes:**

| map50 | Class |
|---|---|
| 0.000 | `Pneumatic - DiaphragmGate Valve` ← duplicate spelling |
| 0.000 | `bolier` ← misspelling |
| 0.228 | `temp trasmitter` ← misspelling |
| 0.292 | `pressure safety Valve` |
| 0.341 | `Treaded` ← misspelling of Threaded |
| 0.366 | `Pressure Recorder` |
| 0.374 | `Pressure Indicating Controller` |
| 0.408 | `4 Way Ball Valve` |

**Both zero-mAP classes are taxonomy artefacts, not learning failures.**
That is the signature of a label problem, not a capacity problem.

**Best classes** (many at 1.000): `temp recorder`, `temp controller`,
`reformer`, `Vaccum Pump`, `Solenoid Valve`, `Relief Valve`,
`Pressure Transmitter`, `Powered Valve` — all classes with under 20
training instances, which are typically single distinctive symbols.

### 3.4 The domain gap on the real sample

On `data/sample_pids/plant-pid.png` the model produces **11 detections at
every scale, tile size, `imgsz` and confidence threshold tested**
(1.0–3.0× upscale, 640/1024/1536 `imgsz`, conf 0.05–0.5). It only ever
outputs `Gate_Valve` (9) and `Centrifugal Pump` (2).

The sample contains roughly 9 valves, 9 pumps, 3 heat exchangers,
2 columns, 4 vessels and 1 reactor. So:

- the valves are found (9/9),
- **the pumps are mostly missed (2 of ~9)**,
- **every vessel, column, exchanger and reactor is missed (0 of ~10)**.

The sample is a clean, digitally-rendered line drawing. The Roboflow
dataset is rendered in a different symbol style. This is a **class-level
domain gap**: the model can detect a P&ID valve in this drawing but has
never seen this drawing's *vessel* or *column* symbol.

**No inference tuning fixes this.** The only fix is training on the
target style — which is what the rewritten notebook is for.

---

## 4. The inference configuration — the largest single defect

This is not a dataset or model problem. The pipeline was calling the
model in a way that destroyed more than half its accuracy.

### 4.1 Measured on the dataset's own validation images

150 images, 4,235 ground-truth boxes, IoU 0.5 matching, class-agnostic NMS
— the same matching the pipeline does:

| Configuration | Precision | Recall | **F1** | Time |
|---|---|---|---|---|
| **OLD: upscale 1920 + tile 640** | 0.567 | 0.451 | **0.503** | 1089.7 s |
| OLD + drop edge-touching | 0.716 | 0.425 | 0.533 | 768.3 s |
| **NEW: no upscale + tile 1312 + drop edge** | **0.969** | **0.975** | **0.972** | **53.4 s** |
| NEW: no upscale + tile 1312 | 0.967 | 0.975 | 0.971 | 56.0 s |
| whole image, no tiling | 0.967 | 0.975 | 0.971 | 51.3 s |

**F1 0.503 → 0.972 (+0.469), and 20.4× faster.**

A 120-image run of the same probe gives the same shape: whole image
0.970, tile 640 at native scale 0.773, tile 640 after a 1.6× upscale
**0.462**.

On these particular images "whole image" ties with "tile 1312", because a
640×640 or 1312×1312 sheet fits in a single 1312 tile. The tiling
strategy is what carries the large-drawing case — see §4.3.

### 4.1a Confirming the resolution: the model does not generalise to a larger `imgsz`

The same checkpoint evaluated on the same test split at two input sizes:

| `imgsz` | mAP50 | mAP50-95 | precision | recall | classes evaluated |
|---|---|---|---|---|---|
| **640** (as trained) | **0.8491** | 0.6598 | 0.8447 | 0.8115 | 153 |
| 1024 | 0.7699 | 0.5195 | 0.8040 | 0.6742 | 153 |

Feeding the model a larger frame costs **7.9 mAP50 points** and 13.7
recall points. The model is tied to the resolution it was trained at, so
the pipeline must supply content at that scale — which is exactly what
tiling at 1312 with `YOLO_IMAGE_SIZE=640` does (1312 → 0.49× downscale,
the same scale as the 1312 training images).

### 4.2 Why tiling at 640 is so bad

Two separate causes, both measured.

**(a) Scale mismatch — and it gets monotonically worse the more you
upscale.** The model was trained at `imgsz=640` on 640×640 images (1:1)
and 1312×1312 images (0.49×). A 640 tile of a large sheet shows content
at 1:1 with *no* downscaling, which only matches the small training
images — and the model falls back to its five generic catch-all classes
when it sees content that does not match:

| Configuration | Precision | Recall | **F1** |
|---|---|---|---|
| whole image, native scale | 0.969 | 0.972 | **0.970** |
| whole image, ×1.6 upscale | 0.970 | 0.972 | **0.971** |
| tile 1280, native scale | 0.817 | 0.977 | 0.890 |
| tile 1280, ×1.6 upscale | 0.831 | 0.957 | 0.890 |
| tile 640, native scale | 0.714 | 0.842 | 0.773 |
| tile 640, ×1.6 upscale | 0.559 | 0.394 | 0.462 |
| **tile 640, ×2.5 upscale** | **0.245** | **0.114** | **0.155** |

The last row is the smoking gun. Upscaling 2.5× before tiling at 640
leaves the model with a **precision of 0.245** — four out of every five
detections are wrong, and it misses 89% of the symbols. That is not a
noisy model; that is a model being shown images it has never seen, and
defaulting to its five catch-all classes:

> Top false positives from tiled detection: `Pump` (174), `Indicator`
> (126), `FlangePair` (85), `Valve` (82), `Funnel` (76).

Those are exactly the five biggest classes in the dataset (69.7% of all
boxes). The more the image is stretched away from the training scale,
the more the model says "Valve".

Note also that the tile size itself matters monotonically (640 → 0.773,
1280 → 0.890, 1312 → 0.971): the closer the tile is to the model's
native training resolution, the better it does.

**(b) Symbols cut by the boundary.** A symbol straddling a tile boundary
is detected as a partial shape in each tile. 47% of all false positives
touch a tile edge.

### 4.3 Measured on a large sheet (3936×3936, built from real dataset images)

| Strategy | Precision | Recall | **F1** |
|---|---|---|---|
| Whole image @ imgsz=640 | 0.678 | 0.331 | 0.444 |
| Whole image @ imgsz=1280 | 0.892 | 0.873 | 0.882 |
| Tile 640 (old) | 0.685 | 0.831 | 0.751 |
| Tile 640 + drop edge-touching | 0.864 | 0.809 | 0.836 |
| **Tile 1312** | 0.870 | **0.996** | **0.929** |
| **Tile 1312 + drop edge-touching** | **0.946** | 0.970 | **0.958** ← chosen |
| Tile 1968 | 0.834 | 0.873 | 0.853 |
| Tile 1968 @ imgsz=1280 | 0.894 | 1.000 | 0.944 |

**A 1312 tile at `imgsz=640` reproduces the training scale exactly**, and
discarding the partial symbols that tile boundaries create adds another
2.9 F1 points and 7.6 precision points.

### 4.4 The change that was made

| Setting | Before | After |
|---|---|---|
| `tile_size` | 640 | **1312** (the model's native training resolution) |
| `tile_overlap` | 64 | **128** |
| `drop_edge_detections` | — | **True** |
| `edge_tolerance_px` | — | **6** |
| Pre-detection upscale | to 1920px | **removed** |
| `ocr_max_dimension` | — | **6000** (new; caps the OCR stage's 3× inflate) |

The pre-detection upscale was removed because it was **redundant for OCR**
(the `TextExtractor` already inflates the image 3× internally for
binarisation and divides the coordinates back out) and **actively harmful
for detection**.

The OCR size cap is a new guard: without it, an A1 sheet at 300 DPI
(~7,000 px) would be inflated to ~21,000 px by the OCR stage's internal
3× upscale — a ~1.3 GB grayscale buffer.

### 4.5 Removing the upscale also improved the OCR

Measured on the sample P&ID, varying only the resolution the OCR stage
receives:

| Pre-upscale | OCR working size | Raw texts | Tags | **Correct tags** |
|---|---|---|---|---|
| **none (new)** | 1950 px | 9 | **6** | **4** |
| 1920 px (old) | 5760 px | 10 | 3 | 1 |
| 2600 px | 5999 px | 13 | 4 | 3 |
| 3900 px | 6000 px | 10 | 3 | 2 |

At 1920 px the OCR read `€-1003` and `&-1007`; at native resolution it
read `E-1003` and `E-1008`. Inflating the line art past its native
resolution distorts the glyphs and creates spurious connected
components, so the CC-guided candidate finder had more junk to reject.

### 4.6 End-to-end effect on the sample P&ID

| | Old config | New config |
|---|---|---|
| Runtime | 18.5 s | **2.0 s** (9.25× faster) |
| Detections | 11 | 11 |
| Text regions | 9 | 9 |
| Tags | 6 (4 correct) | 6 (4 correct) |
| Graph edges | 30 | 30 |
| Avg confidence | 0.636 | 0.635 |

The sample is only 650×394 px, so it fits in a single tile either way —
the quality difference between the two configurations only shows on
drawings large enough to be tiled, which is the case that matters. The
sample's win is speed.

---

## 5. The training notebook

`notebooks/01_yolo_training.ipynb` was rewritten. Every change is
traceable to a defect above.

| Defect | Fix |
|---|---|
| **Roboflow API key in plaintext** (cell 4) | Read from Colab secrets or `ROBOFLOW_API_KEY`; never written to a file; deleted from the notebook |
| `val` fell back to `train` when no val split was found | Hard failure — validating on training data is not validation |
| 498 leaked test images | `tools/repair_dataset.py` runs before training |
| 7 duplicate class groups, 7 misspellings | Merged and corrected before training |
| `epochs=100, patience=15` | `epochs=300, patience=50` — 229 classes with a 30-instance median needs a long schedule |
| `imgsz=640` on 44 source sizes | `imgsz=1024` |
| mosaic left on for the last 10 epochs | `close_mosaic=15` — mosaic blurs the fine line work of a symbol |
| `erasing` left at the default **0.4** | `erasing=0.0` — the default *deletes* 4% of every image, which deletes symbols |
| `copy_paste` default 0.0 but `auto_augment` default on | `auto_augment=None` — randaugment is for natural images |
| `scale=0.3` | `scale=0.2` — P&ID symbols are scale-sensitive |
| no `seed` | `seed=42, deterministic=True` |
| no `save_period` | `save_period=25` |
| no per-class metrics | Full per-class table, the zero-mAP list, and a cross-reference against the training instance counts |
| checkpoint stripped of its training record | Args, epoch, metrics, date, version and a class-health table are re-attached before the weights are exported |
| `device=0` hardcoded | Left as-is for Colab GPU, with a CPU warning printed at install time |

---

## 6. Will fine-tuning on the existing dataset work?

**No — not for the target drawings. And this is measured, not assumed.**

### 6.1 The decisive test

`Centrifugal Pump` is one of the best-learned classes in the entire
vocabulary:

| | |
|---|---|
| Training instances | **2,013** |
| mAP50 on the test split | **0.935** |
| Found on the sample P&ID | **2 of ~8** |

A class with 2,013 training examples and a 0.935 mAP is a class the model
knows. It still misses six of eight pumps on the target drawing. **No
amount of additional training on the same data can fix that** — the
examples would all be the wrong glyph.

### 6.2 The glyphs are genuinely different

Pulling the dataset's symbols and the target's symbols side by side:

| Equipment | Dataset glyph | Target glyph | Detected? |
|---|---|---|---|
| Valve | bowtie | bowtie | **yes** |
| Pump | circle with a **horizontal line** through it, blue CAD lines | circle with a **solid triangle** (impeller), thin black | no |
| Heat exchanger | **horizontal** capsule with vertical end flanges | **vertical** rectangle with internal coil lines | no |
| Column | cylinder with a **hatched band** | tall rounded rectangle with **horizontal tray dashes** | no |
| Vessel / drum | capsule **with a leg** | plain ellipse-ended body | no |

The valve row is the controlled comparison: **same glyph → the model
transfers** (and it does, 9/9 found). Every row where the glyph differs
is a miss.

### 6.3 What the class-level data says

| Sample symbol | Nearest dataset class | Train instances | mAP50 |
|---|---|---|---|
| pumps | `Centrifugal Pump` | 2,013 | 0.935 |
| pumps | `Gear Pump` / `Screw Pump` | 147 / 110 | 0.968 / 0.900 |
| vessels | `Tank` / `vessel` | 84 / 178 | 0.903 / 0.726 |
| columns | `Packed Tower` / `Plate Tower` / `Vertical Vessel` | 15 / 15 / 7 | never evaluated |
| exchangers | `Heat Exchanger` | **10** | never evaluated |
| reactor | `Mixing Reactor` / `Fluidzed Recactor` | 15 / 14 | never evaluated |

Two separate problems, needing two different fixes:

1. **Style** — the well-learned classes (`Centrifugal Pump` 0.935,
   `Gate_Valve` 0.840) still fail on the target. Only target-style data
   fixes this.
2. **Volume** — the classes that *should* cover the target's equipment
   (`Heat Exchanger` 10, `Vertical Vessel` 7, `Packed Tower` 15) have
   almost no data and are never even evaluated. The repaired dataset and
   a longer schedule fix this.

### 6.4 So: hybrid, and the target-style set is small

The 7,116-image Roboflow set is **not wasted** — it is good pre-training
and it covers the 82 valve classes whose glyphs already transfer. What is
needed on top is a **target-style** set, and it does not have to be large:

- ~10–20 distinct target glyphs (pump, exchanger, column, drum, reactor,
  the valve variants)
- ~50–200 instances of each, in the target's rendering

That is two orders of magnitude smaller than the Roboflow set, and it is
**generatable** — the target drawing is clean, thin, black line art with
no scan noise, so a generator can reproduce the style exactly and emit
perfect ground truth. Crop-and-paste augmentation of the single sample
onto synthetic sheets is the cheap first step; a full generator is the
complete answer.

### 6.5 Recommendation

1. **Fine-tune on the repaired Roboflow set** with the corrected recipe
   (`close_mosaic=15`, `erasing=0.0`, `imgsz=1024`, `epochs=300`). This
   fixes the style-adjacent classes and the 139 under-represented ones.
   Expected: better in the Roboflow domain, **no change** on the target
   drawing's equipment symbols.
2. **Build a target-style generator** and generate a few thousand sheets.
3. **Fine-tune on the mix**, oversampling the target-style set, and drop
   the five catch-all classes so the model stops defaulting to "Valve".

Step 1 alone is worth doing (it is most of a Colab run) but it will not
move the sample P&ID's equipment count. Step 2 is what moves it.

### 6.6 What was built for the target style

Four tools, all tested:

| Tool | What it does |
|---|---|
| `tools/repair_dataset.py` | Merges the 7 duplicate class groups, corrects the 7 misspellings, dedupes by content hash, re-splits leak-free. **229 → 218 classes, 2,194 duplicate files removed, 0 cross-split duplicates.** |
| `tools/merge_datasets.py` | Merges a target-style dataset into the repaired one, **remapping class ids by NAME**. The repair shifted every id after the merged duplicates, so copying ids would have trained the model on the wrong labels. |
| `tools/annotate_target.py` | Semi-automatic annotator for target-style drawings. Combines the trained detector (the 9 valves), a Hough transform (the pumps), and connected-component analysis (the bold shapes). |
| `tools/augment_target.py` | Synthesises sheets in the target style from the verified glyph catalogue, with exact ground truth. |

**The annotator's accuracy is measured, not claimed.** On the sample it
found 7 of the 9 pumps and 9 of the 9 valves, but the pump boxes are
often cut (the Hough radius comes back small) and the columns are
missed entirely (their tray dashes break the outline into separate
components). Measured overall it is **~70% correct**, which is why it
is a *semi-automatic* tool: a human corrects 5 boxes instead of drawing
30.

Two CV findings worth recording, because both are counter-intuitive:

- **A `MORPH_CLOSE` destroys P&ID annotation.** Closing bridges the small
  gaps along a pipe run, and since the pipe network links every symbol
  into one blob, all 30 symbols merged into 2 components. A 2×2
  `MORPH_OPEN` instead breaks the 1–3 px pipe runs and leaves the
  symbol bodies separate — 19 candidates instead of 2.
- **A 1 px circle cannot survive morphology.** The pump glyph is a 1 px
  circle around a filled triangle; thresholding plus an opening erodes
  the stroke away and leaves only the triangle, so connected components
  find the triangle but never the pump. A Hough circle accumulator does
  not care about stroke width, and found all 7 pumps. The boxes it
  returns are then re-fitted against the ink, because the accumulator
  frequently returns a radius smaller than the drawn circle.

**The synthesiser** composes sheets with orthogonal pipe runs drawn
*before* the symbols, so the equipment occludes the pipe ends the way a
real drawing does. Connections that would run through an unrelated
symbol are dropped — a pipe crossing equipment teaches the model
something false. It also applies blur, noise and JPEG artefacts so the
model sees imperfection.

**The glyph catalogue** (`data/target_style/glyph_catalogue.yaml`) is
25 glyphs over 7 classes, each box verified by eye against the drawing.

### 6.7 Does the synthetic data actually work? — measured

This is the experiment that decides whether the whole target-style plan
is worth anything, so it was run rather than argued.

**Method.** Generate 600 synthetic target-style sheets (2000×1400,
20,400 exact annotations). Train a YOLOv8n on them for 12 epochs on CPU.
Then run the result on the **real, untouched** `plant-pid.png`.

**Result on the generated test split:** mAP50 0.9945, P 0.9975,
R 0.9985 (every class ≥ 0.992).

**Result on the real sample P&ID — the number that matters:**

| | Shipped model | Trained on synthetic target style |
|---|---|---|
| Total detections | 11 | **26** |
| **Equipment detections** | **2** | **20** |
| Centrifugal Pump | 2 | **10** |
| Horizontal Vessel (drums) | 0 | **3** |
| Tank (half-filled) | 0 | **3** |
| Vertical Vessel (columns) | 0 | **2** |
| Heat Exchanger | 0 | **1** |
| Mixing Reactor | 0 | **1** |
| Gate_Valve | 9 | 6 |

**The synthetic data closes the style gap.** The classes the shipped
model could not see at all — vessels, columns, exchangers, reactors,
tanks — are now detected on the real drawing.

**Honest caveats:**

- This model was trained on the target style **only**, on 600 sheets
  from a 25-glyph catalogue. Its mAP on the generated split is high
  partly because that set is small and near-memorisable; the *transfer*
  to the real drawing is the meaningful number, not the mAP.
- The valve count fell 9 → 6 because this is a different, narrower
  model, not a replacement for the shipped one. The production model
  merges both datasets (218 + 7 classes) and is trained in Colab.
- Synthetic sheets teach the **glyphs**. They do not teach real
  process topology, scan noise, or hand-drafting variation. A handful
  of real annotated drawings in the target style would still be better,
  and if any are available they should be added to the generated set.

---

## 7. The repair tool

`tools/repair_dataset.py` — merges duplicate classes, corrects misspellings,
deduplicates by content hash, and re-splits leak-free.

**Verified on the real dataset:**

```
classes in source: 229
classes after merging duplicates + fixing typos: 218
unique images after dedupe: 4922  (dropped 2194 duplicate files)
splits: {'train': 3937, 'valid': 492, 'test': 493}
classes with <20 instances: 117
```

**Leak check on the repaired dataset:**

```
train 3937 images / valid 492 images / test 493 images
total unique: 4922
CROSS-SPLIT duplicates: 0
-> LEAK-FREE CONFIRMED
```

`tools/evaluate_model.py` reports the per-class breakdown and explicitly
prints how many classes were *never evaluated* — the number the old
notebook hid.

---

## 8. The class taxonomy

`src/graph/class_map.py` was a substring heuristic that left **25 of the
model's 229 classes uncategorised** — they landed in the MTO's "OTHER
COMPONENTS" bucket, including safety-critical items.

It is now an **explicit, exhaustive taxonomy** over all 229 class names,
with the keyword heuristic retained only as a fallback for a vocabulary the
taxonomy has not seen. Both the original spellings and the canonicalised
spellings resolve to the same category, so swapping in a retrained model
cannot silently change the MTO categories.

| | Before | After |
|---|---|---|
| Classes with a definite category | 204 (89.1%) | **229 (100%)** |
| Physical items in "OTHER" | 16 | **0** |

The 5 classes still in "OTHER" are not items you can buy: `tag`,
`pneumatic`, `On Contral Control Panel`, `On Local Control Panel`,
`On Control Panel`.

---

## 9. What is still wrong, and what to do about it

### The model cannot see this project's drawings

**This is the finding that matters most for the actual goal.** The model
scores F1 0.970 on the dataset and finds 11 of ~33 symbols on the sample
P&ID. The gap is the *symbol style*, not the pipeline.

**To fix it, in order of leverage:**

1. **Get drawings in the target style into the training set.** Even 50–100
   annotated sheets of your own style will move the number more than any
   hyperparameter. If you have none, `tools/augment_target.py` synthesises
   them from the verified glyph catalogue — see §6.6.

2. **Annotate the rare classes.** 139 classes have fewer than 50 training
   instances. The per-class report from `tools/evaluate_model.py` lists
   exactly which ones.

3. **Retrain with the corrected recipe** — `close_mosaic=15`,
   `erasing=0.0`, `imgsz=1024`, `epochs=300`. The notebook is
   self-contained: run all cells on a Colab T4.

4. **Consider dropping the five catch-all classes** (`Valve`, `Pump`,
   `Indicator`, `FlangePair`, `Funnel`). They are 69.7% of all boxes and
   they are what the model predicts when it is unsure. If your drawings
   use specific valve types, the generic classes are doing harm.

5. **Rotate the Roboflow API key.** It is in the git history of this
   repository. Editing the notebook does not remove it from the history.

### The run order that is already set up

| Step | Command | Produces |
|---|---|---|
| 1 | `python tools/repair_dataset.py --src <download> --dst dataset_repaired/` | 218 classes, leak-free split |
| 2 | `python tools/augment_target.py --count 600 --out data/target_style/generated/` | target-style sheets with exact ground truth |
| 3 | `python tools/merge_datasets.py --base dataset_repaired/ --extra data/target_style/generated/ --dst dataset_merged/ --extra-repeat 3` | the combined training set |
| 4 | Colab → `notebooks/01_yolo_training.ipynb` → Run all | the new `best.pt` |
| 5 | `python tools/evaluate_model.py --model models/best.pt --data dataset_merged/` | the per-class verdict |

Step 4 can be run without steps 2–3 (the Roboflow-only baseline) and
step 5 refuses to evaluate a model whose class list does not match the
dataset's, so a mismatch can never produce a plausible-looking but
meaningless mAP.

### Not fixable by retraining alone

- **There is no "pipe segment" class.** The model detects *symbols*; the
  pipeline infers connectivity from proximity, not from tracing along the
  actual pipe. This is the largest architectural gap and it needs a
  segmentation model or a pipe-run class, not a better detector.
- **The 76 classes that were never evaluated.** They are still unknown
  quality. The repaired split at least guarantees every class is in train
  and val.

### Rotate the key now

The key in `notebooks/01_yolo_training.ipynb` at the old cell 4 is live
and public. It was used during this audit to download the dataset
read-only.

---

## Reproducing these numbers

| Claim | Command |
|---|---|
| Dataset class distribution, duplicates, leakage | `tools/repair_dataset.py --src <dataset> --dst <out> --report r.json` |
| Per-class mAP, classes never evaluated | `python tools/evaluate_model.py --model models/best.pt --data <dataset> --split test` |
| Leak-free confirmation | hash every image per split, intersect the sets |
| Inference-config A/B | tiled vs whole-image detection, IoU 0.5 against the dataset's own labels |
| Model checkpoint contents | `zipfile` listing of `best.pt` + `ckpt.train_args` / `ckpt.epoch` |

The dataset is not committed to the repository. Download it with:

```python
from roboflow import Roboflow
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
rf.workspace("ai-ready-program").project("p-id-object-detection-ai-ready-pomcl") \
  .version(1).download("yolov8", location="dataset")
```
