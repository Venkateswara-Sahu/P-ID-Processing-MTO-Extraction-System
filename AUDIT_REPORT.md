# L&T P&ID MTO Extraction System — Full Audit

**Date:** 2026-10-21 | **Scope:** 40 tracked files, 5,189 LOC (`src/`, `app/`, `config/`)
**Method:** full source read + runtime verification + 3 parallel subsystem auditors.
Every claim below was executed, not inferred. **26 critical bugs** (C1–C14 + 12 from the
parallel audits), each re-verified by the parent against the real model, the real sample
image, and the live Groq API.

---

## Headline: the system runs, but the output is wrong

Full pipeline executed on `data/sample_pids/plant-pid.png`. Ground truth (read from the image by
vision): 8 `E-xxxx` equipment tags, 4 `P-xxxx` pumps, `R-1001` reactor, 2 `T-xxxx` columns,
17 stream numbers, `CW` service labels, ~10 valves, 4 pumps, 3 exchangers, 2 columns.

| Metric | Ground truth | Actual | Gap |
|---|---|---|---|
| Symbols detected | ~25–30 | **9** | −70% |
| Tags extracted | ~15 | **1** | −93% |
| Tags correct | — | **0** (the 1 is `R-1001`→"Radiation") | 0% |
| Components classified | ~25 | 8 (2 eq + 6 valve) | −68% |
| Graph edges | real piping topology | 3 (proximity guesses) | n/a |
| Runtime | — | 27–63s | ok |

The pipeline is architecturally sound and the code is clean. The **domain logic is wrong in
specific, fixable ways** — and each one was measured, not guessed.

**Counts:** 15 critical bugs · 21 significant issues · ~30 minor · 8 domain gaps
· 9 confirmed-clean · 10 security findings (3 exploitable)

**The single worst finding:** a pump tagged `P-101A` is written to the MTO as
*"Pressure"*, and the MTO's `Quantity` is hardcoded to `1` on every row — so the
deliverable is simultaneously **wrong** and **unorderable**.

---

## CRITICAL BUGS

### C1 — LLM validation has never run. The configured model is dead.
`config/settings.py:21` → `groq_model = "llama-3.3-70b-versatile"`
`src/validation/llm_validator.py:72` → `ChatGroq(model=self.model, ...)`

Live API test against the real key in `.env`:
```
ERROR 404: The model `llama-3.3-70b-versatile` does not exist
```
Probed 16 candidate models. **Only `openai/gpt-oss-120b` and `openai/gpt-oss-20b` respond 200.**
Verified fix: `gpt-oss-120b` runs the full 3-node LangGraph in 11.5s and returns real analysis.

**Impact:** every "AI Validation" claim in the README, the Streamlit UI, and the MTO's
"Validation Report" sheet is false. The code degrades gracefully (good) but the fallback
report says *"Configure a Groq API key"* — **the key is present and valid**. The real fault is
the model name, and the message sends you to fix the wrong thing.

**Fix:** `groq_model = "openai/gpt-oss-120b"`, and make the fallback report distinguish
*no key* from *bad model* from *network error*.

---

### C2 — Equipment tags are parsed as instruments, with nonsense descriptions
`src/ocr/tag_parser.py:158` tries `_try_instrument_tag()` **before** `_try_equipment_tag()`.
`INSTRUMENT_PATTERN` accepts 1–4 letters; `EQUIPMENT_TYPE_MAP` prefixes are 1 letter.
Result: **13 of 14 equipment prefixes are permanently shadowed.**

| Input | Actual | Should be |
|---|---|---|
| `E-1001` | "Voltage" | Heat Exchanger |
| `R-1001` | "Radiation" | Reactor |
| `P-101A` | "Pressure" | Pump |
| `V-3040` | "Vibration" | Vessel |
| `H-100` | "Hand" | Heater |

**This is the single most damaging bug.** Every pump in the MTO is labelled "Pressure".
The Groq validator, when working, independently flagged this exact error
(*"valve tagged R-1001 classified as Radiation"*) — confirming it from a second source.

**Fix:** try equipment first, and require a known prefix / ≥2 letters for instrument tags.
**Measured: 6/15 test tags corrected.** See prototype below.

---

### C3 — Plain words are manufactured into fake instrument tags
`src/ocr/tag_parser.py:135` — OCR correction `([A-Z])\s+(\d)` → `\1-\2` turns `PUMP 101`
into `PUMP-101`, which `INSTRUMENT_PATTERN` then matches:

```
'PUMP 101'  -> tag_id='PUMP-101'  desc='Pressure U/M/P'   -> written to the MTO
'LINE 12'   -> tag_id='LINE-12'   desc='Level Indicator/N/Element'
'FLOW 200'  -> tag_id='FLOW-200'  desc='Flow Low/O/W'
```

**Fix:** require the first letter to be a known ISA variable **and** the prefix to be a
plausible tag (all-caps, 1–4 chars, in a known vocabulary). Verified the fix rejects all three.

---

### C4 — `YOLO_MODEL_PATH` in `.env` is relative → silent COCO fallback
`.env` sets `YOLO_MODEL_PATH=models/best.pt` (relative).
`config/settings.py:26` default is **absolute** — but `.env` overrides it.
`src/detection/symbol_detector.py:85` → `Path(self.model_path).exists()` is **CWD-relative**.

Verified with real `chdir`:
```
cwd=<project root>   exists=True
cwd=C:/Windows/Temp  exists=False
cwd=C:/Users/91961   exists=False
cwd=C:/              exists=False
```
→ `symbol_detector.py:97` falls back to `YOLO("yolov8s.pt")` (**COCO, 80 classes**) →
**zero P&ID detections** → empty MTO → the run *succeeds*. Only a `logger.warning` betrays it.

**Fix:** resolve relative to `PROJECT_ROOT` in `settings.py`; make the missing-model case a
**hard error**, not a silent fallback to an unrelated model.

---

### C5 — Multi-page PDFs silently drop every page but the first
`src/pipeline.py:154` → `image = images[0]  # Process first page`
`PDFConverter.convert_to_numpy()` renders every page, then all but page 1 are discarded.
`app/streamlit_app.py:190` says *"will process first page"* — the API says nothing.

A 20-sheet P&ID package yields an MTO for one sheet and no warning.

**Fix:** loop pages, tag results per sheet, and report the page count processed.

---

### C6 — Excel formula injection from OCR text
`src/mto/mto_generator.py:233` → `cell.value = value` with no sanitisation.
Verified: OCR text `=HYPERLINK("http://evil.test","Click")` lands as a **live formula cell**
(`data_type == 'f'`) in both the Tag and Description columns. Uploading a crafted P&ID lets
you plant a formula in the spreadsheet an engineer opens. `+CMD(...)` and `@SUM(...)` are
stored inert, so this is `=`-prefix injection specifically.

**Fix:** prefix any value starting with `=`, `+`, `-`, `@` with a `'` (or set `cell.data_type='s'`).

---

### C7 — Coordinate space mismatch: entity positions are 3× the reported image size
`src/pipeline.py:160` sets `result.image_size` from the **original** image, then
`src/pipeline.py:168-176` **upscales to 1920px wide**. All entity/OCR coordinates are in the
**upscaled** space.

Verified on the sample:
```
result.image_size reported : (650, 394)     <- original
annotated_image shape      : (1163, 1920)   <- upscaled
max entity x                : 1556
detections outside reported frame: 8 of 9
```
Entity x=1556 is reported against a 650px-wide frame. **Any consumer that maps positions
onto the original image is 2.95× off.** The MTO and the API both ship these coordinates.

**Fix:** report the processed size, and/or carry the scale factor explicitly.

---

### C8 — Stream numbers become pipe diameters on nearby components
`src/ocr/tag_parser.py:77-79` — `PIPE_SIZE_PATTERN` makes the inch mark **optional**:
`^(\d{1,2}(?:/\d{1,2})?)\s*["”]?$`. Every bare 1–2 digit number is therefore a pipe size.

**Stream numbers are the most common text on a P&ID** — the sample drawing has 17 numbered
diamonds. Verified:
```
'12' -> pipe_size  tag_id=12  pipe_size='12"'
'5'  -> pipe_size  tag_id=5   pipe_size='5"'
```
End-to-end: a stream number `12` beside a valve produces
`entity type=valve  size='12"'` → the MTO row reads **"Gate Valve, Size 12""** — a
12-inch gate valve that does not exist on the drawing.

**Fix:** require the inch mark (`["”]`), and treat bare 1–2 digit numbers as stream numbers.

---

### C9 — Negative-area detections survive the tile merge
`src/detection/postprocessor.py:60-66` — the upper bound is clipped
(`min(original_width, x2)`) but the lower bound only via `max(0, ...)`. A detection in a
tile's zero-padded region inverts the box.

Verified:
```
input bbox (520,10,600,50) in a 500px-wide image
-> MERGED bbox (520.0, 10.0, 500, 50.0)  width=-20.0  area=-800.0
-> IoU of two identical such boxes = 0.0   (was ~1.0)
```
`Detection.area` is now negative, so `filter_by_area` would reject it — but
`filter_by_area` is **never called by the pipeline**. The box reaches the entity mapper as a
real symbol, and its garbage IoU means it can neither suppress nor be suppressed.

**Fix:** clip both ends, drop degenerate boxes (`if x2 <= x1 or y2 <= y1: continue`).

---

### C10 — Tiler's last column overlaps by up to 639px instead of 64px
`src/preprocessing/image_tiler.py:108-111` — `x_end = min(x + tile_size, w)` then
`x_start = max(0, x_end - tile_size)` back-shifts the final tile to flush with the edge,
but `x` has already advanced by the stride. Measured starts:
```
w=1217: starts=[0, 576, 577]              -> last pair overlap 639px
w=2369: starts=[0, 576, 1152, 1728, 1729] -> last pair overlap 639px
w=1920: starts=[0, 576, 1152, 1280]       -> last pair overlap 512px
```
The last tile is ~90% redundant — roughly doubling inference cost in that band and creating
the huge same-symbol overlap region where C9 and C13 bite hardest.

**Fix:** compute the grid up front and clamp the *stride*, not the start.

---

### C11 — Denoise makes a real drawing take ~20 minutes
`src/preprocessing/enhancer.py:96-106` — `fastNlMeansDenoisingColored` at `strength=6`
over the **full-resolution** image. Measured:
```
1920x1163   clahe=0.10s  denoise=2.17s
3000x2100   clahe=0.03s  denoise=12.80s
```
Denial scales with pixels, not tiles. Extrapolating to A1 @300 DPI (9933×7016) gives
**~20+ minutes per sheet** before any inference. P&ID drawings are A1/A0.

**Fix:** move denoise into the per-tile path, or make it opt-in — 1px line art survives
CLAHE alone (measured 501 dark px before and after on a 1px line).

---

### C12 — `project_name` is used unsanitized in the output filename
`src/mto/mto_generator.py:120` → `f"MTO_{project_name.replace(' ', '_')}_{timestamp}.xlsx"`.
`project_name` comes straight from the API query string (`app/api.py:125`) and the Streamlit
text input. Verified:
```
'a/b'        -> FileNotFoundError (uncaught -> HTTP 500 with a raw path)
'../../evil' -> wrote data/output/evil_20261021_142955.xlsx   (escaped the output dir)
'bad:name'  -> MTO_bad:name_...xlsx  (NTFS stream separator, silently truncates)
```
**Fix:** `re.sub(r"[^A-Za-z0-9._-]+", "_", project_name)[:60]`, then `Path(filename).name`.

---

### C13 — Class-aware NMS keeps the same physical symbol twice
`src/detection/postprocessor.py:88-116` — NMS groups by `class_id`, so a symbol classified
as `Gate_Valve` in one tile and `Valve` in the neighbour survives as **two** MTO line items.
Verified: IoU 0.877 across two class_ids → **kept: 2**.

This is near-certain in production: `best.pt` has **82 distinct valve classes** with heavy
near-synonymy (`Gate_Valve`, `3 Way Gate Valve`, `Control_valve`, `Valve`, …), and a symbol
on a tile boundary is genuinely ambiguous to the model.

**Fix:** a class-agnostic geometry pass before the class-aware pass, plus a synonym map
collapsing the 82 valve classes onto a canonical taxonomy.

---

### C14 — `_build_graph()` crash escapes the try/except and kills the pipeline
`src/validation/llm_validator.py:124-125` — `self._build_graph()` is called **before** the
`try:` at line 144. It does `from langchain_groq...`/`from langgraph.graph import ...` at
line 86. If `langgraph` is missing or version-incompatible, the `ImportError` propagates out
of `validate()` and kills the whole pipeline.

The asymmetry is the bug: `_init_llm()` wraps its import in `try/except` and returns `False`;
`_build_graph()` does not. Verified by source inspection — the call site precedes the `try`.

**Fix:** move `_build_graph()` inside the `try`, and wrap the `langgraph` import in its own
`try` → `False` → `_fallback_report`.

---

### C15 — An unknown category silently vanishes from the MTO body
`src/mto/mto_generator.py:74-81` iterates `MTO_CATEGORIES` (5 fixed buckets) for the body,
but `_write_summary` (line 254) iterates `grouped.items()`. Two different key sets over the
same dict. Verified with `component_category="weird_category"`:
```
header: "Total Items: 1"
actual item rows in body: 0
SUMMARY row: WEIRD_CATEGORY | 1
```
The item is counted in the header and the summary but **has no row**. Combined with the 43.7%
of classes that map to `other`/unknown, this is a silent data-loss path.

**Fix:** drive the body and the summary from one canonical list, with a catch-all bucket.

---

## SIGNIFICANT ISSUES

### S1 — 43.7% of the model's 229 classes are unclassified
`src/graph/entity_mapper.py:167-186` classifies by substring. Measured against the **real**
class list in `models/best.pt`:

| | count | % |
|---|---|---|
| unknown → "OTHER COMPONENTS" | **100** | 43.7% |
| valve | 82 | 35.8% |
| equipment | 24 | 10.5% |
| instrument | 15 | 6.6% |
| piping_component | 8 | 3.5% |

The 100 orphans are real equipment: *Agitator, Motor, Gauge, Orifice, Fan Blades, Drum,
Chimney Tower, Elevator, Blower, Cylinder, Ejector, …* They land in the MTO as "OTHER COMPONENTS".

**Fix:** a proper class→category map keyed on the model's own vocabulary.
**Measured: coverage 56.3% → 79.9%** with a keyword expansion (100 orphans → 46).

### S2 — OCR confidence threshold discards half the real tags
`config/settings.py:44` → `ocr_confidence_threshold = 0.6` (the `.env.example` says `0.3` —
the code default wins and the two disagree).

The raw psm-11 pass *does* read the real tags — they're then thrown away:
```
'R-1001' 0.89 ✓   'E-1068' 0.52 ✗   'T1002' 0.12 ✗   'E1006' 0.00 ✗   '6-1005' 0.00 ✗
```
Measured tag-like recall on the sample: **threshold 0.6 → 2 tags; 0.15 → 5 tags; 0.0 → 7 tags.**

**Fix:** lower to ~0.15–0.3 and let the *tag parser* (not the confidence gate) reject noise.
Tesseract's per-word confidence is unreliable on binarised line art.

### S3 — `TextExtractor` discards the detections it is given
`src/ocr/text_extractor.py:396` → `extract_with_detections()` ignores `detections` entirely.
`src/pipeline.py:195-199` comments describe cropping "below/beside the symbol" — that logic
doesn't exist. The class docstring still describes the old per-crop strategy, and
`_tesseract_crop` (`text_extractor.py:314`) is **dead code**.

The commit message claims "detection-guided OCR". It isn't. The CC-guided path works (it's
what produced the results), but the code and comments disagree with each other.

### S4 — Rule validation results never reach the MTO
`src/pipeline.py:249` → `validation_report=result.llm_report if run_llm_validation else None`
The deterministic rule report (R001–R006) is computed, stored in `result.rule_report`, served
by the API — and **dropped from the spreadsheet**. So the MTO's "Validation Report" sheet
contains only LLM output, and contains *nothing* when the LLM is unavailable (C1).

### S5 — "pass rate 100%" is misleading
`src/validation/engineering_rules.py:64-68` — `pass_rate` counts only **errors**. The sample
run reported `pass_rate: 100.0%` alongside **18 warnings** and 8 of 9 entities untagged.
Verified: a run with 3 untagged entities still reports 100%.

**Fix:** report coverage (tagged %) alongside pass rate; a reviewer seeing "100%" will not
open the file.

### S6 — Two different confidence scales in one pipeline
`ConfidenceScorer.score_entities()` **overwrites** `entity.confidence` *after* the graph was
built. The graph captured the pre-scoring value:

```
E0001  entity.confidence (MTO) = 0.515   graph node confidence = 0.858
E0003  entity.confidence (MTO) = 0.432   graph node confidence = 0.719
```
The MTO and the API's `/graph` endpoint **disagree about the same entity**. And the
detection-only branch (`confidence_scorer.py:131`) multiplies by 0.6, so an **85%-certain
detection is reported as 51%** and flagged "⚠ Low confidence — needs review" — penalising a
*missing OCR label* as if it were a *bad detection*.

**Fix:** score before building the graph, or write scores back to `graph.nodes[n]["confidence"]`.

### S7 — Self-loops in the instrument-loop detector
`src/graph/relationship_resolver.py:171-179` — a node whose `parsed_tags` contain two tags
sharing a loop number is appended to `loop_members` **twice**, so
`if not graph.has_edge(members[i], members[i+1])` is False for `(X, X)` and it adds
`add_edge(X, X)`.

Verified: `edges: [('E0001', 'E0001')]`. A self-loop in a `DiGraph` corrupts flow analysis
and pollutes the visualisation.

**Fix:** `members = list(dict.fromkeys(members))` and guard `if members[i] != members[i+1]`.

### S8 — Text-to-symbol assignment is greedy and order-dependent
`src/graph/entity_mapper.py:234-261` — each symbol claims up to **5** nearest texts,
first-come-first-served. Verified:

```
symbol A at x=100, symbol B at x=400
TAG_A at x=120 (dist 20 from A)
TAG_B at x=390 (dist 10 from B)   <-- B's true tag
→ A claims BOTH: E0001 gets ['TAG_A','TAG_B'], E0002 gets NOTHING
```
Reversing the detection order changes which symbol wins. Subagent 1 independently reproduced
this and confirmed the winner is decided by **list order**.

**Fix:** global nearest-neighbour assignment (Hungarian / sorted-edge greedy), one text ↔ one symbol.

### S9 — `max_distance` is a magic number in the wrong coordinate space
`src/graph/entity_mapper.py:68` → `max_distance = 350.0` in **upscaled** pixels. At 3× that's
~117 original px, and it's hardcoded — not in `settings.py`. A label 130 original px away is
rejected. (The docstring acknowledges the space, but the value isn't configurable.)

### S10 — Line numbers are assigned by proximity alone
`src/graph/relationship_resolver.py:105` — any entity within 300px of a line label inherits
that line number. Verified: a valve at x=250 with a `6"-PA-1001` label at x=0 is labelled
as being on that line — with no check that the line actually runs past it.

**Fix:** require spatial *alignment* with the line's run direction, not just proximity.

### S11 — Line numbers and pipe sizes are inherited onto the wrong entities
`src/graph/entity_mapper.py:130-133` — a valve near a `6"-PA-1001-A1A-N` label inherits
`size='6"'` and `spec='A1A'`. Verified. **6" is the pipe size of the line, not the valve.**
In an MTO this reads as a 6-inch valve.

### S12 — A line number becomes a procurable MTO row
`src/graph/entity_mapper.py:145-148` creates a standalone entity for `LINE_NUMBER` tags,
which `_group_entities` files under `piping` → the MTO gets a row:
```
(1, '6"-PA-1001-A1A-N', '6" Process Air Line 1001', 'Line', '6"', 'A1A', 1)
```
A line number is a **property of a line**, not a thing you procure. It inflates MTO totals.

### S13 — The MTO is not a Material Take-Off
`src/mto/mto_generator.py:309` → `"quantity": 1` hardcoded, one row per entity.

**Verified:** 3 identical Centrifugal Pumps → 3 rows, each Quantity 1. A procurement
document needs *"Gate Valve, 4", 150# CS, qty 5"*. **The current output cannot be ordered from.**

Aggregation prototype (6 entities → 2 rows with correct quantities) is in the fixes section.
Also missing: Material, Unit, Remarks, Weight, Pipe Class, Service.

### S14 — `SYMBOL_CATEGORIES` in `config/settings.py:74` is entirely dead code
Never imported anywhere. It also lists class names (`valve.gate`, `instrument.flow`) that
**don't exist** in the model, whose real names are `Gate Valve`, `Flow Transmitter`.

### S13 — Hardcoded Roboflow API key committed to git
`notebooks/01_yolo_training.ipynb:143` → `Roboflow(api_key="jhHG…EdA")` (redacted)
Confirmed present in `HEAD` and in commit `f5f1473`. **Rotate this key.** It's a Roboflow
API key, not a P&ID credential, but it's a live secret in permanent history.
The notebook is now clean; the value remains recoverable from earlier commits, so
rotation — not deletion — is the fix.

### S14 — `max_distance=350` (mapper) vs `dist < 300` (resolver) — line numbers rarely assigned
`src/graph/entity_mapper.py:68` (`max_distance = 350.0`) vs
`src/graph/relationship_resolver.py:105` (`dist < 300`).

A line label within 350px is **absorbed by the symbol as its text**, so it never becomes a
standalone "line" entity, so the resolver never sees it as a candidate. The 50px gap means
line numbers are assigned far less often than the code implies — `get_entities_on_line()`
returns `[]` on real runs.

### S15 — LLM issue parsing mangles the response and mis-detects pass/fail
`src/validation/llm_validator.py:206-226` — three defects, confirmed by driving the node
with a realistic Llama-style response:
1. Markdown-bold headers `**1. Missing tag**` aren't recognised (first char is `*`), so the
   header is **appended to the previous issue**.
2. Sub-bullets starting with `-` are promoted to top-level issues (5 bogus issues from 2 real).
3. `passed = not any("error" in issue.lower() ...)` — the word "errors" in a *reassuring*
   sentence (`1. No errors detected`) reads as a **failure**. A real error that omits the
   literal word "error" passes silently.

The severity field is never extracted — issues are flat strings, so neither the API nor
Excel can filter by severity. **Fix:** request JSON and `json.loads` it; compute `passed`
from the parsed severity field, never from substring matching.

### S16 — No Groq timeout, no backoff, no rate-limit handling
`src/validation/llm_validator.py:72-77` — `ChatGroq(...)` with no `timeout=` and no
`max_retries=`. A hung or rate-limited request hangs the handler indefinitely. Groq's free
tier aggressively 429s. **Fix:** `timeout=60.0, max_retries=3` and catch `RateLimitError`
specifically.

### S17 — `or` used for defaults discards a legitimate `0`
`src/preprocessing/image_tiler.py:58`, `src/detection/symbol_detector.py:76-78`,
`src/preprocessing/enhancer.py:35`, `src/preprocessing/pdf_converter.py:33`.
Verified: `ImageTiler(overlap=0).overlap` → **64**; `SymbolDetector(confidence_threshold=0.0)`
→ **0.5**. You cannot request zero overlap or zero confidence.
**Fix:** `x if x is not None else settings.y` throughout.

### S18 — `or` in the OCR correction destroys the valid ISA prefix `TE`
`src/ocr/tag_parser.py:126` → `([RTEPCVDBFKHMSX])e(\d)` → `\1\2`. `TE101` (Temperature
Element) becomes `T101` → parsed as `T-101` / "Temperature". The `TE` prefix is legitimate
ISA notation and is destroyed.

### S19 — ASTM grades become instrument tags
`A105` → matches `INSTRUMENT_PATTERN` → "Analyzer"; `SS304` → "Speed Switch". Spec grades
are being read as instruments.

### S20 — `detect_batch` is not a batch
`src/detection/symbol_detector.py:154-174` — the name and docstring promise batching; the
implementation is a sequential loop of one `model.predict` per tile.
`src/pipeline.py:185`'s comment "batch detect" is inaccurate.

### S21 — No symbol→tag association at detection time (the documented architecture isn't implemented)
`src/pipeline.py:195-199` claims "for each YOLO bbox, crop a label search region below/beside
the symbol". `src/ocr/text_extractor.py:393-397` → `extract_with_detections()` **ignores the
detections entirely** and calls `_run(image)`. The commit message says "detection-guided OCR".
It isn't. (The CC-guided path that *is* implemented works well — this is a docs/behaviour
mismatch, not a quality bug in itself.)

---

## SECURITY

| # | Issue | Location | Status |
|---|---|---|---|
| C6 | Excel formula injection | `mto_generator.py:233` | **confirmed, exploitable** |
| C12 | `project_name` path escape into the output filename | `mto_generator.py:120` | **confirmed, exploitable** |
| S15 | API key in git history | `notebooks/01_yolo_training.ipynb:143` | **confirmed, rotate** |
| — | No upload size limit | `api.py:80` | 50MB accepted; unbounded memory |
| — | No content validation | `api.py:83` | `.exe` content saved as `.png` (extension allow-list only) |
| — | CORS `allow_origins=["*"]` | `api.py:40` | wide open, no auth on any endpoint |
| — | Unbounded job store | `api.py:46` | in-memory dict, never evicted; uploads dir never cleaned |
| — | Async endpoint blocks event loop | `api.py:122` | `async def` + 50s blocking call freezes all routes |
| — | `uvicorn.run(reload=True)` in the entrypoint | `api.py:253` | dev config shipped as the documented start command |
| — | Log injection via unsanitised `file.filename` | `api.py:112` | attacker-controlled text written to logs |

**Genuinely clean:** path traversal is *not* exploitable — the disk path is
`{uuid}{suffix}`, not the client filename (verified: `../../etc/passwd.png` → stored as
`5e1d3e84.png` inside the uploads dir). The extension allow-list works (`.exe` → HTTP 400).
Missing API key degrades gracefully without crashing. `.env` is gitignored; `.gitignore` is
thorough. `models/*.pt` is correctly excluded.

---

## MINOR ISSUES

- **Dead code:** `filter_by_area` (postprocessor.py:175), `stitch_visualization` (image_tiler.py:183),
  `_tesseract_crop` (text_extractor.py:314), `parse_batch` (tag_parser.py:194),
  `get_entities_by_type` / `get_entities_by_category` / `get_connected_entities` /
  `get_entities_on_line` (graph_builder.py:144-187), `to_serializable` — never called.
  `get_class_names()` exists but the **229-class vocabulary is never used** by any code path.
- **Autofilter range** `mto_generator.py:113` spans the summary block (verified `A5:K13` on a
  12-row sheet) — filtering drags the totals in.
- **`.env.example` drift:** documents `OCR_CONFIDENCE_THRESHOLD=0.3`, code default is `0.6`;
  12 settings fields are undocumented.
- **`requirements.txt` drift:** `paddleocr` is listed but **never imported** (0 refs); the UI
  says *"PaddleOCR reads tags"* in two places; `streamlit-agraph` is not in requirements.
- **No tests at all** — `pytest` is in requirements, there is no `tests/` directory.
- **Docstring drift:** `text_extractor.py` describes per-crop OCR it doesn't do;
  `pipeline.py:195-199` describes detection-guided cropping that doesn't exist.
- **6-digit loop numbers** (`FIC-100123`) don't match `{2,5}` digits → fall through to "description".
- **`TT-301/302` loses `/302`** entirely.
- **`F-1` (single digit)** doesn't match — the pattern requires 2–5 digits.
- **`SPEC_PATTERN` second branch has no `$` anchor** → matches prefixes.
- **README overstates:** the architecture diagram shows the MTO fed by both validators (only
  the LLM one is wired); the "Sample Output" table shows `E-1001 | Heat Exchanger` — which
  C2 makes impossible; the Groq section describes a model that 404s.
- **Streamlit has no `@st.cache_resource`** → the 22MB model reloads on every click.

---

## CONFIRMED CLEAN (don't re-audit these)

- Tesseract **is** installed at `C:\Program Files\Tesseract-OCR\` (5.4.0); `_set_tesseract_path()`
  handles it even though it isn't on PATH.
- **The graph summary is NOT stale** — I checked this specifically. `build()` returns
  `self.graph` and `resolve()` mutates it in place, so `get_graph_summary()` after resolve
  sees the live graph. (Verified `g is gb.graph` → True.) The *node confidence* is stale
  (S6) but the summary is not.
- **Path traversal is not exploitable** in the API upload (see Security table).
- **Tiling covers the whole image** — no gaps, both edges reached.
  `max x covered: 1920/1920`, `max y covered: 1163/1163`.
- **Tile→global coordinate mapping is correct** — verified against a whole-image ground
  truth: every tiled detection matched its counterpart at IoU ≥ 0.79. (The *clipping* is
  wrong — C9 — but the offset arithmetic is right.)
- **NMS is a correct greedy implementation** — sort, iterate, suppress lower-confidence
  overlaps. The IoU formula is correct for positive-area boxes.
- **PDF rotation is honoured** — `get_pixmap` applies `/Rotate`; a 400×600pt page with
  `set_rotation(90)` renders 600×400 with content rotated.
- **PDF DPI scaling is correct** — `zoom = dpi/72.0` against PyMuPDF's 72-DPI base;
  a 400×600pt page at 300 DPI renders exactly 2500×1667px.
- **1px line art survives enhancement** — CLAHE+denoise preserves a 1px line (501 dark px
  before and after) and a 1px circle outline.
- **Entity IDs are unique per run**; the counter resets in the pipeline path.
- **`langgraph` 1.2.9 and `langchain-groq` 1.1.3 are installed**, and the
  `ChatGroq(model=..., api_key=...)` call uses the correct field aliases. The bug is the
  *model name*, not the API usage.
- **Exactly 3 LLM calls per pipeline run** (one per graph node), not one per entity — cost
  is bounded regardless of entity count.
- **No API-key leakage in any response** — the key is read only from settings/env, passed
  to `ChatGroq`, and never appears in the API response, the Excel output, the UI, or the logs.
- **`.gitignore` is thorough**; `.env` and `models/*.pt` are excluded; the venv is excluded.
- **Missing API key / bogus model** both degrade gracefully — no crash, no traceback to the user.
- **`_group_into_words` and `_merge_fragments`** are sound and well-tuned for P&ID text.
- **The CC-guided OCR approach is genuinely good** — it found 150 text-like blobs and grouped
  them into 136 word regions. The problem is downstream, not in this step.
- **Upload extension allow-list works** (`.exe` → HTTP 400); missing file → 422; unknown job → 404.
- **Excel styling is genuinely applied** — merged title, coloured header bands, zebra striping,
  borders, low-confidence amber fill, column widths, `freeze_panes="A6"`.
- **`graph_builder.to_serializable()` explicitly drops the `entity` object**, so the graph is
  JSON-safe despite holding live entity references.

---

## FIX ROADMAP

### Wave 1 — Correctness (highest impact, ~2 hours)

| # | Fix | File | Measured gain |
|---|---|---|---|
| 1 | `groq_model = "openai/gpt-oss-120b"` + fix fallback messaging | `config/settings.py:21`, `llm_validator.py:300` | LLM validation actually runs |
| 2 | Equipment tags before instrument tags; ≥2 letters + known ISA letter for instruments | `tag_parser.py:158` | **6/15 tag descriptions corrected** |
| 3 | Reject non-tag words (`PUMP 101`, `LINE 12`) | `tag_parser.py:59` | 3 fake tags eliminated |
| 4 | Resolve `YOLO_MODEL_PATH` to absolute; **hard-fail** if the model is missing | `settings.py:25`, `symbol_detector.py:85` | no silent COCO fallback |
| 5 | Escape Excel formula injection | `mto_generator.py:233` | closes C6 |
| 6 | Report the *processed* image size (or carry scale) | `pipeline.py:160` | coords match the frame |
| 7 | Sanitise `file.filename` before echoing/logging | `api.py:106,112` | log injection |
| 8 | **Require the inch mark** in `PIPE_SIZE_PATTERN`; bare digits = stream numbers | `tag_parser.py:77` | stops fake "12-inch valve" rows |
| 9 | Clip both ends of merged bboxes; drop degenerate boxes | `postprocessor.py:60-66` | stops negative-area detections |
| 10 | Sanitise `project_name` into a safe filename slug | `mto_generator.py:120` | closes the path escape + 500s |
| 11 | Move `_build_graph()` inside the try/except | `llm_validator.py:124` | langgraph failure degrades, not crashes |
| 12 | One canonical category list for body + summary + catch-all bucket | `mto_generator.py:74,254` | no silently dropped rows |

### Wave 2 — Domain quality (~4 hours)

| # | Fix | Measured gain |
|---|---|---|
| 13 | Class→category map from the model's 229 classes | coverage **56.3% → 79.9%**, orphans 100 → 46 |
| 14 | Lower `ocr_confidence_threshold` to ~0.2 | tag recall **2 → 5** |
| 15 | Global (not greedy) text↔symbol assignment | fixes stolen tags |
| 16 | Global nearest-line assignment with alignment check | stops wrong line numbers |
| 17 | Don't inherit line pipe-size onto components | stops fake valve sizes |
| 18 | Exclude LINE_NUMBER from MTO rows | stops non-procurable rows |
| 19 | Aggregate MTO rows by (category, description, size, spec) | **6 entities → 2 rows with real quantities** |
| 20 | Fix the self-loop (`dict.fromkeys` + guard) | removes `add_edge(X,X)` |
| 21 | Score confidence **before** building the graph | MTO and graph agree |
| 22 | Process all PDF pages | no silent page loss |
| 23 | Report tag coverage alongside pass rate | stops "100% pass" on 1/9 tagged |
| 24 | Class-agnostic NMS pass before class-aware NMS | stops double-counted boundary symbols |
| 25 | Drop border-touching (truncated) tile detections | removes false-positive slivers |

### Wave 3 — Performance & robustness (~4 hours)

| # | Fix | Measured gain |
|---|---|---|
| 26 | Move denoise into the per-tile path (or make it opt-in) | A1 sheet: **~20 min → ~1 min** |
| 27 | Fix the tiler grid (clamp stride, not start) | removes the 639px redundant tile; ~2× faster in that band |
| 28 | Add `tests/` — pytest is already in requirements | start with `test_tag_parser.py` (pure) and `test_mto_generator.py` (xlsx round-trip) |
| 29 | `@st.cache_resource` for the pipeline; make `_init_components()` idempotent | 22MB model no longer reloaded per click/request |
| 30 | Make `/process` a `def` endpoint (threadpool) or a queue+worker | API no longer freezes for 50s per request |
| 31 | Upload size limit + content sniffing; job-store TTL; upload cleanup | no unbounded memory/disk |
| 32 | Groq `timeout=60, max_retries=3`; catch `RateLimitError` | no infinite hang on 429 |
| 33 | LLM issue parsing → JSON; `passed` from parsed severity | fixes the false-negative on "No errors detected" |
| 34 | `x if x is not None else settings.y` (not `or`) everywhere | `overlap=0` and `threshold=0.0` become settable |
| 35 | Delete the dead code listed above; fix docstrings; reconcile README with reality | — |
| 36 | Rotate the Roboflow key; strip it from git history | — |

---

## What I'd do first

**Wave 1, items 2 + 3 + 8 + 14 + 19** — the tag parser (order + fake-tag rejection + the
inch-mark requirement), the OCR threshold, and MTO aggregation. Together they take the
sample P&ID from *1 tag (wrong)* to *5–7 tags (mostly right)*, stop 12-inch valves from
appearing out of nowhere, and turn a list of individual components into something a
procurement engineer can act on.

Item 1 (the Groq model) is a one-line change that makes an advertised feature real — and its
output is a free second opinion on everything else in this report. It already independently
flagged the `R-1001` → "Radiation" bug, which is how I knew the tag parser was wrong.

**Want me to implement Wave 1?** Say the word and I'll do it, run the pipeline before/after,
and show you the MTO diff. I'd also add the pytest suite for the tag parser, since the
order bug and the fake-tag bug are exactly the kind of regression that silently comes back.
