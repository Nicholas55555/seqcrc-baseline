# SeqCRC baseline reproduction (Phase 1, Weather Adaptive Autonomous Driving)

This repository reproduces the baseline our course project builds on:

> L. Andéol, L. Mossina, A. Mazoyer, S. Gerchinovitz. **Conformal Object Detection by Sequential
> Risk Control.** arXiv:2505.24038, v2 (31 Oct 2025). Preprint; no peer-reviewed version yet.
> Paper: <https://arxiv.org/abs/2505.24038> · Official code (cods): <https://github.com/leoandeol/cods>

SeqCRC calibrates a frozen, COCO-pretrained detector in two steps with conformal risk control: first
the confidence threshold, then box margins and class sets, so that the expected loss of each task on
a new image stays below a budget. Our project reuses its first (confidence) step per weather
condition; this repository checks that the official code and our environment reproduce the paper's
COCO numbers before we modify anything. **Nothing is trained**: SeqCRC is post-hoc calibration of a
released YOLOv8x checkpoint, so "training" here means calibration and "evaluation" means the
held-out test half.

Results: [`results/RESULTS.md`](results/RESULTS.md) · issues and fixes: [`NOTES.md`](NOTES.md) ·
raw logs: [`logs/`](logs) · per-run JSON lines: [`results/`](results).

## What was reproduced

MS-COCO val2017 (all 5,000 images; first 2,500 for calibration, last 2,500 for test, exactly as
`cods` splits it), YOLOv8x, NMS IoU 0.5, confidence floor 1e-3, alpha_tot = 0.1 with
(alpha_cnf, alpha_loc, alpha_cls) = (0.02, 0.05, 0.05). Four configurations of the paper's appendix
Table VII were run at full size:

| ID | Matching | Confidence loss | Localization loss | Class set | Box set |
| --- | --- | --- | --- | --- | --- |
| F1 | Hausdorff | box_count_recall | pixelwise | APS | additive |
| F2 | LAC | box_count_threshold | pixelwise | LAC | additive |
| F3 | Mix (tau = 0.25) | box_count_recall | pixelwise | LAC | multiplicative |
| F4 | Mix (tau = 0.25) | box_count_threshold | boxwise | APS | additive |

A detector sanity check scores YOLOv8x on val2017 with the COCO API, both through Ultralytics and
through the cods YOLO wrapper that feeds SeqCRC.

## Setup

Tested only on a local laptop, not on the online GPU the assignment asks for (the GPU is needed only for the 6.5-min
detection pass; calibration runs on the CPU): Windows 11 with Python 3.11.15 (uv), an NVIDIA RTX 3050 Ti Laptop GPU (4 GB, driver 566.07,
CUDA 12.7), PyTorch 2.14.1 with CUDA 12.6 wheels. Linux/macOS: replace `.venv/Scripts/python.exe` with
`.venv/bin/python`.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe --index-url https://download.pytorch.org/whl/cu126 torch==2.14.1+cu126 torchvision==0.29.1+cu126
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
bash scripts/setup_cods.sh          # clones cods at commit 269ff11 and applies patches/cods-reproduction.patch
uv pip install --python .venv/Scripts/python.exe -e external/cods
bash scripts/get_coco_val2017.sh    # 1.07 GB, SHA-256 checked, unpacked to data/coco/
```

Weights: `weights/yolov8x.pt` is the official Ultralytics release (SHA-256
`3df4ada6b4dad6d657868f2fdf7faecfb34dcfccf3a25c4b82079064718524c8`); Ultralytics downloads it on first
use if the file is missing (`YOLO("yolov8x.pt")`, then move it to `weights/`).

Plain pip works the same way (`python -m venv .venv`, then the three `pip install` lines).

## Run

Build the detector cache once (YOLOv8x over the 5,000 images, about 6.5 min on the RTX 3050 Ti):

```bash
.venv/Scripts/python.exe scripts/run_seqcrc.py --configs none
```

Calibrate on the 2,500 calibration images and evaluate on the 2,500 test images, one configuration
per process (each runs on one CPU thread; the four can run in parallel):

```bash
.venv/Scripts/python.exe scripts/run_seqcrc.py --configs F1 --calib-device cpu --threads 1
```

(`F2`, `F3`, `F4` likewise; `--alpha-tot 0.2` uses the paper's (0.03, 0.1, 0.1) split;
`--n-calib-test 200` runs a 100/100 smoke test.) Each run appends one JSON line per configuration to
`results/seqcrc_yolov8x_<config>_a0.1_bs1_cpu.jsonl` with test risks, set sizes, calibrated lambdas
and timings.

300/300 subset rows of the results table (first 600 images; F2 and F4 likewise; F3's subset row is
`results/validate_runningsum_n600.jsonl`):

```bash
.venv/Scripts/python.exe scripts/run_seqcrc.py --configs F1 --calib-device cpu --threads 1 --n-calib-test 600 --out results/subset300_F1.jsonl
```

Batch-size diagnostic (F3 with the cods YOLO wrapper at batch 8; the first command builds the batch-8 cache):

```bash
.venv/Scripts/python.exe scripts/run_seqcrc.py --configs none --batch-size 8
.venv/Scripts/python.exe scripts/run_seqcrc.py --configs F3 --calib-device cpu --threads 1 --batch-size 8 --n-calib-test 600 --out results/diag_bs8_n600.jsonl
```

Detector sanity check (COCO box AP of YOLOv8x):

```bash
.venv/Scripts/python.exe scripts/detector_sanity.py native
.venv/Scripts/python.exe scripts/detector_sanity.py cods
```

Other detector-table rows: class-agnostic NMS at IoU 0.5, and the cods wrapper at batch 8 (needs the batch-8 cache
built above):

```bash
.venv/Scripts/python.exe scripts/detector_sanity.py native --iou 0.5 --agnostic --tag _agnostic_iou05
.venv/Scripts/python.exe scripts/detector_sanity.py cods --batch-size 8 --tag _bs8
```

Patch-equivalence tests:

```bash
.venv/Scripts/python.exe -m pytest -q tests
```

Collect all results into `results/RESULTS.md` and the report tables, then build the report (pdfLaTeX twice, or
tectonic):

```bash
.venv/Scripts/python.exe scripts/make_results.py
cd report && tectonic report.tex
```

## Layout

```
scripts/run_seqcrc.py        driver: official cods pipeline (evaluate.py) without wandb, local paths
scripts/detector_sanity.py   COCO AP of YOLOv8x, native Ultralytics vs. the cods wrapper
scripts/setup_cods.sh        clone cods at 269ff11 + apply the patch
scripts/get_coco_val2017.sh  data download with checksums
patches/                     cods-reproduction.patch (both fixes) and 0001 (mask fix alone)
tests/                       the patches must not change results
results/                     JSON lines per run, RESULTS.md (paper vs. ours)
logs/                        every run's console log, environment setup, pip freeze
scripts/make_results.py      results/*.jsonl + paper_values.json -> RESULTS.md and report tables
scripts/render_evidence.py   renders log excerpts as the report's evidence figures
scripts/paper_values.json    the paper's App. Table VII values for F1-F4
report/                      report.tex (Phase 1 PDF source) and its figures
NOTES.md                     setup issues, causes and fixes
```

## Data, weights and caches

`data/`, `weights/`, `cache/` and `external/cods/` are git-ignored: COCO val2017 is 1 GB and the
YOLOv8x prediction cache is about 100 MB. Both are rebuilt by the commands above (COCO from
<https://cocodataset.org>, weights from <https://github.com/ultralytics/assets/releases>).

## Licences

cods: MIT. YOLOv8 weights: Ultralytics, AGPL-3.0. COCO annotations: CC BY 4.0; images: Flickr
terms of use.
