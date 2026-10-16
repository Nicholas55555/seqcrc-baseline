# Setup notes, issues and fixes (Phase 1 baseline reproduction)

Paper reproduced: L. Andéol, L. Mossina, A. Mazoyer, S. Gerchinovitz, *Conformal Object Detection by
Sequential Risk Control*, arXiv:2505.24038 (v2, 31 Oct 2025; preprint, no venue yet). Official code:
<https://github.com/leoandeol/cods>, cloned at commit `269ff11a2967ede680da508ade24f58149959dba`
(HEAD on 2026-10-15). Everything below happened on 2026-10-15 on the machine described in the README.

## What the paper specifies (Sec. V-A, App. E-A of arXiv v2) and what we did

| Setting | Paper | This reproduction |
| --- | --- | --- |
| Data | MS-COCO validation split, halved: 2,500 calibration / 2,500 test | COCO **val2017** (5,000 images), first 2,500 / last 2,500 in annotation-file order (the official `split_dataset(0.5, shuffle=False)`) |
| Detector | YOLOv8x (68 M) and DETR-101 (60 M), COCO-pretrained | YOLOv8x only (`yolov8x.pt`, SHA-256 `3df4ada6...24c8`); DETR-101 not run (time) |
| NMS | IoU 0.5 | IoU 0.5 (official class-agnostic `torchvision.ops.nms`) |
| Confidence floor | predictions below 1e-3 removed | 1e-3 (official `filter_preds` rule) |
| Budgets, alpha_tot = 0.1 | (alpha_cnf, alpha_loc, alpha_cls) = (0.02, 0.05, 0.05) | same |
| Matching / losses / sets | 4 matchings x 2 confidence losses x 3 localization losses x 2 class sets x 2 box sets (96-cell grid; App. Table VII lists 92 YOLOv8x rows at alpha_tot = 0.1, without GIoU + thresholded loss + multiplicative boxes) | 4 configurations at full size (F1-F4, listed in README) chosen to cover both confidence losses, all three non-GIoU matchings, pixelwise and boxwise localization, additive and multiplicative boxes, APS and LAC |
| Hardware | one RTX 4090, ~20 min per run | RTX 3050 Ti Laptop (4 GB) for detection; calibration on the CPU (Ryzen 7 4800H), see issue 4 |
| Splits | one calibration/test draw | the same single draw (the official code does not shuffle) |

## Issues and fixes

1. **Hard-coded dataset path and Weights & Biases.** `cods/od/evaluate.py` reads COCO from
   `/datasets/shared_datasets/coco/` and logs every run to wandb (imported unconditionally, but only in cods' optional `[dev]` extra, not a core dependency).
   *Fix:* `scripts/run_seqcrc.py` reproduces `Benchmark.run_experiment` step for step with a
   `--coco` argument and JSON-lines output instead of wandb. The library itself is used unchanged
   for everything except the patches below.

2. **Regression at HEAD: calibration crashes** with
   `TypeError: expected Tensor as element 0 in argument 0, but got list` (`cods/od/optim.py:62`).
   *Cause:* commit `e70d773` ("Fixing linting issues", 2025-10-20) rewrote
   `all_confidences = list([...],)` as `all_confidences = ([...],)` in
   `ODModel.build_predictions`, so every per-image list became a 1-tuple. The official tutorial
   notebook still runs only because it loads predictions cached before that commit.
   *Fix:* `run_seqcrc.py` wraps `ODPredictions.__init__` to unwrap the 1-tuples (no upstream edit).

3. **GPU memory and device mismatch.** Without a floor, YOLOv8x keeps hundreds to thousands of
   post-NMS boxes per image, each with a 91-way class vector, kept on the GPU; ground truth is put on
   the GPU while our CPU-side boxes are not (`RuntimeError: Expected all tensors to be on the same
   device`). *Fix:* the paper's 1e-3 floor is applied immediately after NMS (identical box set:
   a box below the floor can only suppress boxes that are also below it), boxes are kept on the
   CPU, and `ODPredictions.to()` puts everything on one device before calibration.
   After the floor: mean 50.8, median 30, max 741 boxes per test image.

4. **Calibration far too slow on a laptop.** A 50/50-image smoke test was still calibrating after about 12 min (17:36 to 17:48 in `logs/02`) on
   the GPU (about 20 % GPU utilisation in `nvidia-smi`, not logged). `py-spy` showed two hot spots in
   `cods/od/optim.py` (hot spot (a) in live `py-spy dump` stack traces of the unpatched runs, not saved;
   hot spot (b) in the saved profile `logs/profile_n600.txt`, recorded after fix (a): 27 % of samples in
   `_correct_risk`):
   (a) `pred_cls_i = [x for x, c in zip(pred_cls_i, confidences_i) if c >= 1 - lambda_conf]`
   loops over every box of the image, in Python, for every candidate threshold, with a GPU sync per
   comparison; (b) after each single-image loss update the optimizers recompute
   `torch.sum(torch.stack(losses))` over all n calibration images, which is O(n^2) overall
   (n = 100: 294 s; n = 300: 1,151 s; extrapolated n = 2,500: about 8 h per configuration).
   *Fixes (patches/cods-reproduction.patch):* (a) boolean-mask indexing
   `list(pred_cls_i[confidences_i >= 1 - lambda_conf])`, which selects the same rows in the same
   order; (b) a list subclass that keeps the sum of the losses up to date in O(1) (float64).
   Calibration runs on the CPU with one thread per process (many tiny tensor ops are faster there
   than on a WDDM laptop GPU), four configurations in parallel.
   *Checks:* `tests/test_patch_equivalence.py` (identical selections for 0-300 boxes and thresholds
   0-1.5; running sum equal to `torch.sum(torch.stack())` within 1e-6 after 20,000 updates on 2,500
   images) and a 300/300 run with and without patch (b): identical step-1 thresholds
   (lambda+ = 0.9250815063714981, lambda- = 0.9160988479852676); the calibration risks printed in the
   logs differ at about 2e-9, and every output in the two JSON lines is identical.

5. **Tiny smoke sets are infeasible by construction.** With 10 calibration images the
   1/(n+1) = 0.09 correction exceeds alpha_loc = 0.05 and cods raises `No good lambda found`.
   SeqCRC needs alpha_loc >= alpha_cnf + 1/(n+1), i.e. n >= 33 here; smoke tests use n >= 100.

6. **Python not on PATH / Windows console encoding.** `python` resolves to the Microsoft Store stub
   and printing the notebook fails with `cp1252` errors. *Fix:* uv-managed CPython 3.11.15 in
   `.venv`, `PYTHONIOENCODING=utf-8`.

## Observations relevant to the comparison with the paper

* **The YOLO wrapper ignores letterbox padding.** `cods/od/models/yolo.py` rescales raw head
  outputs by `original_size / letterboxed_size` and never subtracts the padding. With batch size 1
  Ultralytics pads each image only to a multiple of 32 (small error); with a batch of
  differently-sized images it pads all of them to 640 x 640 (large error for non-square images).
  The paper does not state its batch size. This bears on the localization set sizes (stretch),
  which we find much smaller than the paper's YOLOv8x numbers; see the report.
* The YOLOv8x "confidence" in cods is the maximum sigmoid class score of the raw head output, and
  the class vector is a softmax over those sigmoid scores (temperature 1), as in the official code.

## Diagnostics run on 2026-10-15 (all reproducible with the commands in README)

* **Detector check** (`scripts/detector_sanity.py`, COCO API on all 5,000 val2017 images):
  Ultralytics `predict()` (NMS IoU 0.7 per class) 53.2 AP / 69.8 AP50 (published: 53.9);
  Ultralytics with class-agnostic NMS at IoU 0.5 (the paper's NMS) 51.6 AP;
  boxes from the cods YOLO wrapper at batch size 1 (what our SeqCRC runs use) 45.1 AP;
  the same wrapper at batch size 8: 10.7 AP, AP75 1.4.
* **Batch size changes the paper's localization numbers.** F3 configuration on the 300/300 subset:
  batch 1 gives stretch 1.24 (risk 0.038); batch 8 gives stretch 4.70 (risk 0.031), close to the
  paper's 4.54 for the full split. Confidence (11.11 vs. 11.46 boxes) and class sets (66.1 vs. 65.7)
  barely change. So the letterbox rescaling of the cods YOLO wrapper, which is harmless at batch 1 but
  misplaces boxes when a batch mixes image sizes, very likely explains why the paper's YOLOv8x box
  margins are 3-4 times DETR-101's. The paper does not state its batch size.
* **Running-sum patch is exact in practice.** 300/300 run with and without it: identical confidence,
  localization and classification sizes and risks, global risk, and all four lambdas to 7 digits
  (`results/scaling_patched_n600.jsonl` vs. `results/validate_runningsum_n600.jsonl`).
