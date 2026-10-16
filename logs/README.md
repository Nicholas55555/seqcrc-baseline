# Run logs (2026-10-15, in order)

| Log | What it is |
| --- | --- |
| `00_download_coco.log` | COCO val2017 + annotations download (checksums: `data/coco/zips.sha256`, also in `scripts/get_coco_val2017.sh`) |
| `01_env_setup.log` | venv creation, CUDA 12.6 torch install, `pip install -e external/cods`, GPU check |
| `pip_freeze_full.txt` | every installed package version |
| `02_smoke_n100_gpu_unpatched_KILLED.log`, `03_smoke_n100_cpu_unpatched_KILLED.log` | first smoke tests with the unpatched optimizer, killed: `02` (GPU) after about 12 min of calibration, `03` (CPU) after under a minute (issue 4 in NOTES.md) |
| `04_equiv_unpatched_n20.log`, `05_equiv_patched_n20.log` | 10/10-image runs: infeasible by construction (`No good lambda found`, issue 5) |
| `profile_n600.txt` | `py-spy` samples of the 300/300 run, used to find the O(n^2) hot spots |
| `06_scaling_patched_n200.log`, `07_scaling_patched_n600.log` | 100/100 and 300/300 runs with the mask patch only (timing: 294 s, 1,151 s) |
| `08_predict_full_bs1.log` | YOLOv8x over all 5,000 images at batch 1 (prediction cache for the full runs, 390 s) |
| `09_validate_runningsum_n600.log` | 300/300 run with the running-sum patch; identical results to `07` |
| `10_full_F1.log` ... `10_full_F4.log` | the four full-size (2,500 calibration / 2,500 test) SeqCRC runs reported in the results table |
| `11_detector_sanity_native.log` | COCO AP of YOLOv8x through Ultralytics (53.2) |
| `12_pytest.log` | patch-equivalence tests |
| `13_detector_sanity_cods_bs1.log`, `15_detector_sanity_cods_bs8.log` | COCO AP of the cods YOLO wrapper's boxes at batch 1 (45.1) and 8 (10.7) |
| `14_predict_full_bs8.log` | YOLOv8x prediction cache at batch 8 (for the diagnostic) |
| `16_diag_bs8_n600.log` | F3 configuration on the 300/300 subset with batch-8 predictions (stretch 4.70) |
| `17_detector_sanity_native_agnostic05.log` | COCO AP with class-agnostic NMS at IoU 0.5 (51.6) |
| `18_subset300_F1.log`, `18_subset300_F2.log`, `18_subset300_F4.log` | F1, F2, F4 on the 300/300 subset (F3's subset run is `09`) |
| `19_full_F3_bs8_STOPPED.log` | F3 at full size with the batch-8 cache; stopped after 25 min because it could not finish before the deadline (the 300/300 diagnostic `16` is reported instead) |

The `RESULT` line at the end of each run log holds the test risks and set sizes; the same numbers are in
`results/*.jsonl` and are collected by `scripts/make_results.py` into `results/RESULTS.md`.
