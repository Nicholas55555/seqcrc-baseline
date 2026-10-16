# Results: SeqCRC reproduction (YOLOv8x, COCO val2017, alpha_tot = 0.1)

Budgets: alpha_cnf = 0.02, alpha_loc = alpha_cls = 0.05; Global = mean over test images of max(loc, cls) loss, bounded by alpha_loc + alpha_cls = 0.1. Each cell is set size / test risk. Paper: arXiv:2505.24038v2 App. Table VII (one 2,500/2,500 split). Full: our 2,500/2,500 run (same split as the paper's code). Subset: first 300 calibration / next 300 test images.

| ID | Config (matching / conf. loss / loc. loss-set / cls. set) | Source | Cnf. size / risk | Loc. size / risk | Cls. size / risk | Global risk | Calibration time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | Hausdorff / bc-recall / pixel-add / APS | Paper | 11.71 / 0.019 | 3.87 / 0.049 | 41.73 / 0.043 | 0.082 |  |
| F1 | Hausdorff / bc-recall / pixel-add / APS | Ours, full | 11.53 / 0.020 | 1.30 / 0.053 | 40.64 / 0.042 | 0.085 | 253 min |
| F1 | Hausdorff / bc-recall / pixel-add / APS | Ours, 300/300 | 11.11 / 0.025 | 1.30 / 0.040 | 42.67 / 0.033 | 0.066 | 32 min |
| F2 | LAC / bc-thresh / pixel-add / LAC | Paper | 18.86 / 0.012 | 20.64 / 0.047 | 0.72 / 0.049 | 0.086 |  |
| F2 | LAC / bc-thresh / pixel-add / LAC | Ours, full | 18.02 / 0.015 | 17.69 / 0.047 | 0.75 / 0.046 | 0.083 | 143 min |
| F2 | LAC / bc-thresh / pixel-add / LAC | Ours, 300/300 | 14.29 / 0.047 | 13.56 / 0.046 | 1.02 / 0.045 | 0.080 | 19 min |
| F3 | Mix / bc-recall / pixel-mult / LAC | Paper | 11.71 / 0.019 | 4.54 / 0.049 | 67.58 / 0.053 | 0.098 |  |
| F3 | Mix / bc-recall / pixel-mult / LAC | Ours, full | 11.53 / 0.020 | 1.24 / 0.052 | 66.82 / 0.055 | 0.100 | 209 min |
| F3 | Mix / bc-recall / pixel-mult / LAC | Ours, 300/300 | 11.11 / 0.025 | 1.24 / 0.038 | 66.06 / 0.061 | 0.093 | 24 min |
| F4 | Mix / bc-thresh / box-add / APS | Paper | 18.86 / 0.012 | 6.12 / 0.043 | 40.75 / 0.042 | 0.078 |  |
| F4 | Mix / bc-thresh / box-add / APS | Ours, full | 18.02 / 0.015 | 2.56 / 0.053 | 38.28 / 0.043 | 0.088 | 232 min |
| F4 | Mix / bc-thresh / box-add / APS | Ours, 300/300 | 14.29 / 0.047 | 2.21 / 0.055 | 40.66 / 0.038 | 0.085 | 31 min |

## Detector check (COCO val2017, 5,000 images, pycocotools)

| Boxes scored | AP | AP50 | AP75 |
| --- | --- | --- | --- |
| Ultralytics model table (paper's detector) | 53.9 | | |
| Ours: Ultralytics predict, NMS IoU 0.7 per class | 53.2 | 69.8 | 58.0 |
| Ours: Ultralytics predict, class-agnostic NMS IoU 0.5 | 51.6 | 68.8 | 55.6 |
| Ours: cods YOLO wrapper, batch 1 (SeqCRC input) | 45.1 | 65.9 | 48.2 |
| Ours: cods YOLO wrapper, batch 8 | 10.7 | 32.1 | 1.4 |

## Batch-size diagnostic (F3 configuration, 300/300 subset)

| cods YOLO batch size | Cnf. size / risk | Loc. size (stretch) / risk | Cls. size / risk | Global risk |
| --- | --- | --- | --- | --- |
| 1 (ours) | 11.11 / 0.025 | 1.24 / 0.038 | 66.06 / 0.061 | 0.093 |
| 8 | 11.46 / 0.023 | 4.70 / 0.031 | 65.68 / 0.064 | 0.093 |
