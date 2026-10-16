"""Reproduce SeqCRC (Andeol et al., 2025) on MS-COCO val2017 with the official `cods` code.

This is a thin driver around the official library (external/cods, pinned commit in README).
It follows `cods/od/evaluate.py::Benchmark.run_experiment` step for step, with three changes,
all logged in NOTES.md:

1. The COCO root is a command-line argument instead of the hard-coded
   "/datasets/shared_datasets/coco/".
2. Weights & Biases logging is removed; every result is appended to a JSON-lines file instead.
3. The paper's 1e-3 confidence floor (Sec. V-A) is applied right after the official class-agnostic
   NMS, inside prediction building, instead of after the whole split is built. This is the same set
   of boxes (a box below 1e-3 can only suppress boxes that are also below 1e-3), but it keeps the
   cached predictions small enough for a 4 GB GPU.

Usage (see README for the full list of commands):
    python scripts/run_seqcrc.py --configs table3 --alpha-tot 0.1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from cods.od.cp import ODConformalizer
from cods.od.data import MSCOCODataset, ODPredictions
from cods.od.models import YOLOModel

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("seqcrc_repro")

# Fix for a regression in the official repo at HEAD (269ff11): commit e70d773 ("Fixing linting
# issues", 2025-10-20) rewrote `x = list([...],)` as `x = ([...],)` in ODModel.build_predictions,
# which turns every per-image list into a 1-tuple and makes calibration crash
# ("expected Tensor as element 0 in argument 0, but got list"). We unwrap the 1-tuples when the
# ODPredictions object is built, without editing the upstream code.
_PER_IMAGE_FIELDS = ("image_paths", "image_shapes", "true_boxes", "pred_boxes", "confidences",
                     "true_cls", "pred_cls", "pred_boxes_uncertainty")
_orig_odpredictions_init = ODPredictions.__init__


def _odpredictions_init(self, *args, **kwargs):
    for k in _PER_IMAGE_FIELDS:
        v = kwargs.get(k)
        if isinstance(v, tuple) and len(v) == 1 and isinstance(v[0], list):
            kwargs[k] = v[0]
    _orig_odpredictions_init(self, *args, **kwargs)


ODPredictions.__init__ = _odpredictions_init

# (alpha_cnf, alpha_loc, alpha_cls) for each alpha_tot, from the paper (Sec. V-A)
ALPHAS = {0.1: (0.02, 0.05, 0.05), 0.2: (0.03, 0.1, 0.1)}

# Configurations: (matching, confidence loss, localization loss, classification set, localization set)
# "table3": the YOLOv8x configurations that give the best value of each row of the paper's Table III
# at alpha_tot = 0.1 (read off the paper's appendix table "Experimental Results using the YOLOv8x
# model for alpha_tot = 0.1"). Expected values are in scripts/paper_values.json.
CONFIG_SETS = {
    "smoke": [
        ("mix", "box_count_recall", "pixelwise", "lac", "multiplicative"),
    ],
    "none": [],  # build/cache predictions only
    # the four full-size (2,500/2,500) configurations reported in the Phase 1 report
    "F1": [("hausdorff", "box_count_recall", "pixelwise", "aps", "additive")],
    "F2": [("lac", "box_count_threshold", "pixelwise", "lac", "additive")],
    "F3": [("mix", "box_count_recall", "pixelwise", "lac", "multiplicative")],
    "F4": [("mix", "box_count_threshold", "boxwise", "aps", "additive")],
    "table3": [
        ("hausdorff", "box_count_recall", "pixelwise", "aps", "additive"),
        ("mix", "box_count_threshold", "pixelwise", "lac", "multiplicative"),
        ("lac", "box_count_threshold", "pixelwise", "lac", "additive"),
        ("mix", "box_count_recall", "pixelwise", "lac", "multiplicative"),
        ("hausdorff", "box_count_recall", "boxwise", "aps", "additive"),
        ("hausdorff", "box_count_recall", "thresholded", "aps", "additive"),
    ],
}


class FlooredYOLOModel(YOLOModel):
    """Official YOLOModel; NMS as in cods, then the paper's 1e-3 floor, kept on the CPU."""

    conf_floor = 1e-3

    def _filter_preds(self, pred_boxes, pred_cls, confidences, iou_threshold=0.5, method="nms"):
        boxes, cls, confs, unc = super()._filter_preds(
            pred_boxes, pred_cls, confidences, iou_threshold=iou_threshold, method=method
        )
        out_b, out_c, out_s = [], [], []
        for b, c, s in zip(boxes, cls, confs):
            # same rule as cods.od.models.utils.filter_preds
            keep = s > self.conf_floor if (s > self.conf_floor).any() else s.argmin(0)[None]
            out_b.append(b[keep].cpu())
            out_c.append(c[keep].cpu())
            out_s.append(s[keep].cpu())
        return out_b, out_c, out_s, unc


def env_info(device: str) -> dict:
    import torchvision
    import ultralytics

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "ultralytics": ultralytics.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "device": device,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
    return info


def build_preds(args, device):
    dataset = MSCOCODataset(root=str(args.coco), split="val")
    data_cal, data_val = dataset.split_dataset(0.5, shuffle=False, n_calib_test=args.n_calib_test)
    logger.info("COCO val2017: %d images; calibration %d, test %d", len(dataset), len(data_cal), len(data_val))
    model = FlooredYOLOModel(
        model_name=str(args.weights), pretrained=True, device=device, save_dir_path=str(args.pred_cache)
    )
    t0 = time.time()
    preds = {}
    for name, split in (("cal", data_cal), ("test", data_val)):
        preds[name] = model.build_predictions(
            split,
            dataset_name="mscoco",
            split_name=name,
            batch_size=args.batch_size,
            collate_fn=dataset._collate_fn,
            shuffle=False,
            iou_threshold=0.5,  # paper: NMS IoU 0.5
            deletion_method="nms",
        )
        # build_predictions puts ground truth on `device` but our floored boxes on the CPU;
        # the official ODPredictions.to() moves everything to one device for calibration.
        preds[name].to(args.calib_device or device)
    t_pred = time.time() - t0
    n_boxes = [len(c) for c in preds["test"].confidences]
    logger.info(
        "Predictions ready in %.1f s; test boxes/image after floor: mean %.1f, median %d, max %d",
        t_pred, float(np.mean(n_boxes)), int(np.median(n_boxes)), int(np.max(n_boxes)),
    )
    return preds, t_pred, len(data_cal), len(data_val)


def run_config(cfg, alpha_tot, preds, device):
    matching, cnf_loss, loc_loss, cls_set, loc_set = cfg
    a_cnf, a_loc, a_cls = ALPHAS[alpha_tot]
    conf = ODConformalizer(
        guarantee_level="image",
        matching_function=matching,
        confidence_method=cnf_loss,
        localization_method=loc_loss,
        classification_method="binary",  # as in the official notebook; evaluate.py takes it from a config
        localization_prediction_set=loc_set,
        classification_prediction_set=cls_set,
        optimizer="binary_search",
        device=device,
    )
    t0 = time.time()
    params = conf.calibrate(
        preds["cal"], alpha_confidence=a_cnf, alpha_localization=a_loc, alpha_classification=a_cls, verbose=False
    )
    t_cal = time.time() - t0
    cpreds = conf.conformalize(preds["test"], parameters=params, verbose=False)
    res = conf.evaluate(
        preds["test"], parameters=params, conformalized_predictions=cpreds,
        include_confidence_in_global=False, verbose=False,
    )
    t_all = time.time() - t0
    m = lambda x: float(torch.mean(x.float()).item())  # noqa: E731
    return {
        "matching": matching, "confidence_loss": cnf_loss, "localization_loss": loc_loss,
        "classification_set": cls_set, "localization_set": loc_set,
        "alpha_tot": alpha_tot, "alpha_cnf": a_cnf, "alpha_loc": a_loc, "alpha_cls": a_cls,
        "confidence_size": m(res.confidence_set_sizes), "confidence_risk": m(res.confidence_coverages),
        "localization_size": m(res.localization_set_sizes), "localization_risk": m(res.localization_coverages),
        "classification_size": m(res.classification_set_sizes), "classification_risk": m(res.classification_coverages),
        "global_risk": m(res.global_coverage),
        "lambda_cnf_plus": float(params.lambda_confidence_plus), "lambda_cnf_minus": float(params.lambda_confidence_minus),
        "lambda_loc": float(params.lambda_localization), "lambda_cls": float(params.lambda_classification),
        "confidence_threshold": float(params.confidence_threshold),
        "calibration_seconds": round(t_cal, 1), "total_seconds": round(t_all, 1),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coco", type=Path, default=ROOT / "data" / "coco")
    p.add_argument("--weights", type=Path, default=ROOT / "weights" / "yolov8x.pt")
    p.add_argument("--pred-cache", type=Path, default=ROOT / "cache" / "predictions")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--calib-device", default=None,
                   help="device for calibration (default: --device); the step-2 loop is many tiny ops")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--n-calib-test", type=int, default=None, help="use only the first N images (smoke test)")
    p.add_argument("--configs", default="table3", choices=sorted(CONFIG_SETS))
    p.add_argument("--alpha-tot", type=float, default=0.1, choices=sorted(ALPHAS))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--threads", type=int, default=1, help="torch CPU threads (calibration is many tiny ops)")
    args = p.parse_args()

    tag = f"{args.configs}_a{args.alpha_tot}_bs{args.batch_size}_{args.calib_device or args.device}" + (f"_n{args.n_calib_test}" if args.n_calib_test else "")
    out = args.out or ROOT / "results" / f"seqcrc_yolov8x_{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)
    if not logger.handlers:  # cods configures the root logger itself; make ours visible regardless
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(h)
        logger.propagate = False

    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)
    env = env_info(args.device)
    logger.info("Environment: %s", json.dumps(env))
    preds, t_pred, n_cal, n_test = build_preds(args, args.device)
    for cfg in CONFIG_SETS[args.configs]:
        logger.info("Running %s at alpha_tot=%s", cfg, args.alpha_tot)
        row = run_config(cfg, args.alpha_tot, preds, args.calib_device or args.device)
        row.update({"n_cal": n_cal, "n_test": n_test, "batch_size": args.batch_size,
                    "calib_device": args.calib_device or args.device,
                    "prediction_seconds": round(t_pred, 1), "env": env})
        logger.info(
            "RESULT cnf %.4f/%.3f | loc %.4f/%.3f | cls %.4f/%.3f | global %.4f | thr %.4f | %.0fs",
            row["confidence_risk"], row["confidence_size"], row["localization_risk"], row["localization_size"],
            row["classification_risk"], row["classification_size"], row["global_risk"],
            row["confidence_threshold"], row["total_seconds"],
        )
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    logger.info("Results appended to %s", out)


if __name__ == "__main__":
    os.environ.setdefault("YOLO_VERBOSE", "False")
    main()
