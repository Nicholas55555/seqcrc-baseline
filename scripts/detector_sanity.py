"""Sanity check of the detector under the SeqCRC reproduction: COCO val2017 box AP of YOLOv8x.

Two sets of boxes are scored with the official COCO API (pycocotools):

* native   - Ultralytics' own predict() (letterbox undone by Ultralytics), conf 0.001, NMS IoU 0.7,
             300 boxes, imgsz 640: the settings behind Ultralytics' published 53.9 mAP50-95.
* cods     - the boxes the official cods YOLO wrapper hands to SeqCRC (read from the prediction
             cache written by run_seqcrc.py): raw head output rescaled by original/letterboxed size,
             class-agnostic NMS at IoU 0.5, 1e-3 floor; class = argmax of the 91-way class vector.

Usage:
    python scripts/detector_sanity.py native
    python scripts/detector_sanity.py cods
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COCO = ROOT / "data" / "coco"
ANN = COCO / "annotations" / "instances_val2017.json"
# Ultralytics' 80 contiguous classes -> COCO's 91-id category ids
COCO80_TO_91 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
                31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
                57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86,
                87, 88, 89, 90]


def coco_eval(dets: list[dict], img_ids: list[int], tag: str) -> dict:
    from pycocotools.coco import COCO as COCOApi
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCOApi(str(ANN))
    dt = gt.loadRes(dets) if dets else None
    ev = COCOeval(gt, dt, "bbox")
    ev.params.imgIds = img_ids
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    print(f"--- {tag}: COCO bbox evaluation on {len(img_ids)} images ---")
    print(buf.getvalue().split("DONE", 1)[-1].strip())
    names = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100", "AR_s", "AR_m", "AR_l"]
    return {k: round(float(v), 4) for k, v in zip(names, ev.stats)}


def native(args) -> dict:
    import torch
    from ultralytics import YOLO

    ann = json.load(open(ANN, encoding="utf-8"))
    images = ann["images"]
    if args.n:
        images = images[: args.n]
    model = YOLO(str(ROOT / "weights" / "yolov8x.pt"))
    dets, t0 = [], time.time()
    for k in range(0, len(images), args.batch):
        chunk = images[k : k + args.batch]
        res = model.predict([str(COCO / "val2017" / im["file_name"]) for im in chunk], imgsz=640, conf=0.001,
                            iou=args.iou, agnostic_nms=args.agnostic, max_det=300,
                            device=0 if torch.cuda.is_available() else "cpu", verbose=False)
        for im, r in zip(chunk, res):
            b = r.boxes
            for (x1, y1, x2, y2), s, c in zip(b.xyxy.tolist(), b.conf.tolist(), b.cls.int().tolist()):
                dets.append({"image_id": im["id"], "category_id": COCO80_TO_91[c],
                             "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                             "score": round(s, 5)})
    secs = time.time() - t0
    out = coco_eval(dets, [im["id"] for im in images],
                    f"native Ultralytics YOLOv8x (NMS IoU {args.iou}, agnostic={args.agnostic})")
    out.update({"n_images": len(images), "seconds": round(secs, 1), "boxes": len(dets),
                "nms_iou": args.iou, "agnostic_nms": args.agnostic})
    return out


def cods(args) -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_seqcrc  # noqa: F401  (applies the ODPredictions tuple fix before unpickling)

    ann = json.load(open(ANN, encoding="utf-8"))
    file_to_id = {im["file_name"]: im["id"] for im in ann["images"]}
    dets, img_ids = [], []
    if args.pkl:
        files = [Path(x) for x in args.pkl]
    else:
        # the same cache key as cods' ODModel.build_predictions, for the full 2,500/2,500 split at --batch-size
        from hashlib import sha256

        from cods.od.data import MSCOCODataset

        dataset = MSCOCODataset(root=str(COCO), split="val")
        files = []
        for split_name, part in zip(("cal", "test"), dataset.split_dataset(0.5, shuffle=False)):
            key = (f"{part.root}_mscoco_{split_name}_{args.batch_size}_False_{ROOT / 'weights' / 'yolov8x.pt'}"
                   f"_object_detection_{part.image_ids}")
            files.append(ROOT / "cache" / "predictions" / f"{sha256(key.encode()).hexdigest()}.pkl")
        missing = [f for f in files if not f.exists()]
        if missing:
            raise SystemExit(f"no prediction cache for batch size {args.batch_size}: run "
                             f"'run_seqcrc.py --configs none --batch-size {args.batch_size}' first")
    for pkl in files:
        with open(pkl, "rb") as f:
            preds = pickle.load(f)
        for path, boxes, conf, cls in zip(preds.image_paths, preds.pred_boxes, preds.confidences, preds.pred_cls):
            iid = file_to_id[Path(path).name]
            img_ids.append(iid)
            boxes, conf, cls = boxes.cpu().numpy(), conf.cpu().numpy(), cls.cpu().numpy()
            cat = cls.argmax(1)
            for (x1, y1, x2, y2), s, c in zip(boxes.tolist(), conf.tolist(), cat.tolist()):
                if c == 0:
                    continue
                dets.append({"image_id": iid, "category_id": int(c), "score": round(float(s), 5),
                             "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]})
    # COCO keeps at most 100 detections per image for AP; rank them by score per image
    by_img: dict[int, list] = {}
    for d in dets:
        by_img.setdefault(d["image_id"], []).append(d)
    dets = [d for v in by_img.values() for d in sorted(v, key=lambda x: -x["score"])[:300]]
    out = coco_eval(dets, img_ids, "cods YOLOv8x wrapper (input to SeqCRC)")
    out.update({"n_images": len(img_ids), "boxes": len(dets),
                "boxes_per_image_mean": round(float(np.mean([len(v) for v in by_img.values()])), 1)})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("which", choices=["native", "cods"])
    p.add_argument("--n", type=int, default=None, help="first N images only (native)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU (native); Ultralytics val default 0.7")
    p.add_argument("--agnostic", action="store_true", help="class-agnostic NMS (native), as cods does")
    p.add_argument("--batch-size", type=int, default=1, help="(cods) batch size the prediction cache was built with")
    p.add_argument("--pkl", nargs="*", default=None, help="explicit prediction-cache files (cods)")
    p.add_argument("--tag", default="", help="suffix for the output file name")
    args = p.parse_args()
    out = native(args) if args.which == "native" else cods(args)
    print(json.dumps(out))
    res = ROOT / "results" / f"detector_sanity_{args.which}{args.tag}.json"
    res.parent.mkdir(exist_ok=True)
    res.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
