"""Collect run outputs into results/RESULTS.md and the LaTeX tables used by the Phase 1 report.

Reads the full-size runs (results/seqcrc_yolov8x_F*_a0.1_bs1_cpu.jsonl), the 300/300 subset runs
(results/subset300_F*.jsonl, results/validate_runningsum_n600.jsonl for F3), the batch-size diagnostic
(results/diag_bs8_n600.jsonl), the detector checks (results/detector_sanity_*.json) and the paper's values
(scripts/paper_values.json). Nothing is typed in by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PAPER = json.loads((ROOT / "scripts" / "paper_values.json").read_text(encoding="utf-8"))
TASKS = ("confidence", "localization", "classification")
SHORT = {"box_count_recall": "bc-recall", "box_count_threshold": "bc-thresh", "hausdorff": "Hausdorff",
         "lac": "LAC", "mix": "Mix", "pixelwise": "pixel", "boxwise": "box", "thresholded": "thresh",
         "aps": "APS", "additive": "add", "multiplicative": "mult"}
SUBSET = {"F1": "subset300_F1.jsonl", "F2": "subset300_F2.jsonl", "F3": "validate_runningsum_n600.jsonl",
          "F4": "subset300_F4.jsonl"}


def last_row(path: Path):
    if not path.exists():
        return None
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return rows[-1] if rows else None


def describe(cfg):
    m, c, loc, cls, box = cfg
    return f"{SHORT[m]} / {SHORT[c]} / {SHORT[loc]}-{SHORT[box]} / {SHORT[cls]}"


def cells(r, fmt_size="{:.2f}", fmt_risk="{:.3f}"):
    return [fmt_size.format(r[t + "_size"]) for t in TASKS], [fmt_risk.format(r[t + "_risk"]) for t in TASKS]


def main():
    md = ["# Results: SeqCRC reproduction (YOLOv8x, COCO val2017, alpha_tot = 0.1)", "",
          "Budgets: alpha_cnf = 0.02, alpha_loc = alpha_cls = 0.05; Global = mean over test images of "
          "max(loc, cls) loss, bounded by alpha_loc + alpha_cls = 0.1. Each cell is set size / test risk. "
          "Paper: arXiv:2505.24038v2 App. Table VII (one 2,500/2,500 split). Full: our 2,500/2,500 run "
          "(same split as the paper's code). Subset: first 300 calibration / next 300 test images.", "",
          "| ID | Config (matching / conf. loss / loc. loss-set / cls. set) | Source | Cnf. size / risk | Loc. size / risk | Cls. size / risk | Global risk | Calibration time |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    tex = []
    for cid in ("F1", "F2", "F3", "F4"):
        p = PAPER[cid]
        desc = describe(p["config"])
        sources = [("Paper", p), ("Ours, full", last_row(RES / f"seqcrc_yolov8x_{cid}_a0.1_bs1_cpu.jsonl")),
                   ("Ours, 300/300", last_row(RES / SUBSET[cid]))]
        first = True
        for name, r in sources:
            if r is None:
                md.append(f"| {cid} | {desc} | {name} | not finished | | | | |")
                tex.append((rf"\multirow{{3}}{{*}}{{{cid}}} & \multirow{{3}}{{*}}{{\parbox{{3.3cm}}{{\raggedright {desc}}}}}" if first else " & ")
                           + rf" & {name} & \multicolumn{{7}}{{c}}{{(run not finished)}} \\")
                first = False
                continue
            sz, rk = cells(r)
            t = r.get("calibration_seconds")
            tstr = f"{t / 60:.0f} min" if t else ""
            md.append(f"| {cid} | {desc} | {name} | " + " | ".join(f"{a} / {b}" for a, b in zip(sz, rk))
                      + f" | {r['global_risk']:.3f} | {tstr} |")
            lead = (rf"\multirow{{3}}{{*}}{{{cid}}} & \multirow{{3}}{{*}}{{\parbox{{3.3cm}}{{\raggedright {desc}}}}}"
                    if first else " & ")
            tex.append(lead + f" & {name} & " + " & ".join(f"{a} & {b}" for a, b in zip(sz, rk))
                       + f" & {r['global_risk']:.3f} \\\\")
            first = False
        tex.append(r"\addlinespace[3pt]")
    (RES / "results_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    # detector checks
    det = []
    rows = [("Ultralytics model table (paper's detector)", None, 53.9, None),
            ("Ours: Ultralytics predict, NMS IoU 0.7 per class", "detector_sanity_native.json", None, None),
            ("Ours: Ultralytics predict, class-agnostic NMS IoU 0.5", "detector_sanity_native_agnostic_iou05.json", None, None),
            ("Ours: cods YOLO wrapper, batch 1 (SeqCRC input)", "detector_sanity_cods.json", None, None),
            ("Ours: cods YOLO wrapper, batch 8", "detector_sanity_cods_bs8.json", None, None)]
    md += ["", "## Detector check (COCO val2017, 5,000 images, pycocotools)", "",
           "| Boxes scored | AP | AP50 | AP75 |", "| --- | --- | --- | --- |"]
    for name, f, ap, _ in rows:
        if f is None:
            md.append(f"| {name} | {ap:.1f} | | |")
            det.append(rf"{name} & {ap:.1f} & -- & -- \\")
            continue
        path = RES / f
        if not path.exists():
            md.append(f"| {name} | not run | | |")
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        md.append(f"| {name} | {100 * d['AP']:.1f} | {100 * d['AP50']:.1f} | {100 * d['AP75']:.1f} |")
        det.append(rf"{name} & {100 * d['AP']:.1f} & {100 * d['AP50']:.1f} & {100 * d['AP75']:.1f} \\")
    (RES / "detector_table.tex").write_text("\n".join(det) + "\n", encoding="utf-8")

    # batch-size diagnostic on the 300/300 subset, F3 configuration
    b1, b8 = last_row(RES / "validate_runningsum_n600.jsonl"), last_row(RES / "diag_bs8_n600.jsonl")
    if b1 and b8:
        md += ["", "## Batch-size diagnostic (F3 configuration, 300/300 subset)", "",
               "| cods YOLO batch size | Cnf. size / risk | Loc. size (stretch) / risk | Cls. size / risk | Global risk |",
               "| --- | --- | --- | --- | --- |"]
        for name, r in (("1 (ours)", b1), ("8", b8)):
            sz, rk = cells(r)
            md.append(f"| {name} | " + " | ".join(f"{a} / {b}" for a, b in zip(sz, rk)) + f" | {r['global_risk']:.3f} |")
    (RES / "RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()


def write_report_tables():
    tpl = (ROOT / "scripts" / "report_tables_template.tex").read_text(encoding="utf-8")
    rows = (RES / "results_table.tex").read_text(encoding="utf-8").rstrip()
    det = (RES / "detector_table.tex").read_text(encoding="utf-8").rstrip()
    rows = rows.rsplit(r"\addlinespace[3pt]", 1)[0].rstrip()  # no extra space before \bottomrule
    out = tpl.replace("@@SEQCRC_ROWS@@", rows).replace("@@DET_ROWS@@", det)
    (RES / "report_tables.tex").write_text(out, encoding="utf-8")


write_report_tables()
