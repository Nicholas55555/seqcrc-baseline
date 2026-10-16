"""Build the Phase 1 report PDF: refresh the tables and evidence images from results/ and logs/, fill the
status sentence for unfinished full-size runs, and compile report/report.tex with tectonic (or pdflatex).

Usage: python scripts/build_report.py [--tectonic PATH]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tectonic", default=shutil.which("tectonic") or "tectonic")
    a = p.parse_args()
    subprocess.run([PY, str(ROOT / "scripts" / "make_results.py")], check=True, stdout=subprocess.DEVNULL)

    done = {c: (ROOT / "results" / f"seqcrc_yolov8x_{c}_a0.1_bs1_cpu.jsonl").exists() for c in ("F1", "F2", "F3", "F4")}
    missing = [c for c, ok in done.items() if not ok]
    if not missing:
        status = "All four configurations finished at full size."
    else:
        names = " and ".join(missing)
        verb = "was" if len(missing) == 1 else "were"
        rows = "row" if len(missing) == 1 else "rows"
        status = (f"The full-size {names} {'run' if len(missing) == 1 else 'runs'} (APS class sets, the slowest step) "
                  f"{verb} still calibrating when this report was built (\\texttt{{logs/10\\_full\\_*.log}}); Table~1 "
                  f"shows {'its' if len(missing) == 1 else 'their'} 300/300 {rows}.")

    # evidence: the most recent finished full-size run log
    fin = [c for c in ("F3", "F2", "F4", "F1") if done[c]]
    if fin:
        c = fin[0]
        subprocess.run([PY, str(ROOT / "scripts" / "render_evidence.py"), str(ROOT / "logs" / f"10_full_{c}.log"),
                        str(ROOT / "report" / "fig_evidence_run.png"),
                        r"seqcrc_repro: (COCO|Running|RESULT)|lambda_|confidence_threshold",
                        "--max", "9", "--step", "0.16", "--fig-width", "6.2", "--width", "100",
                        "--title", f"$ python scripts/run_seqcrc.py --configs {c} --calib-device cpu --threads 1"],
                       check=True, stdout=subprocess.DEVNULL)
    subprocess.run([PY, str(ROOT / "scripts" / "render_evidence.py"), str(ROOT / "logs" / "11_detector_sanity_native.log"),
                    str(ROOT / "report" / "fig_evidence_detector.png"),
                    r"^---|Average Precision  \(AP\) @\[ IoU=0\.5.*area=   all|Average Recall .*area=   all \| maxDets=100",
                    "--max", "5", "--step", "0.16", "--fig-width", "5.6", "--width", "90",
                    "--title", "$ python scripts/detector_sanity.py native"],
                   check=True, stdout=subprocess.DEVNULL)

    src = (ROOT / "report" / "report.tex").read_text(encoding="utf-8")
    run_log = f"logs/10\\_full\\_{fin[0]}.log" if fin else "logs/09\\_validate\\_runningsum\\_n600.log"
    out = src.replace("F1F4_STATUS", status).replace("RUN_LOG_NAME", run_log)
    build = ROOT / "report" / "report_build.tex"
    build.write_text(out, encoding="utf-8")
    subprocess.run([a.tectonic, build.name], cwd=build.parent, check=True)
    pdf = build.with_suffix(".pdf")
    final = ROOT / "report" / "Phase1_Baseline_Report_Nicholas_Taylor.pdf"
    shutil.move(str(pdf), final)
    print(f"built {final}; full runs finished: {[c for c, ok in done.items() if ok]}")


if __name__ == "__main__":
    main()
