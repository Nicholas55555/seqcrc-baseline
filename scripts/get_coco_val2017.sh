#!/usr/bin/env bash
# Download MS-COCO val2017 images (778 MB) and annotations (241 MB) into data/coco and check SHA-256.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/coco && cd data/coco
curl -L -o annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
curl -L -o val2017.zip http://images.cocodataset.org/zips/val2017.zip
sha256sum -c - <<'SUMS'
113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268  annotations_trainval2017.zip
4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05  val2017.zip
SUMS
unzip -q -o val2017.zip
unzip -q -o annotations_trainval2017.zip annotations/instances_val2017.json
ls val2017 | wc -l   # 5000
