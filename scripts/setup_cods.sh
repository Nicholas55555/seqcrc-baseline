#!/usr/bin/env bash
# Clone the official SeqCRC code (cods) at the exact commit used here and apply the reproduction patch.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p external
if [ ! -d external/cods ]; then
  git clone https://github.com/leoandeol/cods.git external/cods
fi
git -C external/cods checkout 269ff11a2967ede680da508ade24f58149959dba
git -C external/cods apply --whitespace=nowarn "$PWD/patches/cods-reproduction.patch"
echo "cods ready at external/cods (commit 269ff11 + patches/cods-reproduction.patch)"
