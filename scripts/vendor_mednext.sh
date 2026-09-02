#!/usr/bin/env bash
# Vendor MedNeXt's ARCHITECTURE only (MICCAI 2023, arXiv:2303.09975).
# The upstream repo is a fork of the nnU-Net v1 pipeline and its standalone
# import path is documented as broken (MedNeXt issue #22), so we copy the
# network module and skip the pipeline entirely.
# Use kernel 3 ONLY: k5 needs UpKern init from a trained k3 model.
set -euo pipefail
DST="glioseg/vendor/mednext"
rm -rf /tmp/mednext_src "$DST"; mkdir -p "$DST"
git clone --depth 1 https://github.com/MIC-DKFZ/MedNeXt /tmp/mednext_src
SRC=$(find /tmp/mednext_src -type d -name mednextv1 | head -1)
[ -z "$SRC" ] && { echo "mednextv1 dir not found; inspect /tmp/mednext_src"; exit 1; }
cp -r "$SRC"/*.py "$DST"/
touch "$DST/__init__.py"
echo "Vendored -> $DST"; ls "$DST"
echo "NOTE: you may need to fix relative imports inside the copied files."
