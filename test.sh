#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-options/test/EIC-LIE_SDEout.yml}"
WEIGHTS="${2:?Usage: bash test.sh <config.yml> <checkpoint.pth>}"
python basicsr/test.py -opt "$CONFIG" --weights "$WEIGHTS"
