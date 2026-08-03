#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-options/train/EIC-LIE_SDEout.yml}"
python basicsr/train.py -opt "$CONFIG"
