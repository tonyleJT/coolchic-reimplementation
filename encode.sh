#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: ./encode.sh <image> [train.py options]"
    exit 1
fi

python scripts/train.py "$@"
