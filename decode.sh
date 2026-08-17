#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: ./decode.sh <model.pt> [reconstruct.py options]"
    exit 1
fi

python scripts/reconstruct.py "$@"
