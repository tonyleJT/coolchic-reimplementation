#!/usr/bin/env bash
set -euo pipefail

python scripts/benchmark.py --config configs/benchmark.yaml
python scripts/plot_rd.py \
    --input results/kodak_summary.csv \
    --output results/kodak_rd_curve.png
