#!/bin/bash
# Launch all 9 baseline + Betti + Topo GPS experiments on QM7, QM8, QM9

set -e

EXPERIMENTS=(
    # "gps_baseline_qm7"
    # "gps_baseline_qm8"
    # "gps_baseline_qm9"
    # "gps_betti_qm7"
    # "gps_betti_qm8"
    # "gps_betti_qm9"
    "gps_topo_qm7"
    "gps_topo_qm8"
    "gps_topo_qm9"
)

echo "=========================================="
echo "Launching 9 GPS experiments"
echo "=========================================="

for exp in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Starting: $exp"
    echo "=========================================="
    python ph_transfo/main.py experiment="$exp"
    echo "Completed: $exp"
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
