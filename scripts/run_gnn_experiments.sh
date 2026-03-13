#!/bin/bash
# Launcher script for GNN and GNNViT experiments on QM7, QM8, QM9

set -e

# Define color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Array of experiments to run
EXPERIMENTS=(
    # "gnnqm7"
    # "gnnqm8"
    # "gnnqm9"
    "gnnvitqm7"
    "gnnvitqm8"
    "gnnvitqm9"
)

TOTAL=${#EXPERIMENTS[@]}
CURRENT=0

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting GNN/GNNViT experiments${NC}"
echo -e "${BLUE}Total experiments: $TOTAL${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Run each experiment
for exp in "${EXPERIMENTS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo -e "${GREEN}[$CURRENT/$TOTAL]${NC} Running: $exp"
    echo "Command: python reg_transfo/main.py experiment=$exp"
    echo ""

    python reg_transfo/main.py experiment=$exp

    echo -e "${GREEN}✓ Completed: $exp${NC}"
    echo ""
done

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}All experiments completed!${NC}"
echo -e "${BLUE}========================================${NC}"
