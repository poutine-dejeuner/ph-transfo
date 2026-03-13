#!/bin/bash
set -e
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

EXPERIMENTS=(
    "gnnqm7"
    "gnnqm8"
    "gnnqm9"
    "gnnvitqm7"
    "gnnvitqm8"
    "gnnvitqm9"
)

TOTAL=${#EXPERIMENTS[@]}
CURRENT=0

for exp in "${EXPERIMENTS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo -e "${GREEN}[$CURRENT/$TOTAL]${NC} Running: $exp"
    echo "Command: python reg_transfo/main.py experiment=$exp"
    echo ""

    python reg_transfo/main.py experiment=$exp trainer=debug datamodule.num_workers=0
    #datamodule.pin_memory=False

    echo -e "${GREEN}✓ Completed: $exp${NC}"
    echo ""
done

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}All experiments completed!${NC}"
echo -e "${BLUE}========================================${NC}"
