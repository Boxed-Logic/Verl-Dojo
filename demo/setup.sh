#!/usr/bin/env bash
# setup.sh — copies custom tool and reward manager into the installed
# verl-tool package, installs dependencies, and builds the dataset.
#
# Run once before training:   bash setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VERL_TOOL_ROOT=$(python -c "
import verl_tool, os
print(os.path.dirname(os.path.abspath(verl_tool.__file__)))
")
echo "verl_tool root: $VERL_TOOL_ROOT"

echo "Copying arithmetic.py → $VERL_TOOL_ROOT/servers/tools/"
cp "$SCRIPT_DIR/tools/arithmetic.py" "$VERL_TOOL_ROOT/servers/tools/arithmetic.py"

echo "Copying bm25_reward.py → $VERL_TOOL_ROOT/workers/reward_manager/"
cp "$SCRIPT_DIR/reward_manager/bm25_reward.py" "$VERL_TOOL_ROOT/workers/reward_manager/bm25_reward.py"

echo "Installing rank-bm25..."
pip install rank-bm25 --quiet

echo "Generating dataset from verl-team/gsm8k-v0.4.1..."
python "$SCRIPT_DIR/data/prepare_data.py"

echo ""
echo "Setup complete. Run: bash train_granite_grpo.sh"
