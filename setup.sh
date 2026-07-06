#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3.10}"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend"

# Accept any number of extras as positional args, e.g.:
#   ./setup.sh                  -> default: ik (mink)
#   ./setup.sh all              -> everything
#   ./setup.sh none             -> no extras (base install only)
if [ "$#" -eq 0 ]; then
    EXTRAS=("ik")
elif [ "$#" -eq 1 ] && [ "$1" = "none" ]; then
    EXTRAS=()
else
    EXTRAS=("$@")
fi

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 1
    fi
}

require_cmd uv
require_cmd npm

echo "Creating uv venv with $PYTHON..."
uv venv --python "$PYTHON"

if [ "${#EXTRAS[@]}" -gt 0 ]; then
    echo "Syncing project with extras: ${EXTRAS[*]}..."
    extra_args=()
    for e in "${EXTRAS[@]}"; do
        extra_args+=(--extra "$e")
    done
    uv sync "${extra_args[@]}"
else
    echo "Syncing project (no extras)..."
    uv sync
fi

echo "Installing frontend dependencies..."
npm install --prefix "$FRONTEND_DIR"

echo "Building native MuJoCo stepper..."
uv run python - <<'PY'
from mujoco_vr_teleop.native_stepper import NativeStepper

NativeStepper()
print("Native MuJoCo stepper ready.")
PY

echo "Building frontend..."
npm run build --prefix "$FRONTEND_DIR"

echo ""
echo "Setup complete."
echo ""
echo "Start the supervisor TUI (scene picker on first launch):"
echo "  uv run mujoco-vr-tui --builder.arm-type yam_ultra"
