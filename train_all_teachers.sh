#!/usr/bin/env bash

# Train the four unified SF_TRON2A teachers sequentially.
#
# The continuous teacher is trained first.  Its checkpoint initializes the
# stairs, obstacle and gap teachers so that all experts retain a compatible
# nominal gait before specializing.

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-1024}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
SEED="${SEED:-42}"
HEADLESS="${HEADLESS:-1}" # Run in headless mode by default, since this script is intended for training on a server.

CONTINUOUS_ITERATIONS="${CONTINUOUS_ITERATIONS:-15000}"
STAIRS_ITERATIONS="${STAIRS_ITERATIONS:-15000}"
OBSTACLE_ITERATIONS="${OBSTACLE_ITERATIONS:-15000}"
GAP_ITERATIONS="${GAP_ITERATIONS:-15000}"

RUN_TAG="${RUN_TAG:-teachers_$(date +%Y%m%d_%H%M%S)}"
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "[ERROR] RUN_TAG may only contain letters, numbers, '.', '_' and '-'." >&2
    exit 2
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[ERROR] Python executable not found: $PYTHON_BIN" >&2
    echo "Activate the isaaclab environment or set PYTHON_BIN explicitly." >&2
    exit 2
fi

on_error() {
    local exit_code=$?
    echo "[ERROR] Teacher training stopped at line ${BASH_LINENO[0]} (exit ${exit_code})." >&2
    exit "$exit_code"
}
trap on_error ERR

latest_checkpoint_for_run() {
    local experiment_name="$1"
    local run_name="$2"
    "$PYTHON_BIN" - "$PROJECT_ROOT/logs/rsl_rl/$experiment_name" "$run_name" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
run_suffix = "_" + sys.argv[2]
runs = [path for path in root.glob("*") if path.is_dir() and path.name.endswith(run_suffix)]
if not runs:
    raise SystemExit(f"No completed run ending in '{run_suffix}' was found under {root}")
run = max(runs, key=lambda path: path.stat().st_mtime_ns)
models = list(run.glob("model_*.pt"))
if not models:
    raise SystemExit(f"No model checkpoint was produced in {run}")

def iteration(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1

print(max(models, key=lambda path: (iteration(path), path.stat().st_mtime_ns)).resolve())
PY
}

train_teacher() {
    local result_variable="$1"
    local label="$2"
    local task="$3"
    local experiment_name="$4"
    local iterations="$5"
    local initialization_checkpoint="${6:-}"
    local run_name="${RUN_TAG}_${label}"

    echo
    echo "======================================================================"
    echo "[INFO] Training ${label} teacher"
    echo "[INFO] task=${task} envs=${NUM_ENVS} iterations=${iterations} device=${DEVICE}"
    if [[ -n "$initialization_checkpoint" ]]; then
        echo "[INFO] initializing from ${initialization_checkpoint}"
    fi
    echo "======================================================================"

    local train_args=(
        "$PYTHON_BIN" scripts/rsl_rl/train.py
        --task "$task"
        --num_envs "$NUM_ENVS"
        --max_iterations "$iterations"
        --save_interval "$SAVE_INTERVAL"
        --seed "$SEED"
        --device "$DEVICE"
        --run_name "$run_name"
    )
    if [[ "$HEADLESS" == "1" ]]; then
        train_args+=(--headless)
    fi
    if [[ -n "$initialization_checkpoint" ]]; then
        train_args+=(--resume True --checkpoint_path "$initialization_checkpoint")
    fi

    "${train_args[@]}"
    local checkpoint
    checkpoint="$(latest_checkpoint_for_run "$experiment_name" "$run_name")"
    printf -v "$result_variable" '%s' "$checkpoint"
    echo "[INFO] ${label} checkpoint: ${checkpoint}"
}

train_teacher CONTINUOUS_CHECKPOINT \
    continuous \
    Isaac-Limx-SF-TRON2A-Teacher-Continuous-v0 \
    sf_tron_2a_teacher_continuous \
    "$CONTINUOUS_ITERATIONS"

train_teacher STAIRS_CHECKPOINT \
    stairs \
    Isaac-Limx-SF-TRON2A-Teacher-Stairs-v0 \
    sf_tron_2a_teacher_stairs \
    "$STAIRS_ITERATIONS" \
    "$CONTINUOUS_CHECKPOINT"

train_teacher OBSTACLE_CHECKPOINT \
    obstacle \
    Isaac-Limx-SF-TRON2A-Teacher-Obstacle-v0 \
    sf_tron_2a_teacher_obstacle \
    "$OBSTACLE_ITERATIONS" \
    "$CONTINUOUS_CHECKPOINT"

train_teacher GAP_CHECKPOINT \
    gap \
    Isaac-Limx-SF-TRON2A-Teacher-Gap-v0 \
    sf_tron_2a_teacher_gap \
    "$GAP_ITERATIONS" \
    "$CONTINUOUS_CHECKPOINT"

MANIFEST_PATH="$PROJECT_ROOT/logs/rsl_rl/teacher_checkpoints_${RUN_TAG}.env"
{
    printf 'CONTINUOUS_CHECKPOINT=%q\n' "$CONTINUOUS_CHECKPOINT"
    printf 'STAIRS_CHECKPOINT=%q\n' "$STAIRS_CHECKPOINT"
    printf 'OBSTACLE_CHECKPOINT=%q\n' "$OBSTACLE_CHECKPOINT"
    printf 'GAP_CHECKPOINT=%q\n' "$GAP_CHECKPOINT"
} > "$MANIFEST_PATH"

echo
echo "======================================================================"
echo "[DONE] All four teachers finished successfully"
echo "======================================================================"
printf 'Continuous: %s\n' "$CONTINUOUS_CHECKPOINT"
printf 'Stairs:     %s\n' "$STAIRS_CHECKPOINT"
printf 'Obstacle:   %s\n' "$OBSTACLE_CHECKPOINT"
printf 'Gap:        %s\n' "$GAP_CHECKPOINT"
printf 'Manifest:   %s\n' "$MANIFEST_PATH"

echo
echo "Compatibility check:"
printf '%q ' "$PYTHON_BIN" scripts/rsl_rl/check_teacher_compatibility.py \
    --teacher_continuous "$CONTINUOUS_CHECKPOINT" \
    --teacher_stairs "$STAIRS_CHECKPOINT" \
    --teacher_obstacle "$OBSTACLE_CHECKPOINT" \
    --teacher_gap "$GAP_CHECKPOINT"
echo

echo
echo "Multi-teacher distillation:"
printf '%q ' "$PYTHON_BIN" scripts/rsl_rl/distill.py \
    --task Isaac-Limx-SF-TRON2A-MultiTeacher-v0 \
    --num_envs 2048 \
    --teacher_continuous "$CONTINUOUS_CHECKPOINT" \
    --teacher_stairs "$STAIRS_CHECKPOINT" \
    --teacher_obstacle "$OBSTACLE_CHECKPOINT" \
    --teacher_gap "$GAP_CHECKPOINT" \
    --headless
echo
