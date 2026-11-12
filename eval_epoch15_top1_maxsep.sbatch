#!/usr/bin/env bash
#SBATCH --job-name=ramer_floor15_prf
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=0
set -euo pipefail

CODE_DIR="${CODE_DIR:-/datapool/home/tanlijun/RAMER/dec-v2/RAMER_unsupervised_AB}"
RAMER_ROOT="${RAMER_ROOT:-/datapool/home/tanlijun/RAMER}"
TEST_DIR="${TEST_DIR:-$RAMER_ROOT/data/test_data}"
PYTHON="${PYTHON:-/datapool/home/tanlijun/.conda/envs/ramer/bin/python}"
CHECKPOINT="${CHECKPOINT:-$CODE_DIR/experiments_next_full/FLOOR_FULL_seed3408_fp32_full50_v1/seq_only_epoch_015.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-$CODE_DIR/eval_floor15_top1_maxsep}"
BATCH_SIZE="${BATCH_SIZE:-4}"

for file in "$CHECKPOINT" "$CODE_DIR/eval_epoch15_top1_maxsep.py" \
            "$CODE_DIR/legacy_inference_original.py" \
            "$CODE_DIR/eval_full_epoch15_3sets.py"; do
  [[ -f "$file" ]] || { echo "Missing required file: $file" >&2; exit 2; }
done
for f in Price-149.json price.csv NEW-392.json new.csv ram255.json ram255.csv; do
  [[ -f "$TEST_DIR/$f" ]] || { echo "Missing test file: $TEST_DIR/$f" >&2; exit 2; }
done
[[ -x "$PYTHON" ]] || { echo "Python executable not found: $PYTHON" >&2; exit 2; }

cd "$CODE_DIR"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
printf '[EVAL PRF] host=%s checkpoint=%s output=%s\n' "$(hostname)" "$CHECKPOINT" "$OUTPUT_DIR"
ARGS=(
 --checkpoint "$CHECKPOINT"
 --encoder-dir "$RAMER_ROOT/dec-v2"
 --test-dir "$TEST_DIR"
 --output-dir "$OUTPUT_DIR"
 --batch-size "$BATCH_SIZE"
)
if [[ -n "${BACKGROUND_JSON:-}" ]]; then ARGS+=(--background-json "$BACKGROUND_JSON"); fi
if [[ -n "${BACKGROUND_DICT:-}" ]]; then ARGS+=(--background-dict "$BACKGROUND_DICT"); fi
if [[ "${FORCE_EVAL:-0}" == 1 ]]; then ARGS+=(--force); fi
"$PYTHON" "$CODE_DIR/eval_epoch15_top1_maxsep.py" "${ARGS[@]}"
