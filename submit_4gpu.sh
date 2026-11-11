#!/usr/bin/env bash
# Usage: CODE_DIR="$PWD" SEED=3408 EPOCHS=50 bash submit_4gpu.sh MLP_FLOOR
set -euo pipefail
CODE_DIR="${CODE_DIR:-/datapool/home/tanlijun/RAMER/dec-v2/RAMER_unsupervised_AB}"
RUN_ID="${1:?Choose MLP_BASE|MLP_FLOOR|QKV_BASE|QKV_FLOOR|B_RANDOM|B_PAIRED}"
SEED="${SEED:-3408}"
RUN_TAG="${RUN_TAG:-v1}"
TIME_LIMIT="${TIME_LIMIT:-0}"
AFTER_JOB_ID="${AFTER_JOB_ID:-}"
case "$RUN_ID" in
 MLP_BASE|MLP_FLOOR|QKV_BASE|QKV_FLOOR) SBATCH="$CODE_DIR/run_legacy_attention_4gpu.sbatch" ;;
 B_RANDOM|B_PAIRED) SBATCH="$CODE_DIR/run_B_sampler_4gpu.sbatch" ;;
 *) echo "Unknown RUN_ID $RUN_ID" >&2; exit 2 ;;
esac
[[ -f "$SBATCH" ]] || { echo "Missing $SBATCH" >&2; exit 2; }
[[ -z "$AFTER_JOB_ID" || "$AFTER_JOB_ID" =~ ^[0-9]+$ ]] || { echo "AFTER_JOB_ID must be a numeric job ID" >&2; exit 2; }
mkdir -p "$CODE_DIR/slurm_logs"
ARGS=(--parsable --export=ALL --time="$TIME_LIMIT" --job-name="r_${RUN_ID}"
 --output="$CODE_DIR/slurm_logs/${RUN_ID}_${RUN_TAG}_%j.out"
 --error="$CODE_DIR/slurm_logs/${RUN_ID}_${RUN_TAG}_%j.err")
if [[ -n "$AFTER_JOB_ID" ]];then ARGS+=(--dependency="afterok:$AFTER_JOB_ID");fi
RESULT=$(CODE_DIR="$CODE_DIR" RUN_ID="$RUN_ID" SEED="$SEED" RUN_TAG="$RUN_TAG" \
 EPOCHS="${EPOCHS:-50}" PAIRS_PER_BATCH="${PAIRS_PER_BATCH:-2}" \
 sbatch "${ARGS[@]}" "$SBATCH")
JOB_ID="${RESULT%%;*}"
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch result: $RESULT" >&2; exit 2; }
printf 'RUN_ID=%s JOB_ID=%s predecessor=%s\n' "$RUN_ID" "$JOB_ID" "${AFTER_JOB_ID:-none}"