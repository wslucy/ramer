#!/usr/bin/env bash
# 统一资源入口：GPUS=1/2/3/4，GPU_NODE=auto 或 node105。
# 例如 GPUS=2 GPU_NODE=auto bash submit_4gpu.sh MLP_FLOOR
set -euo pipefail

CODE_DIR="${CODE_DIR:-/datapool/home/tanlijun/RAMER/dec-v2/RAMER_unsupervised_AB}"
RUN_ID="${1:?Choose MLP_BASE|MLP_FLOOR|QKV_BASE|QKV_FLOOR|B_RANDOM|B_PAIRED|EVAL15}"
GPUS="${GPUS:-1}"
GPU_NODE="${GPU_NODE:-auto}"
GPU_PARTITION="${GPU_PARTITION:-a100}"
SEED="${SEED:-3408}"
RUN_TAG="${RUN_TAG:-v1}"
TIME_LIMIT="${TIME_LIMIT:-0}"
AFTER_JOB_ID="${AFTER_JOB_ID:-}"
CPUS_PER_GPU="${CPUS_PER_GPU:-8}"
MEM_PER_GPU_GB="${MEM_PER_GPU_GB:-48}"
BATCH_SIZE_NOTE=16

case "$RUN_ID" in
 MLP_BASE|MLP_FLOOR|QKV_BASE|QKV_FLOOR)
   SBATCH="$CODE_DIR/run_legacy_attention_4gpu.sbatch" ;;
 B_RANDOM|B_PAIRED)
   SBATCH="$CODE_DIR/run_B_sampler_4gpu.sbatch" ;;
 EVAL15)
   SBATCH="$CODE_DIR/eval_full_epoch15_3sets.sbatch"
   if [[ "$GPUS" != 1 ]]; then echo "EVAL15 requires GPUS=1" >&2; exit 2; fi ;;
 *) echo "Unknown RUN_ID $RUN_ID" >&2; exit 2 ;;
esac

[[ -f "$SBATCH" ]] || { echo "Missing $SBATCH" >&2; exit 2; }
[[ "$GPUS" =~ ^[1-9][0-9]*$ ]] || { echo "GPUS must be a positive integer" >&2; exit 2; }
[[ "$CPUS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || { echo "CPUS_PER_GPU must be a positive integer" >&2; exit 2; }
[[ "$MEM_PER_GPU_GB" =~ ^[1-9][0-9]*$ ]] || { echo "MEM_PER_GPU_GB must be a positive integer" >&2; exit 2; }
[[ -z "$AFTER_JOB_ID" || "$AFTER_JOB_ID" =~ ^[0-9]+$ ]] || { echo "AFTER_JOB_ID must be a numeric Slurm job ID" >&2; exit 2; }
[[ "$GPU_PARTITION" =~ ^[a-zA-Z0-9,_-]+$ ]] || { echo "GPU_PARTITION contains invalid characters" >&2; exit 2; }
[[ "$GPU_NODE" == auto || "$GPU_NODE" =~ ^[a-zA-Z0-9,_-]+$ ]] || { echo "GPU_NODE must be auto or Slurm nodelist" >&2; exit 2; }

mkdir -p "$CODE_DIR/slurm_logs"
ARGS=(
 --parsable --export=ALL
 --partition="$GPU_PARTITION"
 --nodes=1 --ntasks=1
 --gres="gpu:$GPUS"
 --cpus-per-task="$((GPUS * CPUS_PER_GPU))"
 --mem="$(( GPUS * MEM_PER_GPU_GB < 64 ? 64 : GPUS * MEM_PER_GPU_GB ))G"
 --time="$TIME_LIMIT"
 --job-name="r_${RUN_ID}"
 --output="$CODE_DIR/slurm_logs/${RUN_ID}_${RUN_TAG}_%j.out"
 --error="$CODE_DIR/slurm_logs/${RUN_ID}_${RUN_TAG}_%j.err"
)
if [[ "$GPU_NODE" != auto ]]; then ARGS+=(--nodelist="$GPU_NODE"); fi
if [[ -n "$AFTER_JOB_ID" ]]; then ARGS+=(--dependency="afterok:$AFTER_JOB_ID"); fi

printf '[SUBMIT] id=%s partition=%s node=%s gpus=%s cpus=%s mem=%sG\n' \
 "$RUN_ID" "$GPU_PARTITION" "$GPU_NODE" "$GPUS" "$((GPUS * CPUS_PER_GPU))" \
 "$(( GPUS * MEM_PER_GPU_GB < 64 ? 64 : GPUS * MEM_PER_GPU_GB ))"
RESULT=$(CODE_DIR="$CODE_DIR" RUN_ID="$RUN_ID" SEED="$SEED" RUN_TAG="$RUN_TAG" \
 EPOCHS="${EPOCHS:-50}" PAIRS_PER_BATCH="${PAIRS_PER_BATCH:-2}" GPUS="$GPUS" \
 sbatch "${ARGS[@]}" "$SBATCH")
JOB_ID="${RESULT%%;*}"
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch result: $RESULT" >&2; exit 2; }
printf 'RUN_ID=%s JOB_ID=%s predecessor=%s\n' "$RUN_ID" "$JOB_ID" "${AFTER_JOB_ID:-none}"
