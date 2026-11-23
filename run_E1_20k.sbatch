#!/usr/bin/env bash
#SBATCH --job-name=ramer_E1_20k
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=192G
#SBATCH --time=0
set -euo pipefail

CODE_DIR="${CODE_DIR:-/datapool/home/tanlijun/RAMER/dec-v2/RAMER_unsupervised_AB}"
ENCODER_DIR="${ENCODER_DIR:-/datapool/home/tanlijun/RAMER/dec-v2}"
PYTHON="${PYTHON:-/datapool/home/tanlijun/.conda/envs/ramer/bin/python}"
RAMER_ROOT="${RAMER_ROOT:-/datapool/home/tanlijun/RAMER}"
DATA_DIR="${DATA_DIR:-$RAMER_ROOT/data}"
GRAPH_DIR="${GRAPH_DIR:-/datapool/home/tanlijun/GearNet/pdbs/features/graph}"
NODE_DIR="${NODE_DIR:-/datapool/home/tanlijun/GearNet/pdbs/features/node}"
SUBSET_FILE="${SUBSET_FILE:-$CODE_DIR/subsets/fixed_20k_seed3408.json}"
SEED="${SEED:-3408}"
EPOCHS="${EPOCHS:-50}"
RUN_TAG="${RUN_TAG:-v1}"
ROUTER_MODE="${ROUTER_MODE:-gated}"
ROUTE_WEIGHT="${ROUTE_WEIGHT:-0.1}"
ROUTE_SCALE="${ROUTE_SCALE:-0.25}"
BATCH_SIZE="${BATCH_SIZE:-16}"

case "$ROUTER_MODE" in gated|uniform) ;; *) echo "Invalid ROUTER_MODE=$ROUTER_MODE" >&2; exit 2;; esac
for f in \
  "$CODE_DIR/train_mechpair_light_E1_20k.py" \
  "$ENCODER_DIR/Data2seq_v2.py" \
  "$DATA_DIR/uniprot_20W_struct_seq_reaction_121_without_new392.json" \
  "$DATA_DIR/updated_rhea-reaction-smiles.json" \
  "$RAMER_ROOT/cluster.py" \
  "$SUBSET_FILE" \
  "$DATA_DIR/test_data/Price-149.json" "$DATA_DIR/test_data/price.csv"; do
  [[ -f "$f" ]] || { echo "Missing: $f" >&2; exit 2; }
done
[[ -x "$PYTHON" ]] || { echo "Missing Python executable: $PYTHON" >&2; exit 2; }

GPU_COUNT="${SLURM_GPUS_ON_NODE:-${GPUS:-4}}"
SAVE_DIR="${SAVE_DIR_OVERRIDE:-$CODE_DIR/legacy_20k_controls/E1_${ROUTER_MODE}_seed${SEED}_${RUN_TAG}_g${GPU_COUNT}}"
if [[ -e "$SAVE_DIR/config.json" || -e "$SAVE_DIR/clustering_eval_history.csv" ]]; then
  echo "ERROR: existing E1 output: $SAVE_DIR. Change RUN_TAG." >&2; exit 2
fi
mkdir -p "$SAVE_DIR"
cd "$CODE_DIR"
export ENCODER_DIR TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
printf '[E1] host=%s GPUS=%s visible=%s seed=%s epochs=%s mode=%s batch/gpu=%s save=%s\n' \
  "$(hostname)" "$GPU_COUNT" "${CUDA_VISIBLE_DEVICES:-not-set}" \
  "$SEED" "$EPOCHS" "$ROUTER_MODE" "$BATCH_SIZE" "$SAVE_DIR"
sha256sum "$CODE_DIR/train_mechpair_light_E1_20k.py" "$RAMER_ROOT/cluster.py"

"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=gpu \
  "$CODE_DIR/train_mechpair_light_E1_20k.py" \
  --experiment e1 --selector mlp \
  --json_file "$DATA_DIR/uniprot_20W_struct_seq_reaction_121_without_new392.json" \
  --rhea_smiles_file "$DATA_DIR/updated_rhea-reaction-smiles.json" \
  --graph_embedding_path "$GRAPH_DIR" \
  --residue_embedding_path "$NODE_DIR" \
  --subset_file "$SUBSET_FILE" --subset_seed 3408 \
  --save_dir "$SAVE_DIR" --seed "$SEED" --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" --val_ratio 0 \
  --global_weight 1.0 --local_weight 0.5 --struct_weight 0.3 --export_weight 0.5 \
  --route_weight "$ROUTE_WEIGHT" --route_scale "$ROUTE_SCALE" \
  --router_mode "$ROUTER_MODE" \
  --enable_cluster_eval --cluster_eval_every 1 \
  --cluster_eval_name Price-149 \
  --cluster_eval_json "$DATA_DIR/test_data/Price-149.json" \
  --cluster_eval_truth_csv "$DATA_DIR/test_data/price.csv"
