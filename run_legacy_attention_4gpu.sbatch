#!/usr/bin/env bash
#SBATCH --job-name=ramer_legacy_4gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
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
EPOCHS="${EPOCHS:-50}"
SEED="${SEED:-3408}"
RUN_TAG="${RUN_TAG:-v1}"
RUN_ID="${RUN_ID:?Set RUN_ID=MLP_BASE|MLP_FLOOR|QKV_BASE|QKV_FLOOR}"

case "$RUN_ID" in
  MLP_BASE)  SCRIPT=train_mechpair_light_MLP_20k.py; EXPERIMENT=baseline; SELECTOR=mlp ;;
  MLP_FLOOR) SCRIPT=train_mechpair_light_MLP_20k.py; EXPERIMENT=floor;    SELECTOR=mlp ;;
  QKV_BASE)  SCRIPT=train_mechpair_light_QKV_20k.py; EXPERIMENT=baseline; SELECTOR=qkv ;;
  QKV_FLOOR) SCRIPT=train_mechpair_light_QKV_20k.py; EXPERIMENT=floor;    SELECTOR=qkv ;;
  *) echo "Invalid RUN_ID=$RUN_ID" >&2; exit 2 ;;
esac

for f in "$CODE_DIR/$SCRIPT" "$ENCODER_DIR/Data2seq_v2.py" \
    "$DATA_DIR/uniprot_20W_struct_seq_reaction_121_without_new392.json" \
    "$DATA_DIR/updated_rhea-reaction-smiles.json" \
    "$RAMER_ROOT/cluster.py" "$DATA_DIR/test_data/Price-149.json" \
    "$DATA_DIR/test_data/price.csv" "$CODE_DIR/subsets/fixed_20k_seed3408.json"; do
  [[ -f "$f" ]] || { echo "Missing file: $f" >&2; exit 2; }
done
[[ -x "$PYTHON" ]] || { echo "Python not executable: $PYTHON" >&2; exit 2; }

SAVE_DIR="${SAVE_DIR_OVERRIDE:-$CODE_DIR/legacy_20k_controls/${RUN_ID}_seed${SEED}_${RUN_TAG}}"
if [[ -e "$SAVE_DIR/clustering_eval_history.csv" || -e "$SAVE_DIR/config.json" ]]; then
  echo "Refuse to overwrite: $SAVE_DIR" >&2; exit 2
fi
mkdir -p "$SAVE_DIR"
cd "$CODE_DIR"
export ENCODER_DIR TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
printf '[RUN] %s seed=%s experiment=%s selector=%s 20k 4GPU Epochs=%s\n' \
  "$RUN_ID" "$SEED" "$EXPERIMENT" "$SELECTOR" "$EPOCHS"
printf '[WARN] unchanged old BioData2Seq backbone / shared_mask; 20k fixed indices / 4GPU; baseline has no A or deploy\n'
sha256sum "$CODE_DIR/$SCRIPT" "$RAMER_ROOT/cluster.py"

"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
 "$CODE_DIR/$SCRIPT" \
 --experiment "$EXPERIMENT" --selector "$SELECTOR" --qkv_heads 8 \
 --json_file "$DATA_DIR/uniprot_20W_struct_seq_reaction_121_without_new392.json" \
 --rhea_smiles_file "$DATA_DIR/updated_rhea-reaction-smiles.json" \
 --graph_embedding_path "$GRAPH_DIR" --residue_embedding_path "$NODE_DIR" \
 --subset_file "$CODE_DIR/subsets/fixed_20k_seed3408.json" --subset_seed 3408 \
 --save_dir "$SAVE_DIR" --seed "$SEED" --epochs "$EPOCHS" \
 --batch_size 16 --val_ratio 0 \
 --global_weight 1.0 --local_weight 0.5 --struct_weight 0.3 --export_weight 0.5 \
 --deploy_weight 0.2 --a_weight 0.5 --diversity_weight 0.01 \
 --diversity_floor 0.10 --diversity_floor_penalty 5.0 \
 --num_slots 4 --slot_dim 256 --slot_heads 4 \
 --enable_cluster_eval --cluster_eval_every 1 \
 --cluster_eval_name Price-149 \
 --cluster_eval_json "$DATA_DIR/test_data/Price-149.json" \
 --cluster_eval_truth_csv "$DATA_DIR/test_data/price.csv"