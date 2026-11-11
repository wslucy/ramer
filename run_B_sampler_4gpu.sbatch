#!/usr/bin/env bash
#SBATCH --job-name=ramer_B_sampler
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=0
set -euo pipefail
CODE_DIR="${CODE_DIR:-/datapool/home/tanlijun/RAMER/dec-v2/RAMER_unsupervised_AB}"
ENCODER_DIR="${ENCODER_DIR:-/datapool/home/tanlijun/RAMER/dec-v2}"
PYTHON="${PYTHON:-/datapool/home/tanlijun/.conda/envs/ramer/bin/python}"
RAMER_ROOT="${RAMER_ROOT:-/datapool/home/tanlijun/RAMER}"
DATA_DIR="${DATA_DIR:-$RAMER_ROOT/data}"
GRAPH_DIR="${GRAPH_DIR:-/datapool/home/tanlijun/GearNet/pdbs/features/graph}"
NODE_DIR="${NODE_DIR:-/datapool/home/tanlijun/GearNet/pdbs/features/node}"
RUN_ID="${RUN_ID:?B_RANDOM or B_PAIRED}"
EPOCHS="${EPOCHS:-50}"
SEED="${SEED:-3408}"
RUN_TAG="${RUN_TAG:-v1}"
PAIRS_PER_BATCH="${PAIRS_PER_BATCH:-2}"
case "$RUN_ID" in
 B_RANDOM) PAIR_ARGS=() ;;
 B_PAIRED) PAIR_ARGS=(--relation-pair-sampler --pairs-per-batch "$PAIRS_PER_BATCH" --pair-sampler-seed 3408) ;;
 *) echo "RUN_ID must be B_RANDOM or B_PAIRED" >&2; exit 2 ;;
esac
for f in "$CODE_DIR/train_mechpair_AB_pair_sampler.py" "$CODE_DIR/prepared_inputs/manifest.json" \
 "$CODE_DIR/subsets/fixed_20k_seed3408.json" "$RAMER_ROOT/cluster.py" \
 "$DATA_DIR/test_data/Price-149.json" "$DATA_DIR/test_data/price.csv"; do
 [[ -f "$f" ]] || { echo "Missing file: $f" >&2; exit 2; }
done
[[ -x "$PYTHON" ]] || { echo "Python is not executable: $PYTHON" >&2; exit 2; }
SAVE_DIR="${SAVE_DIR_OVERRIDE:-$CODE_DIR/b_sampler_gpu/${RUN_ID}_seed${SEED}_${RUN_TAG}_g${SLURM_GPUS_ON_NODE:-${GPUS:-1}}}"
if [[ -e "$SAVE_DIR/metrics.jsonl" || -e "$SAVE_DIR/last.pt" ]]; then
 echo "Refuse to overwrite existing run: $SAVE_DIR" >&2;exit 2
fi
mkdir -p "$SAVE_DIR"
cd "$CODE_DIR"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
printf '[RESOURCES] host=%s allocated_gpus=%s cuda_visible=%s\n' "$(hostname)" "${SLURM_GPUS_ON_NODE:-unknown}" "${CUDA_VISIBLE_DEVICES:-not-set}"
printf '[RUN] id=%s rows=20000 seed=%s GPUs=${SLURM_GPUS_ON_NODE:-unknown} batch/gpu=16 pairs=%s\n' "$RUN_ID" "$SEED" "$PAIRS_PER_BATCH"
sha256sum "$CODE_DIR/train_mechpair_AB_pair_sampler.py"
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=gpu \
 "$CODE_DIR/train_mechpair_AB_pair_sampler.py" \
 --variant b --encoder-dir "$ENCODER_DIR" \
 --json-file "$DATA_DIR/uniprot_20W_struct_seq_reaction_121_without_new392.json" \
 --rhea-smiles-file "$DATA_DIR/updated_rhea-reaction-smiles.json" \
 --graph-embedding-path "$GRAPH_DIR" --residue-embedding-path "$NODE_DIR" \
 --protein-model-path "$RAMER_ROOT/T5models/ProtT5" \
 --reaction-model-path "$RAMER_ROOT/T5models/molT5" \
 --token-cache-dir "$CODE_DIR/prepared_inputs" \
 --subset-size 20000 --subset-file "$CODE_DIR/subsets/fixed_20k_seed3408.json" \
 --subset-seed 3408 --seed "$SEED" --save-dir "$SAVE_DIR" \
 --epochs "$EPOCHS" --batch-size 16 --num-workers 2 --precision fp32 \
 --deploy-weight 0 --a-weight 0 --diversity-weight 0 --b-sequence-weight 0 \
 --attention-export-every 0 \
 --cluster-eval --cluster-eval-mode legacy \
 --cluster-eval-module-root "$RAMER_ROOT" \
 --cluster-eval-input "$DATA_DIR/test_data/Price-149.json" \
 --cluster-eval-truth-csv "$DATA_DIR/test_data/price.csv" \
 --cluster-eval-every 1 --cluster-eval-role monitor --log-every 50 \
 "${PAIR_ARGS[@]}"