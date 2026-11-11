#!/usr/bin/env bash
#SBATCH --job-name=ramer_floor15_eval3
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=0
set -euo pipefail
CODE_DIR="${CODE_DIR:-/datapool/home/tanlijun/RAMER/dec-v2/RAMER_unsupervised_AB}"
RAMER_ROOT="${RAMER_ROOT:-/datapool/home/tanlijun/RAMER}"
DATA_ROOT="${DATA_ROOT:-$RAMER_ROOT/data}"
ENCODER_DIR="${ENCODER_DIR:-$RAMER_ROOT/dec-v2}"
PYTHON="${PYTHON:-/datapool/home/tanlijun/.conda/envs/ramer/bin/python}"
CHECKPOINT="${CHECKPOINT:-$CODE_DIR/experiments_next_full/FLOOR_FULL_seed3408_fp32_full50_v1/seq_only_epoch_015.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-$CODE_DIR/eval_full_floor_epoch15_three_sets}"
PRICE_JSON="${PRICE_JSON:-$DATA_ROOT/test_data/Price-149.json}"
PRICE_TRUTH="${PRICE_TRUTH:-$DATA_ROOT/test_data/price.csv}"
find_first(){
  for file in "$@";do
    if [[ -f "$file" ]];then printf '%s' "$file";return 0;fi
  done
  return 1
}
# 与用户原 inference.py 提供的 zero_shot/data 路径兼容，但不假设文件一定存在。
NEW392_JSON="${NEW392_JSON:-$(find_first "$DATA_ROOT/test_data/NEW-392.json" "$RAMER_ROOT/zero_shot/data/NEW-392.json" || true)}"
NEW392_TRUTH="${NEW392_TRUTH:-$(find_first "$DATA_ROOT/test_data/new.csv" "$RAMER_ROOT/zero_shot/data/new.csv" || true)}"
RAM255_JSON="${RAM255_JSON:-$DATA_ROOT/test_data/ram255.json}"
RAM255_TRUTH="${RAM255_TRUTH:-$(find_first "$DATA_ROOT/test_data/ram255.csv" "$DATA_ROOT/test_data/RAM255.csv" "$DATA_ROOT/test_data/ram.csv" "$RAMER_ROOT/zero_shot/data/ram255.csv" || true)}"
for val in CHECKPOINT PRICE_JSON PRICE_TRUTH NEW392_JSON NEW392_TRUTH RAM255_JSON RAM255_TRUTH;do
  file="${!val}"
  [[ -n "$file" && -f "$file" ]] || { echo "Missing $val='$file'. Set $val to its verified actual path." >&2;exit 2;}
done
[[ -f "$CODE_DIR/eval_full_epoch15_3sets.py" ]] || { echo "Missing eval_full_epoch15_3sets.py" >&2;exit 2; }
cd "$CODE_DIR"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
printf '[EVAL] checkpoint=%s\nprice=%s\nnew392=%s\nram255=%s\n' "$CHECKPOINT" "$PRICE_JSON" "$NEW392_JSON" "$RAM255_JSON"
"$PYTHON" "$CODE_DIR/eval_full_epoch15_3sets.py" \
 --checkpoint "$CHECKPOINT" --encoder-dir "$ENCODER_DIR" --cluster-root "$RAMER_ROOT" \
 --output-dir "$OUTPUT_DIR" --epoch 15 --batch-size 4 \
 --price-json "$PRICE_JSON" --price-truth "$PRICE_TRUTH" \
 --new392-json "$NEW392_JSON" --new392-truth "$NEW392_TRUTH" \
 --ram255-json "$RAM255_JSON" --ram255-truth "$RAM255_TRUTH"