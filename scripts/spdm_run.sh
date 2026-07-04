#!/usr/bin/env bash
# Прогон SPDM (ManiMamba) на a100: 2 цели x (3 среза бэктеста + прогноз вперёд).
# Требует: external/spdm с готовым .venv, data/spdm/*.csv (prepare_spdm_data.py).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPDM="$ROOT/external/spdm"
DATA="$ROOT/data/spdm"
RES="$ROOT/results/spdm"
export CUDA_VISIBLE_DEVICES="${SPDM_GPU:-5}"
mkdir -p "$RES"

run_one() {
  local target=$1 tag=$2
  local csv="${target}_${tag}.csv"
  echo "=== SPDM: $csv (GPU $CUDA_VISIBLE_DEVICES) ==="
  cd "$SPDM"
  find . -name real_prediction.npy -delete 2>/dev/null || true
  .venv/bin/python -u scripts/run.py \
    --is_training 1 --do_predict --inverse \
    --model ManiMamba --data custom --features M --target inj42 --freq m \
    --root_path "$DATA/" --data_path "$csv" \
    --model_id "${target}_${tag}" --des TIMESOIL \
    --seq_len 24 --label_len 12 --pred_len 6 \
    --e_layers 2 --enc_in 49 --dec_in 49 --c_out 49 \
    --d_model 128 --d_ff 256 --cov_window 8 --cov_stride 4 \
    --batch_size 8 --dropout 0.2 --learning_rate 3e-4 \
    --optim AdamW --weight_decay 1e-6 \
    --train_epochs 20 --patience 5 --itr 1 --num_workers 2
  local rp
  rp=$(find . -name real_prediction.npy | head -1)
  if [[ -z "$rp" ]]; then echo "ОШИБКА: real_prediction.npy не найден для $csv" >&2; exit 1; fi
  mkdir -p "$RES/${target}_${tag}"
  cp "$rp" "$RES/${target}_${tag}/real_prediction.npy"
  echo "-> $RES/${target}_${tag}/real_prediction.npy"
}

TAGS=(201405 201411 201505 201511)
for target in oil liq; do
  for tag in "${TAGS[@]}"; do
    run_one "$target" "$tag"
  done
done
echo "SPDM: все прогоны завершены."
