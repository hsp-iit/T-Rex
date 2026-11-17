#!/bin/bash
# Train + evaluate the XHand [16,5,3] tactile VQ-VAE (config: ../config/vqvae_xhand_f3.yaml).
#   TAPTIC_DIR=<out>/<task> OUT=<dir> bash tactile_vqvae/scripts/train_vqvae_xhand.sh
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-python}
TAPTIC_DIR=${TAPTIC_DIR:?set TAPTIC_DIR}
OUT=${OUT:?set OUT}
RUN=${RUN:-vqvae_xhand_f3_w16_k64_hand}
cd "$REPO"
$PY -m tactile_vqvae.train --data_format taptic --taptic_dir "$TAPTIC_DIR" \
    --n_fingers 5 --per_finger_dim 3 --window 16 --stride 4 --granularity hand \
    --codebook_size 64 --embed_dim 256 --hidden_channels 128 --bottleneck_channels 256 \
    --n_strided_blocks 2 --commitment_weight 0.25 --decay 0.99 --revive_freq 200 \
    --use_magnitude_weight 1 --weight_alpha 2.0 --weight_tau 0 \
    --epochs 30 --batch_size 256 --lr 3e-4 --min_lr_ratio 0.05 --warmup_steps 500 \
    --weight_decay 1e-4 --grad_clip 1.0 --mixed_precision bf16 --seed 42 \
    --num_workers 4 --val_every 500 --log_every 200 --save_every_epoch 10 \
    --output_dir "$OUT" --run_name "$RUN"
$PY -m tactile_vqvae.eval_taptic --checkpoint "$OUT/$RUN/latest.pt" --taptic_dir "$TAPTIC_DIR" --out_dir "$OUT/$RUN/eval"
