#!/bin/bash

# Colab A100 GPU用の最適化されたStage 2学習コマンド
# 予想学習時間: 30分-2時間（GPU使用時）

echo "🚀 Starting Stage 2 training on Colab A100 GPU..."

python3 -m src.train_stage2 \
  --flow_ckpt checkpoints_paper_m50/flow_stage1_final.pt \
  --m 50 --K 1024 --D 16 --iters 10000 --batch 512 \
  --lr 5e-1 --grad_clip 1.0 --ode_steps 25 \
  --a 20 --atom_size_mode small --n_val 10000 \
  --use_gumbel --tau_start 1.0 --tau_end 0.01 \
  --warmstart --warmstart_grid 500 \
  --reinit_every 1000 --reinit_threshold 0.01 \
  --freeze_beta_iters 1000 \
  --ckpt_every 2000 --log_every 50 --eval_n 2000 \
  --seed 123 \
  --out_dir checkpoints_colab_a100_stage2

echo "✅ Training completed!"
