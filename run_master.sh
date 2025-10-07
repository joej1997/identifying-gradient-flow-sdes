#!/usr/bin/env bash
# run_main_gmm.sh
set -euo pipefail

ROOT="killed"
P0="unif"

#pot_names=(oakley_ohagan quadratic styblinski_tang wavy_plateau bohachevsky)
#pot_flags=(oakley_ohagan poly styblinski_tang wavy_plateau bohachevsky)

pot_names=(quadratic)
pot_flags=(poly)

seeds=(10)
betas=(0.1)           # β (σ² = 2β)
N=4000
dt=0.01
n_steps=5

for idx in "${!pot_names[@]}"; do
  prefix="${pot_names[$idx]}"
  pot_flag="${pot_flags[$idx]}"

  echo "================ POTENTIAL: ${prefix} ================"

  for seed in "${seeds[@]}"; do
    echo "----- SEED: ${seed} -----"
    for beta in "${betas[@]}"; do
      diff=$(awk "BEGIN{printf \"%.1f\", $beta * 2}")   # σ² = 2β
      dataset="killed_${prefix}_diff-${diff}_seed-${seed}_4000"

      echo "=== Generating data: ${dataset}  (β=${beta} ⇒ σ²=${diff}) ==="
      python data_generator.py \
        --root "${ROOT}" \
        --p0 "${P0}" \
        --potential "${pot_flag}" \
        --dataset-name "${dataset}" \
        --beta "${beta}" \
        --internal wiener \
        --seed "${seed}" \
        --killed \
        --dt "${dt}" \
        --initial_length 4 \
        --n-timesteps "${n_steps}" \
        --n-particles "${N}"

      echo ">>> Training on ${dataset}"

#      # 1) JKONet*
#      python train.py \
#        --root "${ROOT}" \
#        --p0 "${P0}" \
#        --solver jkonet-star-potential-internal \
#        --seed "${seed}" \
#        --dataset "${dataset}" \
#        --epochs 1000 \
#        --potential "${pot_flag}" \
#        --sb-iters 0 \
#        --diffusivity "${diff}" \
#        --dt "${dt}" \
#        --method-tag "jkonet_star"

#      # 2) WOT
#      python train.py \
#        --root "${ROOT}" \
#        --p0 "${P0}" \
#        --solver jkonet-star-potential-internal \
#        --seed "${seed}" \
#        --dataset "${dataset}" \
#        --epochs 0 \
#        --potential "${pot_flag}" \
#        --sb-iters 1 \
#        --diffusivity "${diff}" \
#        --activation "silu" \
#        --dt "${dt}" \
#        --method-tag "wot"
#
      # 3) nn_APPEX
      python train.py \
        --root "${ROOT}" \
        --p0 "${P0}" \
        --solver jkonet-star-potential-internal \
        --dataset "${dataset}" \
        --epochs 1000 \
        --seed "${seed}" \
        --potential "${pot_flag}" \
        --sb-iters 10 \
        --diffusivity "${diff}" \
        --activation "silu" \
        --SB_solver "mmot" \
        --dt "${dt}" \
        --method-tag "nn_appex_hybrid_unbalanced_10"

#      # 4) SBIRR (MIT)
#      python train.py \
#        --root "${ROOT}" \
#        --p0 "${P0}" \
#        --solver jkonet-star-potential-internal \
#        --seed "${seed}" \
#        --dataset "${dataset}" \
#        --epochs 0 \
#        --potential "${pot_flag}" \
#        --fix_diffusion \
#        --sb-iters 30 \
#        --diffusivity "${diff}" \
#        --activation "silu" \
#        --dt "${dt}" \
#        --method-tag "sbirr"
#      # 5) nn-APPEX (hybrid)
#      python train.py \
#        --root "${ROOT}" \
#        --p0 "${P0}" \
#        --solver jkonet-star-potential-internal \
#        --dataset "${dataset}" \
#        --epochs 1000 \
#        --seed "${seed}" \
#        --potential "${pot_flag}" \
#        --sb-iters 10 \
#        --diffusivity "${diff}" \
#        --activation "silu" \
#        --dt "${dt}" \
#        --method-tag "nn_appex_hybrid"
      # 6) nn_APPEX (gwot)
#      python train.py \
#        --root "${ROOT}" \
#        --p0 "${P0}" \
#        --solver jkonet-star-potential-internal \
#        --dataset "${dataset}" \
#        --epochs 0 \
#        --seed "${seed}" \
#        --potential "${pot_flag}" \
#        --sb-iters 15 \
#        --diffusivity "${diff}" \
#        --activation "silu" \
#        --dt "${dt}" \
#        --SB_solver 'gwot' \
#        --method-tag "nn_appex_gwot"

      echo
    done
  done
done