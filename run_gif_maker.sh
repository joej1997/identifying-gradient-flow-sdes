#!/usr/bin/env bash
# run_main_gmm.sh
set -euo pipefail

ROOT="just_gifs"
P0="unif"

#pot_names=(oakley_ohagan quadratic styblinski_tang wavy_plateau bohachevsky)
#pot_flags=(oakley_ohagan poly styblinski_tang wavy_plateau bohachevsky)

pot_names=(flat)
pot_flags=(flat)

seeds=(6)
betas=(0.1)           # β (σ² = 2β)
N=1000
dt=0.01
n_steps=50

for idx in "${!pot_names[@]}"; do
  prefix="${pot_names[$idx]}"
  pot_flag="${pot_flags[$idx]}"

  echo "================ POTENTIAL: ${prefix} ================"

  for seed in "${seeds[@]}"; do
    echo "----- SEED: ${seed} -----"
    for beta in "${betas[@]}"; do
      diff=$(awk "BEGIN{printf \"%.1f\", $beta * 2}")   # σ² = 2β
      dataset="${prefix}_diff-${diff}_seed-${seed}"

      echo "=== Generating data: ${dataset}  (β=${beta} ⇒ σ²=${diff}) ==="
      python data_generator.py \
        --root "${ROOT}" \
        --p0 "${P0}" \
        --potential "${pot_flag}" \
        --dataset-name "${dataset}" \
        --beta "${beta}" \
        --internal wiener \
        --seed "${seed}" \
        --dt "${dt}" \
        --initial_length 0 \
        --steps_elapsed_X0 0\
        --n-timesteps "${n_steps}" \
        --n-particles "${N}"
      echo
    done
  done
done