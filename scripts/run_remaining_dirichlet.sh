#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root.
# K=5 is already complete, so this launches only K=3 and K=10.
# Completed runs are skipped, making the script safe to restart.

ALPHAS=(0.1 0.5 1.0 10.0)
SEEDS=(42 43 44)
KS=(3)

for K in "${KS[@]}"; do
  echo "============================================================"
  echo "Creating frozen Dirichlet partitions for K=${K}"
  echo "============================================================"
  python scripts/create_dirichlet_partitions.py \
    --num_clients "${K}" \
    --alphas "${ALPHAS[@]}" \
    --partition_seed 42
done

for K in "${KS[@]}"; do
  for ALPHA in "${ALPHAS[@]}"; do
    ALPHA_SLUG=${ALPHA/./p}
    for SEED in "${SEEDS[@]}"; do
      RESULT_DIR="results/dirichlet_k${K}_alpha_${ALPHA_SLUG}_seed${SEED}"
      RESULT_FILE="${RESULT_DIR}/federated_test_results.csv"

      if [[ -f "${RESULT_FILE}" ]]; then
        echo "SKIP completed: K=${K}, alpha=${ALPHA}, seed=${SEED}"
        continue
      fi

      echo "============================================================"
      echo "Training K=${K}, alpha=${ALPHA}, seed=${SEED}"
      echo "============================================================"
      python scripts/dirichlet_train.py \
        --num_clients "${K}" \
        --alpha "${ALPHA}" \
        --seed "${SEED}" \
        --results_dir "${RESULT_DIR}"
    done
  done
done
