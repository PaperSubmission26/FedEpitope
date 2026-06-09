#!/bin/bash

# python federated_train.py --seed 43 --lora_rank 2 --results_dir results/r2_seed43 2>&1 | tee results/r2_seed43/run.log

# python federated_train.py --seed 44 --lora_rank 2 --results_dir results/r2_seed44 2>&1 | tee results/r2_seed44/run.log

#python baseline.py --seed 43 --lora_rank 2 --results_dir results/baseline_seed43 2>&1 | tee results/baseline_seed43/run.log

#python baseline.py --seed 44 --lora_rank 2 --results_dir results/baseline_seed44 2>&1 | tee results/baseline_seed44/run.log

python personalised_fl_v2.py --seed 42 --lora_rank 2 --fed_checkpoint results/r2_seed42/best_global_weights.pt --results_dir results/pfl_seed42
python personalised_fl_v2.py --seed 43 --lora_rank 2 --fed_checkpoint results/r2_seed43/best_global_weights.pt --results_dir results/pfl_seed43
python personalised_fl_v2.py --seed 44 --lora_rank 2 --fed_checkpoint results/r2_seed44/best_global_weights.pt --results_dir results/pfl_seed44