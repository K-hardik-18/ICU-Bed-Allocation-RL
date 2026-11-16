#!/bin/bash
# Experiment 3: selected algorithm (PPO) under Reward 2 (5 seeds) + alpha/beta sensitivity (3 seeds each). Run from project root with venv active.
ALGO=${1:-ppo}
{
for s in 0 1 2 3 4; do echo "--reward r2 --seed $s"; done
for ab in "1.0 0.25" "0.25 1.0" "1.0 0.0" "0.0 1.0"; do set -- $ab; for s in 0 1 2; do echo "--reward r2 --seed $s --alpha $1 --beta $2 --tag sens_${ALGO}_a$1_b$2_s$s"; done; done
} | xargs -P 6 -L 1 bash -c "python -m src.training.train_$ALGO \$0 \$@ > /dev/null 2>&1"
