#!/bin/bash
# Experiment 1: DQN / DDQN / PPO x 5 seeds under Reward 1 (identical budget). Run from project root with venv active.
for a in dqn ddqn ppo; do for s in 0 1 2 3 4; do echo "$a $s"; done; done | \
  xargs -P 6 -L 1 bash -c 'python -m src.training.train_$0 --reward r1 --seed $1 > results/logs/stdout_$0_r1_s$1.txt 2>&1'
