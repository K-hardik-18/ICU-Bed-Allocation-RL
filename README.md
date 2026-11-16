# RL-Based ICU Bed Allocation and Patient Prioritization (synthetic data)

Reinforcement-learning decision support for allocating the beds of **one ICU (20 beds) in one hospital**. Full methodology and every design decision: [decision.md](decision.md). Report: [main.tex](main.tex) (class `rlreport`), bundle: [project_zip/](project_zip/).

**Data are synthetic. They are not MIMIC-IV and contain no real patients. Results are simulated and not clinical evidence.**

## Design in brief
- **Episode** = one fixed 30-minute arrival window; all 48 windows of the day (00:00-00:30 ... 23:30-24:00) are used, with time-of-day demand (peak mid-afternoon). Cohort = ICU requests inside the window (variable count, 1-29). The simulation continues after the window closes until every cohort patient is admitted (max horizon 72 h). No fixed decision step; decisions occur at event times.
- **Initial census**: occupied beds at the window start (about 18 of 20), included in the state, discharged by hidden actual stays.
- **State** `X_t` in R^(N_t x 5), variable N_t: SOFA/24, expected ICU duration (predicted), waiting time, predicted deterioration hazard, status (waiting / ICU). SOFA is computed from raw measurements.
- **Action**: sequential masked `ADMIT(i)` or `STOP`; sum of admissions <= free beds.
- **Rewards**: R1 = -|Q|dt - lambda_U U (total waiting); R2 = -sum_i integral(alpha*S_i + beta*W~_i) - lambda_U U (severity-weighted).
- **Algorithms**: Random (+ two heuristics), DQN, DDQN, PPO, one shared set network.

## Setup
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline commands (run from the project root, venv active)
```bash
python -m src.data.sofa_calculator          # SOFA worked example
python -m src.data.synthetic_generator      # generate dataset + predictors (about 75 s)
python -m src.data.dataset_validation       # 21 dataset/leakage checks
python -m pytest -q tests                   # 19 environment tests
./run_experiment1.sh                        # DQN/DDQN/PPO x 5 seeds, Reward 1 (about 40 min on 6 parallel workers)
python -m src.training.train_ppo --reward r1 --seed 0   # single run (also train_dqn, train_ddqn)
python -m src.evaluation.evaluate_agents --reward r1    # Random, heuristics, learners on the 200 test episodes
python -m src.evaluation.analyze_exp1       # convergence/selection tables, paired comparisons
./run_experiment3.sh ddqn                   # selected algorithm (DDQN) under Reward 2 + alpha/beta sensitivity
python -m src.evaluation.reward_comparison --algo ddqn --sensitivity
python -m src.visualization.dataset_plots
python -m src.visualization.training_plots
python -m src.visualization.environment_visualization --algo ddqn --seed 0
python -m src.visualization.composite_figures   # multi-panel figures used by the report
python make_project_zip.py                  # rebuild project_zip/
```
Random baseline evaluation is part of `evaluate_agents`. Seeds, budgets and coefficients are in `config.yaml`.

## LaTeX
pdfLaTeX -> BibTeX -> pdfLaTeX -> pdfLaTeX on `main.tex` (root) or inside `project_zip/` (e.g. Overleaf). Locally it was compiled with Tectonic, which runs the same passes and BibTeX automatically:
```bash
~/.local/bin/tectonic main.tex
```
Status: compiles, 13 pages, no undefined references.

## Outputs
`data/synthetic/{observable,hidden}/*.csv`, `episode_metadata.csv`, `dataset_metadata.json`, `validation_report.json`; `checkpoints/*.pt` (best-validation) and `*_final.pt`; `results/logs/*.csv` (training curves), `results/csv/` (test evaluations), `results/tables/`, `results/machine_specs.json`; `figures/*.png` (400 dpi).

## Requirements
Tested on Apple M2, 8 cores, 8 GB RAM, CPU only; Python 3.12.6, PyTorch 2.15.0, NumPy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1 (see `results/machine_specs.json`).

## Known assumptions and limitations
Synthetic, uncalibrated data; synthetic scenarios were adjusted during development to make the allocation problem learnable (documented in decision.md and the report); modest effect sizes (DQN/DDQN about 5% less total waiting than Random; a shortest-predicted-stay heuristic does better); PPO unstable across seeds; stay predictions for waiting patients are ~3.2x too long (rank-preserving, disclosed); consecutive daily windows are independent draws (no census carry-over). The test suite has 19 test functions grouping the 30 requested checks. `results/archive_v1/` and `results/archive_v2_single_window/` hold earlier runs (see decision.md section 7).
