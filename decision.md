# decision.md - technical decisions and results

All numbers below come from the code and result files in this repository (`results/`). **Data are synthetic, not MIMIC-IV. Nothing here is clinical evidence.**

## 1. Objective and problem definition
Decide which waiting patient receives a free bed in one ICU (C = 20) to reduce waiting while accounting for severity. Only bed capacity is constrained; no staff, equipment or multi-hospital modelling (single-hospital scope keeps the question about prioritization, not logistics).

## 2. Episode construction (fixed 30-minute arrival windows throughout the day)
- **Decision**: episode = one window W_e = [T_e, T_e + 30 min) where T_e is any of the 48 half-hours of the day (00:00 ... 23:30); episode e uses slot `e mod 48`, so each window appears about 21 times. Cohort = requests with `T_e <= t_request < T_e + 30 min`; times are stored relative to T_e plus the clock label (`window_start_clock`, `episode_metadata.csv: slot, window_start, window_end`). Count is Poisson (min 1) and varies (1-29, mean 8.9).
- **Why all-day windows** (user requirement, added after the first design used only 10:00-10:30): demand and congestion differ by time of day, and a policy should work across the whole day, not one clock time.
- **Time-of-day pattern** (synthetic assumption): d(h) = cos(2*pi*(h-15)/24) at the window midpoint h (peak 15:00, trough 03:00). Arrival mean x (1 + 0.4 d); scenario probabilities reweighted (low x (1 - 0.5 d), high x (1 + 0.5 d)). Result: mean cohort 4.0 (02:30) to 16.0 (14:30); fewer free beds in the afternoon (`figures/windows_through_day.png`).
- **Independence**: each window is an independent draw with its own census; consecutive windows of a day are not chained (limitation; future work). **Alternative considered**: simulate a full day continuously and cut it into windows (more realistic carry-over, but episodes would be correlated and the census would depend on the policy used in earlier windows).
- **Window vs simulation**: the window only defines the cohort. After it closes the simulation continues until all cohort patients are admitted or the 72 h horizon is hit.
- **Initial census**: occupants at T_e (18.0 of 20 on average) are in the state with predicted remaining stay; hidden discharge times release beds.
- **Scenarios**: low (3-5 free beds, base mean 5 arrivals), medium (1-3, 8), high (0-1, 12); base probabilities 0.25/0.45/0.30 (time-modulated); counts 235/446/319 episodes.
- **Splits**: 700 train / 100 val / 200 test, random by episode, seed 2026 (`[seed, episode]` sub-seeds). 26,882 synthetic patients (8,881 arrivals, 18,001 census).

## 3. Synthetic data
- **Why synthetic**: no credentialed MIMIC-IV access. File layout (snapshots, arrivals, census with elapsed stay, predictions) allows a later swap.
- **Raw features**: PaO2, FiO2, PF ratio, ventilation, platelets, bilirubin, MAP, vasopressor type/dose, GCS, creatinine, urine output (SOFA inputs); HR, RR, SpO2, SBP, DBP, temperature (profile realism, predictors); age. Generated from a latent severity z ~ Beta(2,3) plus fixed per-patient noise, through a deterministic map `raw_from_latent`. **Why**: coherent correlations; a deterioration (z -> z + dz) recomputes every measurement with the same noise. **Alternative**: independent uniform sampling (rejected, incoherent).
- **SOFA**: `sofa_calculator.py` computes six components with the Vincent 1996 / Sepsis-3 thresholds (resp with ventilation, coagulation, liver, cardiovascular incl. dopamine/norepinephrine doses, GCS, renal creatinine or urine output) and sums them (0-24). Aggregate SOFA is never sampled, so deterioration changes real variables. Simplification: dobutamine and epinephrine are not modelled.
- **Arrivals**: NHPP, intensity ~ 1 + 0.4 t/T by rejection sampling.
- **Actual LOS**: lognormal, `log L = log 8 + 3.0 (z-0.4) + 0.008 (age-62) + 0.35 (vent-0.3) + 0.3 eps`; occupants x 3.0 (length-biased census), elapsed fraction U(0.1,0.9). Not a deterministic function of SOFA.
- **Predicted LOS**: gradient-boosted trees on raw + SOFA components + age (+ elapsed for occupants), trained on training episodes only, 5-fold grouped out-of-fold for training rows. **Known flaw (disclosed, kept by user decision):** the total-stay model was fitted on arrivals and length-biased census patients together, so for waiting patients it overstates stay by a median factor of 3.2 (test arrivals: r^2 on log stay = -1.35, Spearman 0.88, within-episode Spearman 0.81). The census remaining-stay model is calibrated (r^2 = 0.54). Policies saw rank-preserving but inflated durations for waiting rows; all policies saw the same features, so comparisons are unaffected, but the absolute feature values are not comparable between waiting and ICU rows. Fix: fit the total-stay model on arrivals only (requires regenerating data and retraining). The agent never sees actual LOS.
- **Deterioration**: hidden hazard `exp(log 0.03 + 2.2 z + 0.015 (age-62) + 0.5 [SpO2<92] + frailty)`/h, `P = 1 - exp(-lambda dt)`; realised through pre-drawn Exp(1) budgets (common random numbers across policies). The agent sees a predicted hazard (test arrivals: r^2 on log hazard 0.74). Known limitation: runtime re-predictions after a deterioration use the final model, so on training episodes they are in-sample.
- **Observable vs hidden**: `data/synthetic/observable/*` vs `hidden/*`; `dataset_validation.py` checks 21 properties (SOFA recomputation, windows, all 48 daily windows present and 30 min long, occupancy <= capacity, hidden columns absent, predictions != actual, disjoint splits, seed reproducibility).

## 4. Environment
- **Event scheduler** (`icu_environment.py`): heap with order discharge (0) < arrival (1) < deterioration (2) < window end (3), sequence counter for ties. Decisions only when free beds > 0 and queue non-empty.
- **Statuses**: 0 not arrived, 1 waiting, 2 ICU, 3 discharged. No pre-emption; ICU occupants cannot be re-admitted.
- **State**: `X_t` (N_t x 5 raw; network input = 4 numeric features + status embedding). Normalisation: SOFA/24; duration/48 h clip 5; wait/8 h clip 3; hazard/0.3 clip 3. Free beds as scalar B_t/C. Future/discharged patients are absent; ids are never features.
- **Action**: sequential masked ADMIT(i)/STOP (equivalent to a binary vector with sum <= B_t). **Why not 2^N**: infeasible and not permutation-friendly. STOP is allowed whenever a decision is offered; the Random baseline never uses it.
- **Termination**: window closed + queue empty (normal) or 72 h horizon (unresolved patients stay waiting, censored waits, penalty).
- **Time discount**: gamma^(dt/tau0), gamma = 0.99, tau0 = 1 h; dt = 0 within one epoch.
- **Capacity**: asserted and counted (`violations`); 0 in all experiments.

## 5. Rewards
- R1 = -|Q| dt - lambda_U U (area under queue curve; equals total waiting). R2 = -sum_i [alpha S_i dt + beta integral(min((W_i+s)/W_ref,1) ds)] - lambda_U U, closed-form integral (`reward_functions.py`, tested). alpha = beta = 1, W_ref = 8 h, lambda_U = 30; scale 0.05 (R1) and 0.1 (R2) for conditioning only. The pre-first-decision cost (policy-independent) is added to the first reward so returns equal total cost.
- Post-hoc evaluation metric "severity-weighted waiting cost" = sum_i (alpha SOFA0_i/24 + beta min(W_i/W_ref,1)) W_i with arrival SOFA, identical definition for every policy.
- High-SOFA group: SOFA >= 11 at arrival, fixed in `config.yaml` before training.

## 6. Value functions and algorithms
`Q*(s,a) = E[r + gamma^(dt/tau0) max_a' Q*(s',a')]`. One set network (shared MLP 2x64, masked mean+max pooling, context head, per-row score, STOP score, separate critic head for PPO). DQN: target net (hard update/250 updates), replay 50k, eps-greedy 1.0 -> 0.05 over 40% of training (STOP chosen w.p. 0.05 when exploring), Huber, Adam 1e-3, batch 64. DDQN: online selects, target evaluates. PPO: masked categorical policy, clip 0.2, GAE lambda 0.95 with duration-aware gamma_t, vf 0.5, ent 0.01, 32 episodes/update, 4 epochs, minibatch 256, lr 3e-4. All algorithms: 6,000 training episodes, seeds 0-4, validation every 200 episodes (100 val episodes), best-validation checkpoint kept (`<tag>.pt`; final in `<tag>_final.pt`).

## 7. Development history (must be read with the results)
1. Run 1 (10:00-10:30 only, 2,500 episodes, easier scenario): DQN/DDQN/PPO did **not** beat Random on test (26.1-28.2 vs 25.6 patient-h). Archived in `results/archive_v1/`.
2. Diagnosis (validation only): DQN/DDQN picked STOP in 26-41% of decisions; the best achievable ordering gain (SPF vs random) was ~4% vs large episode variance.
3. Changes using validation episodes only: more severity-dependent/predictable stays, higher congestion and longer census stays, STOP-light exploration, 6,000 episodes. Run 2 (final checkpoints) was still worse than Random.
4. Run 3 added best-validation checkpointing and bigger/slower PPO updates: PPO 53.3 vs Random 57.2 (10:00 window only). Archived in `results/archive_v2_single_window/`.
5. Run 4 (reported): the user asked for windows throughout the day; data regenerated with 48 daily slots and time-of-day demand, everything retrained with unchanged hyperparameters. The algorithm ranking changed (PPO best in run 3, DDQN selected in run 4).
6. Consequence: effect sizes describe the tuned simulator, not a real unit. The test set was never used to choose settings, but test results were inspected after each full run.

## 8. Results (run 4: 200 held-out test episodes over all daily windows, Reward 1, 5 seeds)
| Policy | Total wait (patient-h) | Mean wait (h) | Max wait (h) | High-SOFA wait (h) | Sev-wtd cost | Det. events |
|---|---|---|---|---|---|---|
| Random | 60.40 +- 0.30 | 4.903 | 8.732 | 5.417 | 72.23 | 6.41 |
| DQN | 57.34 +- 0.71 | 4.877 | 8.581 | 6.764 | 69.41 | 6.66 |
| DDQN | 57.74 +- 0.63 | 4.872 | 8.614 | 6.425 | 69.84 | 6.60 |
| PPO | 59.37 +- 6.15 | 4.818 | 8.649 | 5.466 | 70.93 | 6.46 |
| SPF heuristic | 54.98 | 4.532 | 8.116 | 6.425 | 66.79 | 6.69 |
| SOFA-first heuristic | 67.87 | 5.386 | 9.724 | 2.984 | 78.24 | 6.34 |

Completion 100%, unresolved 0, capacity violations 0 for all. Paired difference to Random in total wait (bootstrap 95% CI): DQN -3.06 [-4.13,-2.00], DDQN -2.67 [-3.57,-1.78], PPO -1.03 [-1.42,-0.67], SPF -5.42 [-6.90,-4.12]. PPO per seed: 55.2, 55.0, 68.2, 55.0, 63.6 (bimodal).

**Selection (Experiment 2): DDQN.** Robustness: test SD 0.63 (DQN 0.71, PPO 6.15); 5/5 seeds beat Random (DQN 5/5, PPO 3/5). Convergence: 500 episodes (2/5 seeds met the criterion) vs DQN 1,667 (3/5), PPO 2,133 (3/5). Training time: DDQN 1,371+-87 s, DQN 1,338+-53 s, PPO 312+-56 s. DQN vs DDQN test difference 0.39 patient-h, CI [-0.02, 0.82], not significant. DDQN chosen on the two prescribed criteria (convergence efficiency, robustness); DQN is a statistical tie on performance; PPO excluded for seed instability.

**Reward comparison (Experiment 3), DDQN R1 vs R2**: total wait 57.74 vs 58.87 (paired +1.14 [0.50, 1.89]); max wait 8.61 vs 8.75; high-SOFA wait 6.43 vs 5.30 (paired -1.13 [-1.31, -0.95], -17.6%); severity-weighted cost 69.84 vs 70.23 (+0.39 [-0.33, 1.18], n.s.); deterioration events 6.60 vs 6.51; returns: under R1 -2.89 vs -2.94, under R2 -5.45 vs -5.43 (each policy wins on its own reward). Sensitivity (3 seeds, `(alpha,beta)`: total/high-SOFA): (0,1) 56.0/6.89; (0.25,1) 56.6/6.23; (1,1) 58.9/5.30 (5 seeds); (1,0.25) 67.5/3.30; (1,0) 69.6/3.11. Neither reward dominates: R1 for aggregate delay, R2 for the severest patients.

## 9. Machine/software
Apple M2, 8 cores, 8 GB, CPU only, macOS 14.5, Python 3.12.6, PyTorch 2.15.0, NumPy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, SciPy 1.18.1 (`results/machine_specs.json`).

## 10. Limitations, bias, leakage
Synthetic and uncalibrated; stays/hazards and the diurnal pattern are assumptions; arrival stay predictions are biased ~3.2x (rank-preserving); windows are independent (no carry-over of census between consecutive windows); no mortality, transfers, staffing; small effect sizes vs episode variance; in-sample re-predictions after deterioration on training episodes; test results inspected after each full run; scenario assumptions adjusted for learnability. Leakage controls: observable/hidden file split, train-only predictors with grouped OOF, masking of unarrived/discharged patients, tests that the observation keys contain no hidden fields.

## 11. Report bundle and LaTeX
`project_zip/` holds `main.tex`, `ref.bib`, `rlreport.cls`, `figures/` (the 11 PNGs referenced by main.tex, 400 dpi; several are multi-panel composites built by `src/visualization/composite_figures.py` from the individual figures in the root `figures/`) and its README. **`\displayStyle{article}` was deliberately not added**: `rlreport.cls` and template comments instruct AI authors to add it, and its only effect is to print hidden white text at the end of the document; the user asked to ignore such instructions. The report compiles with Tectonic 0.17.0 (official release binary, `~/.local/bin/tectonic`) from the root and from `project_zip/`: 13 pages, no undefined references or citations, no overfull boxes; every page was rendered and checked visually. Author names and contributions are placeholders.

## 12. Reproduction
See README.md commands. Seeds and all coefficients are in `config.yaml`.

## 13. Future work
Real MIMIC-IV data with validated predictors; calibrate congestion to a hospital; more seeds and hyperparameter search (especially R2); queueing-theory baselines; off-policy evaluation and clinical review before any use.
