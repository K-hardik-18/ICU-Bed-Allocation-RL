"""Multi-panel figures used by the report (built from the individual PNGs). Command: python -m src.visualization.composite_figures"""
from .style import compose


def main():
    compose([["sofa_distribution.png", "los_distribution.png", "los_pred_vs_actual.png"],
             ["hazard_distribution.png", "patient_arrival_distribution.png", "cohort_arrivals_per_episode.png"],
             ["initial_icu_occupancy.png", "windows_through_day.png"]], "dataset_overview.png")
    compose([["convergence_dqn.png", "convergence_ddqn.png"], ["convergence_ppo.png", "convergence_comparison.png"]], "convergence_all.png")
    compose([["cmp_total_wait.png", "cmp_mean_wait.png", "cmp_high_sofa_wait.png"],
             ["cmp_completion.png", "cmp_unresolved.png", "cmp_utilization.png"]], "algorithm_comparison.png")
    compose([["icu_occupancy_over_time.png", "queue_size_over_time.png"], ["admission_decisions.png", "bed_release_admission.png"]], "agent_snapshots.png")


if __name__ == "__main__":
    main()
