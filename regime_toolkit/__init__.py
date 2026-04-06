from ._version import __version__
from .simulate import simulate, SimulationConfig
from .sweep import parameter_sweep, IndicatorSpec, TRAJECTORY_ONLY, ORACLE
from .observations import (
    state_to_scalar_observation,
    add_observation_noise,
    delay_embed,
    observed_trajectory,
    ObservedTrajectory,
)
from .io import save_json, save_results_json, save_trajectories_npz, save_sweep_bundle
from .boundaries import (
    ChangePointResult,
    BootstrapChangePointResult,
    SensitivityChangePointResult,
    LeadDistanceResult,
    centered_moving_average,
    robust_standardize,
    composite_structural_score,
    piecewise_linear_change_point,
    bootstrap_change_point_from_replicates,
    change_point_sensitivity_scan,
    first_threshold_crossing,
    lead_distance,
)
from .robustness import RobustnessCase, run_robustness_cases, evaluate_robustness_summary
from .contracts import PaperStudyContract, default_paper_study_contract, validate_metric_sources, evaluate_study_contract
from .study import run_study_suite
from .provenance import capture_provenance
from .indicator_factories import make_trajectory_only_indicator_specs
from . import systems
from . import indicators
from . import oracles
from .plots import set_publication_style

__all__ = [
    "__version__",
    "simulate",
    "SimulationConfig",
    "parameter_sweep",
    "IndicatorSpec",
    "TRAJECTORY_ONLY",
    "ORACLE",
    "state_to_scalar_observation",
    "add_observation_noise",
    "delay_embed",
    "observed_trajectory",
    "ObservedTrajectory",
    "save_json",
    "save_results_json",
    "save_trajectories_npz",
    "save_sweep_bundle",
    "ChangePointResult",
    "BootstrapChangePointResult",
    "SensitivityChangePointResult",
    "LeadDistanceResult",
    "centered_moving_average",
    "robust_standardize",
    "composite_structural_score",
    "piecewise_linear_change_point",
    "bootstrap_change_point_from_replicates",
    "change_point_sensitivity_scan",
    "first_threshold_crossing",
    "lead_distance",
    "RobustnessCase",
    "run_robustness_cases",
    "evaluate_robustness_summary",
    "PaperStudyContract",
    "default_paper_study_contract",
    "validate_metric_sources",
    "evaluate_study_contract",
    "run_study_suite",
    "capture_provenance",
    "make_trajectory_only_indicator_specs",
    "systems",
    "indicators",
    "oracles",
    "set_publication_style",
]
