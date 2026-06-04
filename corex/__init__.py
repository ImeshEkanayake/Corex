"""COREX public API."""

from corex.explainers.coalition import CoalitionExplainer
from corex.explainers.kernel import KernelExplainer
from corex.explainers.linear import LinearExplainer
from corex.explainers.proxy import ProxyAuditor
from corex.explainers.quick import QuickExplainer
from corex.explainers.tree import TreeExplainer
from corex.games.game_theory import (
    BanzhafIndex,
    CooperativeGame,
    GameSolution,
    GameTheoreticScorer,
    Nucleolus,
    summarize_coalitions_by_size,
)
from corex.games.value_functions import (
    AblationValueFunction,
    LossReductionValueFunction,
    PredictionValueFunction,
    ProxyValueFunction,
)
from corex.solvers.core import CoreSolver
from corex.solvers.least_core import LeastCoreSolver
from corex.utils.explanation import ExplanationResult, ProxyAuditReport
from corex.utils.fairness import (
    compute_local_fairness_risk,
    mask_proxy_features,
    predictive_nucleolus_allocation,
    resolve_proxy_feature_indices,
)
from corex.utils.validation import (
    ALPHA_RELEASE_ITEMS,
    compare_explainers,
    compute_coalition_interactions,
    release_gate_status,
    validate_game_definition,
)
from corex.utils.benchmarking import benchmark_explainer

__all__ = [
    "AblationValueFunction",
    "BanzhafIndex",
    "CoalitionExplainer",
    "CooperativeGame",
    "CoreSolver",
    "ExplanationResult",
    "GameSolution",
    "GameTheoreticScorer",
    "KernelExplainer",
    "LeastCoreSolver",
    "LinearExplainer",
    "LossReductionValueFunction",
    "Nucleolus",
    "PredictionValueFunction",
    "ProxyAuditReport",
    "ProxyAuditor",
    "ProxyValueFunction",
    "QuickExplainer",
    "TreeExplainer",
    "ALPHA_RELEASE_ITEMS",
    "benchmark_explainer",
    "compare_explainers",
    "compute_local_fairness_risk",
    "compute_coalition_interactions",
    "mask_proxy_features",
    "predictive_nucleolus_allocation",
    "release_gate_status",
    "resolve_proxy_feature_indices",
    "summarize_coalitions_by_size",
    "validate_game_definition",
]

__version__ = "0.3.0"
