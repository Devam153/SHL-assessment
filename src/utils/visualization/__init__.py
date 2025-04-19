
"""
Visualization utilities for displaying model evaluation results and recommendations
"""
from .benchmark_viz import plot_benchmark_comparison
from .catalog_viz import (
    plot_test_type_distribution,
    plot_remote_adaptive_support,
    plot_duration_distribution
)
from .recommendation_viz import display_recommendation_details
from .explanation_viz import add_search_method_explanation

__all__ = [
    'plot_benchmark_comparison',
    'plot_test_type_distribution',
    'plot_duration_distribution',
    'plot_remote_adaptive_support',
    'display_recommendation_details',
    'add_search_method_explanation'
]
