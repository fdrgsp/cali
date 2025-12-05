"""Analysis module for computing metrics from extracted calcium imaging traces.

This module provides tools for analyzing extracted trace data:
- AnalysisRunner: Main class for running analysis on FOVs with existing Traces
- Peak detection in deconvolved traces
- Inter-event interval (IEI) calculation
- Event frequency computation
- Amplitude extraction
- FOV-level correlation and synchrony matrices
"""

from ._analysis_runner import AnalysisRunner
from ._fov_analysis import compute_fov_analysis

__all__ = ["AnalysisRunner", "compute_fov_analysis"]
