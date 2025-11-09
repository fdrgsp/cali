"""SQLModel database schema and utilities for calcium imaging analysis.

This package provides a complete SQLModel-based database schema for storing
and querying calcium imaging analysis data. It includes models for hierarchical
data organization (Experiment → Plate → Well → FOV → ROI), utilities for
migrating existing JSON data to the database, and tools for visualization
and export.

Main Components
---------------
- Models: Experiment, Plate, Well, FOV, ROI, Condition, AnalysisSettings,
  Traces, DataAnalysis, Mask
- Migration: load_analysis_from_json, save_experiment_to_db
- Import: useq_plate_plan_to_plate
- Export: experiment_to_useq_plate, experiment_to_useq_plate_plan
- Visualization: print_experiment_tree, print_model_tree
"""

from ._db_to_plate_map import experiment_to_plate_map_data
from ._db_to_useq_plate import experiment_to_useq_plate, experiment_to_useq_plate_plan
from ._json_to_db import (
    load_analysis_from_json,
    load_experiment_from_db,
    save_experiment_to_db,
)
from ._models import (
    FOV,
    ROI,
    AnalysisSettings,
    Condition,
    DataAnalysis,
    Experiment,
    Mask,
    Plate,
    Traces,
    Well,
    WellCondition,
)
from ._useq_plate_to_db import useq_plate_plan_to_db, useq_plate_to_db
from ._util import (
    check_analysis_settings_consistency,
    create_db_and_tables,
    load_experiment_from_database,
)
from ._visualize_experiment import print_experiment_tree_from_engine, print_experiment_tree

__all__ = [
    "FOV",
    "ROI",
    "AnalysisSettings",
    "Condition",
    "DataAnalysis",
    "Experiment",
    "Mask",
    "Plate",
    "Traces",
    "Well",
    "WellCondition",
    "check_analysis_settings_consistency",
    "create_db_and_tables",
    "experiment_to_plate_map_data",
    "experiment_to_useq_plate",
    "experiment_to_useq_plate_plan",
    "load_analysis_from_json",
    "load_experiment_from_database",
    "load_experiment_from_db",
    "print_experiment_tree_from_engine",
    "print_experiment_tree",
    "save_experiment_to_db",
    "useq_plate_plan_to_db",
    "useq_plate_to_db",
]
