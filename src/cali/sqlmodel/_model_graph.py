from sqlalchemy_data_model_visualizer import generate_data_model_diagram

from cali.sqlmodel._model import (
    FOV,
    ROI,
    AnalysisResult,
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

models = [
    AnalysisResult,
    Experiment,
    AnalysisSettings,
    Plate,
    Condition,
    WellCondition,
    Well,
    FOV,
    ROI,
    Traces,
    DataAnalysis,
    Mask,
]  # list your table classes

generate_data_model_diagram(models, output_file="schema_diagram")
