"""Example script to load an experiment from JSON files and save it to a database."""

import useq
from sqlalchemy import create_engine

from cali.sqlmodel import load_analysis_from_json
from cali.sqlmodel._visualize_experiment import (
    print_cali_results,
)

# Set paths for data, labels, and analysis directory
# data_path = (
#     "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk.tensorstore.zarr"
# )
# output_path = "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk_analysis"

data_path = "/Users/fdrgsp/Documents/git/cali/tests/test_data/spontaneous/spont.tensorstore.zarr"  # noqa: E501
output_path = (
    "/Users/fdrgsp/Documents/git/cali/tests/test_data/spontaneous/spont_analysis"
)

# Create useq.WellPlate that matches the experiment
plate = useq.WellPlate.from_str("96-well")

# Load experiment from JSON files and save to database (with AnalysisResult tracking)
experiment = load_analysis_from_json(str(data_path), str(output_path), plate)

# engine = create_engine(f"sqlite:///{output_path}/evk.tensorstore.zarr.db")
engine = create_engine(f"sqlite:///{output_path}/spont.tensorstore.zarr.db")
print_cali_results(engine)
