from sqlmodel import create_engine

from cali.sqlmodel import print_all_analysis_results

data_path = "/Volumes/T7 Shield/for FG/TSC_hSynLAM77_ACTX250730_D36/results.cali"
engine = create_engine(f"sqlite:///{data_path}")

print_all_analysis_results(engine, show_settings=False, max_experiment_level="fov")
