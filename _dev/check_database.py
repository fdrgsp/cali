"""Check database structure and get analysis settings."""

from sqlmodel import Session, create_engine, select

from cali.sqlmodel._model import Experiment, AnalysisSettings, FOV

database_path = "/Users/fdrgsp/Desktop/cali/results_new.cali"

engine = create_engine(
    f"sqlite:///{database_path}",
    connect_args={"timeout": 30.0, "check_same_thread": False},
    pool_pre_ping=True,
)

with Session(engine) as session:
    # Get the experiment
    experiment = session.exec(select(Experiment)).first()
    if experiment:
        print(f"Experiment: {experiment.name}")
        print(f"Experiment ID: {experiment.id}")
        
        # Get analysis settings from the database
        analysis_settings = session.exec(select(AnalysisSettings)).first()
        if analysis_settings:
            print(f"\nAnalysis Settings found:")
            print(f"  Frame rate: {analysis_settings.frame_rate}")
            print(f"  CCG max lag: {analysis_settings.spikes_sync_cross_corr_lag}")
            print(f"  CCG n_shuffles: {analysis_settings.ccg_n_shuffles}")
            print(f"  Jitter window: {analysis_settings.spikes_sync_jitter_window}")
            print(f"  Rising edge: {analysis_settings.enable_rising_edge_analysis}")
        else:
            print("\nNo Analysis Settings found")
        
        # Get FOVs
        fovs = session.exec(select(FOV)).all()
        print(f"\nFOVs found ({len(fovs)}):")
        for fov in fovs[:20]:  # Limit to first 20
            num_active_rois = sum(1 for roi in fov.rois if roi.active)
            print(f"  - {fov.name} (position_index={fov.position_index}, active ROIs={num_active_rois})")
    else:
        print("No experiment found")

engine.dispose()
