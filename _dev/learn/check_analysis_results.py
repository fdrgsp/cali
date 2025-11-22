from sqlalchemy import create_engine
from sqlmodel import Session, select

from cali.sqlmodel._model import (
    FOV,
    ROI,
    AnalysisResult,
    AnalysisSettings,
    Experiment,
    Plate,
    Well,
)

engine = create_engine("sqlite:///analysis_results/evoked_experiment.db")

with Session(engine) as session:
    # Get all analysis results
    results = session.exec(select(AnalysisResult)).all()
    print(f"\n{'=' * 60}")
    print(f"ANALYSIS RESULTS ({len(results)} total)")
    print(f"{'=' * 60}")

    for i, result in enumerate(results, 1):
        exp = session.get(Experiment, result.experiment)
        settings = session.get(AnalysisSettings, result.analysis_settings)

        print(f"\nResult #{i}:")
        print(
            f"  Experiment ID: {result.experiment} - '{exp.name if exp else 'MISSING'}'"
        )
        print(
            f"  Settings ID: {result.analysis_settings} - dff_window={settings.dff_window if settings else 'MISSING'}"
        )
        print(f"  Positions: {result.positions_analyzed}")

        # Check if FOVs exist for this experiment
        if exp:
            plates = session.exec(
                select(Plate).where(Plate.experiment_id == exp.id)
            ).all()
            for plate in plates:
                wells = session.exec(
                    select(Well).where(Well.plate_id == plate.id)
                ).all()
                print(f"  Plate {plate.id}: {len(wells)} wells")
                for well in wells:
                    fovs = session.exec(select(FOV).where(FOV.well_id == well.id)).all()
                    print(f"    Well {well.id} ({well.name}): {len(fovs)} FOVs")
                    for fov in fovs:
                        rois = session.exec(
                            select(ROI).where(ROI.fov_id == fov.id)
                        ).all()
                        print(f"      FOV {fov.id} ({fov.name}): {len(rois)} ROIs")

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    experiments = session.exec(select(Experiment)).all()
    print(f"Total Experiments: {len(experiments)}")
    for exp in experiments:
        print(f"  Exp {exp.id}: {exp.name}")

    settings_list = session.exec(select(AnalysisSettings)).all()
    print(f"\nTotal Settings: {len(settings_list)}")
    for s in settings_list:
        print(f"  Settings {s.id}: dff_window={s.dff_window}")
