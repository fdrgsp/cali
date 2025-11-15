from sqlalchemy import create_engine, select

from cali.sqlmodel._model import FOV, ROI, Experiment, Plate, Well

engine = create_engine("sqlite:///analysis_results/cali_new.db")
from sqlmodel import Session

with Session(engine) as session:
    # Check full hierarchy
    experiments = list(session.exec(select(Experiment)).all())
    print(f"Total Experiments: {len(experiments)}")
    for exp in experiments:
        plates = list(
            session.exec(select(Plate).where(Plate.experiment_id == exp.id)).all()
        )
        if plates:
            for plate in plates:
                wells = list(
                    session.exec(select(Well).where(Well.plate_id == plate.id)).all()
                )
                print(f"\nExperiment {exp.id} ({exp.name}):")
                print(f"  Plate {plate.id}:")
                for well in wells:
                    fovs = list(
                        session.exec(select(FOV).where(FOV.well_id == well.id)).all()
                    )
                    print(f"    Well {well.id} ({well.name}): {len(fovs)} FOVs")
                    for fov in fovs:
                        rois = list(
                            session.exec(select(ROI).where(ROI.fov_id == fov.id)).all()
                        )
                        print(f"      FOV {fov.id} ({fov.name}): {len(rois)} ROIs")
