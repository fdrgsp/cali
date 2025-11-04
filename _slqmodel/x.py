"""Quick example of using SQLModel for calcium imaging analysis.

This is a minimal example. See SQLMODEL_GUIDE.md for comprehensive documentation
and examples_sqlmodel_usage.py for detailed code examples.
"""

from __future__ import annotations

# Import the models from your new schema
from models import (
    FOV,
    ROI,
    Condition,
    Experiment,
    Plate,
    Well,
    create_tables,
)
from sqlmodel import Session, create_engine, select


def quick_example() -> None:
    """Quick example of creating and querying calcium imaging data."""
    # 1. Create database
    engine = create_engine("sqlite:///example.db")
    create_tables(engine)

    with Session(engine) as session:
        # 2. Create experiment hierarchy
        exp = Experiment(name="MyExperiment", description="Test experiment")
        session.add(exp)
        session.commit()
        session.refresh(exp)

        plate = Plate(experiment_id=exp.id, name="Plate1", plate_type="96-well")
        session.add(plate)
        session.commit()
        session.refresh(plate)

        # 3. Create conditions
        wt = Condition(name="WT", condition_type="genotype", color="blue")
        ko = Condition(name="KO", condition_type="genotype", color="red")
        session.add_all([wt, ko])
        session.commit()

        # 4. Create well with conditions
        well = Well(
            plate_id=plate.id,
            name="B5",
            row=1,
            column=4,
            condition_1_id=wt.id,
        )
        session.add(well)
        session.commit()
        session.refresh(well)

        # 5. Create FOV
        fov = FOV(well_id=well.id, name="B5_0000_p0", position_index=0)
        session.add(fov)
        session.commit()
        session.refresh(fov)

        # 6. Create ROI with analysis data
        roi = ROI(
            fov_id=fov.id,
            label_value=1,
            cell_size=100.5,
            cell_size_units="µm²",
            dec_dff_frequency=1.5,
            active=True,
            raw_trace=[100.0, 101.0, 102.0, 150.0, 103.0],  # Example trace
            dff=[0.0, 0.01, 0.02, 0.5, 0.03],
            elapsed_time_list_ms=[0.0, 33.3, 66.6, 100.0, 133.3],
        )
        session.add(roi)
        session.commit()

        print("✓ Created: Experiment → Plate → Well → FOV → ROI")

    # 7. Query data
    with Session(engine) as session:
        # Get all ROIs
        rois = session.exec(select(ROI)).all()
        print(f"✓ Total ROIs: {len(rois)}")

        # Get active ROIs with high frequency
        active_high_freq = session.exec(
            select(ROI).where(ROI.active, ROI.dec_dff_frequency > 1.0)
        ).all()
        print(f"✓ Active high-frequency ROIs: {len(active_high_freq)}")

        # Get ROIs by condition (with joins)
        wt_rois = session.exec(
            select(ROI)
            .join(FOV)
            .join(Well)
            .join(Condition, Well.condition_1_id == Condition.id)
            .where(Condition.name == "WT")
        ).all()
        print(f"✓ Wild-type ROIs: {len(wt_rois)}")

        # Access relationships
        for roi in rois:
            fov = roi.fov
            well = fov.well
            cond = well.condition_1
            print(
                f"  ROI {roi.label_value}: {roi.dec_dff_frequency} Hz, "
                f"condition={cond.name if cond else 'None'}"
            )


if __name__ == "__main__":
    quick_example()
