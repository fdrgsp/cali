from sqlmodel import Session, create_engine, select

from cali.sqlmodel import FOV, Well

engine = create_engine("sqlite:///cali_cp.db")
session = Session(engine)

# Check FOVs
fovs = list(session.exec(select(FOV)).all())
print(f"Total FOVs in database: {len(fovs)}")
for fov in fovs:
    print(f"  FOV: {fov.name}, well_id: {fov.well_id}, pos_idx: {fov.position_index}")

# Check Wells
wells = list(session.exec(select(Well)).all())
print(f"\nTotal Wells in database: {len(wells)}")
for well in wells:
    print(f"  Well: {well.name}, id: {well.id}, fovs: {len(well.fovs)}")
    for fov in well.fovs:
        print(f"    - {fov.name}")

session.close()
