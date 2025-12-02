"""Debug the pipeline stage availability check."""

from sqlmodel import Session, create_engine, select, col

from cali.sqlmodel._model import FOV, ROI, Traces, DataAnalysis

# Test database
db_path = "tests/test_data/2pos/result_2pos.cali"
engine = create_engine(f"sqlite:///{db_path}")

fov_name = "Well1-Pos_000_000"
run_id = 1

print("=" * 80)
print(f"DEBUGGING QUERIES FOR FOV: {fov_name}, Run ID: {run_id}")
print("=" * 80)

with Session(engine) as session:
    # Check 1: ROIs (detection)
    print("\n1. Checking for ROIs (detection):")
    stmt = select(ROI.id).join(FOV).where(col(FOV.name) == fov_name).limit(1)
    result = session.exec(stmt).first()
    print(f"   Query: {stmt}")
    print(f"   Result: {result}")
    print(f"   Has detection: {bool(result)}")
    
    # Check 2: Traces (extraction)
    print("\n2. Checking for Traces (extraction):")
    stmt = (
        select(Traces.id)
        .join(ROI)
        .join(FOV)
        .where(col(FOV.name) == fov_name)
        .where(col(Traces.analysis_result_id) == run_id)
        .limit(1)
    )
    result = session.exec(stmt).first()
    print(f"   Result: {result}")
    print(f"   Has extraction: {bool(result)}")
    
    # Check 3: DataAnalysis (analysis)
    print("\n3. Checking for DataAnalysis (analysis):")
    stmt = (
        select(DataAnalysis.id)
        .join(ROI)
        .join(FOV)
        .where(col(FOV.name) == fov_name)
        .where(col(DataAnalysis.analysis_result_id) == run_id)
        .limit(1)
    )
    result = session.exec(stmt).first()
    print(f"   Result: {result}")
    print(f"   Has analysis: {bool(result)}")
    
    # Let's check what FOVs actually exist
    print("\n4. Available FOVs in database:")
    stmt = select(FOV.name)
    fovs = session.exec(stmt).all()
    for fov in fovs[:5]:
        print(f"   - {fov}")
    if len(fovs) > 5:
        print(f"   ... and {len(fovs) - 5} more")
    
    # Check what the actual FOV name is for position 0
    print("\n5. Checking FOV for position_index = 0:")
    stmt = select(FOV.name, FOV.position_index).where(col(FOV.position_index) == 0)
    result = session.exec(stmt).first()
    if result:
        print(f"   Found: name='{result[0]}', position_index={result[1]}")
    else:
        print("   No FOV found for position_index = 0")

engine.dispose(close=True)
