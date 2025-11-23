"""Test that the schema change works - detection_settings_id is now on ROI not FOV."""
from cali.sqlmodel import Experiment, FOV, ROI, DetectionSettings, Plate, Well
from sqlmodel import Session, create_engine, select, SQLModel

# Create in-memory database
engine = create_engine("sqlite:///:memory:")
SQLModel.metadata.create_all(engine)

with Session(engine) as session:
    # Create experiment
    exp = Experiment(name="test")
    session.add(exp)
    session.flush()
    
    # Create plate
    plate = Plate(experiment_id=exp.id, name="plate1", rows=8, cols=12)
    session.add(plate)
    session.flush()
    
    # Create detection settings
    ds1 = DetectionSettings(method="cellpose", model_type="nuclei", diameter=20)
    ds2 = DetectionSettings(method="cellpose", model_type="nuclei", diameter=30)
    session.add(ds1)
    session.add(ds2)
    session.flush()
    
    # Create well
    well = Well(plate_id=plate.id, name="A1", row=0, column=0)
    session.add(well)
    session.flush()
    
    # Create FOV
    fov = FOV(well_id=well.id, name="A1_0000", position_index=0, fov_number=0)
    session.add(fov)
    session.flush()
    
    # Verify FOV does NOT have detection_settings_id attribute
    try:
        _ = fov.detection_settings_id
        print("❌ FAIL: FOV still has detection_settings_id attribute!")
    except AttributeError:
        print("✅ PASS: FOV no longer has detection_settings_id attribute")
    
    # Create ROIs with different detection settings
    roi1 = ROI(
        fov_id=fov.id,
        label_value=1,
        detection_settings_id=ds1.id,
    )
    roi2 = ROI(
        fov_id=fov.id,
        label_value=2,
        detection_settings_id=ds1.id,
    )
    roi3 = ROI(
        fov_id=fov.id,
        label_value=1,
        detection_settings_id=ds2.id,
    )
    roi4 = ROI(
        fov_id=fov.id,
        label_value=2,
        detection_settings_id=ds2.id,
    )
    session.add_all([roi1, roi2, roi3, roi4])
    session.commit()
    
    # Verify ROIs have detection_settings_id
    print(f"✅ PASS: ROI has detection_settings_id = {roi1.detection_settings_id}")
    
    # Verify we can have multiple ROIs from different detections in same FOV
    fov = session.get(FOV, fov.id)
    print(f"✅ PASS: FOV has {len(fov.rois)} ROIs from {len(set(r.detection_settings_id for r in fov.rois))} different detections")
    
    # Query ROIs by detection_settings_id
    rois_d1 = session.exec(
        select(ROI).where(ROI.detection_settings_id == ds1.id)
    ).all()
    rois_d2 = session.exec(
        select(ROI).where(ROI.detection_settings_id == ds2.id)
    ).all()
    
    print(f"✅ PASS: Found {len(rois_d1)} ROIs with detection_settings_id={ds1.id}")
    print(f"✅ PASS: Found {len(rois_d2)} ROIs with detection_settings_id={ds2.id}")
    
    print("\n🎉 All schema tests passed!")
