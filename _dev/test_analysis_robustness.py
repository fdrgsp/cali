"""Test analysis robustness with multiple runs and different settings.

This script tests the analysis pipeline with the data that's showing mask loading errors
to identify and fix the root cause.
"""

from pathlib import Path

from sqlmodel import Session, create_engine, select

from cali.runner import CaliRunner
from cali.sqlmodel._model import AnalysisSettings, DetectionSettings, FOV, ROI, Mask

# Test data paths
DATA_PATH = Path(
    "/Volumes/T7 Shield/for FG/TSC_hSynLAM77_ACTX250730_D36/"
    "TSC_hSynLAM77_ACTX250730_D36_DIV54_250923_jRCaMP1b_Spt.tensorstore.zarr"
)
DB_PATH = Path(
    "/Volumes/T7 Shield/for FG/TSC_hSynLAM77_ACTX250730_D36/results_test.cali"
)


def check_database_integrity(db_path: Path) -> None:
    """Check database for orphaned or corrupted mask data."""
    engine = create_engine(f"sqlite:///{db_path}")

    with Session(engine) as session:
        # Check for ROIs without masks
        stmt = select(ROI).where(ROI.roi_mask_id == None)  # noqa: E711
        rois_without_masks = session.exec(stmt).all()
        print(f"\n❌ ROIs without masks: {len(rois_without_masks)}")
        for roi in rois_without_masks[:5]:
            print(f"  ROI {roi.label_value} (id={roi.id}) has no mask")

        # Check for masks that exist but have no ROI
        stmt = select(Mask).outerjoin(ROI, ROI.roi_mask_id == Mask.id).where(ROI.id == None)  # noqa: E711
        orphaned_masks = session.exec(stmt).all()
        print(f"\n❌ Orphaned masks: {len(orphaned_masks)}")

        # Check for ROIs with mask_id but mask doesn't exist
        all_rois = session.exec(select(ROI)).all()
        missing_mask_count = 0
        for roi in all_rois:
            if roi.roi_mask_id is not None:
                mask_exists = session.get(Mask, roi.roi_mask_id)
                if mask_exists is None:
                    missing_mask_count += 1
                    print(
                        f"  ROI {roi.label_value} (id={roi.id}) has "
                        f"roi_mask_id={roi.roi_mask_id} but mask doesn't exist"
                    )
                    if missing_mask_count >= 5:
                        print("  ... (showing first 5)")
                        break

        print(f"\n❌ ROIs pointing to non-existent masks: {missing_mask_count}")

        # Check FOVs and ROI counts
        fovs = session.exec(select(FOV)).all()
        print(f"\n✅ Total FOVs: {len(fovs)}")
        for fov in fovs[:5]:
            roi_count = len(fov.rois) if fov.rois else 0
            print(f"  {fov.name}: {roi_count} ROIs")


def test_fresh_detection_and_analysis() -> None:
    """Test 1: Fresh detection + analysis (clean slate)."""
    print("\n" + "=" * 80)
    print("TEST 1: Fresh Detection + Analysis")
    print("=" * 80)

    # Remove old test database
    if DB_PATH.exists():
        DB_PATH.unlink()

    runner = CaliRunner()

    detection_settings = DetectionSettings(
        cellpose_model="cyto2",
        cellpose_diameter=60,
        cellpose_normalize=True,
    )

    analysis_settings = AnalysisSettings(
        neuropil_inner_radius=3,
        neuropil_min_pixels=10,
        neuropil_correction_factor=0.7,
    )

    # Run on positions 0 and 1
    for message in runner.run(
        data_path=DATA_PATH,
        db_path=DB_PATH,
        detection_settings=detection_settings,
        analysis_settings=analysis_settings,
        global_position_indices=[0, 1],
        threads=2,
    ):
        if isinstance(message, str):
            print(f"  {message}")

    print("\n✅ TEST 1 COMPLETE")
    check_database_integrity(DB_PATH)


def test_reanalysis_same_positions() -> None:
    """Test 2: Re-run analysis on same positions with different settings."""
    print("\n" + "=" * 80)
    print("TEST 2: Re-analysis with Different Settings (Same Positions)")
    print("=" * 80)

    runner = CaliRunner()

    # Use same detection settings (should skip detection)
    detection_settings = DetectionSettings(
        cellpose_model="cyto2",
        cellpose_diameter=60,
        cellpose_normalize=True,
    )

    # Different analysis settings
    analysis_settings = AnalysisSettings(
        neuropil_inner_radius=5,  # Different
        neuropil_min_pixels=20,  # Different
        neuropil_correction_factor=0.8,  # Different
        decay_constant=1.5,  # Different
    )

    for message in runner.run(
        data_path=DATA_PATH,
        db_path=DB_PATH,
        detection_settings=detection_settings,
        analysis_settings=analysis_settings,
        global_position_indices=[0, 1],
        threads=2,
    ):
        if isinstance(message, str):
            print(f"  {message}")

    print("\n✅ TEST 2 COMPLETE")
    check_database_integrity(DB_PATH)


def test_partial_positions() -> None:
    """Test 3: Run analysis on different positions."""
    print("\n" + "=" * 80)
    print("TEST 3: Analysis on Different Positions")
    print("=" * 80)

    runner = CaliRunner()

    detection_settings = DetectionSettings(
        cellpose_model="cyto2",
        cellpose_diameter=60,
        cellpose_normalize=True,
    )

    analysis_settings = AnalysisSettings(
        neuropil_inner_radius=3,
        neuropil_min_pixels=10,
    )

    # Run on position 1 only (0 was already done)
    for message in runner.run(
        data_path=DATA_PATH,
        db_path=DB_PATH,
        detection_settings=detection_settings,
        analysis_settings=analysis_settings,
        global_position_indices=[1],  # Only position 1
        threads=1,
    ):
        if isinstance(message, str):
            print(f"  {message}")

    print("\n✅ TEST 3 COMPLETE")
    check_database_integrity(DB_PATH)


def test_analysis_only_mode() -> None:
    """Test 4: Skip detection, only run analysis."""
    print("\n" + "=" * 80)
    print("TEST 4: Analysis-Only Mode (Skip Detection)")
    print("=" * 80)

    runner = CaliRunner()

    # Same detection settings (will skip)
    detection_settings = DetectionSettings(
        cellpose_model="cyto2",
        cellpose_diameter=60,
        cellpose_normalize=True,
    )

    # New analysis settings
    analysis_settings = AnalysisSettings(
        neuropil_inner_radius=0,  # Disable neuropil
        decay_constant=2.0,
        spike_threshold_value=0.5,
    )

    for message in runner.run(
        data_path=DATA_PATH,
        db_path=DB_PATH,
        detection_settings=detection_settings,
        analysis_settings=analysis_settings,
        global_position_indices=[0, 1],
        threads=2,
    ):
        if isinstance(message, str):
            print(f"  {message}")

    print("\n✅ TEST 4 COMPLETE")
    check_database_integrity(DB_PATH)


if __name__ == "__main__":
    if not DATA_PATH.exists():
        print(f"❌ Data path not found: {DATA_PATH}")
        print("Please update DATA_PATH in the script to point to your data.")
        exit(1)

    print("🧪 Running Analysis Robustness Tests")
    print(f"Data: {DATA_PATH.name}")
    print(f"Test DB: {DB_PATH}")

    try:
        test_fresh_detection_and_analysis()
        test_reanalysis_same_positions()
        test_partial_positions()
        test_analysis_only_mode()

        print("\n" + "=" * 80)
        print("🎉 ALL TESTS COMPLETED")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        check_database_integrity(DB_PATH)
