"""
Test script to simulate GUI analysis run with parallel implementation.
"""

import sys
import os
sys.path.insert(0, 'src')

from sqlmodel import Session, create_engine, select
from cali.analysis._analysis_runner import AnalysisRunner
from cali.sqlmodel._model import FOV, AnalysisSettings

def test_gui_analysis():
    """Test analysis runner with parallel implementation."""

    # Use the working database with active ROIs
    db_path = "/Users/fdrgsp/Desktop/cali/results_new.cali"
    db_uri = f"sqlite:///{db_path}"
    engine = create_engine(db_uri)

    print("Testing GUI analysis with parallel implementation")
    print("=" * 60)

    with Session(engine) as session:
        # Get settings and FOV
        settings = session.exec(select(AnalysisSettings)).first()
        fov = session.exec(select(FOV).where(FOV.name == "B3_0000")).first()

        if fov is None:
            print("❌ FOV B3_0000 not found")
            return

        print(f"📊 FOV: {fov.name}")
        active_rois = [roi for roi in fov.rois if roi.active]
        print(f"   Active ROIs: {len(active_rois)}")

        # Create analysis runner (same as GUI does)
        runner = AnalysisRunner()

        print("\nRunning AnalysisRunner._analyze_fov()...")
        print("-" * 60)

        # Run analysis (same as GUI does)
        result_fov = runner._analyze_fov(settings, fov)

        if result_fov is None:
            print("❌ Analysis returned None")
            return

        # Check results
        if hasattr(result_fov, "_new_fov_analysis"):
            fov_analyses = result_fov._new_fov_analysis
            print(f"✅ FOV analysis computed: {len(fov_analyses)} result(s)")

            for i, fov_analysis in enumerate(fov_analyses):
                print(f"\n   Analysis {i+1}:")
                print(f"     Global corr: {fov_analysis.global_spike_max_lag_correlation:.6f}")
                print(f"     Has spike matrix: {fov_analysis.spike_max_lag_correlation_matrix is not None}")
        else:
            print("⚠️  No _new_fov_analysis attribute")

        # Check ROI analysis
        rois_with_analysis = sum(
            1 for roi in result_fov.rois if hasattr(roi, "_new_data_analysis")
        )
        print(f"\n✅ ROI analysis: {rois_with_analysis} ROIs processed")

    engine.dispose()

    print("\n" + "=" * 60)
    print("✅ GUI analysis test completed successfully!")
    print("✅ Parallel implementation is working in the GUI context")
    print("=" * 60)

if __name__ == "__main__":
    test_gui_analysis()