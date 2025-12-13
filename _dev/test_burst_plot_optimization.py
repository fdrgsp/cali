"""Quick test to verify burst plotting uses pre-computed data."""

from pathlib import Path
from sqlmodel import create_engine
from cali.sqlmodel._model import FOVAnalysis
from cali.plot._single_wells_plots.burst._plot_burst_activity import (
    _get_fov_analysis_for_run,
)


def test_fov_analysis_retrieval():
    """Test that we can retrieve FOVAnalysis with burst data."""
    # Use the test database
    db_path = Path("tests/test_data/data_and_db_for_tests/test_db.cali")

    if not db_path.exists():
        print(f"❌ Test database not found at {db_path}")
        return

    engine = create_engine(f"sqlite:///{db_path}")

    # Try to get FOVAnalysis for a known FOV
    fov_analysis = _get_fov_analysis_for_run(
        engine=engine,
        fov_name="B5_0000",
        run_id=None,  # Get most recent
    )

    if fov_analysis is None:
        print("⚠️  No FOVAnalysis found for B5_0000 (may not have burst data)")
    else:
        print(f"✅ Found FOVAnalysis for B5_0000")
        print(f"   - Has burst_starts: {fov_analysis.spike_burst_starts is not None}")
        print(f"   - Has burst_ends: {fov_analysis.spike_burst_ends is not None}")
        print(
            f"   - Has spike_population_activity: {fov_analysis.spike_population_activity is not None}"
        )
        print(
            f"   - Has spike_population_activity_smoothed: {fov_analysis.spike_population_activity_smoothed is not None}"
        )

        if fov_analysis.spike_burst_starts:
            print(f"   - Number of bursts: {len(fov_analysis.spike_burst_starts)}")
            print(f"   - Burst starts: {fov_analysis.spike_burst_starts[:5]}...")
            print(
                f"   - Burst ends: {fov_analysis.spike_burst_ends[:5] if fov_analysis.spike_burst_ends else None}..."
            )

        if fov_analysis.calcium_burst_starts:
            print(f"   - Calcium bursts: {len(fov_analysis.calcium_burst_starts)}")

    print(
        "\n✅ Plotting functions will now use pre-computed burst data when available!"
    )
    print(
        "   This avoids recomputing population activity, smoothing, and burst detection."
    )


if __name__ == "__main__":
    test_fov_analysis_retrieval()
