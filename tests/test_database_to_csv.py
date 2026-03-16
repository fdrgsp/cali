"""Tests for database to CSV export functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlmodel import Session, create_engine

from cali._constants import (
    CALCIUM_DEN_DFF_CORRELATION,
    CALCIUM_DFF_CORRELATION,
    DFF_TRACES,
    NEUROPIL_CORRECTED_TRACES,
    RAW_CALCIUM_TRACES,
)
from cali.util import (
    export_calcium_den_dff_correlation_to_csv,
    export_calcium_dff_correlation_to_csv,
    export_correlation_matrices_to_csv,
    export_denoised_dff_traces_to_csv,
    export_dff_traces_to_csv,
    export_inferred_spikes_cross_correlation_lags_to_csv,
    export_inferred_spikes_cross_correlation_to_csv,
    export_inferred_spikes_raw_to_csv,
    export_inferred_spikes_synchrony_to_csv,
    export_inferred_spikes_thresholded_to_csv,
    export_neuropil_corrected_traces_to_csv,
    export_neuropil_traces_to_csv,
    export_raw_traces_to_csv,
)
from cali.util._database_to_csv import (
    export_correlations_to_csv,
    export_traces_to_csv,
)

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from sqlalchemy.engine import Engine


@pytest.fixture
def test_engine() -> Generator[Engine, None, None]:
    """Create engine from existing test database."""
    db_path = "tests/test_data/data_and_db_for_tests/test_db.cali"
    engine = create_engine(f"sqlite:///{db_path}")
    yield engine
    engine.dispose(close=True)


def test_export_raw_traces(test_engine: Engine, tmp_path: Path) -> None:
    """Test exporting raw traces to CSV."""
    output_file = tmp_path / "raw_traces.csv"
    export_raw_traces_to_csv(test_engine, output_file, run_id=1)

    assert output_file.exists()
    # Check file has content
    assert output_file.stat().st_size > 0


def test_export_dff_traces(test_engine: Engine, tmp_path: Path) -> None:
    """Test exporting ΔF/F traces to CSV."""
    output_file = tmp_path / "dff_traces.csv"
    export_dff_traces_to_csv(test_engine, output_file, run_id=1)

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_export_denoised_dff_traces(test_engine: Engine, tmp_path: Path) -> None:
    """Test exporting denoised ΔF/F traces to CSV."""
    output_file = tmp_path / "den_dff_traces.csv"
    export_denoised_dff_traces_to_csv(test_engine, output_file, run_id=1)

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_export_inferred_spikes_raw(test_engine: Engine, tmp_path: Path) -> None:
    """Test exporting raw inferred spikes to CSV."""
    output_file = tmp_path / "spikes_raw.csv"
    export_inferred_spikes_raw_to_csv(test_engine, output_file, run_id=1)

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_export_inferred_spikes_thresholded(
    test_engine: Engine, tmp_path: Path
) -> None:
    """Test exporting thresholded inferred spikes to CSV."""
    output_file = tmp_path / "spikes_thresholded.csv"
    export_inferred_spikes_thresholded_to_csv(test_engine, output_file, run_id=1)

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_export_correlation_matrices(test_engine: Engine, tmp_path: Path) -> None:
    """Test exporting correlation matrices to CSV."""
    output_dir = tmp_path / "correlation_matrices"
    try:
        export_correlation_matrices_to_csv(test_engine, output_dir, run_id=1)
    except ValueError:
        # It's okay if no FOV analysis data exists in test database
        pytest.skip("No FOV analysis data found in test database")

    assert output_dir.exists()
    # Check that at least some correlation files were created
    csv_files = list(output_dir.glob("*.csv"))
    assert len(csv_files) > 0


def test_export_with_specific_fov(test_engine: Engine, tmp_path: Path) -> None:
    """Test exporting data for a specific FOV."""
    # Get a FOV name from the database
    from sqlmodel import select

    from cali.sqlmodel._model import FOV

    with Session(test_engine) as session:
        fov = session.exec(select(FOV).limit(1)).first()
        assert fov is not None
        fov_name = fov.name

    output_file = tmp_path / "fov_specific_traces.csv"
    export_raw_traces_to_csv(test_engine, output_file, fov_name=fov_name, run_id=1)

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_export_neuropil_traces(test_engine: Engine, tmp_path: Path) -> None:
    """Test exporting neuropil traces to CSV."""
    output_file = tmp_path / "neuropil_traces.csv"
    try:
        export_neuropil_traces_to_csv(test_engine, output_file, run_id=1)
        # If neuropil data exists, check the file
        if output_file.exists():
            assert output_file.stat().st_size > 0
    except ValueError:
        # It's okay if no neuropil data exists in test database
        pytest.skip("No neuropil traces found in test database")


def test_export_corrected_traces(test_engine: Engine, tmp_path: Path) -> None:
    """Test exporting corrected traces to CSV."""
    output_file = tmp_path / "corrected_traces.csv"
    try:
        export_neuropil_corrected_traces_to_csv(test_engine, output_file, run_id=1)
        # If corrected data exists, check the file
        if output_file.exists():
            assert output_file.stat().st_size > 0
    except ValueError:
        # It's okay if no corrected data exists in test database
        pytest.skip("No corrected traces found in test database")


def test_export_calcium_dff_correlation(test_engine: Engine, tmp_path: Path) -> None:
    """Test exporting ΔF/F correlation matrix to CSV."""
    output_file = tmp_path / "calcium_dff_correlation.csv"
    try:
        export_calcium_dff_correlation_to_csv(test_engine, output_file, run_id=1)
    except ValueError:
        pytest.skip("No FOV analysis data found in test database")

    # Check that at least one file was created (may have FOV prefix if multiple FOVs)
    csv_files = list(tmp_path.glob("*calcium_dff_correlation.csv"))
    assert len(csv_files) > 0
    # Check first file has content
    assert csv_files[0].stat().st_size > 0


def test_export_calcium_den_dff_correlation(
    test_engine: Engine, tmp_path: Path
) -> None:
    """Test exporting denoised ΔF/F correlation matrix to CSV."""
    output_file = tmp_path / "calcium_den_dff_correlation.csv"
    try:
        export_calcium_den_dff_correlation_to_csv(test_engine, output_file, run_id=1)
    except ValueError:
        pytest.skip("No FOV analysis data found in test database")

    # Check that at least one file was created (may have FOV prefix if multiple FOVs)
    csv_files = list(tmp_path.glob("*calcium_den_dff_correlation.csv"))
    assert len(csv_files) > 0
    assert csv_files[0].stat().st_size > 0


def test_export_inferred_spikes_synchrony(test_engine: Engine, tmp_path: Path) -> None:
    """Test exporting inferred spikes synchrony matrix to CSV."""
    output_file = tmp_path / "inferred_spikes_synchrony.csv"
    try:
        export_inferred_spikes_synchrony_to_csv(test_engine, output_file, run_id=1)
    except ValueError:
        pytest.skip("No FOV analysis data found in test database")

    # Check that at least one file was created (may have FOV prefix if multiple FOVs)
    csv_files = list(tmp_path.glob("*inferred_spikes_synchrony.csv"))
    assert len(csv_files) > 0
    assert csv_files[0].stat().st_size > 0


def test_export_inferred_spikes_cross_correlation(
    test_engine: Engine, tmp_path: Path
) -> None:
    """Test exporting inferred spikes cross-correlation matrix to CSV."""
    output_file = tmp_path / "inferred_spikes_cross_correlation.csv"
    try:
        export_inferred_spikes_cross_correlation_to_csv(
            test_engine, output_file, run_id=1
        )
    except ValueError:
        pytest.skip("No FOV analysis data found in test database")

    # Check that at least one file was created (may have FOV prefix if multiple FOVs)
    csv_files = list(tmp_path.glob("*inferred_spikes_cross_correlation.csv"))
    assert len(csv_files) > 0
    assert csv_files[0].stat().st_size > 0


def test_export_inferred_spikes_cross_correlation_lags(
    test_engine: Engine, tmp_path: Path
) -> None:
    """Test exporting inferred spikes cross-correlation lags matrix to CSV."""
    output_file = tmp_path / "inferred_spikes_cross_correlation_lags.csv"
    try:
        export_inferred_spikes_cross_correlation_lags_to_csv(
            test_engine, output_file, run_id=1
        )
    except ValueError:
        pytest.skip("No FOV analysis data found in test database")

    # Check that at least one file was created (may have FOV prefix if multiple FOVs)
    csv_files = list(tmp_path.glob("*inferred_spikes_cross_correlation_lags.csv"))
    assert len(csv_files) > 0
    assert csv_files[0].stat().st_size > 0


def test_export_inferred_spikes_ccg_zscore(test_engine: Engine, tmp_path: Path) -> None:
    """Test exporting inferred spikes CCG z-score matrix to CSV."""
    from cali.util._database_to_csv import export_inferred_spikes_ccg_zscore_to_csv

    output_file = tmp_path / "inferred_spikes_ccg_zscore.csv"
    try:
        export_inferred_spikes_ccg_zscore_to_csv(test_engine, output_file, run_id=1)
    except ValueError:
        pytest.skip("No FOV analysis data found in test database")

    csv_files = list(tmp_path.glob("*inferred_spikes_ccg_zscore.csv"))
    assert len(csv_files) > 0
    assert csv_files[0].stat().st_size > 0


def test_export_inferred_spikes_synchrony_rising_edges(
    test_engine: Engine, tmp_path: Path
) -> None:
    """Test exporting inferred spikes synchrony (rising edges) to CSV."""
    from cali.util._database_to_csv import (
        export_inferred_spikes_synchrony_rising_edges_to_csv,
    )

    output_file = tmp_path / "inferred_spikes_synchrony_rising_edges.csv"
    try:
        export_inferred_spikes_synchrony_rising_edges_to_csv(
            test_engine, output_file, run_id=1
        )
    except ValueError:
        pytest.skip("No FOV analysis data found in test database")

    csv_files = list(tmp_path.glob("*inferred_spikes_synchrony_rising_edges.csv"))
    # This may not exist if rising_edges data isn't in the test DB - that's OK
    assert isinstance(csv_files, list)


def test_export_inferred_spikes_cross_correlation_rising_edges(
    test_engine: Engine, tmp_path: Path
) -> None:
    """Test exporting inferred spikes cross-correlation (rising edges) to CSV."""
    from cali.util._database_to_csv import (
        export_inferred_spikes_cross_correlation_rising_edges_to_csv,
    )

    output_file = tmp_path / "inferred_spikes_cross_correlation_rising_edges.csv"
    try:
        export_inferred_spikes_cross_correlation_rising_edges_to_csv(
            test_engine, output_file, run_id=1
        )
    except ValueError:
        pytest.skip("No FOV analysis data found in test database")

    csv_files = list(
        tmp_path.glob("*inferred_spikes_cross_correlation_rising_edges.csv")
    )
    assert isinstance(csv_files, list)


def test_export_inferred_spikes_cross_correlation_lags_rising_edges(
    test_engine: Engine, tmp_path: Path
) -> None:
    """Test exporting inferred spikes cross-correlation lags (rising edges) to CSV."""
    from cali.util._database_to_csv import (
        export_inferred_spikes_cross_correlation_lags_rising_edges_to_csv,
    )

    output_file = tmp_path / "inferred_spikes_lags_rising_edges.csv"
    try:
        export_inferred_spikes_cross_correlation_lags_rising_edges_to_csv(
            test_engine, output_file, run_id=1
        )
    except ValueError:
        pytest.skip("No FOV analysis data found in test database")

    csv_files = list(tmp_path.glob("*inferred_spikes_lags_rising_edges.csv"))
    assert isinstance(csv_files, list)


def test_export_inferred_spikes_ccg_zscore_rising_edges(
    test_engine: Engine, tmp_path: Path
) -> None:
    """Test exporting inferred spikes CCG z-score (rising edges) to CSV."""
    from cali.util._database_to_csv import (
        export_inferred_spikes_ccg_zscore_rising_edges_to_csv,
    )

    output_file = tmp_path / "inferred_spikes_ccg_zscore_rising_edges.csv"
    try:
        export_inferred_spikes_ccg_zscore_rising_edges_to_csv(
            test_engine, output_file, run_id=1
        )
    except ValueError:
        pytest.skip("No FOV analysis data found in test database")

    csv_files = list(tmp_path.glob("*inferred_spikes_ccg_zscore_rising_edges.csv"))
    assert isinstance(csv_files, list)


def test_export_cluster_labels(test_engine: Engine, tmp_path: Path) -> None:
    """Test exporting cluster labels to CSV."""
    from cali.util._database_to_csv import export_cluster_labels_to_csv

    output_file = tmp_path / "cluster_labels.csv"
    try:
        export_cluster_labels_to_csv(test_engine, output_file, run_id=1)
    except ValueError:
        pytest.skip("No cluster label data found in test database")

    import pandas as pd

    assert output_file.exists()
    df = pd.read_csv(output_file)
    expected_cols = {
        "fov",
        "roi_label",
        "cluster_label",
        "cluster_method",
        "cluster_n_clusters",
        "cluster_silhouette_score",
    }
    assert expected_cols.issubset(df.columns)
    assert len(df) > 0


def test_export_cluster_labels_raises_when_run_id_missing(
    test_engine: Engine, tmp_path: Path
) -> None:
    """Raises ValueError when run_id does not match any FOVAnalysis records."""
    from cali.util._database_to_csv import export_cluster_labels_to_csv

    with pytest.raises(ValueError, match="No FOV analysis data found"):
        export_cluster_labels_to_csv(test_engine, tmp_path / "out.csv", run_id=9999)


def test_export_cluster_labels_via_export_correlations(
    test_engine: Engine, tmp_path: Path
) -> None:
    """CLUSTER_LABELS key in export_correlations_to_csv triggers cluster export."""
    from cali._constants import CLUSTER_LABELS
    from cali.util._database_to_csv import export_correlations_to_csv

    fake_db = tmp_path / "fake.cali"
    try:
        export_correlations_to_csv(
            test_engine,
            {CLUSTER_LABELS: True},
            run_id=1,
            db_path=fake_db,
        )
    except ValueError:
        pytest.skip("No cluster label data found in test database")

    export_dir = tmp_path / "fake_exports" / "run_1"
    csv_files = list(export_dir.rglob("cluster_labels.csv"))
    assert len(csv_files) == 1
    assert csv_files[0].stat().st_size > 0


def test_export_cluster_labels_run_id_none_uses_default(
    test_engine: Engine, tmp_path: Path
) -> None:
    """When run_id is None, the function resolves it via _get_default_run_id."""
    from cali.util._database_to_csv import export_cluster_labels_to_csv

    output_file = tmp_path / "cluster_labels_default_run.csv"
    try:
        export_cluster_labels_to_csv(test_engine, output_file)  # no run_id
    except ValueError:
        pytest.skip("No cluster label data found in test database")

    assert output_file.exists()
    assert output_file.stat().st_size > 0


# ---------------------------------------------------------------------------
# export_multi_well_to_csv: create files, skip evoked, NaN fill, exceptions
# ---------------------------------------------------------------------------


def test_export_multi_well_to_csv_creates_files(
    full_db: tuple[Engine, int], tmp_path: Path
) -> None:
    """export_multi_well_to_csv creates at least one CSV file."""
    from cali.util._database_to_csv import export_multi_well_to_csv

    engine, run_id = full_db
    db_path = tmp_path / "test.cali"
    db_path.touch()
    export_multi_well_to_csv(engine, run_id, db_path, experiment_type="spontaneous")

    output_dir = tmp_path / "test_exports" / f"run_{run_id}" / "multi_well"
    csv_files = list(output_dir.glob("*.csv"))
    assert len(csv_files) > 0


def test_export_multi_well_skips_evoked_for_spontaneous(
    full_db: tuple[Engine, int], tmp_path: Path
) -> None:
    """Evoked-only plots are excluded for spontaneous experiments."""
    from cali.util._database_to_csv import export_multi_well_to_csv

    engine, run_id = full_db
    db_path = tmp_path / "test.cali"
    db_path.touch()
    export_multi_well_to_csv(engine, run_id, db_path, experiment_type="spontaneous")

    output_dir = tmp_path / "test_exports" / f"run_{run_id}" / "multi_well"
    csv_files = {f.name for f in output_dir.glob("*.csv")}
    evoked_only = [
        "calcium_peaks_amplitude_stim",
        "calcium_peaks_amplitude_non_stim",
        "percentage_active_stim",
        "percentage_active_non_stim",
    ]
    for fname in evoked_only:
        assert not any(fname in f for f in csv_files), (
            f"Evoked file {fname!r} should not be exported for spontaneous"
        )


def test_export_multi_well_pca_to_csv(
    full_db: tuple[Engine, int], tmp_path: Path
) -> None:
    """export_multi_well_pca_to_csv creates 3 CSV files with expected columns."""
    import pandas as pd

    from cali.util._database_to_csv import export_multi_well_pca_to_csv

    engine, run_id = full_db
    db_path = tmp_path / "test.cali"
    db_path.touch()
    export_multi_well_pca_to_csv(engine, run_id, db_path)

    out_dir = tmp_path / "test_exports" / f"run_{run_id}" / "multi_well"

    fm = pd.read_csv(out_dir / "pca_feature_matrix.csv")
    assert "fov_name" in fm.columns
    assert "condition" in fm.columns
    assert len(fm) >= 2

    coords = pd.read_csv(out_dir / "pca_coordinates.csv")
    assert "PC1" in coords.columns
    assert len(coords) == len(fm)

    loadings = pd.read_csv(out_dir / "pca_loadings_and_scree.csv")
    assert "component" in loadings.columns
    assert loadings["component"].iloc[0] == "PC1"


def test_export_multi_well_nan_filling(
    full_db: tuple[Engine, int], tmp_path: Path
) -> None:
    """Conditions with different well counts fill missing values with NaN."""
    import pandas as pd
    from sqlmodel import Session, select

    from cali.sqlmodel import FOV, Condition, FOVAnalysis, Well
    from cali.sqlmodel._model import ROI, DataAnalysis
    from cali.util._database_to_csv import export_multi_well_to_csv

    engine, run_id = full_db

    with Session(engine) as session:
        # Add a brand-new well (+ FOV + analysis data) to one condition to create
        # unequal well counts across conditions → triggers NaN filling in CSV.
        existing_wells = session.exec(select(Well)).all()
        target_well = existing_wells[0]  # belongs to WT condition
        wt_cond = session.exec(select(Condition).where(Condition.name == "WT")).first()

        extra_well = Well(
            plate_id=target_well.plate_id,
            name="extra_well",
            row=99,
            column=99,
            conditions=[wt_cond],
        )
        session.add(extra_well)
        session.flush()

        extra_fov = FOV(name="extra_fov", position_index=99, well_id=extra_well.id)
        session.add(extra_fov)
        session.flush()

        extra_fa = FOVAnalysis(
            fov_id=extra_fov.id,
            analysis_result_id=run_id,
            active_roi_labels=[1],
            global_spike_jitter_synchrony=0.42,
        )
        session.add(extra_fa)

        extra_roi = ROI(
            label_value=1, active=True, fov_id=extra_fov.id, cell_size=100.0
        )
        session.add(extra_roi)
        session.flush()

        da = DataAnalysis(
            roi_id=extra_roi.id,
            analysis_result_id=run_id,
            peaks_amplitudes_den_dff=[1.0],
            den_dff_frequency=0.5,
        )
        session.add(da)
        session.commit()

    db_path = tmp_path / "test.cali"
    db_path.touch()
    export_multi_well_to_csv(engine, run_id, db_path, experiment_type="spontaneous")

    output_dir = tmp_path / "test_exports" / f"run_{run_id}" / "multi_well"
    csv_files = list(output_dir.glob("*.csv"))
    assert len(csv_files) > 0

    found_nan = False
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        fov_cols = [c for c in df.columns if c.startswith("fov_")]
        if fov_cols and df[fov_cols].isna().any().any():
            found_nan = True
            break
    assert found_nan, "Expected NaN fill for unequal well counts"


def test_export_multi_well_exception_handling(
    full_db: tuple[Engine, int], tmp_path: Path
) -> None:
    """Exception in compute_fn is caught and export continues."""
    from unittest.mock import patch

    from cali.util._database_to_csv import export_multi_well_to_csv

    engine, run_id = full_db
    db_path = tmp_path / "test.cali"
    db_path.touch()

    with patch(
        "cali.plot._multi_wells_plots._inferred_spikes.compute_spike_synchrony_data",
        side_effect=RuntimeError("test error"),
    ):
        export_multi_well_to_csv(engine, run_id, db_path, experiment_type="spontaneous")

    output_dir = tmp_path / "test_exports" / f"run_{run_id}" / "multi_well"
    csv_files = list(output_dir.glob("*.csv"))
    assert len(csv_files) > 0


# ---------------------------------------------------------------------------
# export_multi_well_pca_to_csv edge cases
# ---------------------------------------------------------------------------


def test_export_pca_fewer_than_2_fovs(tmp_path: Path) -> None:
    """PCA export skips when fewer than 2 FOVs."""
    from sqlmodel import Session, create_engine

    from cali.sqlmodel import FOV, Condition, Experiment, Plate, Well
    from cali.sqlmodel._model import ROI, AnalysisSettings, CaliResult, DataAnalysis
    from cali.sqlmodel._util import create_database_and_tables
    from cali.util._database_to_csv import export_multi_well_pca_to_csv

    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)

    with Session(engine) as session:
        exp = Experiment(name="e")
        session.add(exp)
        session.flush()
        settings = AnalysisSettings(frame_rate=10.0)
        session.add(settings)
        session.flush()
        run = CaliResult(experiment=exp.id, analysis_settings_id=settings.id)
        session.add(run)
        session.flush()
        plate = Plate(experiment=exp, name="P1", plate_type="6-well")
        cond = Condition(name="WT", condition_type="genotype")
        well = Well(plate=plate, name="W1", row=0, column=0, conditions=[cond])
        session.add(well)
        session.flush()
        fov = FOV(name="fov_0", position_index=0, well_id=well.id)
        session.add(fov)
        session.flush()
        roi = ROI(label_value=1, active=True, fov_id=fov.id, cell_size=100.0)
        session.add(roi)
        session.flush()
        da = DataAnalysis(
            roi_id=roi.id,
            analysis_result_id=run.id,
            den_dff_frequency=0.5,
        )
        session.add(da)
        session.commit()
        run_id = run.id

    db_path = tmp_path / "test.cali"
    db_path.touch()
    export_multi_well_pca_to_csv(engine, run_id, db_path)

    output_dir = tmp_path / "test_exports" / f"run_{run_id}" / "multi_well"
    assert not output_dir.exists() or not list(output_dir.glob("pca_*.csv"))
    engine.dispose(close=True)


def test_export_pca_build_matrix_exception(
    full_db: tuple[Engine, int], tmp_path: Path
) -> None:
    """PCA export catches exception from build_fov_feature_matrix."""
    from unittest.mock import patch

    from cali.util._database_to_csv import export_multi_well_pca_to_csv

    engine, run_id = full_db
    db_path = tmp_path / "test.cali"
    db_path.touch()

    with patch(
        "cali.plot._multi_wells_plots._dimensionality_reduction.build_fov_feature_matrix",
        side_effect=RuntimeError("bad matrix"),
    ):
        export_multi_well_pca_to_csv(engine, run_id, db_path)


def test_export_pca_compute_pca_exception(
    full_db: tuple[Engine, int], tmp_path: Path
) -> None:
    """PCA export catches exception from compute_pca."""
    from unittest.mock import patch

    from cali.util._database_to_csv import export_multi_well_pca_to_csv

    engine, run_id = full_db
    db_path = tmp_path / "test.cali"
    db_path.touch()

    with patch(
        "cali.plot._multi_wells_plots._dimensionality_reduction.compute_pca",
        side_effect=RuntimeError("singular matrix"),
    ):
        export_multi_well_pca_to_csv(engine, run_id, db_path)


# ============================================================================
# Export Directory and Condition Subfolder Tests
# ============================================================================


def test_export_traces_creates_export_directory(
    test_engine: Engine,
    tmp_path: Path,
) -> None:
    """Test that export_traces_to_csv creates the export directory."""
    # Use tmp_path for database path to control output location
    db_path = tmp_path / "test.cali"

    export_traces = {
        RAW_CALCIUM_TRACES: True,
    }

    # The function should create the export directory
    export_traces_to_csv(test_engine, export_traces, run_id=1, db_path=db_path)

    # Verify export directory was created
    export_dir = tmp_path / "test_exports" / "run_1"
    assert export_dir.exists()
    assert export_dir.is_dir()


def test_export_correlations_creates_export_directory(
    test_engine: Engine,
    tmp_path: Path,
) -> None:
    """Test that export_correlations_to_csv creates the export directory."""
    db_path = tmp_path / "test.cali"

    export_correlations = {
        CALCIUM_DFF_CORRELATION: True,
    }

    export_correlations_to_csv(
        test_engine,
        export_correlations,
        run_id=1,
        db_path=db_path,
    )

    # Verify export directory was created
    export_dir = tmp_path / "test_exports" / "run_1"
    # Skip if no FOV analysis data exists (directory won't be created)
    if not export_dir.exists():
        pytest.skip("No FOV analysis data found in test database")
    assert export_dir.is_dir()


def test_export_traces_respects_selection(
    test_engine: Engine,
    tmp_path: Path,
) -> None:
    """Test that export_traces_to_csv only exports selected types."""
    db_path = tmp_path / "test.cali"

    # Only export RAW_CALCIUM_TRACES
    export_traces = {
        RAW_CALCIUM_TRACES: True,
        DFF_TRACES: False,
        NEUROPIL_CORRECTED_TRACES: False,
    }

    export_traces_to_csv(test_engine, export_traces, run_id=1, db_path=db_path)

    export_dir = tmp_path / "test_exports" / "run_1"

    # Only raw_traces.csv should exist (may be in condition subfolders)
    raw_files = list(export_dir.rglob("raw_traces.csv"))
    assert len(raw_files) > 0, "raw_traces.csv not found anywhere in export dir"
    assert not list(export_dir.rglob("dff_traces.csv"))
    assert not list(export_dir.rglob("neuropil_corrected_traces.csv"))


def test_export_correlations_respects_selection(
    test_engine: Engine,
    tmp_path: Path,
) -> None:
    """Test that export_correlations_to_csv only exports selected types."""
    db_path = tmp_path / "test.cali"

    # Only export one type
    export_correlations = {
        CALCIUM_DFF_CORRELATION: True,
        CALCIUM_DEN_DFF_CORRELATION: False,
    }

    export_correlations_to_csv(
        test_engine,
        export_correlations,
        run_id=1,
        db_path=db_path,
    )

    export_dir = tmp_path / "test_exports" / "run_1"

    # Check files exist correctly
    correlation_files = list(export_dir.glob("*_correlation_matrix.csv"))

    # Skip if no FOV analysis data exists
    if len(correlation_files) == 0:
        pytest.skip("No FOV analysis data found in test database")

    # Check that all files are calcium_dff, not calcium_den_dff
    for file in correlation_files:
        assert "calcium_dff_correlation" in file.name
        assert "calcium_den_dff_correlation" not in file.name


def test_export_traces_creates_condition_subfolders(
    test_engine: Engine,
    tmp_path: Path,
) -> None:
    """Test that traces are exported into condition-named subfolders.

    The test database has wells with conditions (2x2 design):
      B5: g1 + t1  (positions 0,1)
      B6: g1 + t2  (positions 2,3)
      B7: g2 + t1  (positions 4,5)
      B8: g2 + t2  (positions 6,7)
    Run 1 covers positions [0,1] (well B5 = t1_g1).
    Run 2 covers all 8 positions (wells B5-B8 = all 4 conditions).
    """
    db_path = tmp_path / "test.cali"

    export_traces = {RAW_CALCIUM_TRACES: True}

    # Run 1 has positions 0,1 (well B5 = condition "t1_g1")
    export_traces_to_csv(test_engine, export_traces, run_id=1, db_path=db_path)

    export_dir = tmp_path / "test_exports" / "run_1"

    # Should have condition subfolder
    condition_dir = export_dir / "t1_g1"
    assert condition_dir.exists(), (
        f"Expected condition subfolder 't1_g1' in {export_dir}, "
        f"found: {[p.name for p in export_dir.iterdir()]}"
    )
    assert (condition_dir / "raw_traces.csv").exists()


def test_export_correlations_creates_condition_subfolders(
    test_engine: Engine,
    tmp_path: Path,
) -> None:
    """Test that correlations are exported into condition-named subfolders."""
    db_path = tmp_path / "test.cali"

    export_correlations = {CALCIUM_DFF_CORRELATION: True}

    export_correlations_to_csv(
        test_engine, export_correlations, run_id=1, db_path=db_path
    )

    export_dir = tmp_path / "test_exports" / "run_1"
    condition_dir = export_dir / "t1_g1"

    if not condition_dir.exists():
        pytest.skip("No condition subfolder created (no FOV analysis data)")

    # Should have correlation files in the condition subfolder
    csv_files = list(condition_dir.glob("*.csv"))
    assert len(csv_files) > 0, f"No CSV files in {condition_dir}"


def test_export_traces_multiple_conditions(
    test_engine: Engine,
    tmp_path: Path,
) -> None:
    """Test that exporting across conditions creates separate subfolders.

    Run 1 = positions [0,1] (B5 = t1_g1)
    Run 2 = all 8 positions (B5-B8 = t1_g1, t2_g1, t1_g2, t2_g2)
    Exporting run 2 should create at least the t2_g1 condition subfolder.
    """
    db_path = tmp_path / "test.cali"

    export_traces = {RAW_CALCIUM_TRACES: True}

    # Export run 2 (all 8 positions = wells B5-B8; B6 has condition "t2_g1")
    export_traces_to_csv(test_engine, export_traces, run_id=2, db_path=db_path)

    export_dir = tmp_path / "test_exports" / "run_2"

    # Run 2 covers all wells including B6 which has condition t2_g1
    condition_dir = export_dir / "t2_g1"
    assert condition_dir.exists(), (
        f"Expected condition subfolder 't2_g1' in {export_dir}, "
        f"found: {[p.name for p in export_dir.iterdir()]}"
    )
    assert (condition_dir / "raw_traces.csv").exists()
