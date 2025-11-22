"""Unified runner interface for detection and analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cali.analysis import AnalysisRunner
from cali.detection import DetectionRunner

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cali.sqlmodel import AnalysisSettings, DetectionSettings, Experiment


class CaliRunner:
    """Unified runner for calcium imaging detection and analysis.
    
    This class provides a single interface for running both detection (ROI
    segmentation) and analysis (trace extraction) steps. It delegates to
    specialized runners internally while providing a cleaner API.
    
    Examples
    --------
    Run detection only:
    >>> runner = CaliRunner()
    >>> runner.run(
    ...     experiment=exp,
    ...     detection_settings=d_settings,
    ...     global_position_indices=[0, 1, 2]
    ... )
    
    Run detection + analysis:
    >>> runner.run(
    ...     experiment=exp,
    ...     detection_settings=d_settings,
    ...     analysis_settings=a_settings,
    ...     global_position_indices=[0, 1, 2]
    ... )
    
    Run analysis only (detection must exist):
    >>> runner.run(
    ...     experiment=exp,
    ...     detection_settings=d_settings,  # Specify which detection to use
    ...     analysis_settings=a_settings,
    ...     global_position_indices=[0, 1, 2],
    ...     skip_detection=True
    ... )
    """

    def __init__(self) -> None:
        """Initialize the unified runner."""
        self._detection_runner = DetectionRunner()
        self._analysis_runner = AnalysisRunner()

    def cancel(self) -> None:
        """Cancel both detection and analysis processes."""
        self._detection_runner.cancel()
        self._analysis_runner.cancel()

    def run(
        self,
        experiment: Experiment,
        detection_settings: DetectionSettings,
        global_position_indices: Sequence[int],
        analysis_settings: AnalysisSettings | None = None,
        skip_detection: bool = False,
        force: bool = False,
        overwrite: bool = False,
        echo: bool = False,
    ) -> None:
        """Run detection and/or analysis on the experiment.

        Parameters
        ----------
        experiment : Experiment
            Experiment to process
        detection_settings : DetectionSettings
            Detection parameters (required to specify which ROIs to analyze)
        global_position_indices : Sequence[int]
            Position indices to process
        analysis_settings : AnalysisSettings | None
            Analysis parameters. If None, only detection is run.
            If provided, both detection and analysis are run.
        skip_detection : bool
            If True, skip detection and only run analysis. Useful when detection
            already exists and you only want to run analysis with new settings.
            Default is False.
        force : bool
            If True (detection only), delete all existing analysis results using
            these detection settings and re-run detection. If False (default),
            skip detection if ROIs already exist for these positions.
        overwrite : bool
            Whether to overwrite existing database
        echo : bool
            Enable SQLAlchemy echo for database operations

        Raises
        ------
        ValueError
            If skip_detection=True but no ROIs exist for the specified detection
        """
        # Run detection if not skipped
        if not skip_detection:
            self._detection_runner.run(
                experiment=experiment,
                detection_settings=detection_settings,
                global_position_indices=global_position_indices,
                force=force,
                overwrite=overwrite,
                echo=echo,
            )

        # Run analysis if settings provided
        if analysis_settings is not None:
            self._analysis_runner.run(
                experiment=experiment,
                settings=analysis_settings,
                detection_settings=detection_settings,
                global_position_indices=global_position_indices,
                overwrite=overwrite,
                echo=echo,
            )
